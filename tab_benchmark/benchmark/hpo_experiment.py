import argparse
import os
import time
from pathlib import Path
from typing import Optional
import ray
from ray.tune import Tuner, randint
import mlflow
from tab_benchmark.benchmark.base_experiment import BaseExperiment, log_and_print_msg
from tab_benchmark.benchmark.utils import treat_mlflow, get_search_algorithm_tune_config_run_config
from tab_benchmark.utils import get_git_revision_hash, flatten_dict, extends


class HPOExperiment(BaseExperiment):
    @extends(BaseExperiment.__init__)
    def __init__(self, *args, search_algorithm='random_search', n_trials=30, timeout_experiment=10 * 60 * 60,
                 timeout_trial=2 * 60 * 60, retrain_best_model=False, max_concurrent=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_algorithm = search_algorithm
        self.n_trials = n_trials
        self.timeout_experiment = timeout_experiment  # 10 hours
        self.timeout_trial = timeout_trial  # 2 hours
        self.retrain_best_model = retrain_best_model
        self.max_concurrent = max_concurrent

    def add_arguments_to_parser(self):
        super().add_arguments_to_parser()
        self.parser.add_argument('--search_algorithm', type=str, default=self.search_algorithm)
        self.parser.add_argument('--n_trials', type=int, default=self.n_trials)
        self.parser.add_argument('--timeout_experiment', type=int, default=self.timeout_experiment)
        self.parser.add_argument('--timeout_trial', type=int, default=self.timeout_trial)
        self.parser.add_argument('--retrain_best_model', action='store_true')
        self.parser.add_argument('--max_concurrent', type=int, default=self.max_concurrent)

    def unpack_parser(self):
        args = super().unpack_parser()
        self.search_algorithm = args.search_algorithm
        self.n_trials = args.n_trials
        self.timeout_experiment = args.timeout_experiment
        self.timeout_trial = args.timeout_trial
        self.retrain_best_model = args.retrain_best_model
        self.max_concurrent = args.max_concurrent

    def get_model(self, seed_model, model_params=None, n_jobs=1,
                  logging_to_mlflow=False, create_validation_set=False, output_dir: Optional[Path] = None):
        if output_dir is None:
            if ray.train._internal.session._get_session():
                # When doing HPO with ray we want the output_dir to be configured relative to the ray storage
                output_dir = Path.cwd() / self.output_dir.name
                os.makedirs(output_dir, exist_ok=True)
            # Else we use the default output_dir (artifact location if mlflow or self.output_dir)
        return super().get_model(seed_model, model_params, n_jobs, logging_to_mlflow, create_validation_set, output_dir)

    def get_training_fn_for_hpo(self, is_openml=True):
        def training_fn(config):
            # setup logger on ray worker
            self.setup_logger(log_dir=self.log_dir_dask, filemode='a')
            parent_run_uuid = config.pop('parent_run_uuid', None)
            results = super(HPOExperiment, self).run_combination_with_mlflow(
                create_validation_set=True, parent_run_uuid=parent_run_uuid, is_openml=is_openml, return_results=True,
                **config)
            metrics_results = {metric: value for metric, value in results.items()
                               if metric.startswith('validation_') or metric.startswith('test_')}
            if parent_run_uuid:
                mlflow.log_metrics(metrics_results, step=int(time.time_ns()), run_id=parent_run_uuid)
            metrics_results['was_evaluated'] = True
            return metrics_results

        return training_fn

    def run_combination_hpo(self, seed_model=0, n_jobs=1, create_validation_set=True,
                            model_params=None, is_openml=True, logging_to_mlflow=False,
                            fit_params=None, return_results=False, retrain_best_model=False, **kwargs):
        model_nickname = kwargs.pop('model_nickname', self.model_nickname)
        models_dict = kwargs.pop('models_dict', self.models_dict)
        mlflow_tracking_uri = kwargs.pop('mlflow_tracking_uri', self.mlflow_tracking_uri)
        experiment_name = kwargs.pop('experiment_name', self.experiment_name)
        parent_run_uuid = kwargs.pop('parent_run_uuid', None)
        search_algorithm_str = kwargs.pop('search_algorithm', self.search_algorithm)
        n_trials = kwargs.pop('n_trials', self.n_trials)
        timeout_experiment = kwargs.pop('timeout_experiment', self.timeout_experiment)
        timeout_trial = kwargs.pop('timeout_trial', self.timeout_trial)
        max_concurrent = kwargs.pop('max_concurrent', self.max_concurrent)
        if logging_to_mlflow:
            storage_path = mlflow.get_artifact_uri()
        else:
            storage_path = self.output_dir
        metric = 'validation_default'
        mode = 'min'
        trainable = self.get_training_fn_for_hpo(is_openml=is_openml)
        model_cls = models_dict[model_nickname][0]
        search_space, default_values = model_cls.create_search_space()
        model_params = model_params if model_params is not None else {}
        search_space.update(model_params)
        fit_params = fit_params.copy() if fit_params is not None else {}
        fit_params['report_to_ray'] = True
        param_space = dict(
            seed_model=randint(0, 10000),  # seed for model, seed_model will be passed to the search algorithm
            n_jobs=n_jobs,
            model_params=search_space,
            parent_run_uuid=parent_run_uuid,
            fit_params=fit_params,
        )
        default_param_space = dict(
            seed_model=seed_model,
            model_params=default_values,
        )
        param_space.update(kwargs)
        search_algorithm, tune_config, run_config = get_search_algorithm_tune_config_run_config(default_param_space,
                                                                                                search_algorithm_str,
                                                                                                n_trials,
                                                                                                timeout_experiment,
                                                                                                timeout_trial,
                                                                                                storage_path,
                                                                                                metric,
                                                                                                mode,
                                                                                                seed_model,
                                                                                                max_concurrent)
        tuner = Tuner(trainable=trainable, param_space=param_space, tune_config=tune_config,
                      run_config=run_config)
        # to test the trainable function uncomment the following 3 lines
        # param_space['model_params'] = default_values
        # param_space['seed_model'] = 0
        # trainable(param_space)
        # raise
        results = tuner.fit()
        best_result = results.get_best_result()
        best_result_was_evaluated = best_result.metrics.get('was_evaluated', False)
        best_params_and_seed = best_result.metrics['config']['model_params'].copy()
        best_params_and_seed['seed_best_model'] = best_result.metrics['config']['seed_model']
        best_metric_results = {f'best_{metric}': value for metric, value in best_result.metrics.items()
                               if metric.startswith('validation_') or metric.startswith('test_')}
        if not best_result_was_evaluated:
            # if early stopping was used, the best model may not have been evaluated on the final validation and test
            # sets, because of early stopping managed by the tuner scheduler,
            # so we retrain them starting from the best model file
            # OBS.: This may lead to minor differences between training the entire model from scratch
            # for example, for xgboost, when we load the best model, the random state is not preserved, so the model
            # will build trees differently than the original model would have built

            # first we get the best model file
            best_model_dir = Path(best_result.path) / self.output_dir.name
            # get saved models
            models = list(best_model_dir.glob('model_*'))
            # sort (hopefully) by iteration -> ok for xgboost, catboost, lightgbm
            models.sort(key=lambda f: int(f.stem.split("_")[-1]))
            # get the last model
            best_model_file = models[-1]

            # now we retrain the best model starting from the best model file
            config = best_result.metrics['config']
            config['fit_params']['report_to_ray'] = False
            config['fit_params']['init_model'] = best_model_file
            best_model_results = super(HPOExperiment, self).run_combination_with_mlflow(
                create_validation_set=True, parent_run_uuid=parent_run_uuid, is_openml=is_openml, return_results=True,
                **config)

            # change best_metric_results with new results
            best_metric_results = {f'best_{metric}': value for metric, value in best_model_results.items()
                                   if metric.startswith('validation_') or metric.startswith('test_')}

        if logging_to_mlflow:
            mlflow.log_params(best_params_and_seed)
            mlflow.log_metrics(best_metric_results)
        if retrain_best_model:
            # retrain without the validation set
            # for models that used early stopping, we will still create a validation set if auto_early_stopping is True,
            # so maybe it is not the best idea to retrain them
            # this is most useful for models that do not have early stopping and will benefit from a
            # training set with the validation set included
            config = best_result.metrics['config']
            config['fit_params']['report_to_ray'] = False
            results = super(HPOExperiment, self).run_combination_with_mlflow(
                create_validation_set=False, parent_run_uuid=parent_run_uuid, is_openml=is_openml, return_results=True,
                **config)
            metrics_results = {f'final_{metric}': value for metric, value in results.items()
                               if metric.startswith('validation_') or metric.startswith('test_')}
            if logging_to_mlflow:
                mlflow.log_metrics(metrics_results)
        if return_results:
            return results

    def run_combination_with_mlflow(self, seed_model=0, n_jobs=1, create_validation_set=True,
                                    model_params=None,
                                    parent_run_uuid=None, is_openml=True, return_results=False, **kwargs):
        # setup logger to local dir on dask worker (will not have effect if running from main)
        self.log_dir_dask = Path.cwd() / self.log_dir
        self.setup_logger(log_dir=self.log_dir_dask, filemode='w')

        model_params = model_params if model_params is not None else {}
        experiment_name = kwargs.pop('experiment_name', self.experiment_name)
        mlflow_tracking_uri = kwargs.pop('mlflow_tracking_uri', self.mlflow_tracking_uri)
        check_if_exists = kwargs.pop('check_if_exists', self.check_if_exists)
        model_nickname = kwargs.pop('model_nickname', self.model_nickname)
        search_algorithm = kwargs.pop('search_algorithm', self.search_algorithm)
        n_trials = kwargs.pop('n_trials', self.n_trials)
        timeout_experiment = kwargs.pop('timeout_experiment', self.timeout_experiment)
        timeout_trial = kwargs.pop('timeout_trial', self.timeout_trial)
        retrain_best_model = kwargs.pop('retrain_best_model', self.retrain_best_model)
        if not is_openml:
            kwargs.update({
                'resample_strategy': kwargs.pop('resample_strategy', self.resample_strategy),
                'n_folds': kwargs.pop('n_folds', self.k_folds),
                'pct_test': kwargs.pop('pct_test', self.pct_test),
                'validation_resample_strategy': kwargs.pop('validation_resample_strategy',
                                                           self.validation_resample_strategy),
                'pct_validation': kwargs.pop('pct_validation', self.pct_validation),
            })
        unique_params = dict(model_nickname=model_nickname, model_params=model_params, seed_model=seed_model,
                             search_algorithm=search_algorithm, n_trials=n_trials,
                             timeout_experiment=timeout_experiment, timeout_trial=timeout_trial, **kwargs)
        possible_existent_run, logging_to_mlflow = treat_mlflow(experiment_name, mlflow_tracking_uri, check_if_exists,
                                                                **unique_params)

        if possible_existent_run is not None:
            log_and_print_msg('Run already exists on MLflow. Skipping...')
            if return_results:
                return possible_existent_run.to_dict()
            else:
                return None

        if logging_to_mlflow:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name, artifact_location=str(self.output_dir))
            mlflow.set_experiment(experiment_name)
            run_name = '_'.join([f'{k}={v}' for k, v in unique_params.items()])
            if parent_run_uuid is not None:
                # check if parent run is active, if not start it
                possible_parent_run = mlflow.active_run()
                if possible_parent_run is not None:
                    if possible_parent_run.info.run_uuid != parent_run_uuid:
                        mlflow.start_run(run_id=parent_run_uuid, nested=True)
                else:
                    mlflow.start_run(parent_run_uuid)
                nested = True
            else:
                nested = False
            with mlflow.start_run(run_name=run_name, nested=nested) as run:
                parent_run_uuid = run.info.run_uuid
                mlflow.log_params(flatten_dict(unique_params))
                mlflow.log_param('git_hash', get_git_revision_hash())
                return self.run_combination_hpo(seed_model=seed_model, n_jobs=n_jobs,
                                                create_validation_set=create_validation_set,
                                                model_params=model_params,
                                                parent_run_uuid=parent_run_uuid, is_openml=is_openml,
                                                logging_to_mlflow=logging_to_mlflow,
                                                retrain_best_model=retrain_best_model, return_results=return_results,
                                                **kwargs)
        else:
            return self.run_combination_hpo(seed_model=seed_model, n_jobs=n_jobs,
                                            create_validation_set=create_validation_set,
                                            model_params=model_params,
                                            parent_run_uuid=parent_run_uuid, is_openml=is_openml,
                                            logging_to_mlflow=logging_to_mlflow,
                                            retrain_best_model=retrain_best_model, return_results=return_results,
                                            **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOExperiment(parser=parser)
    experiment.run()
