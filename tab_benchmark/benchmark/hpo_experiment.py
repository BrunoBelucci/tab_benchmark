import argparse
import logging
import time
from ray.tune import Tuner
import mlflow
from base_experiment import BaseExperiment
from tab_benchmark.benchmark.utils import treat_mlflow, get_search_algorithm_tune_config_run_config
from tab_benchmark.utils import get_git_revision_hash, flatten_dict


class HPOExperiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_algorithm = 'random_search'
        self.n_trials = 30
        self.timeout_experiment = 10 * 60 * 60  # 10 hours
        self.timeout_trial = 2 * 60 * 60  # 2 hours

    def add_arguments_to_parser(self):
        super().add_arguments_to_parser()
        self.parser.add_argument('--search_algorithm', type=str, default=self.search_algorithm)
        self.parser.add_argument('--n_trials', type=int, default=self.n_trials)
        self.parser.add_argument('--timeout_experiment', type=int, default=self.timeout_experiment)
        self.parser.add_argument('--timeout_trial', type=int, default=self.timeout_trial)

    def unpack_parser(self):
        args = super().unpack_parser()
        self.search_algorithm = args.search_algorithm
        self.n_trials = args.n_trials
        self.timeout_experiment = args.timeout_experiment
        self.timeout_trial = args.timeout_trial

    def get_training_fn_for_hpo(self, is_openml=True):
        def training_fn(config):
            mlflow_tracking_uri = config.pop('mlflow_tracking_uri', None)
            experiment_name = config.pop('experiment_name', None)
            parent_run_uuid = config.pop('parent_run_uuid', None)
            results = super(HPOExperiment, self).run_combination_with_mlflow(
                create_validation_set=True, return_to_fit=False, parent_run_uuid=parent_run_uuid, is_openml=is_openml,
                **config)
            metrics_results = {metric: value for metric, value in results.items()
                               if metric.startswith('validation_') or metric.startswith('test_')}
            if parent_run_uuid:
                mlflow.log_metrics(metrics_results, step=int(time.time_ns()), run_id=parent_run_uuid)
            return metrics_results

        return training_fn

    def run_combination_hpo(self, seed_model=0, n_jobs=1,
                            parent_run_uuid=None, is_openml=True, logging_to_mlflow=False, **kwargs):
        model_nickname = self.model_nickname
        models_dict = self.models_dict
        mlflow_tracking_uri = self.mlflow_tracking_uri
        experiment_name = self.experiment_name
        search_algorithm_str = kwargs.pop('search_algorithm', self.search_algorithm)
        n_trials = kwargs.pop('n_trials', self.n_trials)
        timeout_experiment = kwargs.pop('timeout_experiment', self.timeout_experiment)
        timeout_trial = kwargs.pop('timeout_trial', self.timeout_trial)
        storage_path = self.output_dir
        metric = 'validation_default'
        mode = 'min'
        trainable = self.get_training_fn_for_hpo(is_openml=is_openml)
        model_cls = models_dict[model_nickname][0]
        search_space, default_values = model_cls.create_search_space()
        param_space = dict(
            model_nickname=model_nickname,
            seed_model=seed_model,
            n_jobs=n_jobs,
            model_params=search_space,
            mlflow_tracking_uri=mlflow_tracking_uri,
            experiment_name=experiment_name,
            parent_run_uuid=parent_run_uuid,
        )
        param_space.update(kwargs)
        search_algorithm, tune_config, run_config = get_search_algorithm_tune_config_run_config(default_values,
                                                                                                search_algorithm_str,
                                                                                                search_space, n_trials,
                                                                                                timeout_experiment,
                                                                                                timeout_trial,
                                                                                                storage_path,
                                                                                                metric,
                                                                                                mode)
        tuner = Tuner(trainable=trainable, param_space=param_space, tune_config=tune_config,
                      run_config=run_config)
        # to test the trainable function uncomment the following 2 lines
        # param_space['model_params'] = default_values
        # trainable(param_space)
        results = tuner.fit()
        return results

    def run_combination_with_mlflow(self, seed_model=0, n_jobs=1, create_validation_set=True, return_to_fit=False,
                                    model_params=None,
                                    parent_run_uuid=None, is_openml=True, **kwargs):
        model_params = model_params if model_params is not None else {}
        experiment_name = kwargs.pop('experiment_name', self.experiment_name)
        mlflow_tracking_uri = kwargs.pop('mlflow_tracking_uri', self.mlflow_tracking_uri)
        check_if_exists = kwargs.pop('check_if_exists', self.check_if_exists)
        model_nickname = kwargs.pop('model_nickname', self.model_nickname)
        search_algorithm = kwargs.pop('search_algorithm', self.search_algorithm)
        n_trials = kwargs.pop('n_trials', self.n_trials)
        timeout_experiment = kwargs.pop('timeout_experiment', self.timeout_experiment)
        timeout_trial = kwargs.pop('timeout_trial', self.timeout_trial)
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
        exists, logging_to_mlflow = treat_mlflow(experiment_name, mlflow_tracking_uri, check_if_exists, **unique_params)

        if exists:
            msg = f"Experiment already exists on MLflow. Skipping..."
            print(msg)
            logging.info(msg)
            return None

        if logging_to_mlflow:
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
                                                return_to_fit=return_to_fit, model_params=model_params,
                                                parent_run_uuid=parent_run_uuid, is_openml=is_openml,
                                                logging_to_mlflow=logging_to_mlflow, **kwargs)
        else:
            return self.run_combination_hpo(seed_model=seed_model, n_jobs=n_jobs,
                                            create_validation_set=create_validation_set,
                                            return_to_fit=return_to_fit, model_params=model_params,
                                            parent_run_uuid=parent_run_uuid, is_openml=is_openml,
                                            logging_to_mlflow=logging_to_mlflow, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOExperiment(parser=parser)
    experiment.run()
