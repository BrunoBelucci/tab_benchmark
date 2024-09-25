import argparse
import os
import time
from pathlib import Path
from random import SystemRandom
import dask
from distributed import get_worker
import mlflow
from tab_benchmark.benchmark.base_experiment import BaseExperiment, log_and_print_msg
from tab_benchmark.benchmark.utils import set_mlflow_tracking_uri_check_if_exists, \
    get_search_algorithm_tune_config_run_config
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.models.dnn_models import max_epochs_dnn
from tab_benchmark.models.xgboost import n_estimators_gbdt
from tab_benchmark.utils import get_git_revision_hash, flatten_dict, extends


class HPOExperiment(BaseExperiment):
    @extends(BaseExperiment.__init__)
    def __init__(self, *args,
                 hpo_framework='optuna',
                 # general
                 n_trials=30, timeout_experiment=10 * 60 * 60, timeout_trial=2 * 60 * 60, max_concurrent_trials=1,
                 # optuna
                 sampler='tpe', pruner='hyperband',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hpo_framework = hpo_framework
        # general
        self.n_trials = n_trials
        self.timeout_experiment = timeout_experiment
        self.timeout_trial = timeout_trial
        self.max_concurrent_trials = max_concurrent_trials
        # optuna
        self.sampler = sampler
        self.pruner = pruner
        self.log_dir_dask = None

    def add_arguments_to_parser(self):
        super().add_arguments_to_parser()
        self.parser.add_argument('--hpo_framework', type=str, default=self.hpo_framework)
        # general
        self.parser.add_argument('--n_trials', type=int, default=self.n_trials)
        self.parser.add_argument('--timeout_experiment', type=int, default=self.timeout_experiment)
        self.parser.add_argument('--timeout_trial', type=int, default=self.timeout_trial)
        self.parser.add_argument('--max_concurrent_trials', type=int, default=self.max_concurrent_trials)
        # optuna
        self.parser.add_argument('--sampler', type=str, default=self.sampler)
        self.parser.add_argument('--pruner', type=str, default=self.pruner)

    def unpack_parser(self):
        args = super().unpack_parser()
        self.hpo_framework = args.hpo_framework
        # general
        self.n_trials = args.n_trials
        self.timeout_experiment = args.timeout_experiment
        self.timeout_trial = args.timeout_trial
        self.max_concurrent_trials = args.max_concurrent_trials
        # optuna
        self.sampler = args.sampler
        self.pruner = args.pruner

    def get_training_fn_for_hpo(self):
        def training_fn(config):
            # setup logger on ray worker
            config = config.copy()
            self.setup_logger(log_dir=self.log_dir_dask, filemode='a')
            parent_run_uuid = config.pop('parent_run_uuid', None)
            # n_jobs = 1, create_validation_set = False,
            # model_params = None,
            # fit_params = None, return_results = False, clean_output_dir = True,
            # parent_run_uuid = None,
            # experiment_name = None, mlflow_tracking_uri = None, check_if_exists = None,
            # fn_to_train_model = None
            results = super(HPOExperiment, self).run_mlflow_and_train_model(
                fn_to_train_model=BaseExperiment.train_model, **config)
            metrics_results = {metric: value for metric, value in results['evaluate_return'].items()
                               if metric.startswith('final_validation_') or metric.startswith('final_test_')}
            if parent_run_uuid:
                mlflow.log_metrics(metrics_results, step=int(time.time_ns()), run_id=parent_run_uuid)
            metrics_results['was_evaluated'] = True
            return metrics_results

        return training_fn

    def train_model(self,
                    n_jobs=1, create_validation_set=False,
                    model_params=None,
                    fit_params=None, return_results=False, clean_output_dir=True, log_to_mlflow=False,
                    # hpo parameters
                    hpo_framework='optuna', n_trials=5, timeout_experiment=5 * 60, timeout_trial=60,
                    max_concurrent_trials=1, sampler='tpe', pruner='hyperband',
                    **kwargs):

        model_nickname = kwargs.get('model_nickname')
        model_params = model_params if model_params else self.models_params.get(model_nickname, {}).copy()
        fit_params = fit_params if fit_params else self.fits_params.get(kwargs.get('model_nickname'), {}).copy()

        if hpo_framework == 'optuna':


        if logging_to_mlflow:
            storage_path = Path(mlflow.get_artifact_uri())
        else:
            storage_path = self.output_dir
        model_cls = models_dict[model_nickname][0]
        # check if model_cls is a model that has early stopping
        if model_cls.has_early_stopping:
            # metric is reported inside the model
            metric = 'validation_default'
            trainable = self.get_training_fn_for_hpo(is_openml=is_openml, has_early_stopping=True)
            last_metrics_actor = LastMetricsActor.remote()
            callbacks = [LastMetricsActorCallback(last_metrics_actor)]
            with_parameters = {'last_metrics_actor': last_metrics_actor}
        else:
            # metric is reported by the training function
            metric = 'final_validation_default'
            trainable = self.get_training_fn_for_hpo(is_openml=is_openml, has_early_stopping=False)
            callbacks = []
            with_parameters = {}
        mode = 'min'
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
            model_nickname=model_nickname,
        )
        default_param_space = dict(
            seed_model=seed_model,
            model_params=default_values,
        )
        data_params = kwargs.copy()  # hopefully it only rest them
        param_space.update(**data_params)
        if issubclass(model_cls, DNNModel):
            max_t = max_epochs_dnn
        else:
            max_t = n_estimators_gbdt
        search_algorithm, tune_config, run_config = get_search_algorithm_tune_config_run_config(
            default_param_space, search_algorithm_str, trial_scheduler_str, max_t, n_trials, timeout_experiment,
            timeout_trial, storage_path, metric, mode, seed_model, max_concurrent_trials, callbacks)
        trainable_with_parameters = tune.with_parameters(trainable, **with_parameters)
        trainable_with_resources = tune.with_resources(trainable_with_parameters, {'cpu': num_cpus, 'gpu': num_gpus})
        tuner = Tuner(trainable_with_resources, param_space=param_space,
                      tune_config=tune_config, run_config=run_config)
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
                               if metric.startswith('final_validation_') or metric.startswith('final_test_')}
        if not best_result_was_evaluated:
            # if early stopping was used, the best model may not have been evaluated on the final validation and test
            # sets, because of early stopping managed by the tuner scheduler,
            # so we retrain them from scratch
            # OBS.: We could start from the last saved model but:
            # This may lead to minor differences between training the entire model from scratch
            # for example, for xgboost, when we load the best model, the random state is not preserved, so the model
            # will build trees differently than the original model would have built
            # And we need to save as many models as the number of trials, which can be a lot
            # now we retrain the best model starting from the best model file
            config = best_result.metrics['config']
            config['fit_params']['report_to_ray'] = False
            parent_run_uuid = config.pop('parent_run_uuid', None)
            if is_openml:
                args = (config.pop('model_nickname'), config.pop('seed_model'), config.pop('task_id'),
                        config.pop('task_fold'), config.pop('task_repeat'), config.pop('task_sample'))
            else:
                args = (config.pop('model_nickname'), config.pop('seed_model'), config.pop('dataset_name_or_id'),
                        config.pop('seed_dataset'), config.pop('fold'))
            best_model_results = super(HPOExperiment, self).run_mlflow_and_train_model(*args,
                                                                                       create_validation_set=True,
                                                                                       return_results=True,
                                                                                       parent_run_uuid=parent_run_uuid,
                                                                                       is_openml=is_openml, **config)

            # change best_metric_results with new results
            best_metric_results = {f'best_{metric}': value for metric, value in best_model_results.items()
                                   if metric.startswith('final_validation_') or metric.startswith('final_test_')}

        if logging_to_mlflow:
            mlflow.log_params(best_params_and_seed)
            mlflow.log_metrics(best_metric_results)
        if retrain_best_model:
            # retrain without the validation set
            # for models that used early stopping, we will still create a validation set if auto_early_stopping is True,
            # so maybe it is not the best idea to retrain them
            # this is most useful for models that do not have early stopping and will benefit from a
            # training set with the validation set included
            config = best_result.metrics['config'].copy()
            config['fit_params']['report_to_ray'] = False
            results = super(HPOExperiment, self).run_mlflow_and_train_model(create_validation_set=False,
                                                                            return_results=True,
                                                                            parent_run_uuid=parent_run_uuid,
                                                                            is_openml=is_openml, **config)
            metrics_results = {f'final_{metric}': value for metric, value in results.items()
                               if metric.startswith('validation_') or metric.startswith('test_')}
            if logging_to_mlflow:
                mlflow.log_metrics(metrics_results)
        if return_results:
            return results
        else:
            return True

    def run_mlflow_and_train_model(self,
                                   n_jobs=1, create_validation_set=False,
                                   model_params=None,
                                   fit_params=None, return_results=False, clean_output_dir=True,
                                   parent_run_uuid=None,
                                   experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                                   # hpo parameters
                                   hpo_framework='optuna', n_trials=5, timeout_experiment=5 * 60, timeout_trial=60,
                                   max_concurrent_trials=1, sampler='tpe', pruner='hyperband',
                                   **kwargs):
        return super().run_mlflow_and_train_model(n_jobs=n_jobs, create_validation_set=create_validation_set,
                                                  model_params=model_params, fit_params=fit_params,
                                                  return_results=return_results, clean_output_dir=clean_output_dir,
                                                  parent_run_uuid=parent_run_uuid,
                                                  experiment_name=experiment_name,
                                                  mlflow_tracking_uri=mlflow_tracking_uri,
                                                  check_if_exists=check_if_exists,
                                                  hpo_framework=hpo_framework, n_trials=n_trials,
                                                  timeout_experiment=timeout_experiment, timeout_trial=timeout_trial,
                                                  max_concurrent_trials=max_concurrent_trials, sampler=sampler,
                                                  pruner=pruner,
                                                  **kwargs)

    def run_openml_task_combination(self, model_nickname, seed_model, task_id,
                                    task_fold=0, task_repeat=0, task_sample=0,
                                    n_jobs=1, create_validation_set=False,
                                    model_params=None,
                                    fit_params=None, return_results=False, clean_output_dir=True,
                                    log_to_mlflow=False, parent_run_uuid=None,
                                    experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                                    # hpo parameters
                                    hpo_framework='optuna', n_trials=5, timeout_experiment=5 * 60, timeout_trial=60,
                                    max_concurrent_trials=1, sampler='tpe', pruner='hyperband',
                                    **kwargs):
        return super().run_openml_task_combination(model_nickname, seed_model, task_id,
                                                   task_fold=task_fold, task_repeat=task_repeat,
                                                   task_sample=task_sample,
                                                   n_jobs=n_jobs, create_validation_set=create_validation_set,
                                                   model_params=model_params, fit_params=fit_params,
                                                   return_results=return_results, clean_output_dir=clean_output_dir,
                                                   log_to_mlflow=log_to_mlflow, parent_run_uuid=parent_run_uuid,
                                                   experiment_name=experiment_name,
                                                   mlflow_tracking_uri=mlflow_tracking_uri,
                                                   check_if_exists=check_if_exists,
                                                   hpo_framework=hpo_framework, n_trials=n_trials,
                                                   timeout_experiment=timeout_experiment, timeout_trial=timeout_trial,
                                                   max_concurrent_trials=max_concurrent_trials, sampler=sampler,
                                                   pruner=pruner,
                                                   **kwargs)

    def run_openml_dataset_combination(self, model_nickname, seed_model, dataset_name_or_id, seed_dataset,
                                       fold=0,
                                       resample_strategy='k-fold_cv', n_folds=10, pct_test=0.2,
                                       validation_resample_strategy='next_fold', pct_validation=0.1,
                                       n_jobs=1, create_validation_set=False,
                                       model_params=None,
                                       fit_params=None, return_results=False, clean_output_dir=True,
                                       log_to_mlflow=False, parent_run_uuid=None,
                                       experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                                       # hpo parameters
                                       hpo_framework='optuna', n_trials=5, timeout_experiment=5 * 60, timeout_trial=60,
                                       max_concurrent_trials=1, sampler='tpe', pruner='hyperband',
                                       **kwargs):
        return super().run_openml_dataset_combination(model_nickname, seed_model, dataset_name_or_id, seed_dataset,
                                                      fold=fold,
                                                      resample_strategy=resample_strategy, n_folds=n_folds,
                                                      pct_test=pct_test,
                                                      validation_resample_strategy=validation_resample_strategy,
                                                      pct_validation=pct_validation,
                                                      n_jobs=n_jobs, create_validation_set=create_validation_set,
                                                      model_params=model_params, fit_params=fit_params,
                                                      return_results=return_results, clean_output_dir=clean_output_dir,
                                                      log_to_mlflow=log_to_mlflow, parent_run_uuid=parent_run_uuid,
                                                      experiment_name=experiment_name,
                                                      mlflow_tracking_uri=mlflow_tracking_uri,
                                                      check_if_exists=check_if_exists,
                                                      hpo_framework=hpo_framework, n_trials=n_trials,
                                                      timeout_experiment=timeout_experiment,
                                                      timeout_trial=timeout_trial,
                                                      max_concurrent_trials=max_concurrent_trials, sampler=sampler,
                                                      pruner=pruner,
                                                      **kwargs)

    def get_combinations(self):
        combinations, extra_params = super().get_combinations()
        extra_params.update(dict(hpo_framework=self.hpo_framework, n_trials=self.n_trials,
                                 timeout_experiment=self.timeout_experiment, timeout_trial=self.timeout_trial,
                                 max_concurrent_trials=self.max_concurrent_trials, sampler=self.sampler,
                                 pruner=self.pruner))
        return combinations, extra_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOExperiment(parser=parser)
    experiment.run()
