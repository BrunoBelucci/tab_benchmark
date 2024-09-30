import argparse
import time
from functools import partial
from math import floor
import optuna
from optuna_integration import DaskStorage
from distributed import get_client, worker_client, get_worker
import mlflow
from tab_benchmark.benchmark.base_experiment import BaseExperiment
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.models.dnn_models import max_epochs_dnn
from tab_benchmark.models.xgboost import n_estimators_gbdt
from tab_benchmark.utils import extends


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

    def training_fn(self, config):
        log_to_mlflow = config.pop('log_to_mlflow', False)
        if log_to_mlflow:
            results = super(HPOExperiment, self).run_mlflow_and_train_model(
                fn_to_train_model=partial(BaseExperiment.train_model, self=self), **config)
            metrics_results = {metric: value for metric, value in results['evaluate_return'].items()
                               if metric.startswith('final_validation_') or metric.startswith('final_test_')}
            parent_run_uuid = config.get('parent_run_uuid', None)
            if parent_run_uuid:
                mlflow.log_metrics(metrics_results, step=int(time.time_ns()), run_id=parent_run_uuid)
        else:
            results = super(HPOExperiment, self).train_model(log_to_mlflow=False, **config)
            metrics_results = {metric: value for metric, value in results['evaluate_return'].items()
                               if metric.startswith('final_validation_') or metric.startswith('final_test_')}
        metrics_results['was_evaluated'] = True
        return metrics_results

    def train_model(self,
                    n_jobs=1, create_validation_set=True,
                    model_params=None,
                    fit_params=None, return_results=False, clean_output_dir=True, log_to_mlflow=False,
                    # hpo parameters
                    hpo_framework='optuna', n_trials=5, timeout_experiment=5 * 60, timeout_trial=60,
                    max_concurrent_trials=1, sampler='tpe', pruner='hyperband',
                    **kwargs):

        model_nickname = kwargs.get('model_nickname')
        model_params = model_params if model_params else self.models_params.get(model_nickname, {}).copy()
        fit_params = fit_params if fit_params else self.fits_params.get(kwargs.get('model_nickname'), {}).copy()
        seed_model = kwargs.get('seed_model')
        model_cls = self.models_dict[model_nickname][0]

        if hpo_framework == 'optuna':
            # sampler
            if sampler == 'tpe':
                sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=seed_model)
            elif sampler == 'random':
                sampler = optuna.samplers.RandomSampler(seed=seed_model)
            else:
                raise NotImplementedError(f'Sampler {sampler} not implemented for optuna')

            # pruner
            if pruner == 'hyperband':
                if issubclass(model_cls, DNNModel):
                    max_resources = max_epochs_dnn
                else:
                    max_resources = n_estimators_gbdt
                n_brackets = 5
                min_resources = 1
                # the following should give us the desired number of brackets
                reduction_factor = floor((max_resources / min_resources) ** (1 / (n_brackets - 1)))
                pruner = optuna.pruners.HyperbandPruner(min_resource=min_resources, max_resource=max_resources,
                                                        reduction_factor=reduction_factor)
            elif pruner == 'sha':
                pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=10)
            elif pruner is None:
                pruner = None
            else:
                raise NotImplementedError(f'Pruner {pruner} not implemented for optuna')

            # storage
            if self.dask_cluster_type is not None:
                client = get_client()
                storage = DaskStorage(client=client)
            else:
                storage = None

            # study
            study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, direction='minimize')

            # objective and search space (distribution)
            search_space, default_values = model_cls.create_search_space()
            search_space.update(model_params)
            search_space['seed_model'] = optuna.distributions.IntUniformDistribution(0, 10000)
            default_values['seed_model'] = seed_model
            if log_to_mlflow:
                parent_run_uuid = mlflow.active_run().info.run_id
            else:
                parent_run_uuid = None
            fit_params['report_to_optuna'] = True
            config = kwargs.copy()
            config.update(dict(
                # model_params and seed_model defined below
                n_jobs=n_jobs,
                create_validation_set=create_validation_set,
                fit_params=fit_params,
                return_results=True,
                clean_output_dir=clean_output_dir,
                log_to_mlflow=log_to_mlflow,
                parent_run_uuid=parent_run_uuid,
            ))
            study.enqueue_trial(default_values)

            # create mlflow runs
            run_uuids = []
            for _ in range(n_trials):
                if log_to_mlflow:
                    with mlflow.start_run(nested=True) as run:
                        run_uuids.append(run.info.run_id)
                else:
                    run_uuids.append(None)

            # run
            start_time = time.perf_counter()
            for n_trial in range(n_trials):
                if self.dask_cluster_type is not None:
                    with worker_client() as client:
                        futures = []
                        trial_numbers = []
                        for _ in range(max_concurrent_trials):
                            trial = study.ask(search_space)
                            trial_numbers.append(trial.number)
                            model_params = trial.params
                            seed_model = model_params.pop('seed_model')
                            config_trial = config.copy()
                            config_trial.update(dict(model_params=model_params, seed_model=seed_model))
                            config_trial['fit_params']['optuna_trial'] = trial
                            config_trial['run_uuid'] = run_uuids[n_trial]
                            resources = {'cores': n_jobs, 'gpus': self.n_gpus / (self.n_cores / n_jobs)}
                            # send task to workers different from the current one
                            workers = client.scheduler_info()['workers'].keys()
                            current_worker = get_worker().worker_address
                            workers = [worker for worker in workers if worker != current_worker]
                            futures.append(client.submit(self.training_fn, resources=resources, workers=workers,
                                                         **dict(config=config_trial)))
                        results = client.gather(futures)
                    for trial_number, result in zip(trial_numbers, results):
                        study_id = storage.get_study_id_from_name(study.study_name)
                        trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, trial_number)
                        storage.set_trial_user_attr(trial_id, 'was_evaluated', True)
                        for metric, value in result.items():
                            storage.set_trial_user_attr(trial_id, metric, value)
                        study.tell(trial_number, result['final_validation_default'])
                    elapsed_time = time.perf_counter() - start_time
                    if elapsed_time > timeout_experiment:
                        break
                else:
                    trial = study.ask(search_space)
                    model_params = trial.params
                    seed_model = model_params.pop('seed_model')
                    config_trial = config.copy()
                    config_trial.update(dict(model_params=model_params, seed_model=seed_model))
                    config_trial['fit_params']['optuna_trial'] = trial
                    results = self.training_fn(config_trial)
                    trial.set_user_attr('was_evaluated', True)
                    for metric, value in results.items():
                        trial.set_user_attr(metric, value)
                    study.tell(trial, results['final_validation_default'])
                    elapsed_time = time.perf_counter() - start_time
                    if elapsed_time > timeout_experiment:
                        break

            best_trial = study.best_trial
            best_model_params_and_seed = best_trial.params.copy()
            best_model_params_and_seed['seed_best_model'] = best_model_params_and_seed.pop('seed_model')
            best_metric_results = {f'best_{metric}': value for metric, value in best_trial.user_attrs.items()
                                   if metric.startswith('final_validation_') or metric.startswith('final_test_')}
            if log_to_mlflow:
                mlflow.log_params(best_model_params_and_seed, run_id=parent_run_uuid)
                mlflow.log_metrics(best_metric_results, run_id=parent_run_uuid)

            if return_results:
                return study
            else:
                return True
        else:
            raise NotImplementedError(f'HPO framework {hpo_framework} not implemented')

    def run_mlflow_and_train_model(self,
                                   n_jobs=1, create_validation_set=True,
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
                                    task_fold=0, task_repeat=0, task_sample=0, run_uuid=None,
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
                                                   log_to_mlflow=log_to_mlflow,
                                                   run_uuid=run_uuid, parent_run_uuid=parent_run_uuid,
                                                   experiment_name=experiment_name,
                                                   mlflow_tracking_uri=mlflow_tracking_uri,
                                                   check_if_exists=check_if_exists,
                                                   hpo_framework=hpo_framework, n_trials=n_trials,
                                                   timeout_experiment=timeout_experiment, timeout_trial=timeout_trial,
                                                   max_concurrent_trials=max_concurrent_trials, sampler=sampler,
                                                   pruner=pruner,
                                                   **kwargs)

    def run_openml_dataset_combination(self, model_nickname, seed_model, dataset_name_or_id, seed_dataset,
                                       fold=0, run_uuid=None,
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
                                                      log_to_mlflow=log_to_mlflow,
                                                      run_uuid=run_uuid, parent_run_uuid=parent_run_uuid,
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
                                 pruner=self.pruner, create_validation_set=True))
        return combinations, extra_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOExperiment(parser=parser)
    experiment.run()
