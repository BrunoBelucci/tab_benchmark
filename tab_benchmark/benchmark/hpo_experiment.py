import argparse
import time
from functools import partial
from math import floor
from typing import Optional
import optuna
import pandas as pd
from optuna_integration import DaskStorage
from distributed import get_client, worker_client
import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
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
        """HPO experiment.

        This class allows to perform experiments with HPO for tabular data.It allows to perform experiments with
        different models, datasets and
        resampling strategies. It also allows to log the results to mlflow and to parallelize the experiments with
        dask. We can also run a single experiment with the run_* meth

        Parameters
        ----------
        models_nickname :
            The nickname of the models to be used in the experiment. The nickname must be one of the keys of the
            models_dict.
        seeds_models :
            The seeds to be used in the models.
        n_jobs :
            Number of threads/cores to be used by the model if it supports it. Defaults to 1.
        models_params :
            Dictionary with the parameters of the models. The keys must be the nickname of the model and the values
            must be a dictionary with the parameters of the model. In case only one dictionary is passed, it will be
            used for all models. Defaults to None.
        fits_params :
            Dictionary with the parameters of the fits. The keys must be the nickname of the model and the values
            must be a dictionary with the parameters of the fit. In case only one dictionary is passed, it will be
            used for all models. Defaults to None.
        datasets_names_or_ids :
            The names or ids of the datasets to be used in the experiment. Defaults to None.
        seeds_datasets :
            The seeds to be used in the datasets. Defaults to None.
        resample_strategy :
            The resampling strategy to be used. Defaults to 'k-fold_cv'.
        k_folds :
            The number of folds to be used in the k-fold cross-validation. Defaults to 10.
        folds :
            The folds to be used in the resampling. Defaults to None.
        pct_test :
            The percentage of the test set. Defaults to 0.2.
        validation_resample_strategy :
            The resampling strategy to be used to create the validation set. Defaults to 'next_fold'.
        pct_validation :
            The percentage of the validation set. Defaults to 0.1.
        tasks_ids :
            The ids of the tasks to be used in the experiment. Defaults to None.
        task_repeats :
            The repeats to be used in the tasks. Defaults to None.
        task_folds :
            The folds to be used in the tasks. Defaults to None.
        task_samples :
            The samples to be used in the tasks. Defaults to None.
        experiment_name :
            The name of the experiment. Defaults to 'base_experiment'.
        create_validation_set :
            If True, create a validation set. Defaults to False.
        models_dict :
            The dictionary with the models to be used in the experiment, it must be a dictionary with the keys being
            the nickname of the model and the values being another dictionary with the class of the model and the
            parameters of the model.
        hpo_framework :
            The hyperparameter optimization framework to be used. It must be 'optuna' for the moment.
        n_trials :
            The number of trials to be run.
        timeout_experiment :
            The timeout of the experiment in seconds.
        timeout_trial :
            The timeout of each trial in seconds.
        max_concurrent_trials :
            The maximum number of concurrent trials that can be run.
        sampler :
            The sampler to be used in the hyperparameter optimization. It can be 'tpe' or 'random'.
        pruner :
            The pruner to be used in the hyperparameter optimization. It can be 'hyperband', 'sha' or None.
        """
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

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--hpo_framework', type=str, default=self.hpo_framework)
        # general
        self.parser.add_argument('--n_trials', type=int, default=self.n_trials)
        self.parser.add_argument('--timeout_experiment', type=int, default=self.timeout_experiment)
        self.parser.add_argument('--timeout_trial', type=int, default=self.timeout_trial)
        self.parser.add_argument('--max_concurrent_trials', type=int, default=self.max_concurrent_trials)
        # optuna
        self.parser.add_argument('--sampler', type=str, default=self.sampler)
        self.parser.add_argument('--pruner', type=str, default=self.pruner)

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.hpo_framework = args.hpo_framework
        # general
        self.n_trials = args.n_trials
        self.timeout_experiment = args.timeout_experiment
        self.timeout_trial = args.timeout_trial
        self.max_concurrent_trials = args.max_concurrent_trials
        # optuna
        self.sampler = args.sampler
        self.pruner = args.pruner

    def _training_fn(self, config):
        log_to_mlflow = config.pop('log_to_mlflow', False)
        if log_to_mlflow:
            results = super(HPOExperiment, self)._run_mlflow_and_train_model(
                fn_to_train_model=partial(BaseExperiment._train_model, self=self), **config)
            metrics_results = {metric: value for metric, value in results['evaluate_return'].items()
                               if metric.startswith('final_validation_') or metric.startswith('final_test_')}
            parent_run_id = config.get('parent_run_id', None)
            if parent_run_id:
                mlflow.log_metrics(metrics_results, step=int(time.time_ns()), run_id=parent_run_id)
        else:
            results = super(HPOExperiment, self)._train_model(log_to_mlflow=False, **config)
            metrics_results = {metric: value for metric, value in results['evaluate_return'].items()
                               if metric.startswith('final_validation_') or metric.startswith('final_test_')}
        metrics_results['was_evaluated'] = True
        return metrics_results

    def _get_optuna_config_trial(self, search_space, study, model_params, config, child_run_id):
        optuna_distributions_search_space = {}
        conditional_distributions_search_space = {}
        for name, value in search_space.items():
            if isinstance(value, optuna.distributions.BaseDistribution):
                optuna_distributions_search_space[name] = value
            else:
                conditional_distributions_search_space[name] = value
        trial = study.ask(optuna_distributions_search_space)
        conditional_params = {name: fn(trial) for name, fn
                              in conditional_distributions_search_space.items()}
        trial_model_params = trial.params
        trial_model_params.update(model_params.copy())
        seed_model = trial_model_params.pop('seed_model')
        config_trial = config.copy()
        config_trial.update(dict(model_params=trial_model_params, seed_model=seed_model,
                                 run_id=child_run_id))
        config_trial['fit_params']['optuna_trial'] = trial
        return trial, config_trial

    def _train_model(self,
                     n_jobs=1, create_validation_set=True,
                     model_params=None,
                     fit_params=None, return_results=False, clean_output_dir=True, log_to_mlflow=False, run_id=None,
                     # hpo parameters
                     hpo_framework='optuna', n_trials=5, timeout_experiment=5 * 60, timeout_trial=60,
                     max_concurrent_trials=1, sampler='tpe', pruner='hyperband',
                     **kwargs):
        try:
            results = {}
            start_time = time.perf_counter()
            model_nickname = kwargs.get('model_nickname')
            model_params = model_params if model_params else self.models_params.get(model_nickname, {}).copy()
            fit_params = fit_params if fit_params else self.fits_params.get(kwargs.get('model_nickname'), {}).copy()
            seed_model = kwargs.get('seed_model')
            model_cls = self.models_dict[model_nickname][0]
            if hasattr(model_cls, 'has_early_stopping'):
                model_params['max_time'] = timeout_trial

            if hpo_framework == 'optuna':
                # sampler
                if sampler == 'tpe':
                    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=seed_model)
                elif sampler == 'random':
                    sampler = optuna.samplers.RandomSampler(seed=seed_model)
                else:
                    raise NotImplementedError(f'Sampler {sampler} not implemented for optuna')
                results['sampler'] = sampler

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
                results['pruner'] = pruner

                # storage
                if self.dask_cluster_type is not None:
                    client = get_client()
                    storage = DaskStorage(client=client)
                else:
                    storage = None
                results['storage'] = storage

                # study
                study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, direction='minimize')
                results['study'] = study

                # objective and search space (distribution)
                search_space, default_values = model_cls.create_search_space()
                search_space['seed_model'] = optuna.distributions.IntUniformDistribution(0, 10000)
                default_values['seed_model'] = seed_model
                if log_to_mlflow:
                    parent_run_id = run_id
                    parent_run = mlflow.get_run(parent_run_id)
                    child_runs = parent_run.data.tags
                    child_runs_ids = [child_run_id for key, child_run_id in child_runs.items()
                                      if key.startswith('child_run_id_')]
                else:
                    parent_run_id = None
                    child_runs_ids = [None for _ in range(n_trials)]
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
                ))
                study.enqueue_trial(default_values)

                # run
                n_trial = 0
                first_trial = True
                while n_trial < n_trials:
                    if self.dask_cluster_type is not None:
                        with worker_client() as client:
                            futures = []
                            trial_numbers = []
                            for _ in range(max_concurrent_trials):
                                trial, config_trial = self._get_optuna_config_trial(search_space, study, model_params,
                                                                                    config, child_run_id=child_runs_ids[
                                        n_trial])
                                trial_numbers.append(trial.number)
                                resources = {'cores': n_jobs, 'gpus': self.n_gpus / (self.n_cores / n_jobs)}
                                key = '_'.join([str(value) for value in kwargs.values()])  # shared prefix
                                key = key + f'-{config_trial["run_id"]}'  # unique key (child_run_id)
                                futures.append(
                                    client.submit(self._training_fn, resources=resources, key=key, pure=False,
                                                  **dict(config=config_trial)))
                                n_trial += 1
                                if n_trial >= n_trials or first_trial:
                                    # we have already enqueued all the trials, or it is the first trial,
                                    # and we want to run it before the others
                                    first_trial = False
                                    break
                            results = client.gather(futures)
                            for future in futures:
                                future.release()
                        for trial_number, result in zip(trial_numbers, results):
                            study_id = storage.get_study_id_from_name(study.study_name)
                            trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, trial_number)
                            storage.set_trial_user_attr(trial_id, 'was_evaluated', True)
                            for metric, value in result.items():
                                storage.set_trial_user_attr(trial_id, metric, value)
                            study.tell(trial_number, result['final_validation_reported'])
                        elapsed_time = time.perf_counter() - start_time
                        if elapsed_time > timeout_experiment:
                            break
                    else:
                        trial, config_trial = self._get_optuna_config_trial(search_space, study, model_params, config,
                                                                            child_run_id=child_runs_ids[n_trial])
                        results = self._training_fn(config_trial)
                        trial.set_user_attr('was_evaluated', True)
                        for metric, value in results.items():
                            trial.set_user_attr(metric, value)
                        study.tell(trial, results['final_validation_reported'])
                        n_trial += 1
                        elapsed_time = time.perf_counter() - start_time
                        if elapsed_time > timeout_experiment:
                            break

                best_trial = study.best_trial
                best_model_params_and_seed = best_trial.params.copy()
                best_model_params_and_seed['seed_best_model'] = best_model_params_and_seed.pop('seed_model')
                best_metric_results = {f'best_{metric}': value for metric, value in best_trial.user_attrs.items()
                                       if metric.startswith('final_validation_') or metric.startswith('final_test_')}
                if log_to_mlflow:
                    mlflow.log_params(best_model_params_and_seed, run_id=parent_run_id)
                    mlflow.log_metrics(best_metric_results, run_id=parent_run_id)

            else:
                raise NotImplementedError(f'HPO framework {hpo_framework} not implemented')
        except Exception as exception:
            total_time = time.perf_counter() - start_time
            if log_to_mlflow:
                log_tags = {'was_evaluated': False, 'EXCEPTION': str(exception)}
                mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
                for tag, value in log_tags.items():
                    mlflow_client.set_tag(run_id, tag, value)
                log_metrics = {'elapsed_time': total_time}
                mlflow.log_metrics(log_metrics, run_id=run_id)  # run_id should be the same as parent_run_id
                mlflow_client.set_terminated(run_id, status='FAILED')
            if self.raise_on_fit_error:
                raise exception
            if return_results:
                return results
            else:
                return False
        else:
            total_time = time.perf_counter() - start_time
            if log_to_mlflow:
                log_tags = {'was_evaluated': True}
                mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
                for tag, value in log_tags.items():
                    mlflow_client.set_tag(run_id, tag, value)
                log_metrics = {'elapsed_time': total_time}
                mlflow.log_metrics(log_metrics, run_id=run_id)
                mlflow_client.set_terminated(parent_run_id, status='FINISHED')
            if return_results:
                return study
            else:
                return True

    def _run_mlflow_and_train_model(self,
                                    n_jobs=1, create_validation_set=True,
                                    model_params=None,
                                    fit_params=None, return_results=False, clean_output_dir=True,
                                    run_id=None,
                                    experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                                    # hpo parameters
                                    hpo_framework='optuna', n_trials=5, timeout_experiment=5 * 60, timeout_trial=60,
                                    max_concurrent_trials=1, sampler='tpe', pruner='hyperband',
                                    **kwargs):
        return super()._run_mlflow_and_train_model(n_jobs=n_jobs, create_validation_set=create_validation_set,
                                                   model_params=model_params, fit_params=fit_params,
                                                   return_results=return_results, clean_output_dir=clean_output_dir,
                                                   experiment_name=experiment_name,
                                                   mlflow_tracking_uri=mlflow_tracking_uri,
                                                   check_if_exists=check_if_exists, run_id=run_id,
                                                   hpo_framework=hpo_framework, n_trials=n_trials,
                                                   timeout_experiment=timeout_experiment, timeout_trial=timeout_trial,
                                                   max_concurrent_trials=max_concurrent_trials, sampler=sampler,
                                                   pruner=pruner, **kwargs)

    def run_openml_task_combination(self, model_nickname, seed_model, task_id,
                                    task_fold=0, task_repeat=0, task_sample=0, run_id=None,
                                    n_jobs=1, create_validation_set=False,
                                    model_params=None,
                                    fit_params=None, return_results=False, clean_output_dir=True,
                                    log_to_mlflow=False,
                                    experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                                    # hpo parameters
                                    hpo_framework='optuna', n_trials=5, timeout_experiment=5 * 60, timeout_trial=60,
                                    max_concurrent_trials=1, sampler='tpe', pruner='hyperband',
                                    **kwargs):
        """Run the experiment using an OpenML task.

        This function can be used to run the experiment using an OpenML task in an interactive way, without the need to
        run the experiment.

        Parameters
        ----------
        model_nickname :
            The nickname of the model to be used in the experiment. It must be a key of the models_dict attribute.
        seed_model :
            The seed of the model to be used in the experiment.
        task_id :
            The id of the OpenML task.
        task_fold :
            The fold of the OpenML task.
        task_repeat :
            The repeat of the OpenML task.
        task_sample :
            The sample of the OpenML task.
        run_id :
            The run_id of the mlflow run.
        n_jobs :
            Number of threads/cores to be used by the model if it supports it.
        create_validation_set :
            If True, create a validation set.
        model_params :
            Dictionary with the parameters of the model. Defaults models_params defined at the initialization of the
            class if None.
        fit_params :
            Dictionary with the parameters of the fit. Defaults fits_params defined at the initialization of the
            class if None.
        return_results :
            If True, return the results of the experiment.
        clean_output_dir :
            If True, clean the output directory after the experiment.
        log_to_mlflow :
            If True, log the results to mlflow.
        experiment_name :
            The name of the experiment.
        mlflow_tracking_uri :
            The uri of the mlflow tracking server.
        check_if_exists :
            If True, check if the run already exists on mlflow.
        hpo_framework :
            The hyperparameter optimization framework to be used. It must be 'optuna' for the moment.
        n_trials :
            The number of trials to be run.
        timeout_experiment :
            The timeout of the experiment in seconds.
        timeout_trial :
            The timeout of each trial in seconds.
        max_concurrent_trials :
            The maximum number of concurrent trials that can be run.
        sampler :
            The sampler to be used in the hyperparameter optimization. It can be 'tpe' or 'random'.
        pruner :
            The pruner to be used in the hyperparameter optimization. It can be 'hyperband', 'sha' or None.
        kwargs :
            Additional arguments from the experiment.

        Returns
        -------
        results :
            If return_results is True, return a dictionary with the results of the experiment, otherwise return True
            if the experiment was successful and False otherwise. It still tries to return the results when
            return_results is true and an exception is raised. The dictionary contains the following
            keys:
            data_return :
                The data returned by the load_data method.
            model :
                The model returned by the get_model method.
            metrics :
                The metrics returned by the get_metrics method.
            report_metric :
                The report_metric returned by the get_metrics method.
            fit_return :
                The data returned by the fit_model method.
            evaluate_return :
                The data returned by the evaluate_model method.
        """
        if create_validation_set is False:
            raise NotImplementedError('HPOExperiment requires a validation set, please set create_validation_set=True')
        return super().run_openml_task_combination(model_nickname, seed_model, task_id,
                                                   task_fold=task_fold, task_repeat=task_repeat,
                                                   task_sample=task_sample, run_id=run_id,
                                                   n_jobs=n_jobs, create_validation_set=create_validation_set,
                                                   model_params=model_params, fit_params=fit_params,
                                                   return_results=return_results, clean_output_dir=clean_output_dir,
                                                   log_to_mlflow=log_to_mlflow,
                                                   experiment_name=experiment_name,
                                                   mlflow_tracking_uri=mlflow_tracking_uri,
                                                   check_if_exists=check_if_exists, hpo_framework=hpo_framework,
                                                   n_trials=n_trials, timeout_experiment=timeout_experiment,
                                                   timeout_trial=timeout_trial,
                                                   max_concurrent_trials=max_concurrent_trials, sampler=sampler,
                                                   pruner=pruner, **kwargs)

    def run_openml_dataset_combination(self, model_nickname, seed_model, dataset_name_or_id, seed_dataset,
                                       fold=0, run_id=None,
                                       resample_strategy='k-fold_cv', n_folds=10, pct_test=0.2,
                                       validation_resample_strategy='next_fold', pct_validation=0.1,
                                       n_jobs=1, create_validation_set=False,
                                       model_params=None,
                                       fit_params=None, return_results=False, clean_output_dir=True,
                                       log_to_mlflow=False,
                                       experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                                       # hpo parameters
                                       hpo_framework='optuna', n_trials=5, timeout_experiment=5 * 60, timeout_trial=60,
                                       max_concurrent_trials=1, sampler='tpe', pruner='hyperband',
                                       **kwargs):
        """Run the experiment using an OpenML dataset.

        This function can be used to run the experiment using an OpenML dataset in an interactive way, without the need
        to run the experiment.

        Parameters
        ----------
        model_nickname :
            The nickname of the model to be used in the experiment. It must be a key of the models_dict attribute.
        seed_model :
            The seed of the model to be used in the experiment.
        dataset_name_or_id :
            The name or id of the OpenML dataset. If it is the name, it must be defined in our openml_tasks.csv file.
        seed_dataset :
            The seed of the dataset to be used in the experiment.
        fold :
            The fold of the OpenML dataset.
        run_id :
            The run_id of the mlflow run.
        resample_strategy :
            The resample strategy to be used in the experiment, it can be 'k-fold_cv' or 'hold_out'.
        n_folds :
            The number of folds to be used in the experiment.
        pct_test :
            The percentage of the test set.
        validation_resample_strategy :
            The resample strategy to be used in the validation set, it can be 'next_fold' or 'hold_out'.
        pct_validation :
            The percentage of the validation set.
        n_jobs :
            Number of threads/cores to be used by the model if it supports it.
        create_validation_set :
            If True, create a validation set.
        model_params :
            Dictionary with the parameters of the model. Defaults models_params defined at the initialization of the
            class if None.
        fit_params :
            Dictionary with the parameters of the fit. Defaults fits_params defined at the initialization of the
            class if None.
        return_results :
            If True, return the results of the experiment.
        clean_output_dir :
            If True, clean the output directory after the experiment.
        log_to_mlflow :
            If True, log the results to mlflow.
        experiment_name :
            The name of the experiment.
        mlflow_tracking_uri :
            The uri of the mlflow tracking server.
        check_if_exists :
            If True, check if the run already exists on mlflow.
        hpo_framework :
            The hyperparameter optimization framework to be used. It must be 'optuna' for the moment.
        n_trials :
            The number of trials to be run.
        timeout_experiment :
            The timeout of the experiment in seconds.
        timeout_trial :
            The timeout of each trial in seconds.
        max_concurrent_trials :
            The maximum number of concurrent trials that can be run.
        sampler :
            The sampler to be used in the hyperparameter optimization. It can be 'tpe' or 'random'.
        pruner :
            The pruner to be used in the hyperparameter optimization. It can be 'hyperband', 'sha' or None.
        kwargs :
            Additional arguments from the experiment.

        Returns
        -------
        results :
            If return_results is True, return a dictionary with the results of the experiment, otherwise return True
            if the experiment was successful and False otherwise. It still tries to return the results when
            return_results is true and an exception is raised. The dictionary contains the following
            keys:
            data_return :
                The data returned by the load_data method.
            model :
                The model returned by the get_model method.
            metrics :
                The metrics returned by the get_metrics method.
            report_metric :
                The report_metric returned by the get_metrics method.
            fit_return :
                The data returned by the fit_model method.
            evaluate_return :
                The data returned by the evaluate_model method.
        """
        if create_validation_set is False:
            raise NotImplementedError('HPOExperiment requires a validation set, please set create_validation_set=True')
        return super().run_openml_dataset_combination(model_nickname, seed_model, dataset_name_or_id, seed_dataset,
                                                      fold=fold, run_id=run_id, resample_strategy=resample_strategy,
                                                      n_folds=n_folds, pct_test=pct_test,
                                                      validation_resample_strategy=validation_resample_strategy,
                                                      pct_validation=pct_validation, n_jobs=n_jobs,
                                                      create_validation_set=create_validation_set,
                                                      model_params=model_params, fit_params=fit_params,
                                                      return_results=return_results, clean_output_dir=clean_output_dir,
                                                      log_to_mlflow=log_to_mlflow,
                                                      experiment_name=experiment_name,
                                                      mlflow_tracking_uri=mlflow_tracking_uri,
                                                      check_if_exists=check_if_exists, hpo_framework=hpo_framework,
                                                      n_trials=n_trials, timeout_experiment=timeout_experiment,
                                                      timeout_trial=timeout_trial,
                                                      max_concurrent_trials=max_concurrent_trials, sampler=sampler,
                                                      pruner=pruner, **kwargs)

    def run_pandas_combination(self, model_nickname: str, seed_model: int, seed_dataset: int, dataframe: pd.DataFrame,
                               dataset_name: str, target: str, task: str,
                               fold: int = 0, run_id: Optional[str] = None,
                               resample_strategy: str = 'k-fold_cv', n_folds: int = 10, pct_test: float = 0.2,
                               validation_resample_strategy: str = 'next_fold', pct_validation: float = 0.1,
                               n_jobs: int = 1, create_validation_set: bool = False,
                               model_params: Optional[dict] = None,
                               fit_params: Optional[dict] = None, return_results: bool = False,
                               clean_output_dir: bool = True,
                               log_to_mlflow: bool = False,
                               experiment_name: Optional[str] = None, mlflow_tracking_uri: Optional[str] = None,
                               check_if_exists: Optional[bool] = None,
                               # hpo parameters
                               hpo_framework='optuna', n_trials=5, timeout_experiment=5 * 60, timeout_trial=60,
                               max_concurrent_trials=1, sampler='tpe', pruner='hyperband',
                               **kwargs):
        """Run the experiment using an OpenML dataset.

        This function can be used to run the experiment using an OpenML dataset in an interactive way, without the need
        to run the experiment.

        Parameters
        ----------
        model_nickname :
            The nickname of the model to be used in the experiment. It must be a key of the models_dict attribute.
        seed_model :
            The seed of the model to be used in the experiment.
        dataframe :
            The dataframe to be used in the experiment.
        dataset_name :
            The name of the dataset to be used in the experiment.
        target :
            The name of the target in the dataframe.
        task :
            The task of the dataset, it can be 'classification', 'binary_classification' or 'regression'.
        seed_dataset :
            The seed of the dataset to be used in the experiment.
        fold :
            The fold of the OpenML dataset.
        run_id :
            The run_id of the mlflow run.
        resample_strategy :
            The resample strategy to be used in the experiment, it can be 'k-fold_cv' or 'hold_out'.
        n_folds :
            The number of folds to be used in the experiment.
        pct_test :
            The percentage of the test set.
        validation_resample_strategy :
            The resample strategy to be used in the validation set, it can be 'next_fold' or 'hold_out'.
        pct_validation :
            The percentage of the validation set.
        n_jobs :
            Number of threads/cores to be used by the model if it supports it.
        create_validation_set :
            If True, create a validation set.
        model_params :
            Dictionary with the parameters of the model. Defaults models_params defined at the initialization of the
            class if None.
        fit_params :
            Dictionary with the parameters of the fit. Defaults fits_params defined at the initialization of the
            class if None.
        return_results :
            If True, return the results of the experiment.
        clean_output_dir :
            If True, clean the output directory after the experiment.
        log_to_mlflow :
            If True, log the results to mlflow.
        experiment_name :
            The name of the experiment.
        mlflow_tracking_uri :
            The uri of the mlflow tracking server.
        check_if_exists :
            If True, check if the run already exists on mlflow.
        hpo_framework :
            The hyperparameter optimization framework to be used. It must be 'optuna' for the moment.
        n_trials :
            The number of trials to be run.
        timeout_experiment :
            The timeout of the experiment in seconds.
        timeout_trial :
            The timeout of each trial in seconds.
        max_concurrent_trials :
            The maximum number of concurrent trials that can be run.
        sampler :
            The sampler to be used in the hyperparameter optimization. It can be 'tpe' or 'random'.
        pruner :
            The pruner to be used in the hyperparameter optimization. It can be 'hyperband', 'sha' or None.
        kwargs :
            Additional arguments from the experiment.

        Returns
        -------
        results :
            If return_results is True, return a dictionary with the results of the experiment, otherwise return True
            if the experiment was successful and False otherwise. It still tries to return the results when
            return_results is true and an exception is raised. The dictionary contains the following
            keys:
            data_return :
                The data returned by the load_data method.
            model :
                The model returned by the get_model method.
            metrics :
                The metrics returned by the get_metrics method.
            report_metric :
                The report_metric returned by the get_metrics method.
            fit_return :
                The data returned by the fit_model method.
            evaluate_return :
                The data returned by the evaluate_model method.
        """
        if create_validation_set is False:
            raise NotImplementedError('HPOExperiment requires a validation set, please set create_validation_set=True')
        return super().run_pandas_combination(model_nickname, seed_model, seed_dataset, dataframe, dataset_name, target,
                                              task, fold=fold, run_id=run_id, resample_strategy=resample_strategy,
                                              n_folds=n_folds, pct_test=pct_test,
                                              validation_resample_strategy=validation_resample_strategy,
                                              pct_validation=pct_validation, n_jobs=n_jobs,
                                              create_validation_set=create_validation_set, model_params=model_params,
                                              fit_params=fit_params, return_results=return_results,
                                              clean_output_dir=clean_output_dir, log_to_mlflow=log_to_mlflow,
                                              experiment_name=experiment_name, mlflow_tracking_uri=mlflow_tracking_uri,
                                              check_if_exists=check_if_exists, hpo_framework=hpo_framework,
                                              n_trials=n_trials, timeout_experiment=timeout_experiment,
                                              timeout_trial=timeout_trial, max_concurrent_trials=max_concurrent_trials,
                                              sampler=sampler, pruner=pruner, **kwargs)

    def _get_combinations(self):
        combinations, extra_params = super()._get_combinations()
        extra_params.update(dict(hpo_framework=self.hpo_framework, n_trials=self.n_trials,
                                 timeout_experiment=self.timeout_experiment, timeout_trial=self.timeout_trial,
                                 max_concurrent_trials=self.max_concurrent_trials, sampler=self.sampler,
                                 pruner=self.pruner, create_validation_set=self.create_validation_set))
        if not self.create_validation_set:
            raise NotImplementedError('HPOExperiment requires a validation set, please set create_validation_set=True'
                                      'or pass --create_validation_set')
        return combinations, extra_params

    def _create_mlflow_run(self, *args,
                           create_validation_set=False,
                           model_params=None,
                           fit_params=None,
                           experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                           n_jobs=1, log_to_mlflow=True, return_results=False, clean_work_dir=True,
                           **kwargs):
        parent_run_id = super()._create_mlflow_run(*args, create_validation_set=create_validation_set,
                                                   model_params=model_params, fit_params=fit_params,
                                                   experiment_name=experiment_name,
                                                   mlflow_tracking_uri=mlflow_tracking_uri,
                                                   check_if_exists=check_if_exists,
                                                   **kwargs)
        mlflow_client = mlflow.client.MlflowClient(tracking_uri=mlflow_tracking_uri)
        experiment_id = mlflow_client.get_experiment_by_name(experiment_name).experiment_id
        # we will initialize the nested runs from the trials
        for trial in range(self.n_trials):
            run = mlflow_client.create_run(experiment_id, tags={MLFLOW_PARENT_RUN_ID: parent_run_id})
            run_id = run.info.run_id
            mlflow_client.set_tag(parent_run_id, f'child_run_id_{trial}', run_id)
            mlflow_client.update_run(run_id, status='SCHEDULED')
        return parent_run_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOExperiment(parser=parser)
    experiment.run()
