import argparse
from itertools import product
from pathlib import Path
from typing import Optional
from shutil import rmtree
import tempfile

import mlflow
import numpy as np
from ml_experiments.base_experiment import BaseExperiment
import torch
from torch.cuda import reset_peak_memory_stats, max_memory_reserved, max_memory_allocated

from tab_benchmark.benchmark.utils import load_openml_task, load_own_task, load_json_task, get_model, fit_model, \
    evaluate_model
from tab_benchmark.benchmark.benchmarked_models import models_dict


class TabularExperiment(BaseExperiment):
    @property
    def models_dict(self):
        return models_dict

    def __init__(
            self,
            *args,
            # when performing our own resampling
            datasets_names_or_ids: Optional[list[int]] = None,
            seeds_datasets: Optional[list[int]] = None,
            resample_strategy: str = 'k-fold_cv',
            k_folds: int = 10,
            folds: Optional[list[int]] = None,
            pct_test: float = 0.2,
            validation_resample_strategy: str = 'next_fold',
            pct_validation: float = 0.1,
            # when using openml tasks
            tasks_ids: Optional[list[int]] = None,
            task_repeats: Optional[list[int]] = None,
            task_folds: Optional[list[int]] = None,
            task_samples: Optional[list[int]] = None,
            # custom for tab_benchmark models
            max_time: Optional[int] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        # when performing our own resampling
        self.datasets_names_or_ids = datasets_names_or_ids
        self.seeds_datasets = seeds_datasets if seeds_datasets else [0]
        self.resample_strategy = resample_strategy
        self.k_folds = k_folds
        self.folds = folds if folds else [0]
        self.pct_test = pct_test
        self.validation_resample_strategy = validation_resample_strategy
        self.pct_validation = pct_validation

        # when using openml tasks
        self.tasks_ids = tasks_ids
        self.task_repeats = task_repeats if task_repeats else [0]
        self.task_folds = task_folds if task_folds else [0]
        self.task_samples = task_samples if task_samples else [0]

        self.max_time = max_time

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--datasets_names_or_ids', nargs='*', choices=self.datasets_names_or_ids,
                                 type=str, default=self.datasets_names_or_ids)
        self.parser.add_argument('--seeds_datasets', nargs='*', type=int, default=self.seeds_datasets)
        self.parser.add_argument('--resample_strategy', default=self.resample_strategy, type=str)
        self.parser.add_argument('--k_folds', default=self.k_folds, type=int)
        self.parser.add_argument('--folds', nargs='*', type=int, default=self.folds)
        self.parser.add_argument('--pct_test', type=float, default=self.pct_test)
        self.parser.add_argument('--validation_resample_strategy', type=str, default=self.validation_resample_strategy)
        self.parser.add_argument('--pct_validation', type=float, default=self.pct_validation)

        self.parser.add_argument('--tasks_ids', nargs='*', type=int, default=self.tasks_ids)
        self.parser.add_argument('--task_repeats', nargs='*', type=int, default=self.task_repeats)
        self.parser.add_argument('--task_samples', nargs='*', type=int, default=self.task_samples)
        self.parser.add_argument('--task_folds', nargs='*', type=int, default=self.task_folds)

        self.parser.add_argument('--max_time', type=int, default=self.max_time)

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.datasets_names_or_ids = args.datasets_names_or_ids
        self.seeds_datasets = args.seeds_datasets
        self.seeds_models = args.seeds_models
        self.resample_strategy = args.resample_strategy
        self.k_folds = args.k_folds
        self.folds = args.folds
        self.pct_test = args.pct_test
        self.validation_resample_strategy = args.validation_resample_strategy
        self.pct_validation = args.pct_validation

        self.tasks_ids = args.tasks_ids
        self.task_repeats = args.task_repeats
        self.task_folds = args.task_folds
        self.task_samples = args.task_samples

        self.max_time = args.max_time
        return args

    def _on_train_start(self, combination: dict, unique_params: Optional[dict] = None,
                        extra_params: Optional[dict] = None, **kwargs):
        if torch.cuda.is_available() or self.n_gpus > 0:
            reset_peak_memory_stats()
        return {}

    def _load_data(self, combination: dict, unique_params: Optional[dict] = None,
                   extra_params: Optional[dict] = None, **kwargs):
        create_validation_set = unique_params.get('create_validation_set', False)
        if 'task_id' in combination:
            task_id = combination['task_id']
            task_repeat = combination['task_repeat']
            task_sample = combination['task_sample']
            task_fold = combination['task_fold']
            (X, y, cat_ind, att_names, cat_features_names, cat_dims, task_name, dataset_name, n_classes, train_indices,
             test_indices, validation_indices) = load_openml_task(task_id=task_id, task_repeat=task_repeat,
                                                                  task_sample=task_sample, task_fold=task_fold,
                                                                  create_validation_set=create_validation_set)
        elif 'dataset_name_or_id' in combination:
            dataset_name_or_id = combination['dataset_name_or_id']
            seed_dataset = combination['seed_dataset']
            fold = combination['fold']
            resample_strategy = unique_params.get('resample_strategy', self.resample_strategy)
            k_folds = unique_params.get('k_folds', self.k_folds)
            pct_test = unique_params.get('pct_test', self.pct_test)
            validation_resample_strategy = unique_params.get('validation_resample_strategy',
                                                             self.validation_resample_strategy)
            pct_validation = unique_params.get('pct_validation', self.pct_validation)

            (X, y, cat_ind, att_names, cat_features_names, cat_dims, task_name, dataset_name, n_classes, train_indices,
             test_indices, validation_indices) = (
                load_own_task(dataset_name_or_id=dataset_name_or_id, seed_dataset=seed_dataset, fold=fold,
                              resample_strategy=resample_strategy, k_folds=k_folds, pct_test=pct_test,
                              validation_resample_strategy=validation_resample_strategy, pct_validation=pct_validation,
                              create_validation_set=create_validation_set)
            )
        elif 'json_path' in combination:
            json_path = combination['csv_path']
            seed_dataset = combination['seed_dataset']
            fold = combination['fold']
            resample_strategy = unique_params.get('resample_strategy', self.resample_strategy)
            k_folds = unique_params.get('k_folds', self.k_folds)
            pct_test = unique_params.get('pct_test', self.pct_test)
            validation_resample_strategy = unique_params.get('validation_resample_strategy',
                                                             self.validation_resample_strategy)
            pct_validation = unique_params.get('pct_validation', self.pct_validation)
            (X, y, cat_ind, att_names, cat_features_names, cat_dims, task_name, dataset_name, n_classes, train_indices,
             test_indices, validation_indices) = (
                load_json_task(json_path=json_path, seed_dataset=seed_dataset, fold=fold,
                               resample_strategy=resample_strategy, k_folds=k_folds, pct_test=pct_test,
                               validation_resample_strategy=validation_resample_strategy, pct_validation=pct_validation,
                               create_validation_set=create_validation_set)
            )
        else:
            raise ValueError('Combination must have either task_id or dataset_name_or_id or json_path')

        return {

            'X': X,
            'y': y,
            'cat_ind': cat_ind,
            'att_names': att_names,
            'cat_features_names': cat_features_names,
            'cat_dims': cat_dims,
            'task_name': task_name,
            'dataset_name': dataset_name,
            'n_classes': n_classes,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'validation_indices': validation_indices
        }

    def _load_model(self, combination: dict, unique_params: Optional[dict] = None,
                    extra_params: Optional[dict] = None, **kwargs):
        model_nickname = combination['model_nickname']
        seed_model = combination['seed_model']
        model_params = combination['model_params']
        create_validation_set = unique_params.get('create_validation_set', False)
        mlflow_run_id = extra_params.get('mlflow_run_id', None)
        work_dir = self.get_local_work_dir(combination=combination, mlflow_run_id=mlflow_run_id,
                                           unique_params=unique_params)
        n_jobs = extra_params.get('n_jobs', self.n_jobs)
        max_time = extra_params.get('max_time', self.max_time)
        model = get_model(model_nickname=model_nickname, seed_model=seed_model, model_params=model_params,
                          models_dict=self.models_dict, n_jobs=n_jobs, output_dir=work_dir, max_time=max_time)
        if self.log_to_mlflow:
            mlflow_run_id = extra_params.get('mlflow_run_id')
            if hasattr(model, 'mlflow_run_id'):
                setattr(model, 'mlflow_run_id', mlflow_run_id)
        if create_validation_set:
            # we disable auto early stopping when creating a validation set, because we will use it to validate
            if hasattr(model, 'auto_early_stopping'):
                model.auto_early_stopping = False

        return {
            'model': model
        }

    def _get_metrics(self, combination: dict, unique_params: Optional[dict] = None,
                     extra_params: Optional[dict] = None, **kwargs):
        task_name = kwargs['load_data_return']['task_name']
        if task_name in ('classification', 'binary_classification'):
            metrics = ['logloss', 'auc', 'auc_micro', 'auc_weighted', 'accuracy', 'balanced_accuracy',
                       'balanced_accuracy_adjusted', 'f1_micro', 'f1_macro', 'f1_weighted']
            report_metric = 'logloss'
        elif task_name == 'regression':
            metrics = ['rmse', 'r2_score', 'mae', 'mape']
            report_metric = 'rmse'
        else:
            raise NotImplementedError
        return {
            'metrics': metrics,
            'report_metric': report_metric
        }

    def _fit_model(self, combination: dict, unique_params: Optional[dict] = None,
                   extra_params: Optional[dict] = None, **kwargs):
        fit_params = combination['fit_params']
        model = kwargs['load_model_return']['model']
        cat_features_names = kwargs['load_data_return']['cat_features_names']
        X = kwargs['load_data_return']['X']
        y = kwargs['load_data_return']['y']
        task_name = kwargs['load_data_return']['task_name']
        cat_ind = kwargs['load_data_return']['cat_ind']
        att_names = kwargs['load_data_return']['att_names']
        cat_dims = kwargs['load_data_return']['cat_dims']
        n_classes = kwargs['load_data_return']['n_classes']
        train_indices = kwargs['load_data_return']['train_indices']
        test_indices = kwargs['load_data_return']['test_indices']
        validation_indices = kwargs['load_data_return']['validation_indices']
        # we will already convert categorical features to codes to avoid missing categories when splitting the data
        # one can argue if the model alone should account for this (not observing all the categories in the training
        # set), but for many applications this is fine and if we really want to do this we could simply always add
        # a category for missing values
        for cat_feature in cat_features_names:
            X[cat_feature] = X[cat_feature].cat.codes
            X[cat_feature] = X[cat_feature].replace(-1, np.nan).astype('category')
        if task_name in ('classification', 'binary_classification'):
            if y.dtypes != 'category':
                y = y.astype('category')
            y = y.cat.codes
            y = y.replace(-1, np.nan).astype('category')
        # if we are just using ordinal encoding, we can disable it
        # otherwise the encoder or the model must take care of possible missing categories
        if model.categorical_encoder == 'ordinal':
            model.categorical_encoder = None  # we already encoded the categorical features
        if model.categorical_target_encoder == 'ordinal':
            model.categorical_target_encoder = None  # we already encoded the categorical target
        # fit model
        # data here is already preprocessed
        model, X_train, y_train, X_test, y_test, X_validation, y_validation = fit_model(
            model, X, y, cat_ind, att_names, cat_dims, n_classes, task_name, train_indices, test_indices,
            validation_indices,
            **fit_params)
        # maybe do not return preprocessed data to avoid memory issues
        return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    X_validation=X_validation, y_validation=y_validation)

    def _evaluate_model(self, combination: dict, unique_params: Optional[dict] = None,
                        extra_params: Optional[dict] = None, **kwargs):
        result = {}
        create_validation_set = unique_params.get('create_validation_set', False)
        model = kwargs['load_model_return']['model']
        X_test = kwargs['fit_model_return']['X_test']
        y_test = kwargs['fit_model_return']['y_test']
        n_classes = kwargs['load_data_return']['n_classes']
        X_validation = kwargs['fit_model_return']['X_validation']
        y_validation = kwargs['fit_model_return']['y_validation']
        metrics = kwargs['get_metrics_return']['metrics']
        report_metric = kwargs['get_metrics_return']['report_metric']
        test_results = evaluate_model(model=model, eval_set=(X_test, y_test), eval_name='final_test',
                                      metrics=metrics, n_classes=n_classes,
                                      error_score=self.error_score)
        result.update(test_results)
        if create_validation_set:
            validation_results = evaluate_model(model=model, eval_set=(X_validation, y_validation),
                                                eval_name='final_validation', metrics=metrics,
                                                report_metric=report_metric, n_classes=n_classes,
                                                error_score=self.error_score)
            result.update(validation_results)
        return result

    def _get_combinations(self):
        if self.datasets_names_or_ids is not None and self.tasks_ids is None:
            self.using_own_resampling = True
        elif self.datasets_names_or_ids is None and self.tasks_ids is not None:
            self.using_own_resampling = False
        else:
            raise ValueError("You must provide either datasets_names_or_ids or tasks_ids, but not both.")
        if self.using_own_resampling:
            combinations = list(product(self.models_nickname, self.seeds_models, self.datasets_names_or_ids,
                                        self.seeds_datasets, self.folds))
            combination_names = ['model_nickname', 'seed_model', 'dataset_name_or_id', 'seed_dataset', 'fold']
            unique_params = dict(resample_strategy=self.resample_strategy, k_folds=self.k_folds, pct_test=self.pct_test,
                                 validation_resample_strategy=self.validation_resample_strategy,
                                 pct_validation=self.pct_validation)
        else:
            combinations = list(product(self.models_nickname, self.seeds_models, self.tasks_ids, self.task_folds,
                                        self.task_repeats, self.task_samples))
            combination_names = ['model_nickname', 'seed_model', 'task_id', 'task_fold', 'task_repeat', 'task_sample']
            unique_params = dict()

        combinations = [list(combination) + [self.models_params[combination[0]]] + [self.fits_params[combination[0]]]
                        for combination in combinations]
        combination_names += ['model_params', 'fit_params']

        unique_params.update(dict(create_validation_set=self.create_validation_set))

        extra_params = dict(n_jobs=self.n_jobs, return_results=False, max_time=self.max_time,
                            timeout_combination=self.timeout_combination, timeout_fit=self.timeout_fit)
        return combinations, combination_names, unique_params, extra_params

    def _log_run_start_params(self, mlflow_run_id, **run_unique_params):
        self._log_base_experiment_start_params(mlflow_run_id, **run_unique_params)
        params_to_log = dict(
            cuda_available=torch.cuda.is_available(),
        )
        mlflow.log_params(params_to_log, run_id=mlflow_run_id)

    def _log_run_results(self, combination: dict, unique_params: Optional[dict] = None, mlflow_run_id=None,
                         extra_params: Optional[dict] = None, **kwargs):
        if mlflow_run_id is None:
            return
        self._log_base_experiment_run_results(combination=combination, unique_params=unique_params,
                                              mlflow_run_id=mlflow_run_id, extra_params=extra_params, **kwargs)

        log_params = {}
        log_metrics = {}

        # cuda memory
        if torch.cuda.is_available() or self.n_gpus > 0:
            log_metrics['max_cuda_memory_reserved'] = max_memory_reserved() / (1024 ** 2)  # in MB
            log_metrics['max_cuda_memory_allocated'] = max_memory_allocated() / (1024 ** 2)  # in MB

        # model name to facilitate filtering
        model_nickname = combination['model_nickname']
        if model_nickname.find('TabBenchmark') != -1:
            log_params.update({'model_name': model_nickname[len('TabBenchmark'):]})

        # task and dataset_name
        load_data_return = kwargs['load_data_return']
        task_name = load_data_return['task_name']
        dataset_name = load_data_return['dataset_name']
        log_params.update({'task_name': task_name, 'dataset_name': dataset_name})

        # report metric
        get_metrics_return = kwargs.get('get_metrics_return', {})
        report_metric = get_metrics_return.get('report_metric', None)
        log_params.update({'report_metric': report_metric})

        # evaluation results
        eval_results_dict = kwargs.get('evaluate_model_return', {}).copy()
        eval_results_dict.pop('elapsed_time', None)
        log_metrics.update(eval_results_dict)

        mlflow.log_params(log_params, run_id=mlflow_run_id)
        mlflow.log_metrics(log_metrics, run_id=mlflow_run_id)

    def _on_train_end(self, combination: dict, unique_params: Optional[dict] = None,
                      extra_params: Optional[dict] = None, **kwargs):
        mlflow_run_id = extra_params.get('mlflow_run_id', None)
        self._log_run_results(combination=combination, unique_params=unique_params, extra_params=extra_params,
                              mlflow_run_id=mlflow_run_id, **kwargs)

        # save and/or clean work_dir
        work_dir = self.get_local_work_dir(combination, mlflow_run_id, unique_params)
        if self.clean_work_dir:
            if work_dir.exists():
                rmtree(work_dir)

        model = kwargs['load_model_return'].get('model', None)
        if self.save_root_dir and model is not None:
            if mlflow_run_id is not None:
                # will log the model to mlflow artifacts
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    model.save_model(temp_dir)
                    mlflow.log_artifacts(str(temp_dir.resolve()), artifact_path='model', run_id=mlflow_run_id)
            else:
                save_dir = self.save_root_dir / work_dir.name
                model.save(save_dir)
        return {}

    def run_openml_task_combination(self, model_nickname: str, seed_model: int, task_id: int,
                                    task_fold: int = 0, task_repeat: int = 0, task_sample: int = 0,
                                    run_id: Optional[str] = None,
                                    n_jobs: int = 1, create_validation_set: bool = False,
                                    max_time: Optional[int] = None, timeout_combination: Optional[int] = None,
                                    timeout_fit: Optional[int] = None,
                                    model_params: Optional[dict] = None,
                                    fit_params: Optional[dict] = None, return_results: bool = True,
                                    log_to_mlflow: bool = False):
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
        log_to_mlflow :
            If True, log the results to mlflow.

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
        combination = {
            'model_nickname': model_nickname,
            'seed_model': seed_model,
            'task_id': task_id,
            'task_fold': task_fold,
            'task_repeat': task_repeat,
            'task_sample': task_sample,
            'model_params': model_params,
            'fit_params': fit_params,
        }

        unique_params = {
            'create_validation_set': create_validation_set,
        }

        extra_params = {
            'n_jobs': n_jobs,
            'max_time': max_time,
            'timeout_combination': timeout_combination,
            'timeout_fit': timeout_fit,
        }
        if log_to_mlflow:
            return self._run_mlflow_and_train_model(combination=combination, mlflow_run_id=run_id,
                                                    unique_params=unique_params, return_results=return_results,
                                                    extra_params=extra_params)
        return self._train_model(combination=combination, unique_params=unique_params, return_results=return_results,
                                 extra_params=extra_params)

    def run_openml_dataset_combination(self, model_nickname: str, seed_model: int, dataset_name_or_id: str | int,
                                       seed_dataset: int,
                                       fold: int = 0, run_id: Optional[str] = None,
                                       resample_strategy: str = 'k-fold_cv', k_folds: int = 10, pct_test: float = 0.2,
                                       validation_resample_strategy: str = 'next_fold', pct_validation: float = 0.1,
                                       n_jobs: int = 1, create_validation_set: bool = False,
                                       max_time: Optional[int] = None,
                                       timeout_combination: Optional[int] = None, timeout_fit: Optional[int] = None,
                                       model_params: Optional[dict] = None,
                                       fit_params: Optional[dict] = None, return_results: bool = False,
                                       log_to_mlflow: bool = False):
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
        k_folds :
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
        log_to_mlflow :
            If True, log the results to mlflow.

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
        combination = {
            'model_nickname': model_nickname,
            'seed_model': seed_model,
            'dataset_name_or_id': dataset_name_or_id,
            'seed_dataset': seed_dataset,
            'fold': fold,
            'model_params': model_params,
            'fit_params': fit_params,
        }

        unique_params = {
            'resample_strategy': resample_strategy,
            'k_folds': k_folds,
            'pct_test': pct_test,
            'validation_resample_strategy': validation_resample_strategy,
            'pct_validation': pct_validation,
            'create_validation_set': create_validation_set,
        }

        extra_params = {
            'n_jobs': n_jobs,
            'max_time': max_time,
            'timeout_combination': timeout_combination,
            'timeout_fit': timeout_fit,
        }

        if log_to_mlflow:
            return self._run_mlflow_and_train_model(combination=combination, mlflow_run_id=run_id,
                                                    unique_params=unique_params, return_results=return_results,
                                                    extra_params=extra_params)
        return self._train_model(combination=combination, unique_params=unique_params, return_results=return_results,
                                 extra_params=extra_params)

    def run_json_combination(self, model_nickname: str, seed_model: int, seed_dataset: int, json_path: str | Path,
                             fold: int = 0, run_id: Optional[str] = None,
                             resample_strategy: str = 'k-fold_cv', k_folds: int = 10, pct_test: float = 0.2,
                             validation_resample_strategy: str = 'next_fold', pct_validation: float = 0.1,
                             n_jobs: int = 1, create_validation_set: bool = False,
                             max_time: Optional[int] = None,
                             timeout_combination: Optional[int] = None, timeout_fit: Optional[int] = None,
                             model_params: Optional[dict] = None,
                             fit_params: Optional[dict] = None, return_results: bool = False,
                             log_to_mlflow: bool = False):
        """Run the experiment using an OpenML dataset.

        This function can be used to run the experiment using an OpenML dataset in an interactive way, without the need
        to run the experiment.

        Parameters
        ----------
        model_nickname :
            The nickname of the model to be used in the experiment. It must be a key of the models_dict attribute.
        seed_model :
            The seed of the model to be used in the experiment.
        seed_dataset :
            The seed of the dataset to be used in the experiment.
        fold :
            The fold of the OpenML dataset.
        run_id :
            The run_id of the mlflow run.
        resample_strategy :
            The resample strategy to be used in the experiment, it can be 'k-fold_cv' or 'hold_out'.
        k_folds :
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
        log_to_mlflow :
            If True, log the results to mlflow.

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
        combination = {
            'model_nickname': model_nickname,
            'seed_model': seed_model,
            'json_path': json_path,
            'seed_dataset': seed_dataset,
            'fold': fold,
            'model_params': model_params,
            'fit_params': fit_params,
        }

        unique_params = {
            'resample_strategy': resample_strategy,
            'k_folds': k_folds,
            'pct_test': pct_test,
            'validation_resample_strategy': validation_resample_strategy,
            'pct_validation': pct_validation,
            'create_validation_set': create_validation_set,
        }

        extra_params = {
            'n_jobs': n_jobs,
            'max_time': max_time,
            'timeout_combination': timeout_combination,
            'timeout_fit': timeout_fit,
        }

        if log_to_mlflow:
            return self._run_mlflow_and_train_model(combination=combination, mlflow_run_id=run_id,
                                                    unique_params=unique_params, return_results=return_results,
                                                    extra_params=extra_params)
        return self._train_model(combination=combination, unique_params=unique_params, return_results=return_results,
                                 extra_params=extra_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = TabularExperiment(parser=parser)
    experiment.run()
