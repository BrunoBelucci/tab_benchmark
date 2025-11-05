import argparse
from itertools import product
from pathlib import Path
from typing import Optional
from shutil import rmtree
import tempfile
import mlflow
import numpy as np
from ml_experiments.base_experiment import BaseExperiment
from tab_benchmark.benchmark.utils import load_openml_task, load_own_task, load_json_task, get_model, fit_model, \
    evaluate_model
from tab_benchmark.benchmark.benchmarked_models import models_dict
from tab_benchmark.models.dnn_models import DNNMixin
from sklearn.base import BaseEstimator
import json


class TabularExperiment(BaseExperiment):
    @property
    def models_dict(self):
        return models_dict

    def __init__(
        self,
        *args,
        # model
        model: Optional[str | BaseEstimator | type[BaseEstimator] | list[str]] = None,
        model_params: Optional[dict] = None,
        fit_params: Optional[dict] = None,
        seed_model: int | list[int] = 0,
        n_jobs: int = 1,
        # when performing our own resampling
        dataset_name_or_id: Optional[list[int]] = None,
        seed_dataset: Optional[list[int]] = None,
        resample_strategy: str = "k-fold_cv",
        k_fold: int = 10,
        fold: Optional[list[int]] = None,
        pct_test: float = 0.2,
        validation_resample_strategy: str = "next_fold",
        pct_validation: float = 0.1,
        # when using openml tasks
        task_id: Optional[list[int]] = None,
        task_repeat: Optional[list[int]] = None,
        task_fold: Optional[list[int]] = None,
        task_sample: Optional[list[int]] = None,
        create_validation_set: bool = False,
        # custom for tab_benchmark models
        max_time: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # model
        self.model = model
        self.model_params = model_params if model_params else dict()
        self.fit_params = fit_params if fit_params else dict()
        self.seed_model = seed_model if isinstance(seed_model, list) else [seed_model]
        self.n_jobs = n_jobs

        # when performing our own resampling
        self.dataset_name_or_id = dataset_name_or_id
        self.seed_dataset = seed_dataset if seed_dataset else [0]
        self.resample_strategy = resample_strategy
        self.k_fold = k_fold
        self.fold = fold if fold else [0]
        self.pct_test = pct_test
        self.validation_resample_strategy = validation_resample_strategy
        self.pct_validation = pct_validation

        # when using openml tasks
        self.task_id = task_id
        self.task_repeat = task_repeat if task_repeat else [0]
        self.task_fold = task_fold if task_fold else [0]
        self.task_sample = task_sample if task_sample else [0]
        self.create_validation_set = create_validation_set

        self.max_time = max_time

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--model', default=self.model, type=str)
        self.parser.add_argument("--model_params", default=self.model_params, type=json.loads)
        self.parser.add_argument("--fit_params", default=self.fit_params, type=json.loads)
        self.parser.add_argument('--seed_model', nargs='*', type=int, default=self.seed_model)
        self.parser.add_argument('--n_jobs', type=int, default=self.n_jobs)
        self.parser.add_argument('--dataset_name_or_id', nargs='*', choices=self.dataset_name_or_id,
                                 type=str, default=self.dataset_name_or_id)
        self.parser.add_argument('--seed_dataset', nargs='*', type=int, default=self.seed_dataset)
        self.parser.add_argument('--resample_strategy', default=self.resample_strategy, type=str)
        self.parser.add_argument('--k_fold', default=self.k_fold, type=int)
        self.parser.add_argument('--fold', nargs='*', type=int, default=self.fold)
        self.parser.add_argument('--pct_test', type=float, default=self.pct_test)
        self.parser.add_argument('--validation_resample_strategy', type=str, default=self.validation_resample_strategy)
        self.parser.add_argument('--pct_validation', type=float, default=self.pct_validation)
        self.parser.add_argument('--create_validation_set', type=bool, default=self.create_validation_set)

        self.parser.add_argument('--task_id', nargs='*', type=int, default=self.task_id)
        self.parser.add_argument('--task_repeat', nargs='*', type=int, default=self.task_repeat)
        self.parser.add_argument('--task_sample', nargs='*', type=int, default=self.task_sample)
        self.parser.add_argument('--task_fold', nargs='*', type=int, default=self.task_fold)

        self.parser.add_argument('--max_time', type=int, default=self.max_time)

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.model = args.model
        self.model_params = args.model_params
        self.fit_params = args.fit_params
        self.seed_model = args.seed_model
        self.n_jobs = args.n_jobs
        self.dataset_name_or_id = args.dataset_name_or_id
        self.seed_dataset = args.seed_dataset
        self.resample_strategy = args.resample_strategy
        self.k_fold = args.k_fold
        self.fold = args.fold
        self.pct_test = args.pct_test
        self.validation_resample_strategy = args.validation_resample_strategy
        self.pct_validation = args.pct_validation

        self.task_id = args.task_id
        self.task_repeat = args.task_repeat
        self.task_fold = args.task_fold
        self.task_sample = args.task_sample
        self.create_validation_set = args.create_validation_set

        self.max_time = args.max_time
        return args

    def _load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
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
            k_folds = unique_params.get('k_folds', self.k_fold)
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
            k_folds = unique_params.get('k_folds', self.k_fold)
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

    def _load_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        model = combination["model"]
        seed_model = combination['seed_model']
        model_params = unique_params['model_params']
        create_validation_set = unique_params.get("create_validation_set", False)
        work_dir = self.get_local_work_dir(combination=combination, mlflow_run_id=mlflow_run_id,
                                           unique_params=unique_params)
        n_jobs = unique_params.get("n_jobs", self.n_jobs)
        max_time = unique_params.get("max_time", self.max_time)
        model = get_model(
            model=model,
            seed_model=seed_model,
            model_params=model_params,
            models_dict=self.models_dict,
            n_jobs=n_jobs,
            output_dir=work_dir,
            max_time=max_time,
        )
        if mlflow_run_id is not None:
            if hasattr(model, 'mlflow_run_id'):
                setattr(model, 'mlflow_run_id', mlflow_run_id)
        if create_validation_set:
            # we disable auto early stopping when creating a validation set, because we will use it to validate
            if hasattr(model, 'auto_early_stopping'):
                setattr(model, 'auto_early_stopping', False)
        return {
            'model': model
        }

    def _get_metrics(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        task_name = kwargs['load_data_return']['task_name']
        model = kwargs['load_model_return']['model']
        if task_name in ('classification', 'binary_classification'):
            metrics = ['logloss', 'auc', 'auc_micro', 'auc_weighted', 'accuracy', 'balanced_accuracy',
                       'balanced_accuracy_adjusted', 'f1_micro', 'f1_macro', 'f1_weighted']
            if hasattr(model, 'es_eval_metric'):
                if isinstance(model, DNNMixin):
                    setattr(model, 'es_eval_metric', 'loss')
                else:
                    setattr(model, 'es_eval_metric', 'logloss')
        elif task_name == 'regression':
            metrics = ['rmse', 'r2_score', 'mae', 'mape']
            if hasattr(model, 'es_eval_metric'):
                if isinstance(model, DNNMixin):
                    setattr(model, 'es_eval_metric', 'loss')
                else:
                    setattr(model, 'es_eval_metric', 'rmse')
        else:
            raise NotImplementedError
        return {
            'metrics': metrics,
        }

    def _fit_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        fit_params = unique_params["fit_params"]
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

    def _evaluate_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        result = {}
        create_validation_set = unique_params.get('create_validation_set', False)
        model = kwargs['load_model_return']['model']
        X_test = kwargs['fit_model_return']['X_test']
        y_test = kwargs['fit_model_return']['y_test']
        n_classes = kwargs['load_data_return']['n_classes']
        X_validation = kwargs['fit_model_return']['X_validation']
        y_validation = kwargs['fit_model_return']['y_validation']
        metrics = kwargs['get_metrics_return']['metrics']
        test_results = evaluate_model(model=model, eval_set=(X_test, y_test), eval_name='final_test',
                                      metrics=metrics, n_classes=n_classes,
                                      error_score='raise')
        result.update(test_results)
        if create_validation_set:
            validation_results = evaluate_model(model=model, eval_set=(X_validation, y_validation),
                                                eval_name='final_validation', metrics=metrics,
                                                n_classes=n_classes, error_score='raise')
            result.update(validation_results)
        return result

    def _get_combinations_names(self):
        combination_names = super()._get_combinations_names()
        if self.dataset_name_or_id is not None and self.task_id is None:
            combination_names += ['model', 'seed_model', 'dataset_name_or_id', 'seed_dataset', 'fold']
        elif self.dataset_name_or_id is None and self.task_id is not None:
           combination_names += ['model', 'seed_model', 'task_id', 'task_repeat', 'task_sample', 'task_fold']
        else:
            raise ValueError("You must provide either datasets_names_or_ids or tasks_ids, but not both.")
        return combination_names

    def _get_unique_params(self):
        unique_params = super()._get_unique_params()
        unique_params.update(dict(
            create_validation_set=self.create_validation_set,
            model_params=self.model_params,
            fit_params=self.fit_params,
            n_jobs=self.n_jobs,
        ))
        return unique_params

    def _get_extra_params(self):
        extra_params = super()._get_extra_params()
        return extra_params

    def _log_run_results(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        if mlflow_run_id is None:
            return
        self._log_base_experiment_run_results(combination=combination, unique_params=unique_params,
                                              mlflow_run_id=mlflow_run_id, extra_params=extra_params, **kwargs)

        log_params = {}

        # model name to facilitate filtering
        model_nickname = combination['model_nickname']
        if model_nickname.find('TabBenchmark') != -1:
            log_params.update({'model_name': model_nickname[len('TabBenchmark'):]})

        # task and dataset_name
        load_data_return = kwargs['load_data_return']
        task_name = load_data_return['task_name']
        dataset_name = load_data_return['dataset_name']
        log_params.update({'task_name': task_name, 'dataset_name': dataset_name})

        mlflow.log_params(log_params, run_id=mlflow_run_id)

    def _on_exception_or_train_end(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        mlflow_run_id = extra_params.get('mlflow_run_id', None)
        self._log_run_results(combination=combination, unique_params=unique_params, extra_params=extra_params,
                              mlflow_run_id=mlflow_run_id, **kwargs)

        # save and/or clean work_dir
        work_dir = self.get_local_work_dir(combination, mlflow_run_id, unique_params)
        load_model_return = kwargs.get('load_model_return', dict())
        model = load_model_return.get('model', None)
        if self.save_root_dir and model is not None:
            if mlflow_run_id is not None:
                # will log the model to mlflow artifacts
                with tempfile.TemporaryDirectory(dir=str(work_dir.resolve())) as temp_dir:
                    temp_dir = Path(temp_dir)
                    model.save_model(temp_dir)
                    mlflow.log_artifacts(str(temp_dir.resolve()), artifact_path='model', run_id=mlflow_run_id)
            else:
                save_dir = self.save_root_dir / work_dir.name
                model.save(save_dir)
        if self.clean_work_dir:
            if work_dir.exists():
                rmtree(work_dir)
        return {}


if __name__ == '__main__':
    experiment = TabularExperiment()
    experiment.run_from_cli()
