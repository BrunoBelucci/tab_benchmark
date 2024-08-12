import os
from copy import deepcopy
from pathlib import Path
import mlflow
import numpy as np
import openml
from ray.air import FailureConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.bohb import TuneBOHB
from ray.train import RunConfig, SyncConfig
from sklearn.model_selection import StratifiedKFold, KFold
from tab_benchmark.benchmark.benchmarked_models import models_dict as benchmarked_models_dict
from tab_benchmark.datasets import get_dataset
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.utils import set_seeds, evaluate_set, train_test_split_forced, flatten_dict


def check_if_exists_mlflow(experiment_name, **kwargs):
    filter_string = " AND ".join([f'params."{k}" = "{v}"' for k, v in flatten_dict(kwargs).items()])
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter_string)
    # remove ./mlruns if it is automatically created
    if os.path.exists('./mlruns'):
        os.rmdir('./mlruns')
    runs = runs.loc[runs['status'] == 'FINISHED']
    if not runs.empty:
        return runs.iloc[0]
    else:
        return None


def get_model(model_nickname, seed_model, model_params=None, models_dict=None, n_jobs=1, output_dir=None):
    model_params = model_params if model_params is not None else {}
    models_dict = models_dict if models_dict is not None else benchmarked_models_dict.copy()
    set_seeds(seed_model)
    model_class, model_default_params = deepcopy(models_dict[model_nickname])
    if callable(model_default_params):
        model_default_params = model_default_params(model_class)
    model_default_params.update(model_params)
    model_default_params['random_state'] = seed_model
    model = model_class(**model_default_params)
    if hasattr(model, 'n_jobs'):
        if isinstance(model, DNNModel) and n_jobs == 1:
            # set n_jobs to 0 for DNNModel (no parallelism)
            setattr(model, 'n_jobs', 0)
        setattr(model, 'n_jobs', n_jobs)
    if output_dir is not None:
        if hasattr(model, 'output_dir'):
            setattr(model, 'output_dir', output_dir)
    return model


def treat_mlflow(experiment_name, mlflow_tracking_uri, check_if_exists, **kwargs):
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    if mlflow_tracking_uri and experiment_name:
        run = None
        if check_if_exists:
            run = check_if_exists_mlflow(experiment_name, **kwargs)
        if run is not None:
            return run, False
        else:
            return run, True
    else:
        return None, False


def fit_model(model, X, y, cat_ind, att_names, task_name, train_indices, test_indices, validation_indices=None,
              logging_to_mlflow=False, **kwargs):
    if logging_to_mlflow:
        mlflow.log_param('task_name', task_name)
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    if validation_indices is not None:
        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]
    else:
        X_validation = None
        y_validation = None
    cat_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is True]
    cont_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is False]
    if not (('classifier' in model._estimator_type and task_name in ('classification', 'binary_classification')) or (
            'regressor' in model._estimator_type and task_name in ('regression', 'multi_regression'))):
        raise ValueError('Model class and task do not match')
    # safer to preprocess and fit the model separately
    model.create_preprocess_pipeline(task_name, cat_features_names, cont_features_names, att_names)
    model.create_model_pipeline()
    data_preprocess_pipeline_ = model.data_preprocess_pipeline_
    target_preprocess_pipeline_ = model.target_preprocess_pipeline_
    X_train = data_preprocess_pipeline_.fit_transform(X_train)
    y_train = target_preprocess_pipeline_.fit_transform(y_train.to_frame())
    X_test = data_preprocess_pipeline_.transform(X_test)
    y_test = target_preprocess_pipeline_.transform(y_test.to_frame())
    if validation_indices is not None:
        X_validation = data_preprocess_pipeline_.transform(X_validation)
        y_validation = target_preprocess_pipeline_.transform(y_validation.to_frame())
        eval_set = [(X_validation, y_validation)]
        eval_name = ['validation']
    else:
        X_validation = None
        y_validation = None
        eval_set = None
        eval_name = None

    model.fit(X_train, y_train, task=task_name, cat_features=cat_features_names, eval_set=eval_set, eval_name=eval_name,
              **kwargs)
    return model, X_train, y_train, X_test, y_test, X_validation, y_validation


def evaluate_model(model, eval_set, eval_name, metrics, default_metric=None, n_classes=None,
                   logging_to_mlflow=False):
    results = evaluate_set(model, eval_set, metrics, n_classes)
    if default_metric is not None:
        results['default'] = results[default_metric]
        if logging_to_mlflow:
            mlflow.log_param('default_metric', default_metric)
    results_dict = {f'{eval_name}_{metric}': value for metric, value in results.items()}
    if logging_to_mlflow:
        mlflow.log_metrics(results_dict)
    return results_dict


def load_own_task(dataset_name_or_id, seed_dataset, resample_strategy, n_folds, pct_test, fold,
                  create_validation_set=False, validation_resample_strategy='next_fold', pct_validation=0.1):
    dataset, task_name, target = get_dataset(dataset_name_or_id)
    X, y, cat_ind, att_names = dataset.get_data(target=target)
    if resample_strategy == 'hold_out':
        test_size = int(pct_test * len(dataset.qualities['NumberOfInstances']))
        if task_name in ('classification', 'binary_classification'):
            stratify = y
        elif task_name in ('regression', 'multi_regression'):
            stratify = None
        else:
            raise NotImplementedError
        X_train, X_test, y_train, y_test = train_test_split_forced(X, y, test_size_pct=test_size,
                                                                   random_state=seed_dataset, stratify=stratify)
        train_indices = X_train.index
        test_indices = X_test.index
        folds = None
    elif resample_strategy == 'k-fold_cv':
        if task_name in ('classification', 'binary_classification'):
            kf = StratifiedKFold(n_splits=n_folds, random_state=seed_dataset, shuffle=True)
        elif task_name in ('regression', 'multi_regression'):
            kf = KFold(n_splits=n_folds, random_state=seed_dataset, shuffle=True)
        else:
            raise NotImplementedError
        folds = list(kf.split(X, y))
        train_indices, test_indices = folds[fold]
    else:
        raise NotImplementedError
    if create_validation_set:
        if validation_resample_strategy == 'next_fold':
            if resample_strategy == 'k-fold_cv':
                next_fold = (fold + 1) % n_folds
                _, validation_indices = folds[next_fold]
                train_indices = np.setdiff1d(train_indices, validation_indices, assume_unique=True)
            else:
                raise NotImplementedError
        elif validation_resample_strategy == 'hold_out':
            validation_size = int(pct_validation * (len(train_indices) + len(test_indices)) / (1 - pct_test))
            if task_name in ('classification', 'binary_classification'):
                stratify = y.iloc[train_indices]
            elif task_name in ('regression', 'multi_regression'):
                stratify = None
            else:
                raise NotImplementedError
            X_train, X_validation, y_train, y_validation = train_test_split_forced(
                X.iloc[train_indices], y.iloc[train_indices], test_size_pct=validation_size,
                random_state=seed_dataset,
                stratify=stratify)
            train_indices = X_train.index
            validation_indices = X_validation.index
        else:
            raise NotImplementedError
    else:
        validation_indices = None
    return X, y, cat_ind, att_names, task_name, train_indices, test_indices, validation_indices


def load_openml_task(task_id, task_repeat, task_sample, task_fold, create_validation_set=False):
    task = openml.tasks.get_task(task_id)
    split = task.get_train_test_split_indices(task_fold, task_repeat, task_sample)
    train_indices = split.train
    test_indices = split.test
    if task.task_type == 'Supervised Classification':
        n_classes = len(task.class_labels)
        if n_classes == 2:
            task_name = 'binary_classification'
        elif n_classes > 2:
            task_name = 'classification'
        else:
            raise ValueError('Task has less than 2 classes')
    elif task.task_type == 'Supervised Regression':
        task_name = 'regression'
    else:
        raise NotImplementedError
    if create_validation_set:
        n_folds = int(task.estimation_procedure['parameters']['number_folds'])
        split_validation = task.get_train_test_split_indices((task_fold + 1) % n_folds, task_repeat,
                                                             task_sample)
        validation_indices = split_validation.test
        train_indices = np.setdiff1d(train_indices, validation_indices, assume_unique=True)
    else:
        validation_indices = None
    dataset = task.get_dataset()
    X, y, cat_ind, att_names = dataset.get_data(target=task.target_name)
    return X, y, cat_ind, att_names, task_name, train_indices, test_indices, validation_indices


def get_search_algorithm_tune_config_run_config(default_param_space, search_algorithm_str, n_trials,
                                                timeout_experiment, timeout_trial, storage_path, metric, mode,
                                                seed, max_concurrent):
    if isinstance(storage_path, Path):
        storage_path = str(storage_path.resolve())
    if search_algorithm_str == 'random_search':
        search_algorithm = BasicVariantGenerator(points_to_evaluate=[default_param_space], random_state=seed,
                                                 max_concurrent=max_concurrent)
        scheduler = None
    elif search_algorithm_str == 'bohb':
        search_algorithm = TuneBOHB(metric=metric, mode=mode, seed=seed, max_concurrent=max_concurrent,
                                    points_to_evaluate=[flatten_dict(default_param_space)])
        scheduler = HyperBandForBOHB(metric=metric, mode=mode)
    else:
        raise NotImplementedError(f"Search algorithm {search_algorithm_str} not implemented.")
    sync_config = SyncConfig(sync_artifacts=True)
    tune_config = TuneConfig(mode=mode, metric=metric, search_alg=search_algorithm,
                             scheduler=scheduler, num_samples=n_trials, time_budget_s=timeout_experiment)
    run_config = RunConfig(stop={'time_total_s': timeout_trial}, storage_path=storage_path, log_to_file=True,
                           failure_config=FailureConfig(fail_fast='raise'), sync_config=sync_config)
    return search_algorithm, tune_config, run_config
