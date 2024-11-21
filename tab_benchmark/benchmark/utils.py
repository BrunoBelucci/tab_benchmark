import json
from copy import deepcopy
import mlflow
import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from tab_benchmark.benchmark.benchmarked_models import models_dict as benchmarked_models_dict
from tab_benchmark.datasets import get_dataset
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.utils import set_seeds, evaluate_set, train_test_split_forced, flatten_dict


def check_if_exists_mlflow(experiment_name, **kwargs):
    filter_string = " AND ".join([f'params."{k}" = "{v}"' for k, v in flatten_dict(kwargs).items()])
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter_string)
    # remove ./mlruns if it is automatically created
    # if os.path.exists('./mlruns'):
    #     os.rmdir('./mlruns')
    if 'tags.was_evaluated' in runs.columns:
        runs = runs.loc[(runs['status'] == 'FINISHED') & (runs['tags.was_evaluated'])]
        if not runs.empty:
            return runs.iloc[0]
        else:
            return None
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
    model = model_class(**model_default_params)
    if hasattr(model, 'n_jobs'):
        n_jobs = model_params.get('n_jobs', n_jobs)
        if isinstance(model, DNNModel) and n_jobs == 1:
            # set n_jobs to 0 for DNNModel (no parallelism)
            setattr(model, 'n_jobs', 0)
        else:
            setattr(model, 'n_jobs', n_jobs)
    if output_dir is not None:
        output_dir = model_params.get('output_dir', output_dir)
        if hasattr(model, 'output_dir'):
            setattr(model, 'output_dir', output_dir)
    return model


def set_mlflow_tracking_uri_check_if_exists(experiment_name, mlflow_tracking_uri, check_if_exists, **kwargs):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    if check_if_exists:
        run = check_if_exists_mlflow(experiment_name, **kwargs)
    else:
        run = None
    if run is not None:
        return run
    else:
        return None


def fit_model(model, X, y, cat_ind, att_names, cat_dims, n_classes, task_name, train_indices, test_indices,
              validation_indices=None, **kwargs):
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
    # check if there is any label in y_test or y_validation that is not in y_train, if there is, copy the first
    # appearance of this instance to the training set (preferably from the validation set)
    if task_name in ('classification', 'binary_classification'):
        y_train_labels = y_train.unique()
        y_labels = y.unique()
        y_missing_labels = np.setdiff1d(y_labels, y_train_labels)
        for y_missing_label in y_missing_labels:
            if y_validation is not None:
                if y_missing_label in y_validation.values:
                    X_train = pd.concat((X_train, X_validation[y_validation == y_missing_label].head(1)), axis=0)
                    y_train = pd.concat((y_train, y_validation[y_validation == y_missing_label].head(1)), axis=0)
                else:
                    X_train = pd.concat((X_train, X_test[y_test == y_missing_label].head(1)), axis=0)
                    y_train = pd.concat((y_train, y_test[y_test == y_missing_label].head(1)), axis=0)
            else:
                X_train = pd.concat((X_train, X_test[y_test == y_missing_label].head(1)), axis=0)
                y_train = pd.concat((y_train, y_test[y_test == y_missing_label].head(1)), axis=0)
    cat_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is True]
    cont_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is False]
    if not (('classifier' in model._estimator_type and task_name in ('classification', 'binary_classification')) or (
            'regressor' in model._estimator_type and task_name in ('regression', 'multi_regression'))):
        raise ValueError('Model class and task do not match')
    # safer to preprocess and fit the model separately
    model.create_preprocess_pipeline(task_name, cat_features_names, cont_features_names, att_names)
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

    model.fit(X_train, y_train, task=task_name, cat_features=cat_features_names, cat_dims=cat_dims, n_classes=n_classes,
              eval_set=eval_set, eval_name=eval_name, **kwargs)
    return model, X_train, y_train, X_test, y_test, X_validation, y_validation


def evaluate_model(model, eval_set, eval_name, metrics, report_metric=None, n_classes=None, error_score='raise'):
    results = evaluate_set(model, eval_set, metrics, n_classes, error_score)
    if report_metric is not None:
        results['reported'] = results[report_metric]
    results_dict = {f'{eval_name}_{metric}': value for metric, value in results.items()}
    return results_dict


# just so we can reuse the same function for loading tasks from OpenML and from pandas
def load_task_from_X_y_cat_ind_att_names(X, y, cat_ind, att_names, dataset_name, n_classes, task_name, seed_dataset,
                                         resample_strategy, k_folds, pct_test, fold,
                                         create_validation_set=False, validation_resample_strategy='next_fold',
                                         pct_validation=0.1):
    cat_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is True]
    cat_dims = [len(X[cat_feature].cat.categories) for cat_feature in cat_features_names]
    if resample_strategy == 'hold_out':
        test_size = int(pct_test * len(X))
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
            kf = StratifiedKFold(n_splits=k_folds, random_state=seed_dataset, shuffle=True)
        elif task_name in ('regression', 'multi_regression'):
            kf = KFold(n_splits=k_folds, random_state=seed_dataset, shuffle=True)
        else:
            raise NotImplementedError
        folds = list(kf.split(X, y))
        train_indices, test_indices = folds[fold]
    else:
        raise NotImplementedError
    if create_validation_set:
        if validation_resample_strategy == 'next_fold':
            if resample_strategy == 'k-fold_cv':
                next_fold = (fold + 1) % k_folds
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
    return (X, y, cat_ind, att_names, cat_features_names, cat_dims, task_name, dataset_name, n_classes, train_indices,
            test_indices, validation_indices)


def load_own_task(dataset_name_or_id, seed_dataset, resample_strategy, k_folds, pct_test, fold,
                  create_validation_set=False, validation_resample_strategy='next_fold', pct_validation=0.1):
    dataset, task_name, target, n_classes = get_dataset(dataset_name_or_id)
    X, y, cat_ind, att_names = dataset.get_data(target=target)
    return load_task_from_X_y_cat_ind_att_names(X, y, cat_ind, att_names, dataset.name, n_classes, task_name,
                                                seed_dataset, resample_strategy, k_folds,
                                                pct_test, fold, create_validation_set, validation_resample_strategy,
                                                pct_validation)


def load_openml_task(task_id, task_repeat, task_sample, task_fold, create_validation_set=False):
    task = openml.tasks.get_task(task_id)
    split = task.get_train_test_split_indices(task_fold, task_repeat, task_sample)
    train_indices = split.train
    test_indices = split.test
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
    cat_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is True]
    cat_dims = [len(X[cat_feature].cat.categories) for cat_feature in cat_features_names]
    if task.task_type == 'Supervised Classification':
        n_classes = len(task.class_labels)
        if n_classes == 2:
            task_name = 'binary_classification'
        elif n_classes > 2:
            task_name = 'classification'
        else:
            raise ValueError('Task has less than 2 classes')
    elif task.task_type == 'Supervised Regression':
        n_classes = 1  # there isn't any multi-output regression task in OpenML for the moment
        task_name = 'regression'
    else:
        raise NotImplementedError
    dataset_name = dataset.name
    return (X, y, cat_ind, att_names, cat_features_names, cat_dims, task_name, dataset_name, n_classes, train_indices,
            test_indices, validation_indices)


def load_json_task(json_path, seed_dataset, resample_strategy, k_folds, pct_test, fold,
                   create_validation_set=False, validation_resample_strategy='next_fold', pct_validation=0.1):
    with open(json_path, 'r') as json_file:
        json_content = json.load(json_file)
    X = pd.read_csv(json_content['X'])
    y = pd.read_csv(json_content['y'])
    task = json_content['task']
    dataset_name = json_content['dataset_name']
    cat_ind = [True if X[feature].dtype.name == 'category' else False for feature in X.columns]
    att_names = X.columns
    if task in ('classification', 'binary_classification'):
        n_classes = len(y.unique())
    else:
        n_classes = 1
    return load_task_from_X_y_cat_ind_att_names(X, y, cat_ind, att_names, dataset_name, n_classes, task, seed_dataset,
                                                resample_strategy, k_folds,
                                                pct_test, fold, create_validation_set, validation_resample_strategy,
                                                pct_validation)
