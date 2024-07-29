import mlflow
import numpy as np
import openml
import time
from ray.tune import TuneConfig, Tuner
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.bohb import TuneBOHB
from ray.train import RunConfig
from sklearn.model_selection import StratifiedKFold, KFold
from ray.air.integrations.mlflow import setup_mlflow
from tab_benchmark.benchmark.benchmarked_models import models_dict
from tab_benchmark.datasets import get_dataset
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.utils import set_seeds, evaluate_set, get_git_revision_hash, train_test_split_forced


def check_if_exists_mlflow(experiment_name, **kwargs):
    filter_string = " AND ".join([f"params.{k} = '{v}'" for k, v in kwargs.items()])
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter_string)
    if not runs.empty:
        return True
    else:
        return False


def get_model(model_nickname, model_params, models_dict=models_dict, n_jobs=None):
    model_class, model_default_params = models_dict[model_nickname]
    if callable(model_default_params):
        model_default_params = model_default_params(model_class)
    model_default_params.update(model_params)
    model = model_class(**model_default_params)
    if hasattr(model, 'n_jobs'):
        if isinstance(model, DNNModel) and n_jobs == 1:
            # set n_jobs to 0 for DNNModel (no parallelism)
            setattr(model, 'n_jobs', 0)
        setattr(model, 'n_jobs', n_jobs)
    return model


def setup_mlflow_run(experiment_name, parent_run_uuid=None, **kwargs):
    mlflow.set_experiment(experiment_name)
    run_name = '_'.join([f'{k}={v}' for k, v in kwargs.items()])
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
    mlflow.start_run(run_name=run_name, nested=nested)
    model_params = kwargs.pop('model_params', {})
    mlflow.log_params(model_params)
    mlflow.log_params(kwargs)
    mlflow.log_param('git_hash', get_git_revision_hash())


def run_openml_combination(model_nickname, model_params, seed_model, task_id, task_repeat, task_sample, task_fold,
                           create_validation_set=False, n_jobs=1, experiment_name=None, parent_run_uuid=None,
                           mlflow_tracking_uri=None,
                           check_if_exists=False, return_to_fit=False):
    exists, logging_to_mlflow = treat_mlflow(experiment_name, mlflow_tracking_uri, check_if_exists, parent_run_uuid,
                                             model_nickname=model_nickname, model_params=model_params,
                                             seed_model=seed_model, task_id=task_id, task_repeat=task_repeat,
                                             task_sample=task_sample, task_fold=task_fold,
                                             create_validation_set=create_validation_set, n_jobs=n_jobs)
    if exists:
        return None

    try:
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
            split_validation = task.get_train_test_split_indices((task_fold + 1) % n_folds, task_repeat, task_sample)
            validation_indices = split_validation.test
            train_indices = np.setdiff1d(train_indices, validation_indices, assume_unique=True)
        else:
            validation_indices = None
        dataset = task.get_dataset()
        X, y, cat_ind, att_names = dataset.get_data(target=task.target_name)

        result = fit_model(X, y, cat_ind, att_names, model_nickname, model_params, seed_model, task_name, train_indices,
                           test_indices, validation_indices, logging_to_mlflow=logging_to_mlflow, n_jobs=n_jobs,
                           return_to_fit=return_to_fit)
    except Exception as exception:
        if logging_to_mlflow:
            mlflow.end_run('FAILED')
        raise exception
    else:
        if logging_to_mlflow:
            mlflow.end_run('FINISHED')
        return result


def treat_mlflow(experiment_name, mlflow_tracking_uri, check_if_exists, parent_run_uuid=None, **kwargs):
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    if mlflow_tracking_uri and experiment_name:
        exists = False
        if check_if_exists:
            exists = check_if_exists_mlflow(experiment_name, **kwargs)
        if exists:
            return True, False
        else:
            setup_mlflow_run(experiment_name, parent_run_uuid, **kwargs)
            return False, True
    else:
        return False, False


def run_own_combination(model_nickname, model_params, seed_model, dataset_name_or_id, seed_dataset, resample_strategy,
                        fold, n_folds, pct_test,
                        create_validation_set=False, validation_resample_strategy='next_fold', pct_validation=0.1,
                        n_jobs=1, experiment_name=None, parent_run_uuid=None,
                        mlflow_tracking_uri=None, check_if_exists=False, return_to_fit=False):
    exists, logging_to_mlflow = treat_mlflow(experiment_name, mlflow_tracking_uri, check_if_exists, parent_run_uuid,
                                             model_nickname=model_nickname, model_params=model_params,
                                             seed_model=seed_model, dataset_name_or_id=dataset_name_or_id,
                                             resample_strategy=resample_strategy, seed_dataset=seed_dataset, fold=fold,
                                             n_folds=n_folds, pct_test=pct_test,
                                             create_validation_set=create_validation_set,
                                             validation_resample_strategy=validation_resample_strategy,
                                             pct_validation=pct_validation, n_jobs=n_jobs)
    if exists:
        return None

    try:
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

        result = fit_model(X, y, cat_ind, att_names, model_nickname, model_params, seed_model, task_name, train_indices,
                           test_indices, validation_indices, logging_to_mlflow=logging_to_mlflow, n_jobs=n_jobs,
                           return_to_fit=return_to_fit)
    except Exception as exception:
        if logging_to_mlflow:
            mlflow.end_run('FAILED')
        raise exception
    else:
        if logging_to_mlflow:
            mlflow.end_run('FINISHED')
        return result


def fit_model(X, y, cat_ind, att_names, model_nickname, model_params, seed_model, task_name,
              train_indices, test_indices, validation_indices=None, n_jobs=1, logging_to_mlflow=False,
              return_to_fit=False):
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
    set_seeds(seed_model)
    model = get_model(model_nickname, model_params, n_jobs=n_jobs)
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
    else:
        eval_set = None
    if return_to_fit:
        return model, X_train, y_train, X_test, y_test, eval_set, task_name, cat_features_names
    model.fit(X_train, y_train, task=task_name, cat_features=cat_features_names, eval_set=eval_set)
    if task_name in ('classification', 'binary_classification'):
        metrics = ['logloss', 'auc']
        validation_default = 'logloss'
        n_classes = len(y.unique())
    elif task_name == 'regression':
        metrics = ['rmse', 'r2_score']
        validation_default = 'rmse'
        n_classes = None
    else:
        raise NotImplementedError
    test_results = evaluate_set(model, (X_test, y_test), metrics, n_classes)
    if logging_to_mlflow:
        mlflow.log_metrics({f'test_{metric}': value for metric, value in test_results.items()})
    if validation_indices is not None:
        validation_results = evaluate_set(model, (X_validation, y_validation), metrics, n_classes)
        validation_results['default'] = validation_results[validation_default]
        if logging_to_mlflow:
            mlflow.log_metrics({f'validation_{metric}': value for metric, value in validation_results.items()})
            mlflow.log_param('validation_default_metric', validation_default)
        return model, test_results, validation_results
    return model, test_results


def training_fn_for_openml_hpo(config):
    mlflow_tracking_uri = config.pop('mlflow_tracking_uri', None)
    experiment_name = config.pop('experiment_name', None)
    parent_run_uuid = config.pop('parent_run_uuid', None)
    model_nickname = config['model_nickname']
    seed_model = config['seed_model']
    task_id = config['task_id']
    task_repeat = config['task_repeat']
    task_sample = config['task_sample']
    task_fold = config['task_fold']
    model_params = config['model_params']
    n_jobs = config['n_jobs']
    model, test_results, validation_results = run_openml_combination(model_nickname, model_params, seed_model, task_id,
                                                                     task_repeat, task_sample, task_fold,
                                                                     create_validation_set=True, n_jobs=n_jobs,
                                                                     experiment_name=experiment_name,
                                                                     parent_run_uuid=parent_run_uuid,
                                                                     mlflow_tracking_uri=mlflow_tracking_uri,
                                                                     check_if_exists=False, return_to_fit=False)
    results = {f'test_{metric}': value for metric, value in test_results.items()}
    results.update({f'validation_{metric}': value for metric, value in validation_results.items()})
    if parent_run_uuid:
        mlflow.log_metrics(results, step=int(time.time_ns()), run_id=parent_run_uuid)
    return results


def training_fn_for_own_hpo(config):
    mlflow_tracking_uri = config.pop('mlflow_tracking_uri', None)
    experiment_name = config.pop('experiment_name', None)
    if mlflow_tracking_uri:
        setup_mlflow(config, experiment_name=experiment_name, tracking_uri=mlflow_tracking_uri)
    mlflow.log_params(config['model_params'])
    model_nickname = config['model_nickname']
    seed_model = config['seed_model']
    dataset_name_or_id = config['dataset_name_or_id']
    seed_dataset = config['seed_dataset']
    resample_strategy = config['resample_strategy']
    fold = config['fold']
    n_folds = config['n_folds']
    pct_test = config['pct_test']
    model_params = config['model_params']
    n_jobs = config['n_jobs']
    parent_run_uuid = config['parent_run_uuid']
    model, test_results, validation_results = run_own_combination(model_nickname, model_params, seed_model,
                                                                  dataset_name_or_id, seed_dataset, resample_strategy,
                                                                  fold, n_folds, pct_test, create_validation_set=True,
                                                                  n_jobs=n_jobs, experiment_name=experiment_name,
                                                                  parent_run_uuid=parent_run_uuid,
                                                                  mlflow_tracking_uri=mlflow_tracking_uri,
                                                                  check_if_exists=False, return_to_fit=False)
    results = {f'test_{metric}': value for metric, value in test_results.items()}
    results.update({f'validation_{metric}': value for metric, value in validation_results.items()})
    return results


def run_openml_combination_hpo(search_algorithm_str, n_trials, timeout_experiment, timeout_trial, storage_path,
                               model_nickname, seed_model, n_jobs, task_id, task_repeat, task_sample, task_fold,
                               mlflow_tracking_uri, experiment_name, parent_run_uuid,
                               metric='validation_default', mode='min', models_dict=models_dict):
    model_cls = models_dict[model_nickname][0]
    search_space, default_values = model_cls.create_search_space()
    param_space = dict(
        model_nickname=model_nickname,
        seed_model=seed_model,
        n_jobs=n_jobs,
        task_id=task_id,
        task_repeat=task_repeat,
        task_sample=task_sample,
        task_fold=task_fold,
        model_params=search_space,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        parent_run_uuid=parent_run_uuid,
    )
    search_algorithm, tune_config, run_config = get_search_algorithm_tune_config_run_config(default_values,
                                                                                            search_algorithm_str,
                                                                                            search_space, n_trials,
                                                                                            timeout_experiment,
                                                                                            timeout_trial,
                                                                                            storage_path,
                                                                                            metric,
                                                                                            mode)
    tuner = Tuner(trainable=training_fn_for_openml_hpo, param_space=param_space, tune_config=tune_config,
                  run_config=run_config)
    results = tuner.fit()
    return results


def run_own_combination_hpo(search_algorithm_str, n_trials, timeout_experiment, timeout_trial, storage_path,
                            model_nickname, seed_model, n_jobs, dataset_name_or_id, seed_dataset, resample_strategy,
                            fold, n_folds, pct_test, mlflow_tracking_uri, experiment_name, parent_run_uuid,
                            metric='validation_default', mode='min', models_dict=models_dict):
    model_cls = models_dict[model_nickname][0]
    search_space, default_values = model_cls.create_search_space()
    param_space = dict(
        model_nickname=model_nickname,
        seed_model=seed_model,
        n_jobs=n_jobs,
        dataset_name_or_id=dataset_name_or_id,
        seed_dataset=seed_dataset,
        resample_strategy=resample_strategy,
        fold=fold,
        n_folds=n_folds,
        pct_test=pct_test,
        model_params=search_space,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        parent_run_uuid=parent_run_uuid,
    )
    search_algorithm, tune_config, run_config = get_search_algorithm_tune_config_run_config(default_values,
                                                                                            search_algorithm_str,
                                                                                            search_space, n_trials,
                                                                                            timeout_experiment,
                                                                                            timeout_trial,
                                                                                            storage_path,
                                                                                            metric,
                                                                                            mode)
    tuner = Tuner(trainable=training_fn_for_own_hpo, param_space=param_space, tune_config=tune_config,
                  run_config=run_config)
    results = tuner.fit()
    return results


def get_search_algorithm_tune_config_run_config(default_values, search_algorithm_str, search_space, n_trials,
                                                timeout_experiment, timeout_trial, storage_path, metric, mode):
    param_space_default = dict(model_params=default_values)
    if search_algorithm_str == 'random_search':
        search_algorithm = BasicVariantGenerator(points_to_evaluate=[param_space_default])
        scheduler = None
    elif search_algorithm_str == 'bohb':
        search_algorithm = TuneBOHB(space=search_space, metric=metric, mode=mode,
                                    points_to_evaluate=[param_space_default])
        scheduler = HyperBandForBOHB()
    else:
        raise NotImplementedError(f"Search algorithm {search_algorithm_str} not implemented.")
    tune_config = TuneConfig(mode=mode, metric=metric, search_alg=search_algorithm,
                             scheduler=scheduler, num_samples=n_trials, time_budget_s=timeout_experiment)
    run_config = RunConfig(stop={'time_total_s': timeout_trial}, storage_path=storage_path, log_to_file=True)
    return search_algorithm, tune_config, run_config
