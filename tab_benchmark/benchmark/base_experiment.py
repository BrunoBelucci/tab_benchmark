from __future__ import annotations
import argparse
import shlex
import time
from pathlib import Path
from shutil import rmtree
import mlflow
import os
import logging
import warnings
import numpy as np
from distributed import WorkerPlugin, Worker, Client
import dask
from tab_benchmark.benchmark.utils import set_mlflow_tracking_uri_check_if_exists, get_model, load_openml_task, \
    fit_model, evaluate_model, \
    load_own_task
from tab_benchmark.benchmark.benchmarked_models import models_dict as benchmarked_models_dict
from tab_benchmark.utils import get_git_revision_hash, flatten_dict
from dask.distributed import LocalCluster, get_worker, as_completed
from dask_jobqueue import SLURMCluster
from tqdm.auto import tqdm
from torch.cuda import (set_per_process_memory_fraction, max_memory_reserved, max_memory_allocated,
                        reset_peak_memory_stats)
from resource import getrusage, RUSAGE_SELF
from itertools import product
from random import SystemRandom
import json

warnings.simplefilter(action='ignore', category=FutureWarning)


class MLFlowCleanupPlugin(WorkerPlugin):
    def teardown(self, worker: Worker):
        if mlflow.active_run() is not None:
            mlflow.log_param('EXCEPTION', f'KILLED, worker status {worker.status}')
            mlflow.end_run('KILLED')


class LoggingSetterPlugin(WorkerPlugin):
    def __init__(self, logging_config=None):
        self.logging_config = logging_config if logging_config is not None else {}
        super().__init__()

    def setup(self, worker: Worker):
        logging.basicConfig(**self.logging_config)


def log_and_print_msg(first_line, **kwargs):
    slurm_job_id = os.getenv('SLURM_JOB_ID', None)
    if slurm_job_id is not None:
        first_line = f"SLURM_JOB_ID: {slurm_job_id}\n{first_line}"
    first_line = f"{first_line}\n"
    first_line += "".join([f"{key}: {value}\n" for key, value in kwargs.items()])
    print(first_line)
    logging.info(first_line)


class BaseExperiment:
    def __init__(
            self,
            # model specific
            models_nickname=None, seeds_models=None, n_jobs=1,
            models_params=None, fits_params=None,
            # when performing our own resampling
            datasets_names_or_ids=None, seeds_datasets=None,
            resample_strategy='k-fold_cv', k_folds=10, folds=None, pct_test=0.2,
            validation_resample_strategy='next_fold', pct_validation=0.1,
            # when using openml tasks
            tasks_ids=None,
            task_repeats=None, task_folds=None, task_samples=None,
            # parameters of experiment
            experiment_name='base_experiment',
            models_dict=None,
            log_dir=Path.cwd() / 'logs',
            output_dir=Path.cwd() / 'output',
            clean_output_dir=True,
            raise_on_fit_error=False, parser=None,
            error_score='raise',
            # mlflow specific
            log_to_mlflow=True,
            mlflow_tracking_uri='sqlite:///' + str(Path.cwd().resolve()) + '/tab_benchmark.db', check_if_exists=True,
            # parallelization
            dask_cluster_type=None,
            n_workers=1,
            n_processes=1,
            n_cores=1,
            dask_memory=None,
            dask_job_extra_directives=None,
            dask_address=None,
            # gpu specific
            n_gpus=0,
    ):
        self.models_nickname = models_nickname if models_nickname else []
        self.seeds_model = seeds_models if seeds_models else [0]
        self.n_jobs = n_jobs

        self.models_params = self.validate_dict_of_models_params(models_params, self.models_nickname)
        self.fits_params = self.validate_dict_of_models_params(fits_params, self.models_nickname)

        # when performing our own resampling0
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

        self.using_own_resampling = None

        # parallelization
        self.dask_cluster_type = dask_cluster_type
        self.n_workers = n_workers
        self.n_cores = n_cores
        self.n_processes = n_processes
        self.dask_memory = dask_memory
        self.dask_job_extra_directives = dask_job_extra_directives
        self.dask_address = dask_address
        self.n_gpus = n_gpus

        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.clean_output_dir = clean_output_dir
        self.log_to_mlflow = log_to_mlflow
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.check_if_exists = check_if_exists
        self.parser = parser
        self.models_dict = models_dict if models_dict else benchmarked_models_dict.copy()
        self.raise_on_fit_error = raise_on_fit_error
        self.error_score = error_score
        self.client = None
        self.logger_filename = None

    def validate_dict_of_models_params(self, dict_to_validate, models_nickname):
        dict_to_validate = dict_to_validate or {model_nickname: {} for model_nickname in models_nickname}
        models_missing = [model_nickname for model_nickname in models_nickname
                          if model_nickname not in dict_to_validate]
        if len(models_missing) == len(models_nickname):
            dict_to_validate = {model_nickname: dict_to_validate.copy() for model_nickname in models_nickname}
        else:
            for model_missing in models_missing:
                dict_to_validate[model_missing] = {}
        return dict_to_validate

    def add_arguments_to_parser(self):
        self.parser.add_argument('--experiment_name', type=str, default=self.experiment_name)
        self.parser.add_argument('--models_nickname', type=str, choices=self.models_dict.keys(),
                                 nargs='*', default=self.models_nickname)
        self.parser.add_argument('--seeds_model', nargs='*', type=int, default=self.seeds_model)
        self.parser.add_argument('--n_jobs', type=int, default=self.n_jobs,
                                 help='Number of threads/cores to be used by the model if it supports it. '
                                      'Obs.: In the CEREMADE cluster the minimum number of cores that can be requested'
                                      'are 2, so it is a good idea to set at least n_jobs to 2 if we want to use all '
                                      'the resources available.')
        self.parser.add_argument('--models_params', type=json.loads, default=self.models_params,
                                 help='Dictionary with the parameters of the models. The keys must be the nickname '
                                      'of the model and the values must be a dictionary with the parameters of '
                                      'the model. In case only one dictionary is passed, it will be used for '
                                      'all models. The dictionary is passed as a string in the json format, depending '
                                      'on your shell interpreter you may '
                                      'have to escape the quotes and not use spaces if you are using the command line.'
                                      'Examples of usage from cli: '
                                      '--models_params {\\"[MODEL_NICKNAME]\\":{\\"[NAME_PARAM_1]\\":[VALUE_PARAM_1],'
                                      '\\"[NAME_PARAM_2]\\":[VALUE_PARAM_2]},'
                                      '\\"[MODEL_NICKNAME_2]\\":{\\"[NAME_PARAM_1]\\":[VALUE_PARAM_1]}}'
                                      '--models_params {\\"[NAME_PARAM_1]\\":[VALUE_PARAM_1]}')
        self.parser.add_argument('--fits_params', type=json.loads, default=self.fits_params,
                                 help='Dictionary with the parameters of the fits. The keys must be the nickname '
                                      'of the model and the values must be a dictionary with the parameters of '
                                      'the fit. In case only one dictionary is passed, it will be used for '
                                      'all models. The dictionary is passed as a string in the json format, you may'
                                      'have to escape the quotes and not use spaces if you are using the command line. '
                                      'Please refer to the --models_params argument for examples of usage.')
        self.parser.add_argument('--error_score', type=str, default=self.error_score)

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

        self.parser.add_argument('--log_dir', type=Path, default=self.log_dir)
        self.parser.add_argument('--output_dir', type=Path, default=self.output_dir)
        self.parser.add_argument('--do_not_clean_output_dir', action='store_true')
        self.parser.add_argument('--do_not_log_to_mlflow', action='store_true')
        self.parser.add_argument('--mlflow_tracking_uri', type=str, default=self.mlflow_tracking_uri)
        self.parser.add_argument('--do_not_check_if_exists', action='store_true')
        self.parser.add_argument('--do_not_retry_on_oom', action='store_true')
        self.parser.add_argument('--raise_on_fit_error', action='store_true')

        self.parser.add_argument('--dask_cluster_type', type=str, default=self.dask_cluster_type)
        self.parser.add_argument('--n_workers', type=int, default=self.n_workers,
                                 help='Maximum number of workers to be used.')
        self.parser.add_argument('--n_cores', type=int, default=self.n_cores,
                                 help='Number of cores per job that will be requested in the cluster. It is ignored'
                                      'if the cluster type is local.')
        self.parser.add_argument('--n_processes', type=int, default=self.n_processes,
                                 help='Number of processes to use when parallelizing with dask.')
        self.parser.add_argument('--dask_memory', type=str, default=self.dask_memory)
        self.parser.add_argument('--dask_job_extra_directives', type=str, default=self.dask_job_extra_directives)
        self.parser.add_argument('--dask_address', type=str, default=self.dask_address)
        self.parser.add_argument('--n_gpus', type=int, default=self.n_gpus,
                                 help='Number of GPUs per job that will be requested in the cluster.'
                                      'Note that this will not allocate the GPU in the cluster, we must still pass '
                                      'the required resource allocation parameter to the cluster (we can do this via'
                                      'the dask_job_extra_directives argument, for example with '
                                      '--dask_job_extra_directives "-G 1").')

    def unpack_parser(self):
        args = self.parser.parse_args()
        self.experiment_name = args.experiment_name
        self.models_nickname = args.models_nickname
        self.n_jobs = args.n_jobs
        models_params = args.models_params
        self.models_params = self.validate_dict_of_models_params(models_params, self.models_nickname)
        fits_params = args.fits_params
        self.fits_params = self.validate_dict_of_models_params(fits_params, self.models_nickname)
        error_score = args.error_score
        if error_score == 'nan':
            error_score = np.nan
        self.error_score = error_score

        self.datasets_names_or_ids = args.datasets_names_or_ids
        self.seeds_datasets = args.seeds_datasets
        self.seeds_model = args.seeds_model
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

        self.log_dir = args.log_dir
        output_dir = args.output_dir
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        self.output_dir = output_dir
        self.clean_output_dir = not args.do_not_clean_output_dir
        self.log_to_mlflow = not args.do_not_log_to_mlflow
        self.mlflow_tracking_uri = args.mlflow_tracking_uri
        self.check_if_exists = not args.do_not_check_if_exists
        self.raise_on_fit_error = args.raise_on_fit_error

        self.dask_cluster_type = args.dask_cluster_type
        self.n_workers = args.n_workers
        self.n_cores = args.n_cores
        self.n_processes = args.n_processes
        self.dask_memory = args.dask_memory
        dask_job_extra_directives = args.dask_job_extra_directives
        # parse dask_job_extra_directives
        if isinstance(dask_job_extra_directives, str):
            # the following was generated by chatgpt, it seems to work
            dask_job_extra_directives = shlex.split(dask_job_extra_directives)
            dask_job_extra_directives = [
                f"{dask_job_extra_directives[i]} {dask_job_extra_directives[i + 1]}"
                if i + 1 < len(dask_job_extra_directives) and not dask_job_extra_directives[i + 1].startswith('-')
                else dask_job_extra_directives[i]
                for i in range(len(dask_job_extra_directives)) if dask_job_extra_directives[i].startswith('-')
            ]
        else:
            dask_job_extra_directives = []
        self.dask_job_extra_directives = dask_job_extra_directives
        self.dask_address = args.dask_address
        self.n_gpus = args.n_gpus
        return args

    def treat_parser(self):
        if self.parser is not None:
            self.add_arguments_to_parser()
            self.unpack_parser()

    def setup_logger(self, log_dir=None, filemode='w'):
        if log_dir is None:
            log_dir = self.log_dir
        os.makedirs(log_dir, exist_ok=True)
        if self.logger_filename is None:
            name = self.experiment_name
            if (log_dir / f'{name}.log').exists():
                file_names = sorted(log_dir.glob(f'{name}_????.log'))
                if file_names:
                    file_name = file_names[-1].name
                    id_file = int(file_name.split('_')[-1].split('.')[0])
                    name = f'{name}_{id_file + 1:04d}'
                else:
                    name = name + '_0001'
            logger_filename = f'{name}.log'
            self.logger_filename = logger_filename
        else:
            logger_filename = self.logger_filename
        logging.basicConfig(filename=log_dir / logger_filename,
                            format='%(asctime)s - %(levelname)s\n%(message)s\n',
                            level=logging.INFO, filemode=filemode)

    def get_model(self, model_params=None, n_jobs=1,
                  log_to_mlflow=False, run_id=None, create_validation_set=False, output_dir=None, data_return=None,
                  **kwargs):
        model_nickname = kwargs.get('model_nickname')
        seed_model = kwargs.get('seed_model')
        model_params = model_params if model_params else self.models_params.get(model_nickname, {}).copy()
        if data_return:
            data_params = data_return.get('data_params', None).copy()
        else:
            data_params = None
        models_dict = self.models_dict.copy()
        if output_dir is None:
            # if logging to mlflow we use the mlflow artifact directory
            if log_to_mlflow:
                # this is already unique
                run = mlflow.get_run(run_id)
                artifact_uri = run.info.artifact_uri
                output_dir = Path(artifact_uri)
                unique_name = output_dir.parts[-2]
            else:
                # if we only save the model in the output_dir we will have some problems when running in parallel
                # because the workers will try to save the model in the same directory, so we must create a unique
                # directory for each combination model/dataset
                if data_params is not None:
                    unique_name = '_'.join([f'{k}={v}' for k, v in data_params.items()])
                else:
                    # not sure, but I think it will generate a true random number and even be thread safe
                    unique_name = f'{SystemRandom().randint(0, 1000000):06d}'
                output_dir = self.output_dir / f'{model_nickname}_{unique_name}'
            # if running on a worker, we use the worker's local directory
            try:
                worker = get_worker()
                output_dir = Path(worker.local_directory) / unique_name
            except ValueError:
                # if running on the main process, we use the output_dir
                output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        model = get_model(model_nickname, seed_model, model_params, models_dict, n_jobs, output_dir=output_dir)
        if log_to_mlflow:
            if hasattr(model, 'run_id'):
                setattr(model, 'run_id', run_id)
        if create_validation_set:
            # we disable auto early stopping when creating a validation set, because we will use it to validate
            if hasattr(model, 'auto_early_stopping'):
                model.auto_early_stopping = False
        if log_to_mlflow:
            model_params = vars(model).copy()
            if hasattr(model, 'loss_fn'):
                # will be logged after
                del model_params['loss_fn']
            mlflow.log_params(model_params, run_id=run_id)
            # just to make it easier to filter after
            if model_nickname.find('TabBenchmark') != -1:
                log_params = {'model_name': model_nickname[len('TabBenchmark'):]}
                mlflow.log_params(log_params, run_id=run_id)
        return model

    def load_data(self, create_validation_set=False, log_to_mlflow=False, run_id=None, **kwargs):
        is_openml_task = kwargs.get('is_openml_task', False)
        if is_openml_task:
            keys = ['task_id', 'task_repeat', 'task_fold', 'task_sample']
        else:
            keys = ['dataset_name_or_id', 'seed_dataset', 'resample_strategy', 'n_folds', 'fold', 'pct_test',
                    'validation_resample_strategy', 'pct_validation']
        data_params = {key: kwargs.get(key) for key in keys}
        if is_openml_task:
            (X, y, cat_ind, att_names, cat_features_names, cat_dims, task_name, n_classes, train_indices,
             test_indices, validation_indices) = load_openml_task(create_validation_set=create_validation_set,
                                                                  **data_params)
        else:
            (X, y, cat_ind, att_names, cat_features_names, cat_dims, task_name, n_classes, train_indices,
             test_indices, validation_indices) = (
                load_own_task(**data_params,
                              create_validation_set=create_validation_set, log_to_mlflow=log_to_mlflow, run_id=run_id)
            )
        data_return = dict(X=X, y=y, cat_ind=cat_ind, att_names=att_names, cat_features_names=cat_features_names,
                           cat_dims=cat_dims, task_name=task_name, n_classes=n_classes, train_indices=train_indices,
                           test_indices=test_indices, validation_indices=validation_indices,
                           data_params=data_params)
        return data_return

    def get_metrics(self, data_return, **kwargs):
        task_name = data_return['task_name']
        if task_name in ('classification', 'binary_classification'):
            metrics = ['logloss', 'auc']
            default_metric = 'logloss'
        elif task_name == 'regression':
            metrics = ['rmse', 'r2_score']
            default_metric = 'rmse'
        else:
            raise NotImplementedError
        return metrics, default_metric

    def fit_model(self, model, fit_params, data_return, metrics, default_metric, **kwargs):
        cat_features_names = data_return['cat_features_names']
        X = data_return['X']
        y = data_return['y']
        task_name = data_return['task_name']
        cat_ind = data_return['cat_ind']
        att_names = data_return['att_names']
        cat_dims = data_return['cat_dims']
        n_classes = data_return['n_classes']
        train_indices = data_return['train_indices']
        test_indices = data_return['test_indices']
        validation_indices = data_return['validation_indices']
        fit_params = fit_params if fit_params else self.fits_params.get(kwargs.get('model_nickname'), {}).copy()
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
        fit_return = dict(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                          X_validation=X_validation, y_validation=y_validation)
        return fit_return

    def evaluate_model(self, metrics, default_metric, fit_return, data_return, create_validation_set=False,
                       log_to_mlflow=False, run_id=None, **kwargs):
        model = fit_return['model']
        X_test = fit_return['X_test']
        y_test = fit_return['y_test']
        n_classes = data_return['n_classes']
        X_validation = fit_return['X_validation']
        y_validation = fit_return['y_validation']
        evaluate_results = evaluate_model(model=model, eval_set=(X_test, y_test), eval_name='final_test',
                                          metrics=metrics, default_metric=default_metric, n_classes=n_classes,
                                          error_score=self.error_score, log_to_mlflow=log_to_mlflow, run_id=run_id)
        if create_validation_set:
            validation_results = evaluate_model(model=model, eval_set=(X_validation, y_validation),
                                                eval_name='final_validation', metrics=metrics,
                                                default_metric=default_metric, n_classes=n_classes,
                                                error_score=self.error_score, log_to_mlflow=log_to_mlflow,
                                                run_id=run_id)
            evaluate_results.update(validation_results)
        return evaluate_results

    def train_model(self,
                    n_jobs=1, create_validation_set=False,
                    model_params=None,
                    fit_params=None, return_results=False, clean_output_dir=True, log_to_mlflow=False, run_id=None,
                    **kwargs):
        try:
            results = {}
            start_time = time.perf_counter()
            # logging
            log_and_print_msg('Running...', **kwargs)
            if self.n_gpus > 0:
                # Number of gpus in this job / Number of models being trained in parallel
                fraction_of_gpu_being_used = self.n_gpus / (self.n_cores / self.n_jobs)
                set_per_process_memory_fraction(fraction_of_gpu_being_used)
                reset_peak_memory_stats()
            model_nickname = kwargs.get('model_nickname')
            model_params = model_params if model_params else self.models_params.get(model_nickname, {}).copy()
            fit_params = fit_params if fit_params else self.fits_params.get(kwargs.get('model_nickname'), {}).copy()

            # load data
            data_return = self.load_data(create_validation_set=create_validation_set, log_to_mlflow=log_to_mlflow,
                                         run_id=run_id,
                                         **kwargs)
            results['data_return'] = data_return

            # load model
            model = self.get_model(model_params=model_params, n_jobs=n_jobs, log_to_mlflow=log_to_mlflow, run_id=run_id,
                                   create_validation_set=create_validation_set, data_return=data_return, **kwargs)
            results['model'] = model

            # get metrics
            metrics, default_metric = self.get_metrics(data_return, **kwargs)
            results['metrics'] = metrics
            results['default_metric'] = default_metric

            # fit model
            fit_return = self.fit_model(model=model, fit_params=fit_params, data_return=data_return, metrics=metrics,
                                        default_metric=default_metric, **kwargs)
            results['fit_return'] = fit_return

            # evaluate model
            evaluate_return = self.evaluate_model(metrics=metrics, default_metric=default_metric, fit_return=fit_return,
                                                  data_return=data_return, create_validation_set=create_validation_set,
                                                  log_to_mlflow=log_to_mlflow, run_id=run_id, **kwargs)
            results['evaluate_return'] = evaluate_return

            if log_to_mlflow:
                log_params = {'was_evaluated': True}
                mlflow.log_params(log_params, run_id=run_id)
                # in MB (in linux getrusage seems to returns in KB)
                log_metrics = {'max_memory_used': getrusage(RUSAGE_SELF).ru_maxrss / 1000}
                if self.n_gpus > 0:
                    log_metrics['max_cuda_memory_reserved'] = max_memory_reserved() / (1024 ** 2)  # in MB
                    log_metrics['max_cuda_memory_allocated'] = max_memory_allocated() / (1024 ** 2)  # in MB
                mlflow.log_metrics(log_metrics, run_id=run_id)

        except Exception as exception:
            if log_to_mlflow:
                log_params = {'was_evaluated': False, 'EXCEPTION': str(exception)}
                mlflow.log_params(log_params, run_id=run_id)
                mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
                mlflow_client.set_terminated(run_id, status='FAILED')
            try:
                total_time = time.perf_counter() - start_time
                if log_to_mlflow:
                    log_params = {'elapsed_time': total_time}
                    mlflow.log_params(log_params, run_id=run_id)
            except UnboundLocalError:
                total_time = 'unknown'
            try:
                kwargs_with_error = kwargs.copy()
            except UnboundLocalError:
                kwargs_with_error = kwargs.copy()
            log_and_print_msg('Error while running', exception=exception, elapsed_time=total_time, **kwargs_with_error)
            if self.raise_on_fit_error:
                raise exception
            if return_results:
                try:
                    return results
                except UnboundLocalError:
                    return {}
            else:
                return False
        else:
            total_time = time.perf_counter() - start_time
            if log_to_mlflow:
                log_params = {'elapsed_time': total_time}
                mlflow.log_params(log_params, run_id=run_id)
                mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
                mlflow_client.set_terminated(run_id, status='FINISHED')
            log_and_print_msg('Finished!', elapsed_time=total_time, **kwargs)
            if clean_output_dir:
                output_dir = model.output_dir
                if output_dir.exists():
                    rmtree(output_dir)
            if return_results:
                return results
            else:
                return True

    def log_run_start_params(self, run_id, **run_unique_params):
        params_to_log = flatten_dict(run_unique_params).copy()
        params_to_log.update(dict(
            git_hash=get_git_revision_hash(),
            # slurm parameters
            SLURM_JOB_ID=os.getenv('SLURM_JOB_ID', None),
            SLURMD_NODENAME=os.getenv('SLURMD_NODENAME', None),
            # dask parameters
            dask_cluster_type=self.dask_cluster_type,
            n_workers=self.n_workers,
            n_cores=self.n_cores,
            n_processes=self.n_processes,
            dask_memory=self.dask_memory,
            dask_job_extra_directives=self.dask_job_extra_directives,
            dask_address=self.dask_address,
            n_gpus=self.n_gpus,
        ))
        mlflow.log_params(params_to_log, run_id=run_id)

    def run_mlflow_and_train_model(self,
                                   n_jobs=1, create_validation_set=False,
                                   model_params=None,
                                   fit_params=None, return_results=False, clean_output_dir=True,
                                   run_id=None,
                                   experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                                   fn_to_train_model=None,
                                   **kwargs):
        if fn_to_train_model is None:
            fn_to_train_model = self.train_model
        # get unique parameters for mlflow and check if the run already exists
        model_nickname = kwargs.get('model_nickname')
        model_params = model_params if model_params else self.models_params.get(model_nickname, {}).copy()
        fit_params = fit_params if fit_params else self.fits_params.get(model_nickname, {}).copy()
        run_unique_params = dict(model_params=model_params, fit_params=fit_params,
                                 create_validation_set=create_validation_set, **kwargs)
        possible_existent_run = set_mlflow_tracking_uri_check_if_exists(experiment_name, mlflow_tracking_uri,
                                                                        check_if_exists, **run_unique_params)
        if possible_existent_run is not None:
            log_and_print_msg('Run already exists on MLflow. Skipping...')
            if return_results:
                return possible_existent_run.to_dict()
            else:
                return True

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location=str(self.output_dir))
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        mlflow_client = mlflow.client.MlflowClient(tracking_uri=mlflow_tracking_uri)
        if run_id is None:
            run = mlflow_client.create_run(experiment_id)
            run_id = run.info.run_id

        mlflow_client.update_run(run_id, status='RUNNING')
        self.log_run_start_params(run_id, **run_unique_params)
        return fn_to_train_model(n_jobs=n_jobs,
                                 create_validation_set=run_unique_params.pop('create_validation_set'),
                                 model_params=run_unique_params.pop('model_params'),
                                 fit_params=run_unique_params.pop('fit_params'), return_results=return_results,
                                 clean_output_dir=clean_output_dir,
                                 log_to_mlflow=True, run_id=run_id,
                                 experiment_name=experiment_name,
                                 mlflow_tracking_uri=mlflow_tracking_uri, check_if_exists=check_if_exists,
                                 **run_unique_params)

    def setup_dask(self, n_workers, cluster_type='local', address=None):
        if address is not None:
            client = Client(address)
        else:
            if cluster_type == 'local':
                threads_per_worker = self.n_jobs
                processes = self.n_processes
                cores = self.n_cores
                if cores < threads_per_worker * n_workers:
                    warnings.warn(f"n_workers * threads_per_worker (n_jobs) is greater than the number of cores "
                                  f"available ({cores}). This may lead to performance issues.")
                resources = {'cores': cores, 'gpus': self.n_gpus, 'processes': processes}
                cluster = LocalCluster(n_workers=0, memory_limit=self.dask_memory, processes=True,
                                       threads_per_worker=threads_per_worker, resources=resources)
                cluster.scale(n_workers)
            elif cluster_type == 'slurm':
                cores = self.n_cores
                processes = self.n_processes
                resources_per_work = {'cores': cores, 'gpus': self.n_gpus, 'processes': processes}
                job_extra_directives = dask.config.get(
                    "jobqueue.slurm.job-extra-directives", []
                )
                job_script_prologue = dask.config.get(
                    "jobqueue.slurm.job-script-prologue", []
                )
                worker_extra_args = dask.config.get(
                    "jobqueue.slurm.worker-extra-args", []
                )
                job_extra_directives = job_extra_directives + self.dask_job_extra_directives
                job_script_prologue = job_script_prologue + ['eval "$(conda shell.bash hook)"',
                                                             'conda activate tab_benchmark']
                resources_per_work_string = ' '.join([f'{key}={value}' for key, value in resources_per_work.items()])
                worker_extra_args = worker_extra_args + [f'--resources "{resources_per_work_string}"']
                walltime = '364-23:59:59'
                job_name = f'dask-worker-{self.experiment_name}'
                cluster = SLURMCluster(cores=cores, memory=self.dask_memory, processes=processes,
                                       job_extra_directives=job_extra_directives,
                                       job_script_prologue=job_script_prologue, walltime=walltime,
                                       job_name=job_name, worker_extra_args=worker_extra_args)
                log_and_print_msg(f"Cluster script generated:\n{cluster.job_script()}")
                cluster.scale(n_workers)
            else:
                raise ValueError("cluster_type must be either 'local' or 'slurm'.")
            log_and_print_msg("Cluster dashboard address", dashboard_address=cluster.dashboard_link)
            client = cluster.get_client()
        logging_plugin = LoggingSetterPlugin(logging_config={'level': logging.INFO})
        client.register_plugin(logging_plugin)
        mlflow_plugin = MLFlowCleanupPlugin()
        client.register_plugin(mlflow_plugin)
        client.forward_logging()
        return client

    def run_openml_task_combination(self, model_nickname, seed_model, task_id,
                                    task_fold=0, task_repeat=0, task_sample=0, run_id=None,
                                    n_jobs=1, create_validation_set=False,
                                    model_params=None,
                                    fit_params=None, return_results=False, clean_output_dir=True,
                                    log_to_mlflow=False,
                                    experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                                    **kwargs):
        task_combination = dict(model_nickname=model_nickname, seed_model=seed_model, task_id=task_id,
                                task_fold=task_fold, task_repeat=task_repeat, task_sample=task_sample,
                                is_openml_task=True)
        task_combination.update(kwargs)
        if log_to_mlflow:
            return self.run_mlflow_and_train_model(n_jobs=n_jobs, create_validation_set=create_validation_set,
                                                   model_params=model_params, fit_params=fit_params,
                                                   return_results=return_results, clean_output_dir=clean_output_dir,
                                                   run_id=run_id, experiment_name=experiment_name,
                                                   mlflow_tracking_uri=mlflow_tracking_uri,
                                                   check_if_exists=check_if_exists,
                                                   **task_combination)
        return self.train_model(n_jobs=n_jobs, create_validation_set=create_validation_set,
                                model_params=model_params, logging_to_mlflow=False,
                                fit_params=fit_params, return_results=return_results, clean_output_dir=clean_output_dir,
                                **task_combination)

    def run_openml_dataset_combination(self, model_nickname, seed_model, dataset_name_or_id, seed_dataset,
                                       fold=0, run_id=None,
                                       resample_strategy='k-fold_cv', n_folds=10, pct_test=0.2,
                                       validation_resample_strategy='next_fold', pct_validation=0.1,
                                       n_jobs=1, create_validation_set=False,
                                       model_params=None,
                                       fit_params=None, return_results=False, clean_output_dir=True,
                                       log_to_mlflow=False,
                                       experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                                       **kwargs):
        dataset_combination = dict(model_nickname=model_nickname, seed_model=seed_model,
                                   dataset_name_or_id=dataset_name_or_id,
                                   seed_dataset=seed_dataset, fold=fold, resample_strategy=resample_strategy,
                                   n_folds=n_folds,
                                   pct_test=pct_test, validation_resample_strategy=validation_resample_strategy,
                                   pct_validation=pct_validation, is_openml_task=False)
        dataset_combination.update(kwargs)
        if log_to_mlflow:
            return self.run_mlflow_and_train_model(n_jobs=n_jobs, create_validation_set=create_validation_set,
                                                   model_params=model_params, fit_params=fit_params,
                                                   return_results=return_results, clean_output_dir=clean_output_dir,
                                                   run_id=run_id, experiment_name=experiment_name,
                                                   mlflow_tracking_uri=mlflow_tracking_uri,
                                                   check_if_exists=check_if_exists,
                                                   **dataset_combination)
        return self.train_model(n_jobs=n_jobs, create_validation_set=create_validation_set,
                                model_params=model_params, logging_to_mlflow=False,
                                fit_params=fit_params, return_results=return_results, clean_output_dir=clean_output_dir,
                                **dataset_combination)

    def run_combination(self, *args, **kwargs):
        is_openml_task = kwargs.get('is_openml_task', False)
        if is_openml_task:
            return self.run_openml_task_combination(*args, **kwargs)
        else:
            return self.run_openml_dataset_combination(*args, **kwargs)

    def get_combinations(self):
        if self.using_own_resampling:
            # (model_nickname, seed_model, dataset_name_or_id, seed_dataset, fold)
            combinations = list(product(self.models_nickname, self.seeds_model, self.datasets_names_or_ids,
                                        self.seeds_datasets, self.folds))
            extra_params = dict(is_openml_task=False, resample_strategy=self.resample_strategy,
                                n_folds=self.k_folds, pct_test=self.pct_test,
                                validation_resample_strategy=self.validation_resample_strategy,
                                pct_validation=self.pct_validation)

        else:
            # (model_nickname, seed_model, task_id, task_fold, task_repeat, task_sample)
            combinations = list(product(self.models_nickname, self.seeds_model, self.tasks_ids, self.task_folds,
                                        self.task_repeats, self.task_samples))
            extra_params = dict(is_openml_task=True)
        extra_params.update(dict(n_jobs=self.n_jobs, log_to_mlflow=self.log_to_mlflow,
                                 return_results=False, clean_output_dir=self.clean_output_dir,
                                 create_validation_set=False, experiment_name=self.experiment_name,
                                 mlflow_tracking_uri=self.mlflow_tracking_uri, check_if_exists=self.check_if_exists))
        return combinations, extra_params

    def create_mlflow_run(self, *args,
                          create_validation_set=False,
                          model_params=None,
                          fit_params=None,
                          experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                          **kwargs):
        is_openml_task = kwargs.get('is_openml_task', False)
        if is_openml_task:
            model_nickname, seed_model, task_id, task_fold, task_repeat, task_sample = args
            data_combination = dict(model_nickname=model_nickname, seed_model=seed_model, task_id=task_id,
                                    task_fold=task_fold, task_repeat=task_repeat, task_sample=task_sample,
                                    is_openml_task=True)
        else:
            model_nickname, seed_model, dataset_name_or_id, seed_dataset, fold = args
            # resample_strategy = kwargs.get('resample_strategy')
            # n_folds = kwargs.get('n_folds')
            # pct_test = kwargs.get('pct_test')
            # validation_resample_strategy = kwargs.get('validation_resample_strategy')
            # pct_validation = kwargs.get('pct_validation')
            data_combination = dict(model_nickname=model_nickname, seed_model=seed_model,
                                    dataset_name_or_id=dataset_name_or_id,
                                    seed_dataset=seed_dataset, fold=fold, is_openml_task=False)
        # get unique parameters for mlflow and check if the run already exists
        model_nickname = kwargs.get('model_nickname')
        model_params = model_params if model_params else self.models_params.get(model_nickname, {}).copy()
        fit_params = fit_params if fit_params else self.fits_params.get(model_nickname, {}).copy()
        run_unique_params = dict(model_params=model_params, fit_params=fit_params,
                                 create_validation_set=create_validation_set, **kwargs)
        run_unique_params.update(data_combination)
        possible_existent_run = set_mlflow_tracking_uri_check_if_exists(experiment_name, mlflow_tracking_uri,
                                                                        check_if_exists, **run_unique_params)
        if possible_existent_run is not None:
            return possible_existent_run.run_id
        else:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name, artifact_location=str(self.output_dir))
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_name)
            mlflow_client = mlflow.client.MlflowClient(tracking_uri=mlflow_tracking_uri)
            run = mlflow_client.create_run(experiment_id)
            run_id = run.info.run_id
            mlflow_client.update_run(run_id, status='SCHEDULED')
            return run_id

    def run_experiment(self, client=None):

        combinations, extra_params = self.get_combinations()
        n_args = len(combinations[0])

        total_combinations = len(combinations)
        n_combinations_successfully_completed = 0
        n_combinations_failed = 0
        n_combinations_none = 0
        if client is not None:
            first_args = list(combinations[0])
            list_of_args = [[combination[i] for combination in combinations[1:]] for i in range(n_args)]
            # we will first create the mlflow runs to avoid threading problems
            if self.log_to_mlflow:
                resources_per_task = {'processes': 1}
                first_future = client.submit(self.create_mlflow_run, *first_args, resources=resources_per_task,
                                             **extra_params)
                futures = [first_future]
                if total_combinations > 1:
                    time.sleep(5)
                    other_futures = client.map(self.create_mlflow_run, *list_of_args,
                                               batch_size=self.n_workers, resources=resources_per_task, **extra_params)
                    futures.extend(other_futures)
                run_ids = client.gather(futures)
                first_args.append(run_ids[0])  # add the run_id to the first args
                list_of_args.append(run_ids[1:])  # add the run_ids to the list of args
            if hasattr(self, 'n_trials'):
                # the resources are actually used when training the models, here we will launch the hpo framework
                resources_per_task = {'cores': 0, 'gpus': 0, 'processes': 1}
            else:
                resources_per_task = {'cores': self.n_jobs, 'gpus': self.n_gpus / (self.n_cores / self.n_jobs)}
            log_and_print_msg(f'{total_combinations} models are being trained and evaluated in parallel, '
                              f'check the logs for real time information. We will display information about the '
                              f'completion of the tasks right after sending all the tasks to the cluster. '
                              f'Note that this can take a while if a lot of tasks are being submitted. '
                              f'You can check the dask dashboard to get more information about the progress and '
                              f'the workers.')

            first_future = client.submit(self.run_combination, *first_args,
                                         resources=resources_per_task, **extra_params)
            futures = [first_future]
            if total_combinations > 1:
                # wait a little bit for the first submission to create folders, experiments, etc
                time.sleep(5)
                other_futures = client.map(self.run_combination, *list_of_args,
                                           batch_size=self.n_workers, resources=resources_per_task, **extra_params)
                futures.extend(other_futures)
        else:
            progress_bar = tqdm(combinations, desc='Combinations completed')
            for combination in progress_bar:
                combination_success = self.run_combination(*combination, **extra_params)
                if combination_success is True:
                    n_combinations_successfully_completed += 1
                elif combination_success is False:
                    n_combinations_failed += 1
                else:
                    n_combinations_none += 1
                log_and_print_msg(str(progress_bar), succesfully_completed=n_combinations_successfully_completed,
                                  failed=n_combinations_failed, none=n_combinations_none)

        if client is not None:
            progress_bar = tqdm(as_completed(futures), total=len(futures), desc='Combinations completed')
            for i, future in enumerate(progress_bar):
                log_and_print_msg(str(progress_bar), succesfully_completed=n_combinations_successfully_completed,
                                  failed=n_combinations_failed, none=n_combinations_none, i=i)
                combination_success = future.result()
                if combination_success is True:
                    n_combinations_successfully_completed += 1
                elif combination_success is False:
                    n_combinations_failed += 1
                else:
                    n_combinations_none += 1
                # future.release()  # release the memory of the future
                # del future  # to free memory
                # scale down the cluster if there is fewer tasks than workers
                # n_remaining_tasks = total_combinations - i
                # if n_remaining_tasks < self.n_workers:
                #     n_remaining_workers = max(n_remaining_tasks, 1)
                #     client.cluster.scale(n_remaining_workers)
            client.close()

        return total_combinations, n_combinations_successfully_completed, n_combinations_failed, n_combinations_none

    def get_kwargs_to_log_experiment(self):
        kwargs_to_log = dict(experiment_name=self.experiment_name, models_nickname=self.models_nickname,
                             seeds_model=self.seeds_model)
        if self.using_own_resampling:
            kwargs_to_log.update(dict(datasets_names_or_ids=self.datasets_names_or_ids,
                                      seeds_datasets=self.seeds_datasets,
                                      resample_strategy=self.resample_strategy, k_folds=self.k_folds, folds=self.folds,
                                      pct_test=self.pct_test))
        else:
            kwargs_to_log.update(dict(tasks_ids=self.tasks_ids, task_repeats=self.task_repeats,
                                      task_samples=self.task_samples, task_folds=self.task_folds))
        return kwargs_to_log

    def run(self):
        self.treat_parser()
        if self.datasets_names_or_ids is not None and self.tasks_ids is None:
            self.using_own_resampling = True
        elif self.datasets_names_or_ids is None and self.tasks_ids is not None:
            self.using_own_resampling = False
        else:
            raise ValueError("You must provide either datasets_names_or_ids or tasks_ids, but not both.")
        self.output_dir = self.output_dir / self.experiment_name
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logger()
        kwargs_to_log = self.get_kwargs_to_log_experiment()
        start_time = time.perf_counter()
        log_and_print_msg('Starting experiment...', **kwargs_to_log)
        if self.dask_cluster_type is not None:
            client = self.setup_dask(self.n_workers, self.dask_cluster_type, self.dask_address)
        else:
            client = None
        total_combinations, n_combinations_successfully_completed, n_combinations_failed, n_combinations_none = (
            self.run_experiment(client=client))
        total_time = time.perf_counter() - start_time
        log_and_print_msg('Experiment finished!', total_elapsed_time=total_time,
                          total_combinations=total_combinations,
                          sucessfully_completed=n_combinations_successfully_completed,
                          failed=n_combinations_failed, none=n_combinations_none,
                          **kwargs_to_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = BaseExperiment(parser=parser)
    experiment.run()
