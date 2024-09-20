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
import ray
from distributed import WorkerPlugin, Worker, Client
import dask
from oslo_concurrency import lockutils
from tab_benchmark.benchmark.utils import treat_mlflow, get_model, load_openml_task, fit_model, evaluate_model, \
    load_own_task
from tab_benchmark.benchmark.benchmarked_models import models_dict as benchmarked_models_dict
from tab_benchmark.utils import get_git_revision_hash, flatten_dict
from dask.distributed import LocalCluster, get_worker, as_completed
from dask_jobqueue import SLURMCluster
from tqdm.auto import tqdm
from multiprocessing import cpu_count
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
            mlflow.log_param('EXCEPTION', 'KILLED')
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
            mlflow_tracking_uri='sqlite:///' + str(Path.cwd().resolve()) + '/tab_benchmark.db', check_if_exists=True,
            retry_on_oom=True,
            raise_on_fit_error=False, parser=None,
            error_score='raise',
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
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.check_if_exists = check_if_exists
        self.retry_on_oom = retry_on_oom
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
        self.mlflow_tracking_uri = args.mlflow_tracking_uri
        self.check_if_exists = not args.do_not_check_if_exists
        self.retry_on_oom = not args.do_not_retry_on_oom
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
                  logging_to_mlflow=False, create_validation_set=False, output_dir=None, data_return=None, **kwargs):
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
            if logging_to_mlflow:
                # this is already unique
                output_dir = Path(mlflow.get_artifact_uri())
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
        if create_validation_set:
            # we disable auto early stopping when creating a validation set, because we will use it to validate
            if hasattr(model, 'auto_early_stopping'):
                model.auto_early_stopping = False
        if logging_to_mlflow:
            model_params = vars(model).copy()
            if hasattr(model, 'loss_fn'):
                # will be logged after
                del model_params['loss_fn']
            mlflow.log_params(model_params)
            # just to make it easier to filter after
            if model_nickname.find('TabBenchmark') != -1:
                mlflow.log_param('model_name', model_nickname[len('TabBenchmark'):])
        return model

    def combination_args_to_kwargs(self, *args, **kwargs):
        is_openml = kwargs.get('is_openml', False)
        if is_openml:
            model_nickname, seed_model, task_id, task_fold, task_repeat, task_sample = args
            unique_params = dict(task_id=task_id, task_repeat=task_repeat, task_sample=task_sample,
                                 task_fold=task_fold)
        else:
            model_nickname, seed_model, dataset_name_or_id, seed_dataset, fold = args
            unique_params = dict(dataset_name_or_id=dataset_name_or_id, seed_dataset=seed_dataset, fold=fold)
        unique_params.update(model_nickname=model_nickname, seed_model=seed_model, **kwargs)
        return unique_params

    def load_data(self, create_validation_set=False, logging_to_mlflow=False, **kwargs):
        is_openml = kwargs.get('is_openml', False)
        data_params = kwargs.copy()
        data_params.pop('model_nickname')
        data_params.pop('seed_model')
        data_params.pop('is_openml')
        if is_openml:
            (X, y, cat_ind, att_names, cat_features_names, cat_dims, task_name, n_classes, train_indices,
             test_indices, validation_indices) = load_openml_task(**data_params,
                                                                  create_validation_set=create_validation_set,
                                                                  logging_to_mlflow=logging_to_mlflow)
        else:
            resample_strategy = self.resample_strategy
            n_folds = self.k_folds
            pct_test = self.pct_test
            validation_resample_strategy = self.validation_resample_strategy
            pct_validation = self.pct_validation
            (X, y, cat_ind, att_names, cat_features_names, cat_dims, task_name, n_classes, train_indices,
             test_indices, validation_indices) = (
                load_own_task(**data_params,
                              resample_strategy=resample_strategy, n_folds=n_folds, pct_test=pct_test,
                              create_validation_set=create_validation_set,
                              validation_resample_strategy=validation_resample_strategy,
                              pct_validation=pct_validation, logging_to_mlflow=logging_to_mlflow)
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
                       logging_to_mlflow=False, **kwargs):
        model = fit_return['model']
        X_test = fit_return['X_test']
        y_test = fit_return['y_test']
        n_classes = data_return['n_classes']
        X_validation = fit_return['X_validation']
        y_validation = fit_return['y_validation']
        evaluate_results = evaluate_model(model, (X_test, y_test), 'test', metrics, default_metric, n_classes,
                                          self.error_score, logging_to_mlflow)
        if create_validation_set:
            validation_results = evaluate_model(model, (X_validation, y_validation), 'validation', metrics,
                                                default_metric, n_classes, self.error_score, logging_to_mlflow)
            evaluate_results.update(validation_results)
        return evaluate_results

    def run_combination(self,
                        n_jobs=1, create_validation_set=False,
                        model_params=None, logging_to_mlflow=False,
                        fit_params=None, return_results=False, **kwargs):
        """

        Parameters
        ----------
        args
        return_results
        fit_params
        fold
        seed_dataset
        dataset_name_or_id
        task_sample
        task_repeat
        task_fold
        task_id
        seed_model
        model_nickname
        n_jobs
        create_validation_set
        model_params
        is_openml
        logging_to_mlflow
        kwargs
        Returns
        -------

        """
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
            clean_output_dir = kwargs.pop('clean_output_dir', self.clean_output_dir)

            # load data
            data_return = self.load_data(**kwargs, create_validation_set=create_validation_set,
                                         logging_to_mlflow=logging_to_mlflow)
            results['data_return'] = data_return

            # load model
            model = self.get_model(model_params=model_params,
                                   n_jobs=n_jobs, logging_to_mlflow=logging_to_mlflow,
                                   create_validation_set=create_validation_set, data_return=data_return, **kwargs)
            results['model'] = model

            # get metrics
            metrics, default_metric = self.get_metrics(data_return, **kwargs)
            results['metrics'] = metrics
            results['default_metric'] = default_metric

            # fit model
            fit_return = self.fit_model(model, fit_params, data_return, metrics, default_metric, **kwargs)
            results['fit_return'] = fit_return

            # evaluate model
            evaluate_return = self.evaluate_model(metrics, default_metric, fit_return, data_return,
                                                  create_validation_set, logging_to_mlflow, **kwargs)
            results['evaluate_return'] = evaluate_return

            if logging_to_mlflow:
                mlflow.log_param('was_evaluated', True)
                # in MB (in linux getrusage seems to returns in KB)
                mlflow.log_metric('max_memory_used', getrusage(RUSAGE_SELF).ru_maxrss / 1000)
                if self.n_gpus > 0:
                    mlflow.log_metric('max_cuda_memory_reserved', max_memory_reserved() / (1024 ** 2))  # in MB
                    mlflow.log_metric('max_cuda_memory_allocated', max_memory_allocated() / (1024 ** 2))  # in MB

        except Exception as exception:
            if logging_to_mlflow:
                mlflow.log_param('EXCEPTION', str(exception))
                mlflow.end_run('FAILED')
            try:
                total_time = time.perf_counter() - start_time
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
            log_and_print_msg('Finished!', elapsed_time=total_time, **kwargs)
            if clean_output_dir:
                output_dir = model.output_dir
                if output_dir.exists():
                    rmtree(output_dir)
            if return_results:
                return results
            else:
                return True

    @lockutils.synchronized('run_combination_with_mlflow')
    def run_combination_with_mlflow(self, *args,
                                    n_jobs=1, create_validation_set=False,
                                    model_params=None, logging_to_mlflow=False,
                                    fit_params=None, return_results=False, parent_run_uuid=None, **kwargs):
        experiment_name = kwargs.pop('experiment_name', self.experiment_name)
        mlflow_tracking_uri = kwargs.pop('mlflow_tracking_uri', self.mlflow_tracking_uri)
        check_if_exists = kwargs.pop('check_if_exists', self.check_if_exists)
        clean_output_dir = kwargs.pop('clean_output_dir', self.clean_output_dir)
        unique_params = self.combination_args_to_kwargs(*args, **kwargs)
        model_nickname = unique_params.get('model_nickname')
        model_params = model_params if model_params else self.models_params.get(model_nickname, {}).copy()
        fit_params = fit_params if fit_params else self.fits_params.get(model_nickname, {}).copy()
        if 'n_jobs' in model_params:
            n_jobs = model_params.pop('n_jobs')
        unique_params.update(model_params=model_params, create_validation_set=create_validation_set,
                             fit_params=fit_params)
        possible_existent_run, logging_to_mlflow = treat_mlflow(experiment_name, mlflow_tracking_uri, check_if_exists,
                                                                **unique_params)
        unique_params.pop('model_params')
        unique_params.pop('fit_params')
        unique_params.pop('create_validation_set')

        if possible_existent_run is not None:
            log_and_print_msg('Run already exists on MLflow. Skipping...')
            if return_results:
                return possible_existent_run.to_dict()
            else:
                return True

        if logging_to_mlflow:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name, artifact_location=str(self.output_dir))
            mlflow.set_experiment(experiment_name)
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

            # If in a ray session, set name of the run as the trial name
            if ray.train._internal.session._get_session():
                train_context = ray.train.get_context()
                run_name = train_context.get_trial_name()
            else:
                run_name = '_'.join([f'{k}={v}' for k, v in unique_params.items()])

            with mlflow.start_run(run_name=run_name, nested=nested) as run:
                mlflow.log_params(flatten_dict(unique_params))
                mlflow.log_params(flatten_dict(fit_params))
                mlflow.log_params(flatten_dict(model_params))
                mlflow.log_param('create_validation_set', create_validation_set)
                mlflow.log_param('git_hash', get_git_revision_hash())
                # slurm parameters
                mlflow.log_param('SLURM_JOB_ID', os.getenv('SLURM_JOB_ID', None))
                mlflow.log_param('SLURMD_NODENAME', os.getenv('SLURMD_NODENAME', None))
                # dask parameters
                mlflow.log_param('dask_cluster_type', self.dask_cluster_type)
                mlflow.log_param('n_workers', self.n_workers)
                mlflow.log_param('n_cores', self.n_cores)
                mlflow.log_param('n_processes', self.n_processes)
                mlflow.log_param('dask_memory', self.dask_memory)
                mlflow.log_param('dask_job_extra_directives', self.dask_job_extra_directives)
                mlflow.log_param('dask_address', self.dask_address)
                mlflow.log_param('n_gpus', self.n_gpus)
                return self.run_combination(
                    n_jobs=n_jobs,
                    create_validation_set=create_validation_set,
                    model_params=model_params, fit_params=fit_params,
                    logging_to_mlflow=logging_to_mlflow, return_results=return_results,
                    clean_output_dir=clean_output_dir,
                    **unique_params
                )
        else:
            return self.run_combination(
                n_jobs=n_jobs,
                create_validation_set=create_validation_set,
                model_params=model_params, fit_params=fit_params,
                logging_to_mlflow=logging_to_mlflow, return_results=return_results,
                clean_output_dir=clean_output_dir,
                **unique_params
            )

    def setup_dask(self, n_workers, cluster_type='local', address=None):
        if address is not None:
            client = Client(address)
        else:
            if cluster_type == 'local':
                threads_per_worker = self.n_jobs
                processes = self.n_processes
                if cpu_count() < threads_per_worker * n_workers:
                    warnings.warn(f"n_workers * threads_per_worker (n_jobs) is greater than the number of cores "
                                  f"available ({cpu_count}). This may lead to performance issues.")
                resources = {'cores': cpu_count()}
                if self.n_gpus > 0:
                    resources['gpus'] = self.n_gpus
                cluster = LocalCluster(n_workers=0, memory_limit=self.dask_memory, processes=False,
                                       threads_per_worker=threads_per_worker, resources=resources)
                cluster.scale(n_workers)
            elif cluster_type == 'slurm':
                cores = self.n_cores
                processes = self.n_processes
                resources_per_work = {'cores': cores}
                if self.n_gpus > 0:
                    resources_per_work['gpus'] = self.n_gpus
                job_extra_directives = dask.config.get(
                    "jobqueue.%s.job-extra-directives" % 'slurm', []
                )
                job_script_prologue = dask.config.get(
                    "jobqueue.%s.job-script-prologue" % 'slurm', []
                )
                worker_extra_args = dask.config.get(
                    "jobqueue.%s.worker-extra-args" % 'slurm', []
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

    def get_combinations(self):
        if self.using_own_resampling:
            # (model_nickname, seed_model, dataset_name_or_id, seed_dataset, fold)
            combinations = list(product(self.models_nickname, self.seeds_model, self.datasets_names_or_ids,
                                        self.seeds_datasets, self.folds))
            extra_params = dict(is_openml=False, n_jobs=self.n_jobs, resample_strategy=self.resample_strategy,
                                n_folds=self.k_folds, pct_test=self.pct_test,
                                validation_resample_strategy=self.validation_resample_strategy,
                                pct_validation=self.pct_validation)

        else:
            # (model_nickname, seed_model, task_id, task_fold, task_repeat, task_sample)
            combinations = list(product(self.models_nickname, self.seeds_model, self.tasks_ids, self.task_folds,
                                        self.task_repeats, self.task_samples))
            extra_params = dict(is_openml=True, n_jobs=self.n_jobs)
        return combinations, extra_params

    def run_experiment(self, client=None):

        combinations, extra_params = self.get_combinations()
        n_args = len(combinations[0])

        total_combinations = len(combinations)
        n_combinations_successfully_completed = 0
        n_combinations_failed = 0
        n_combinations_none = 0
        if client is not None:
            resources_per_task = {'cores': self.n_jobs}
            if self.n_gpus > 0:
                resources_per_task['gpus'] = self.n_gpus / (self.n_cores / self.n_jobs)
            log_and_print_msg(f'{total_combinations} models are being trained and evaluated in parallel, '
                              f'check the logs for real time information. We will display information about the '
                              f'completion of the tasks right after sending all the tasks to the cluster. '
                              f'Note that this can take a while if a lot of tasks are being submitted. '
                              f'You can check the dask dashboard to get more information about the progress and '
                              f'the workers.')

            list_of_args = [[combination[i] for combination in combinations[1:]] for i in range(n_args)]
            first_future = client.submit(self.run_combination_with_mlflow, *combinations[0],
                                         resources=resources_per_task, **extra_params)
            # wait a little bit for the first submission to create folders, experiments, etc
            time.sleep(5)
            futures = client.map(self.run_combination_with_mlflow, *list_of_args,
                                 batch_size=self.n_workers, resources=resources_per_task, **extra_params)
            futures = [first_future] + futures
        else:
            progress_bar = tqdm(combinations, desc='Combinations completed')
            for combination in progress_bar:
                combination_success = self.run_combination_with_mlflow(*combination, **extra_params)
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
                combination_success = future.result()
                if combination_success is True:
                    n_combinations_successfully_completed += 1
                elif combination_success is False:
                    n_combinations_failed += 1
                else:
                    n_combinations_none += 1
                del future  # to free memory
                log_and_print_msg(str(progress_bar), succesfully_completed=n_combinations_successfully_completed,
                                  failed=n_combinations_failed, none=n_combinations_none, i=i)
                # scale down the cluster if there is fewer tasks than workers
                n_remaining_tasks = total_combinations - i
                if n_remaining_tasks < self.n_workers:
                    n_remaining_workers = max(n_remaining_tasks, 1)
                    client.cluster.scale(n_remaining_workers)
            # ensure all futures are done
            client.gather(futures)
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
