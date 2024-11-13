from __future__ import annotations
import argparse
import shlex
import tempfile
import time
from multiprocessing import cpu_count
from pathlib import Path
from shutil import rmtree
from typing import Optional
import mlflow
import os
import logging
import warnings
import numpy as np
import pandas as pd
import torch
from distributed import WorkerPlugin, Worker, Client
import dask
from tab_benchmark.benchmark.utils import set_mlflow_tracking_uri_check_if_exists, get_model, load_openml_task, \
    fit_model, evaluate_model, \
    load_own_task, load_pandas_task
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
            models_nickname: Optional[list[str]] = None, seeds_models: Optional[list[int]] = None, n_jobs: int = 1,
            models_params: Optional[dict] = None, fits_params: Optional[dict] = None,
            # when performing our own resampling
            datasets_names_or_ids: Optional[list[int]] = None, seeds_datasets: Optional[list[int]] = None,
            resample_strategy: str = 'k-fold_cv', k_folds: int = 10, folds: Optional[list[int]] = None,
            pct_test: float = 0.2,
            validation_resample_strategy: str = 'next_fold', pct_validation: float = 0.1,
            # when using openml tasks
            tasks_ids: Optional[list[int]] = None,
            task_repeats: Optional[list[int]] = None, task_folds: Optional[list[int]] = None,
            task_samples: Optional[list[int]] = None,
            # parameters of experiment
            experiment_name: str = 'base_experiment',
            create_validation_set: bool = False,
            models_dict: Optional[dict] = None,
            log_dir: str | Path = Path.cwd() / 'logs',
            log_file_name: Optional[str] = None,
            work_dir: str | Path = Path.cwd() / 'work',
            save_dir: Optional[str | Path] = None,
            clean_work_dir: bool = True,
            raise_on_fit_error: bool = False, parser: Optional = None,
            error_score: str = 'raise',
            # mlflow specific
            log_to_mlflow: bool = True,
            mlflow_tracking_uri: str = 'sqlite:///' + str(Path.cwd().resolve()) + '/tab_benchmark.db',
            check_if_exists: bool = True,
            # parallelization
            dask_cluster_type: Optional[str] = None,
            n_workers: int = 1,
            n_processes: int = 1,
            n_cores: int = 1,
            dask_memory: Optional[str] = None,
            dask_job_extra_directives: Optional[str] = None,
            dask_address: Optional[str] = None,
            # gpu specific
            n_gpus: int = 0,
    ):
        """Base experiment.

        This class allows to perform experiments with tabular data. It is a base class that can be inherited to
        perform more specific experiments. It allows to perform experiments with different models, datasets and
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
        log_dir :
            The directory where the logs will be saved. Defaults to 'logs'.
        log_file_name :
            The name of the log file. If None, it will be the experiment_name. Defaults to None.
        work_dir :
            The directory where the intermediate outputs will be saved. Defaults to 'work'.
        save_dir :
            The directory where the final trained models will be saved.
        clean_work_dir :
            If True, clean the work directory after running the experiment. Defaults to True.
        raise_on_fit_error :
            If True, raise an error if it is encountered when fitting the model. Defaults to False.
        parser :
            The parser to be used in the experiment. Defaults to None, which means that the parser will be created.
        error_score :
            The default value to be used if a error occurs when evaluating the model. Defaults to 'raise' which
            raises an error.
        log_to_mlflow :
            If True, log the results to mlflow. Defaults to True.
        mlflow_tracking_uri :
            The uri of the mlflow server. Defaults to 'sqlite:///' + str(Path.cwd().resolve()) + '/tab_benchmark.db'.
        check_if_exists :
            If True, check if the experiment already exists in mlflow. Defaults to True.
        dask_cluster_type :
            The type of the dask cluster to be used. It can be 'local' or 'slurm'. Defaults to None, which means
            that dask will not be used.
        n_workers :
            The number of workers to be used in the dask cluster. Defaults to 1.
        n_processes :
            The number of processes to be used in the dask cluster. Defaults to 1.
        n_cores :
            The number of cores to be used in the dask cluster. Defaults to 1.
        dask_memory :
            The memory to be used in the dask cluster. Defaults to None.
        dask_job_extra_directives :
            The extra directives to be used in the dask cluster. Defaults to None.
        dask_address :
            The address of an initialized dask cluster. Defaults to None.
        n_gpus :
            The number of gpus to be used in the dask cluster. Defaults to 0.
        """
        self.models_nickname = models_nickname if models_nickname else []
        self.seeds_models = seeds_models if seeds_models else [0]
        self.n_jobs = n_jobs

        self.models_params = self._validate_dict_of_models_params(models_params, self.models_nickname)
        self.fits_params = self._validate_dict_of_models_params(fits_params, self.models_nickname)

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
        self.create_validation_set = create_validation_set
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        self.log_dir = log_dir
        self.log_file_name = log_file_name
        if isinstance(work_dir, str):
            work_dir = Path(work_dir)
        self.work_dir = work_dir
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        self.save_dir = save_dir
        self.clean_work_dir = clean_work_dir
        self.log_to_mlflow = log_to_mlflow
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.check_if_exists = check_if_exists
        self.parser = parser
        self.models_dict = models_dict if models_dict else benchmarked_models_dict.copy()
        self.raise_on_fit_error = raise_on_fit_error
        self.error_score = error_score
        self.client = None
        self.logger_filename = None

    def _validate_dict_of_models_params(self, dict_to_validate, models_nickname):
        """Validate the dictionary of models parameters or fit parameters, broadcasting it to the number of models
        if a single dictionary is passed instead of a dictionary of dictionaries."""
        dict_to_validate = dict_to_validate or {model_nickname: {} for model_nickname in models_nickname}
        models_missing = [model_nickname for model_nickname in models_nickname
                          if model_nickname not in dict_to_validate]
        if len(models_missing) == len(models_nickname):
            dict_to_validate = {model_nickname: dict_to_validate.copy() for model_nickname in models_nickname}
        else:
            for model_missing in models_missing:
                dict_to_validate[model_missing] = {}
        return dict_to_validate

    def _add_arguments_to_parser(self):
        """Add the arguments to the parser."""
        self.parser.add_argument('--experiment_name', type=str, default=self.experiment_name)
        self.parser.add_argument('--create_validation_set', action='store_true')
        self.parser.add_argument('--models_nickname', type=str, choices=self.models_dict.keys(),
                                 nargs='*', default=self.models_nickname)
        self.parser.add_argument('--seeds_models', nargs='*', type=int, default=self.seeds_models)
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
        self.parser.add_argument('--log_file_name', type=str, default=self.log_file_name)
        self.parser.add_argument('--work_dir', type=Path, default=self.work_dir)
        self.parser.add_argument('--save_dir', type=Path, default=self.save_dir)
        self.parser.add_argument('--do_not_clean_work_dir', action='store_true')
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
                                      '--dask_job_extra_directives "-G 1"). For the case of a local cluster, it will'
                                      'be the total number of GPUs that will be used, each worker will have access'
                                      'to n_gpus / n_workers GPUs.')

    def _unpack_parser(self):
        """Unpack the parser."""
        args = self.parser.parse_args()
        self.experiment_name = args.experiment_name
        self.create_validation_set = args.create_validation_set
        self.models_nickname = args.models_nickname
        self.n_jobs = args.n_jobs
        models_params = args.models_params
        self.models_params = self._validate_dict_of_models_params(models_params, self.models_nickname)
        fits_params = args.fits_params
        self.fits_params = self._validate_dict_of_models_params(fits_params, self.models_nickname)
        error_score = args.error_score
        if error_score == 'nan':
            error_score = np.nan
        self.error_score = error_score

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

        self.log_dir = args.log_dir
        self.work_dir = args.work_dir
        self.save_dir = args.save_dir
        self.clean_work_dir = not args.do_not_clean_work_dir
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

    def _treat_parser(self):
        """Treat the parser."""
        if self.parser is not None:
            self._add_arguments_to_parser()
            self._unpack_parser()

    def _setup_logger(self, log_dir=None, filemode='w'):
        """Setup the logger."""
        if log_dir is None:
            log_dir = self.log_dir
        os.makedirs(log_dir, exist_ok=True)
        if self.log_file_name is None:
            name = self.experiment_name
        else:
            name = self.log_file_name
        if (log_dir / f'{name}.log').exists():
            file_names = sorted(log_dir.glob(f'{name}_????.log'))
            if file_names:
                file_name = file_names[-1].name
                id_file = int(file_name.split('_')[-1].split('.')[0])
                name = f'{name}_{id_file + 1:04d}'
            else:
                name = name + '_0001'
        log_file_name = f'{name}.log'
        logging.basicConfig(filename=log_dir / log_file_name,
                            format='%(asctime)s - %(levelname)s\n%(message)s\n',
                            level=logging.INFO, filemode=filemode)

    def _get_model(self, model_params: Optional[dict] = None, n_jobs: int = 1,
                   log_to_mlflow: bool = False, run_id: Optional[str] = None, create_validation_set: bool = False,
                   work_dir: Optional[str | Path] = None, data_return: Optional[dict] = None,
                   **kwargs):
        """Get the model to be used in the experiment.

        Parameters
        ----------
        model_params :
            Dictionary with the parameters of the model. Defaults models_params defined at the initialization of the
            class if None.
        n_jobs :
            Number of threads/cores to be used by the model if it supports it.
        log_to_mlflow :
            If True, log the model to mlflow.
        run_id :
            The run_id of the mlflow run.
        create_validation_set :
            If True, create a validation set.
        work_dir :
            The output directory where the model will save outputs.
        data_return :
            The data returned by the load_data method.
        kwargs :
            Additional arguments from the experiment, it must contain the model_nickname and seed_model.

        Returns
        -------
        model :
            The instantiated model to be used in the experiment.
        """
        # if we reuse the package to other experiments, we don't necessarily need model_nickname and seed_model
        model_nickname = kwargs.get('model_nickname')
        seed_model = kwargs.get('seed_model')
        model_params = model_params if model_params else self.models_params.get(model_nickname, {}).copy()
        if data_return:
            data_params = data_return.get('data_params', None).copy()
        else:
            data_params = None
        models_dict = self.models_dict.copy()

        if work_dir is None:
            # if running on a dask worker, we use the worker's local directory as work_dir, irrespective of
            # self.work_dir
            try:
                worker = get_worker()
                work_dir = Path(worker.local_directory)
            except ValueError:
                # if running on the main process, we use the work_dir defined in the class
                work_dir = self.work_dir

        # if logging to mlflow we use the run_id as the name of the output_dir in the work_dir
        if log_to_mlflow:
            # this is already unique
            unique_name = run_id
        else:
            # if we only save the model in the output_dir we will have some problems when running in parallel
            # because the workers will try to save the model in the same directory, so we must create a unique
            # directory for each combination model/dataset
            if data_params is not None:
                unique_name = '_'.join([f'{k}={v}' for k, v in data_params.items() if k != 'dataframe'])
            else:
                # not sure, but I think it will generate a true random number and even be thread safe
                unique_name = f'{SystemRandom().randint(0, 1000000):06d}'
            unique_name = f'{model_nickname}_{unique_name}'

        output_dir = work_dir / unique_name
        os.makedirs(output_dir, exist_ok=True)
        model = get_model(model_nickname, seed_model, model_params, models_dict, n_jobs, output_dir=output_dir)
        if log_to_mlflow:
            if hasattr(model, 'mlflow_run_id'):
                setattr(model, 'mlflow_run_id', run_id)
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

    def _load_data(self, create_validation_set=False, log_to_mlflow=False, run_id=None, **kwargs):
        """Load the data to be used in the experiment.

        Parameters
        ----------
        create_validation_set :
            If True, create a validation set.
        log_to_mlflow :
            If True, log the data to mlflow.
        run_id :
            The run_id of the mlflow run.
        kwargs :
            Additional arguments from the experiment, it must contain the data parameters.

        Returns
        -------
        data_return :
            A dictionary with the data to be used in the experiment. It contains the following keys:
            X :
                The features of the dataset.
            y :
                The target of the dataset.
            cat_ind :
                The indices of the categorical features.
            att_names :
                The names of the features.
            cat_features_names :
                The names of the categorical features.
            cat_dims :
                The dimensions of the categorical features.
            task_name :
                The name of the task.
            n_classes :
                The number of classes.
            train_indices :
                The indices of the training set.
            test_indices :
                The indices of the test set.
            validation_indices :
                The indices of the validation set.
            data_params :
                The parameters of the data.
        """
        is_openml_task = kwargs.get('is_openml_task', False)
        if is_openml_task:
            keys = ['task_id', 'task_repeat', 'task_fold', 'task_sample']
        else:
            if 'dataset_name_or_id' in kwargs:
                keys = ['dataset_name_or_id',
                        'seed_dataset', 'resample_strategy', 'n_folds', 'fold', 'pct_test',
                        'validation_resample_strategy', 'pct_validation']
            else:
                keys = ['dataframe', 'target', 'task', 'dataset_name',
                        'seed_dataset', 'resample_strategy', 'n_folds', 'fold', 'pct_test',
                        'validation_resample_strategy', 'pct_validation']
        data_params = {key: kwargs.get(key) for key in keys}
        if is_openml_task:
            (X, y, cat_ind, att_names, cat_features_names, cat_dims, task_name, n_classes, train_indices,
             test_indices, validation_indices) = load_openml_task(create_validation_set=create_validation_set,
                                                                  log_to_mlflow=log_to_mlflow, run_id=run_id,
                                                                  **data_params)
        else:
            if 'dataset_name_or_id' in kwargs:
                (X, y, cat_ind, att_names, cat_features_names, cat_dims, task_name, n_classes, train_indices,
                 test_indices, validation_indices) = (
                    load_own_task(**data_params,
                                  create_validation_set=create_validation_set, log_to_mlflow=log_to_mlflow,
                                  run_id=run_id)
                )
            else:
                (X, y, cat_ind, att_names, cat_features_names, cat_dims, task_name, n_classes, train_indices,
                 test_indices, validation_indices) = (
                    load_pandas_task(**data_params,
                                     create_validation_set=create_validation_set, log_to_mlflow=log_to_mlflow,
                                     run_id=run_id)
                )
        data_return = dict(X=X, y=y, cat_ind=cat_ind, att_names=att_names, cat_features_names=cat_features_names,
                           cat_dims=cat_dims, task_name=task_name, n_classes=n_classes, train_indices=train_indices,
                           test_indices=test_indices, validation_indices=validation_indices,
                           data_params=data_params)
        return data_return

    def _get_metrics(self, data_return, **kwargs):
        """Get the metrics to be used in the experiment.

        Parameters
        ----------
        data_return :
            The data returned by the load_data method.
        kwargs :
            Additional arguments from the experiment.

        Returns
        -------
        metrics :
            The metrics to be used in the experiment.
        report_metric :
            The metric to be used to report the results, used for example when performing hyperparameter optimization.
        """
        task_name = data_return['task_name']
        if task_name in ('classification', 'binary_classification'):
            metrics = ['logloss', 'auc', 'auc_micro', 'auc_weighted', 'accuracy', 'balanced_accuracy',
                       'balanced_accuracy_adjusted', 'f1_micro', 'f1_macro', 'f1_weighted']
            report_metric = 'logloss'
        elif task_name == 'regression':
            metrics = ['rmse', 'r2_score', 'mae', 'mape']
            report_metric = 'rmse'
        else:
            raise NotImplementedError
        return metrics, report_metric

    def _fit_model(self, model, fit_params, data_return, metrics, report_metric, log_to_mlflow=False, run_id=None,
                   **kwargs):
        """Fit the model to the data.

        Parameters
        ----------
        model :
            The model to be fitted, returned by the get_model method.
        fit_params :
            The parameters of the fit. Defaults fits_params defined at the initialization of the class if None.
        data_return :
            The data returned by the load_data method.
        metrics :
            The metrics to be used in the experiment, returned by the get_metrics method.
        report_metric :
            The metric to be used to report the results, returned by the get_metrics method.
        kwargs :
            Additional arguments from the experiment.

        Returns
        -------
        fit_return :
            A dictionary with the results of the fit. It contains the following keys:
            model :
                The fitted model.
            X_train :
                The features of the training set, already preprocessed.
            y_train :
                The target of the training set, already preprocessed.
            X_test :
                The features of the test set, already preprocessed.
            y_test :
                The target of the test set, already preprocessed.
            X_validation :
                The features of the validation set, already preprocessed.
            y_validation :
                The target of the validation set, already preprocessed.
        """
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
        if self.save_dir:
            if log_to_mlflow:
                # will log the model to mlflow artifacts
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    model.save_model(temp_dir)
                    mlflow.log_artifacts(str(temp_dir.resolve()), artifact_path='model', run_id=run_id)
            else:
                # will log the model directly to the save_dir
                model.save(self.save_dir)
        return fit_return

    def _evaluate_model(self, metrics, report_metric, fit_return, data_return, create_validation_set=False,
                        log_to_mlflow=False, run_id=None, **kwargs):
        """Evaluate the model.

        Parameters
        ----------
        metrics :
            The metrics to be used in the experiment, returned by the get_metrics method.
        report_metric :
            The metric to be used to report the results, returned by the get_metrics method.
        fit_return :
            The data returned by the fit_model method.
        data_return :
            The data returned by the load_data method.
        create_validation_set :
            If True, create a validation set.
        log_to_mlflow :
            If True, log the results to mlflow.
        run_id :
            The run_id of the mlflow run.
        kwargs :
            Additional arguments from the experiment.

        Returns
        -------
        evaluate_results :
            A dictionary with the results of the evaluation. It contains the following keys:
            final_test :
                The results of the evaluation on the test set.
            final_validation :
                The results of the evaluation on the validation set, if create_validation_set is True.
        """
        model = fit_return['model']
        X_test = fit_return['X_test']
        y_test = fit_return['y_test']
        n_classes = data_return['n_classes']
        X_validation = fit_return['X_validation']
        y_validation = fit_return['y_validation']
        evaluate_results = evaluate_model(model=model, eval_set=(X_test, y_test), eval_name='final_test',
                                          metrics=metrics, n_classes=n_classes,
                                          error_score=self.error_score, log_to_mlflow=log_to_mlflow, run_id=run_id)
        if create_validation_set:
            validation_results = evaluate_model(model=model, eval_set=(X_validation, y_validation),
                                                eval_name='final_validation', metrics=metrics,
                                                report_metric=report_metric, n_classes=n_classes,
                                                error_score=self.error_score, log_to_mlflow=log_to_mlflow,
                                                run_id=run_id)
            evaluate_results.update(validation_results)
        return evaluate_results

    def _train_model(self,
                     n_jobs=1, create_validation_set=False,
                     model_params=None,
                     fit_params=None, return_results=False, clean_work_dir=True, log_to_mlflow=False, run_id=None,
                     **kwargs):
        """Train the model.

        It executes the main steps of the experiment: load_data, get_model, get_metrics, fit_model and evaluate_model.

        Parameters
        ----------
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
        clean_work_dir :
            If True, clean the work directory after the experiment.
        log_to_mlflow :
            If True, log the results to mlflow.
        run_id :
            The run_id of the mlflow run.
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
        try:
            results = {}
            start_time = time.perf_counter()
            # logging
            log_and_print_msg('Running...', **kwargs)
            if self.n_gpus > 0 and self.dask_cluster_type is not None:
                # Number of gpus in this job / Number of models being trained in parallel
                fraction_of_gpu_being_used = self.n_gpus / (self.n_cores / self.n_jobs)
                set_per_process_memory_fraction(fraction_of_gpu_being_used)
            if torch.cuda.is_available() or self.n_gpus > 0:
                reset_peak_memory_stats()
            model_nickname = kwargs.get('model_nickname')
            model_params = model_params if model_params else self.models_params.get(model_nickname, {}).copy()
            fit_params = fit_params if fit_params else self.fits_params.get(kwargs.get('model_nickname'), {}).copy()

            # load data
            data_return = self._load_data(create_validation_set=create_validation_set, log_to_mlflow=log_to_mlflow,
                                          run_id=run_id, **kwargs)
            results['data_return'] = data_return

            # load model
            model = self._get_model(model_params=model_params, n_jobs=n_jobs, log_to_mlflow=log_to_mlflow,
                                    run_id=run_id, create_validation_set=create_validation_set, data_return=data_return,
                                    **kwargs)
            results['model'] = model

            # get metrics
            metrics, report_metric = self._get_metrics(data_return, **kwargs)
            results['metrics'] = metrics
            results['report_metric'] = report_metric

            # fit model
            fit_return = self._fit_model(model=model, fit_params=fit_params, data_return=data_return, metrics=metrics,
                                         report_metric=report_metric, log_to_mlflow=log_to_mlflow, run_id=run_id,
                                         **kwargs)
            results['fit_return'] = fit_return

            # evaluate model
            evaluate_return = self._evaluate_model(metrics=metrics, report_metric=report_metric, fit_return=fit_return,
                                                   data_return=data_return, create_validation_set=create_validation_set,
                                                   log_to_mlflow=log_to_mlflow, run_id=run_id, **kwargs)
            results['evaluate_return'] = evaluate_return

        except Exception as exception:
            if log_to_mlflow:
                log_tags = {'was_evaluated': False, 'EXCEPTION': str(exception)}
                mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
                for tag, value in log_tags.items():
                    mlflow_client.set_tag(run_id, tag, value)
                mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
                mlflow_client.set_terminated(run_id, status='FAILED')
            try:
                total_time = time.perf_counter() - start_time
                if log_to_mlflow:
                    log_metrics = {'elapsed_time': total_time}
                    mlflow.log_metrics(log_metrics, run_id=run_id)
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
                log_tags = {'was_evaluated': True}
                mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
                for tag, value in log_tags.items():
                    mlflow_client.set_tag(run_id, tag, value)
                # in MB (in linux getrusage seems to returns in KB)
                log_metrics = {'elapsed_time': total_time, 'max_memory_used': getrusage(RUSAGE_SELF).ru_maxrss / 1000}
                if torch.cuda.is_available() or self.n_gpus > 0:
                    log_metrics['max_cuda_memory_reserved'] = max_memory_reserved() / (1024 ** 2)  # in MB
                    log_metrics['max_cuda_memory_allocated'] = max_memory_allocated() / (1024 ** 2)  # in MB
                mlflow.log_metrics(log_metrics, run_id=run_id)
                mlflow_client.set_terminated(run_id, status='FINISHED')
            log_and_print_msg('Finished!', elapsed_time=total_time, **kwargs)
            if clean_work_dir:
                output_dir = model.output_dir
                if output_dir.exists():
                    rmtree(output_dir)
            if return_results:
                return results
            else:
                return True

    def _log_run_start_params(self, run_id, **run_unique_params):
        """Log the parameters of the run to mlflow."""
        params_to_log = flatten_dict(run_unique_params).copy()
        params_to_log.update(dict(
            git_hash=get_git_revision_hash(),
            # dask parameters
            dask_cluster_type=self.dask_cluster_type,
            n_workers=self.n_workers,
            n_cores=self.n_cores,
            n_processes=self.n_processes,
            dask_memory=self.dask_memory,
            dask_job_extra_directives=self.dask_job_extra_directives,
            dask_address=self.dask_address,
            n_gpus=self.n_gpus,
            cuda_available=torch.cuda.is_available(),
        ))
        tags_to_log = dict(
            # slurm parameters
            SLURM_JOB_ID=os.getenv('SLURM_JOB_ID', None),
            SLURMD_NODENAME=os.getenv('SLURMD_NODENAME', None),
        )
        mlflow.log_params(params_to_log, run_id=run_id)
        mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
        for tag, value in tags_to_log.items():
            mlflow_client.set_tag(run_id, tag, value)

    def _run_mlflow_and_train_model(self,
                                    n_jobs=1, create_validation_set=False,
                                    model_params=None,
                                    fit_params=None, return_results=False, clean_work_dir=True,
                                    run_id=None,
                                    experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                                    fn_to_train_model=None,
                                    **kwargs):
        """Run the experiment using mlflow and train the model."""
        if fn_to_train_model is None:
            fn_to_train_model = self._train_model
        # get unique parameters for mlflow and check if the run already exists
        model_nickname = kwargs.get('model_nickname')
        model_params = model_params if model_params else self.models_params.get(model_nickname, {}).copy()
        fit_params = fit_params if fit_params else self.fits_params.get(model_nickname, {}).copy()
        run_unique_params = dict(model_params=model_params, fit_params=fit_params,
                                 create_validation_set=create_validation_set, **kwargs)
        if 'dataframe' in run_unique_params:
            # we will not log the dataframe
            dataframe = run_unique_params.pop('dataframe', None)
        else:
            dataframe = None
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
            if self.save_dir:
                artifact_location = str(self.save_dir / experiment_name)
            else:
                artifact_location = None
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        mlflow_client = mlflow.client.MlflowClient(tracking_uri=mlflow_tracking_uri)
        if run_id is None:
            run = mlflow_client.create_run(experiment_id)
            run_id = run.info.run_id

        mlflow_client.update_run(run_id, status='RUNNING')
        self._log_run_start_params(run_id, **run_unique_params)

        if dataframe != None:
            # we reinsert the dataframe
            run_unique_params['dataframe'] = dataframe

        return fn_to_train_model(n_jobs=n_jobs,
                                 create_validation_set=run_unique_params.pop('create_validation_set'),
                                 model_params=run_unique_params.pop('model_params'),
                                 fit_params=run_unique_params.pop('fit_params'), return_results=return_results,
                                 clean_work_dir=clean_work_dir,
                                 log_to_mlflow=True, run_id=run_id,
                                 experiment_name=experiment_name,
                                 mlflow_tracking_uri=mlflow_tracking_uri, check_if_exists=check_if_exists,
                                 **run_unique_params)

    def _setup_dask(self, n_workers, cluster_type='local', address=None):
        """Setup the dask cluster."""
        if address is not None:
            client = Client(address)
        else:
            if cluster_type == 'local':
                threads_per_worker = self.n_cores
                processes_per_worker = self.n_processes
                gpus_per_worker = self.n_gpus / n_workers
                if n_workers * threads_per_worker > cpu_count():
                    warnings.warn(f"n_workers * threads_per_worker is greater than the number of cores "
                                  f"available ({cpu_count()}). This may lead to performance issues.")
                resources_per_worker = {'cores': threads_per_worker, 'gpus': gpus_per_worker,
                                        'processes': processes_per_worker}
                cluster = LocalCluster(n_workers=0, memory_limit=self.dask_memory, processes=True,
                                       threads_per_worker=threads_per_worker, resources=resources_per_worker)
                cluster.scale(n_workers)
            elif cluster_type == 'slurm':
                cores_per_worker = self.n_cores
                processes_per_worker = self.n_processes
                gpus_per_worker = self.n_gpus
                resources_per_work = {'cores': cores_per_worker, 'gpus': gpus_per_worker,
                                      'processes': processes_per_worker}
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
                cluster = SLURMCluster(cores=cores_per_worker, memory=self.dask_memory, processes=processes_per_worker,
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

    def run_openml_task_combination(self, model_nickname: str, seed_model: int, task_id: int,
                                    task_fold: int = 0, task_repeat: int = 0, task_sample: int = 0,
                                    run_id: Optional[str] = None,
                                    n_jobs: int = 1, create_validation_set: bool = False,
                                    model_params: Optional[dict] = None,
                                    fit_params: Optional[dict] = None, return_results: bool = False,
                                    clean_work_dir: bool = True,
                                    log_to_mlflow: bool = False,
                                    experiment_name: Optional[str] = None, mlflow_tracking_uri: Optional[str] = None,
                                    check_if_exists: Optional[bool] = None,
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
        clean_work_dir :
            If True, clean the work directory after the experiment.
        log_to_mlflow :
            If True, log the results to mlflow.
        experiment_name :
            The name of the experiment.
        mlflow_tracking_uri :
            The uri of the mlflow tracking server.
        check_if_exists :
            If True, check if the run already exists on mlflow.
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
        task_combination = dict(model_nickname=model_nickname, seed_model=seed_model, task_id=task_id,
                                task_fold=task_fold, task_repeat=task_repeat, task_sample=task_sample,
                                is_openml_task=True)
        task_combination.update(kwargs)
        if log_to_mlflow:
            return self._run_mlflow_and_train_model(n_jobs=n_jobs, create_validation_set=create_validation_set,
                                                    model_params=model_params, fit_params=fit_params,
                                                    return_results=return_results, clean_work_dir=clean_work_dir,
                                                    run_id=run_id, experiment_name=experiment_name,
                                                    mlflow_tracking_uri=mlflow_tracking_uri,
                                                    check_if_exists=check_if_exists, **task_combination)
        return self._train_model(n_jobs=n_jobs, create_validation_set=create_validation_set, model_params=model_params,
                                 fit_params=fit_params, return_results=return_results,
                                 clean_work_dir=clean_work_dir, logging_to_mlflow=False, **task_combination)

    def run_openml_dataset_combination(self, model_nickname: str, seed_model: int, dataset_name_or_id: str | int,
                                       seed_dataset: int,
                                       fold: int = 0, run_id: Optional[str] = None,
                                       resample_strategy: str = 'k-fold_cv', n_folds: int = 10, pct_test: float = 0.2,
                                       validation_resample_strategy: str = 'next_fold', pct_validation: float = 0.1,
                                       n_jobs: int = 1, create_validation_set: bool = False,
                                       model_params: Optional[dict] = None,
                                       fit_params: Optional[dict] = None, return_results: bool = False,
                                       clean_work_dir: bool = True,
                                       log_to_mlflow: bool = False,
                                       experiment_name: Optional[str] = None, mlflow_tracking_uri: Optional[str] = None,
                                       check_if_exists: Optional[bool] = None,
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
        clean_work_dir :
            If True, clean the work directory after the experiment.
        log_to_mlflow :
            If True, log the results to mlflow.
        experiment_name :
            The name of the experiment.
        mlflow_tracking_uri :
            The uri of the mlflow tracking server.
        check_if_exists :
            If True, check if the run already exists on mlflow.
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
        dataset_combination = dict(model_nickname=model_nickname, seed_model=seed_model,
                                   dataset_name_or_id=dataset_name_or_id,
                                   seed_dataset=seed_dataset, fold=fold, resample_strategy=resample_strategy,
                                   n_folds=n_folds,
                                   pct_test=pct_test, validation_resample_strategy=validation_resample_strategy,
                                   pct_validation=pct_validation, is_openml_task=False)
        dataset_combination.update(kwargs)
        if log_to_mlflow:
            return self._run_mlflow_and_train_model(n_jobs=n_jobs, create_validation_set=create_validation_set,
                                                    model_params=model_params, fit_params=fit_params,
                                                    return_results=return_results, clean_work_dir=clean_work_dir,
                                                    run_id=run_id, experiment_name=experiment_name,
                                                    mlflow_tracking_uri=mlflow_tracking_uri,
                                                    check_if_exists=check_if_exists, **dataset_combination)
        return self._train_model(n_jobs=n_jobs, create_validation_set=create_validation_set, model_params=model_params,
                                 fit_params=fit_params, return_results=return_results,
                                 clean_work_dir=clean_work_dir, logging_to_mlflow=False, **dataset_combination)

    def run_pandas_combination(self, model_nickname: str, seed_model: int, seed_dataset: int, dataframe: pd.DataFrame,
                               dataset_name: str, target: str, task: str,
                               fold: int = 0, run_id: Optional[str] = None,
                               resample_strategy: str = 'k-fold_cv', n_folds: int = 10, pct_test: float = 0.2,
                               validation_resample_strategy: str = 'next_fold', pct_validation: float = 0.1,
                               n_jobs: int = 1, create_validation_set: bool = False,
                               model_params: Optional[dict] = None,
                               fit_params: Optional[dict] = None, return_results: bool = False,
                               clean_work_dir: bool = True,
                               log_to_mlflow: bool = False,
                               experiment_name: Optional[str] = None, mlflow_tracking_uri: Optional[str] = None,
                               check_if_exists: Optional[bool] = None,
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
        clean_work_dir :
            If True, clean the work directory after the experiment.
        log_to_mlflow :
            If True, log the results to mlflow.
        experiment_name :
            The name of the experiment.
        mlflow_tracking_uri :
            The uri of the mlflow tracking server.
        check_if_exists :
            If True, check if the run already exists on mlflow.
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
        dataset_combination = dict(model_nickname=model_nickname, seed_model=seed_model,
                                   dataframe=dataframe, dataset_name=dataset_name, target=target, task=task,
                                   seed_dataset=seed_dataset, fold=fold, resample_strategy=resample_strategy,
                                   n_folds=n_folds,
                                   pct_test=pct_test, validation_resample_strategy=validation_resample_strategy,
                                   pct_validation=pct_validation, is_openml_task=False)
        dataset_combination.update(kwargs)
        if log_to_mlflow:
            return self._run_mlflow_and_train_model(n_jobs=n_jobs, create_validation_set=create_validation_set,
                                                    model_params=model_params, fit_params=fit_params,
                                                    return_results=return_results, clean_work_dir=clean_work_dir,
                                                    run_id=run_id, experiment_name=experiment_name,
                                                    mlflow_tracking_uri=mlflow_tracking_uri,
                                                    check_if_exists=check_if_exists, **dataset_combination)
        return self._train_model(n_jobs=n_jobs, create_validation_set=create_validation_set, model_params=model_params,
                                 fit_params=fit_params, return_results=return_results,
                                 clean_work_dir=clean_work_dir, logging_to_mlflow=False, **dataset_combination)

    def _run_combination(self, *args, **kwargs):
        """Run the experiment using an OpenML task or an OpenML dataset."""
        is_openml_task = kwargs.get('is_openml_task', False)
        if is_openml_task:
            return self.run_openml_task_combination(*args, **kwargs)
        else:
            return self.run_openml_dataset_combination(*args, **kwargs)

    def _get_combinations(self):
        """Get the combinations of the experiment."""
        if self.using_own_resampling:
            # (model_nickname, seed_model, dataset_name_or_id, seed_dataset, fold)
            combinations = list(product(self.models_nickname, self.seeds_models, self.datasets_names_or_ids,
                                        self.seeds_datasets, self.folds))
            extra_params = dict(is_openml_task=False, resample_strategy=self.resample_strategy,
                                n_folds=self.k_folds, pct_test=self.pct_test,
                                validation_resample_strategy=self.validation_resample_strategy,
                                pct_validation=self.pct_validation)

        else:
            # (model_nickname, seed_model, task_id, task_fold, task_repeat, task_sample)
            combinations = list(product(self.models_nickname, self.seeds_models, self.tasks_ids, self.task_folds,
                                        self.task_repeats, self.task_samples))
            extra_params = dict(is_openml_task=True)
        extra_params.update(dict(n_jobs=self.n_jobs, log_to_mlflow=self.log_to_mlflow,
                                 return_results=False, clean_work_dir=self.clean_work_dir,
                                 create_validation_set=self.create_validation_set, experiment_name=self.experiment_name,
                                 mlflow_tracking_uri=self.mlflow_tracking_uri, check_if_exists=self.check_if_exists))
        return combinations, extra_params

    def _create_mlflow_run(self, *args,
                           create_validation_set=False,
                           model_params=None,
                           fit_params=None,
                           experiment_name=None, mlflow_tracking_uri=None, check_if_exists=None,
                           # just add some kwargs that are not unique to the run to ignore them when checking
                           # for existent runs
                           n_jobs=1, log_to_mlflow=True, return_results=False, clean_work_dir=True,
                           **kwargs):
        """Create a mlflow run."""
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
                if self.save_dir:
                    artifact_location = str(self.save_dir / experiment_name)
                else:
                    artifact_location = None
                experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_name)
            mlflow_client = mlflow.client.MlflowClient(tracking_uri=mlflow_tracking_uri)
            run = mlflow_client.create_run(experiment_id)
            run_id = run.info.run_id
            mlflow_client.update_run(run_id, status='SCHEDULED')
            return run_id

    def _run_experiment(self, client=None):
        """Run the experiment."""

        combinations, extra_params = self._get_combinations()
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
                first_future = client.submit(self._create_mlflow_run, *first_args, resources=resources_per_task,
                                             pure=False, **extra_params)
                futures = [first_future]
                if total_combinations > 1:
                    time.sleep(5)
                    other_futures = client.map(self._create_mlflow_run, *list_of_args, pure=False,
                                               batch_size=self.n_workers, resources=resources_per_task, **extra_params)
                    futures.extend(other_futures)
                run_ids = client.gather(futures)
                for future in futures:
                    future.release()  # release the memory of the future
                combinations = [list(combination) + [run_id] for combination, run_id in zip(combinations, run_ids)]
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

            workers = {value['name']: value['resources'] for worker_address, value
                       in client.scheduler_info()['workers'].items()}
            free_workers = list(workers.keys())
            futures = []
            submitted_combinations = 0
            finished_combinations = 0
            with tqdm(total=len(combinations), desc='Combinations completed') as progress_bar:
                while finished_combinations < total_combinations:
                    # submit tasks to free workers
                    while free_workers and submitted_combinations < total_combinations:
                        worker_name = free_workers.pop()
                        worker = workers[worker_name]
                        combination = list(combinations[submitted_combinations])
                        key = '_'.join(str(arg) for arg in combination)
                        future = client.submit(self._run_combination, *combination, pure=False, key=key,
                                               resources=resources_per_task, workers=[worker_name],
                                               allow_other_workers=True, **extra_params)
                        future.worker = worker_name
                        futures.append(future)
                        worker_can_still_work = True
                        for resource in resources_per_task:
                            worker[resource] -= resources_per_task[resource]
                            if worker[resource] < resources_per_task[resource]:
                                worker_can_still_work = False
                        if worker_can_still_work:
                            free_workers.append(worker_name)
                        submitted_combinations += 1

                    # wait for at least one task to finish
                    completed_future = next(as_completed(futures))
                    combination_success = completed_future.result()
                    if combination_success is True:
                        n_combinations_successfully_completed += 1
                    elif combination_success is False:
                        n_combinations_failed += 1
                    else:
                        n_combinations_none += 1
                    finished_combinations += 1
                    progress_bar.update(1)
                    log_and_print_msg(str(progress_bar), succesfully_completed=n_combinations_successfully_completed,
                                      failed=n_combinations_failed, none=n_combinations_none)
                    completed_worker_name = completed_future.worker
                    worker = workers[completed_worker_name]
                    worker_can_work = True
                    for resource in resources_per_task:
                        worker[resource] += resources_per_task[resource]
                        if worker[resource] < resources_per_task[resource]:
                            worker_can_work = False
                    if worker_can_work:
                        free_workers.append(completed_worker_name)
                    futures.remove(completed_future)
                    completed_future.release()  # release the memory of the future

            client.close()
        else:
            progress_bar = tqdm(combinations, desc='Combinations completed')
            for combination in progress_bar:
                run_id = self._create_mlflow_run(*combination, **extra_params)
                combination_with_run_id = list(combination) + [run_id]
                combination_success = self._run_combination(*combination_with_run_id, **extra_params)
                if combination_success is True:
                    n_combinations_successfully_completed += 1
                elif combination_success is False:
                    n_combinations_failed += 1
                else:
                    n_combinations_none += 1
                log_and_print_msg(str(progress_bar), succesfully_completed=n_combinations_successfully_completed,
                                  failed=n_combinations_failed, none=n_combinations_none)

        return total_combinations, n_combinations_successfully_completed, n_combinations_failed, n_combinations_none

    def _get_kwargs_to_log_experiment(self):
        """Get the kwargs to log the experiment."""
        kwargs_to_log = dict(experiment_name=self.experiment_name, models_nickname=self.models_nickname,
                             seeds_models=self.seeds_models)
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
        """Run the entire pipeline."""
        self._treat_parser()
        if self.datasets_names_or_ids is not None and self.tasks_ids is None:
            self.using_own_resampling = True
        elif self.datasets_names_or_ids is None and self.tasks_ids is not None:
            self.using_own_resampling = False
        else:
            raise ValueError("You must provide either datasets_names_or_ids or tasks_ids, but not both.")
        os.makedirs(self.work_dir, exist_ok=True)
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self._setup_logger()
        kwargs_to_log = self._get_kwargs_to_log_experiment()
        start_time = time.perf_counter()
        log_and_print_msg('Starting experiment...', **kwargs_to_log)
        if self.dask_cluster_type is not None:
            client = self._setup_dask(self.n_workers, self.dask_cluster_type, self.dask_address)
        else:
            client = None
        total_combinations, n_combinations_successfully_completed, n_combinations_failed, n_combinations_none = (
            self._run_experiment(client=client))
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
