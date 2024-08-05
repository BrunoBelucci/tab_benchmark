from __future__ import annotations
import argparse
from pathlib import Path
import mlflow
import os
import logging
import warnings
from distributed import WorkerPlugin, Worker, Client
from tab_benchmark.benchmark.utils import treat_mlflow, get_model, load_openml_task, fit_model, evaluate_model, \
    load_own_task
from tab_benchmark.benchmark.benchmarked_models import models_dict
from tab_benchmark.utils import get_git_revision_hash, flatten_dict
from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster

warnings.simplefilter(action='ignore', category=FutureWarning)


class LoggingSetter(WorkerPlugin):
    def __init__(self, logging_config=None):
        self.logging_config = logging_config if logging_config is not None else {}
        super().__init__()

    def setup(self, worker: Worker):
        logging.basicConfig(**self.logging_config)


def log_and_print_msg(fist_line, **kwargs):
    fist_line = f"{fist_line}\n"
    fist_line += "".join([f"{key}: {value}\n" for key, value in kwargs.items()])
    print(fist_line)
    logging.info(fist_line)


class BaseExperiment:
    def __init__(
            self,
            # model specific
            model_nickname=None, seeds_models=None, n_jobs=1,
            # when performing our own resampling
            datasets_names_or_ids=None, seeds_datasets=None,
            resample_strategy='k-fold_cv', k_folds=10, folds=None, pct_test=0.2,
            validation_resample_strategy='next_fold', pct_validation=0.1,
            # when using openml tasks
            tasks_ids=None,
            task_repeats=None, task_folds=None, task_samples=None,
            # parameters of experiment
            experiment_name='base_experiment',
            models_dict=models_dict,
            log_dir=Path.cwd() / 'logs',
            output_dir=Path.cwd() / 'output',
            mlflow_tracking_uri='sqlite:///' + str(Path.cwd().resolve()) + '/tab_benchmark.db', check_if_exists=True,
            retry_on_oom=True,
            raise_on_fit_error=False, parser=None,
            # parallelization
            dask_cluster_type=None,
            n_workers=1,
            slurm_config_name=None,
            dask_address=None,
    ):
        self.model_nickname = model_nickname
        self.seeds_model = seeds_models if seeds_models else [0]
        self.n_jobs = n_jobs

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
        self.slurm_config_name = slurm_config_name
        self.dask_address = dask_address

        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.check_if_exists = check_if_exists
        self.retry_on_oom = retry_on_oom
        self.parser = parser
        self.models_dict = models_dict
        self.raise_on_fit_error = raise_on_fit_error
        self.client = None

    def add_arguments_to_parser(self):
        self.parser.add_argument('--experiment_name', type=str, default=self.experiment_name)
        self.parser.add_argument('--model_nickname', type=str, choices=self.models_dict.keys(),
                                 default=self.model_nickname)
        self.parser.add_argument('--seeds_model', nargs='*', type=int, default=self.seeds_model)
        self.parser.add_argument('--n_jobs', type=int, default=self.n_jobs)

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
        self.parser.add_argument('--mlflow_tracking_uri', type=str, default=self.mlflow_tracking_uri)
        self.parser.add_argument('--do_not_check_if_exists', action='store_true')
        self.parser.add_argument('--do_not_retry_on_oom', action='store_true')
        self.parser.add_argument('--raise_on_fit_error', action='store_true')

        self.parser.add_argument('--dask_cluster_type', type=str, default=self.dask_cluster_type)
        self.parser.add_argument('--n_workers', type=int, default=self.n_workers)
        self.parser.add_argument('--slurm_config_name', type=str, default=self.slurm_config_name)
        self.parser.add_argument('--dask_address', type=str, default=self.dask_address)

    def unpack_parser(self):
        args = self.parser.parse_args()
        self.experiment_name = args.experiment_name
        self.model_nickname = args.model_nickname
        self.n_jobs = args.n_jobs

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
        self.output_dir = args.output_dir
        self.mlflow_tracking_uri = args.mlflow_tracking_uri
        self.check_if_exists = not args.do_not_check_if_exists
        self.retry_on_oom = not args.do_not_retry_on_oom
        self.raise_on_fit_error = args.raise_on_fit_error

        self.dask_cluster_type = args.dask_cluster_type
        self.n_workers = args.n_workers
        self.slurm_config_name = args.slurm_config_name
        self.dask_address = args.dask_address
        return args

    def treat_parser(self):
        if self.parser is not None:
            self.add_arguments_to_parser()
            self.unpack_parser()

    def create_logger(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.output_dir = self.output_dir / self.experiment_name
        os.makedirs(self.output_dir, exist_ok=True)
        name = self.experiment_name
        if (self.log_dir / f'{name}.log').exists():
            file_names = sorted(self.log_dir.glob(f'{name}_????.log'))
            if file_names:
                file_name = file_names[-1].name
                id_file = int(file_name.split('_')[-1].split('.')[0])
                name = f'{name}_{id_file + 1:04d}'
            else:
                name = name + '_0001'
        logging.basicConfig(filename=self.log_dir / f'{name}.log',
                            format='%(asctime)s - %(levelname)s\n%(message)s\n',
                            level=logging.INFO, filemode='w')
        kwargs_to_log = dict(experiment_name=self.experiment_name, model_nickname=self.model_nickname,
                             seeds_model=self.seeds_model)
        if self.using_own_resampling:
            kwargs_to_log.update(dict(datasets_names_or_ids=self.datasets_names_or_ids,
                                      seeds_datasets=self.seeds_datasets,
                                      resample_strategy=self.resample_strategy, k_folds=self.k_folds, folds=self.folds,
                                      pct_test=self.pct_test))
        else:
            kwargs_to_log.update(dict(tasks_ids=self.tasks_ids, task_repeats=self.task_repeats,
                                      task_samples=self.task_samples, task_folds=self.task_folds))
        log_and_print_msg('Starting experiment...', **kwargs_to_log)

    def get_model(self, model_nickname, seed_model, model_params=None, models_dict=models_dict, n_jobs=1,
                  logging_to_mlflow=False):
        model = get_model(model_nickname, seed_model, model_params, models_dict, n_jobs, output_dir=self.output_dir)
        if logging_to_mlflow:
            model_params = vars(model).copy()
            if hasattr(model, 'loss_fn'):
                # will be logged after
                del model_params['loss_fn']
            mlflow.log_params(model_params)
        return model

    def run_combination(self, seed_model=0, n_jobs=1, create_validation_set=False,
                        model_params=None, is_openml=True, logging_to_mlflow=False,
                        fit_params=None, return_results=False, **kwargs):
        """

        Parameters
        ----------
        seed_model
        n_jobs
        create_validation_set
        model_params
        is_openml
        logging_to_mlflow
        kwargs:
            must contain task_id, task_repeat, task_sample, task_fold if is_openml is True
            must contain dataset_name_or_id, seed_dataset, fold if is_openml is False

        Returns
        -------

        """
        try:
            fit_params = fit_params if fit_params is not None else {}
            model_params = model_params if model_params is not None else {}
            # logging
            kwargs_to_log = dict(model_nickname=self.model_nickname, seed_model=seed_model, **kwargs)
            log_and_print_msg('Running...', **kwargs_to_log)
            # load data
            if is_openml:
                task_id = kwargs['task_id']
                task_repeat = kwargs['task_repeat']
                task_sample = kwargs['task_sample']
                task_fold = kwargs['task_fold']
                X, y, cat_ind, att_names, task_name, train_indices, test_indices, validation_indices = (
                    load_openml_task(task_id, task_repeat, task_sample, task_fold,
                                     create_validation_set=create_validation_set)
                )
            else:
                dataset_name_or_id = kwargs['dataset_name_or_id']
                seed_dataset = kwargs['seed_dataset']
                fold = kwargs['fold']
                resample_strategy = self.resample_strategy
                n_folds = self.k_folds
                pct_test = self.pct_test
                validation_resample_strategy = self.validation_resample_strategy
                pct_validation = self.pct_validation
                X, y, cat_ind, att_names, task_name, train_indices, test_indices, validation_indices = (
                    load_own_task(dataset_name_or_id, seed_dataset, resample_strategy, n_folds, pct_test, fold,
                                  create_validation_set=create_validation_set,
                                  validation_resample_strategy=validation_resample_strategy,
                                  pct_validation=pct_validation)
                )

            # load model
            model = self.get_model(self.model_nickname, seed_model, model_params=model_params,
                                   models_dict=self.models_dict,
                                   n_jobs=n_jobs, logging_to_mlflow=logging_to_mlflow)

            # fit model
            # data here is already preprocessed
            model, X_train, y_train, X_test, y_test, X_validation, y_validation = fit_model(
                model, X, y, cat_ind, att_names, task_name, train_indices, test_indices, validation_indices,
                logging_to_mlflow, **fit_params)

            # evaluate model
            if task_name in ('classification', 'binary_classification'):
                metrics = ['logloss', 'auc']
                default_metric = 'logloss'
                n_classes = len(y.unique())
            elif task_name == 'regression':
                metrics = ['rmse', 'r2_score']
                default_metric = 'rmse'
                n_classes = None
            else:
                raise NotImplementedError

            results = evaluate_model(model, (X_test, y_test), 'test', metrics, default_metric, n_classes,
                                     logging_to_mlflow)
            if create_validation_set:
                validation_results = evaluate_model(model, (X_validation, y_validation), 'validation', metrics,
                                                    default_metric, n_classes, logging_to_mlflow)
                results.update(validation_results)
            results.update({
                'model': model,
                'task_name': task_name,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'X_validation': X_validation,
                'y_validation': y_validation,
                'cat_features_names': [att_names[i] for i, value in enumerate(cat_ind) if value is True],
                'n_classes': n_classes,
                'metrics': metrics,
                'default_metric': default_metric,
            })
        except Exception as exception:
            if self.raise_on_fit_error:
                raise exception
            log_and_print_msg('Error while running', **kwargs_to_log)
            return None
        else:
            log_and_print_msg('Finished!', **kwargs_to_log)
            if return_results:
                return results

    def run_combination_with_mlflow(self, seed_model=0, n_jobs=1, create_validation_set=False,
                                    model_params=None, fit_params=None,
                                    parent_run_uuid=None, is_openml=True, **kwargs):
        fit_params = fit_params if fit_params is not None else {}
        model_params = model_params if model_params is not None else {}
        experiment_name = kwargs.pop('experiment_name', self.experiment_name)
        mlflow_tracking_uri = kwargs.pop('mlflow_tracking_uri', self.mlflow_tracking_uri)
        check_if_exists = kwargs.pop('check_if_exists', self.check_if_exists)
        model_nickname = kwargs.pop('model_nickname', self.model_nickname)
        if not is_openml:
            kwargs.update({
                'resample_strategy': kwargs.pop('resample_strategy', self.resample_strategy),
                'n_folds': kwargs.pop('n_folds', self.k_folds),
                'pct_test': kwargs.pop('pct_test', self.pct_test),
                'validation_resample_strategy': kwargs.pop('validation_resample_strategy',
                                                           self.validation_resample_strategy),
                'pct_validation': kwargs.pop('pct_validation', self.pct_validation),
            })
        unique_params = dict(model_nickname=model_nickname, model_params=model_params, seed_model=seed_model, **kwargs)
        exists, logging_to_mlflow = treat_mlflow(experiment_name, mlflow_tracking_uri, check_if_exists, **unique_params)

        if exists:
            log_and_print_msg('Experiment already exists on MLflow. Skipping...')
            return None

        if logging_to_mlflow:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name, artifact_location=str(self.output_dir))
            mlflow.set_experiment(experiment_name)
            run_name = '_'.join([f'{k}={v}' for k, v in unique_params.items()])
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
            with mlflow.start_run(run_name=run_name, nested=nested) as run:
                parent_run_uuid = run.info.run_uuid
                mlflow.log_params(flatten_dict(unique_params))
                mlflow.log_param('git_hash', get_git_revision_hash())
                return self.run_combination(seed_model=seed_model, n_jobs=n_jobs,
                                            create_validation_set=create_validation_set,
                                            model_params=model_params, fit_params=fit_params,
                                            parent_run_uuid=parent_run_uuid, is_openml=is_openml,
                                            logging_to_mlflow=logging_to_mlflow, **kwargs)
        else:
            return self.run_combination(seed_model=seed_model, n_jobs=n_jobs,
                                        create_validation_set=create_validation_set,
                                        model_params=model_params, fit_params=fit_params,
                                        parent_run_uuid=parent_run_uuid, is_openml=is_openml,
                                        logging_to_mlflow=logging_to_mlflow, **kwargs)

    def setup_dask(self, n_workers, cluster_type='local', slurm_config_name=None, address=None):
        if address is not None:
            client = Client(address)
        else:
            if cluster_type == 'local':
                cluster = LocalCluster(n_workers=0, threads_per_worker=1)
            elif cluster_type == 'slurm':
                cluster = SLURMCluster(config_name=slurm_config_name)
            else:
                raise ValueError("cluster_type must be either 'local' or 'slurm'.")
            cluster.scale(n_workers)
            client = cluster.get_client()
        plugin = LoggingSetter(logging_config={'level': logging.INFO})
        client.register_plugin(plugin)
        client.forward_logging()
        return client

    def run_experiment(self, client=None):
        if client is not None:
            futures = []
        if self.using_own_resampling:
            for seed_dataset in self.seeds_datasets:
                for seed_model in self.seeds_model:
                    for fold in self.folds:
                        for dataset_name_or_id in self.datasets_names_or_ids:
                            if client is not None:
                                futures.append(client.submit(self.run_combination_with_mlflow,
                                                             seed_model=seed_model,
                                                             dataset_name_or_id=dataset_name_or_id,
                                                             seed_dataset=seed_dataset, fold=fold, n_jobs=self.n_jobs,
                                                             is_openml=False))
                            else:
                                self.run_combination_with_mlflow(
                                    seed_model=seed_model, dataset_name_or_id=dataset_name_or_id,
                                    seed_dataset=seed_dataset, fold=fold, n_jobs=self.n_jobs, is_openml=False)
        else:
            for task_repeat in self.task_repeats:
                for task_sample in self.task_samples:
                    for seed_model in self.seeds_model:
                        for task_fold in self.task_folds:
                            for task_id in self.tasks_ids:
                                if client is not None:
                                    futures.append(client.submit(self.run_combination_with_mlflow,
                                                                 seed_model=seed_model, task_id=task_id,
                                                                 task_repeat=task_repeat,
                                                                 task_sample=task_sample, task_fold=task_fold,
                                                                 n_jobs=self.n_jobs, is_openml=True))
                                else:
                                    self.run_combination_with_mlflow(seed_model=seed_model,
                                                                     task_id=task_id, task_repeat=task_repeat,
                                                                     task_sample=task_sample,
                                                                     task_fold=task_fold,
                                                                     n_jobs=self.n_jobs, is_openml=True)
        if client is not None:
            client.gather(futures)
            client.close()

    def run(self):
        self.treat_parser()
        if self.datasets_names_or_ids is not None and self.tasks_ids is None:
            self.using_own_resampling = True
        elif self.datasets_names_or_ids is None and self.tasks_ids is not None:
            self.using_own_resampling = False
        else:
            raise ValueError("You must provide either datasets_names_or_ids or tasks_ids, but not both.")
        self.create_logger()
        if self.dask_cluster_type is not None:
            client = self.setup_dask(self.n_workers, self.dask_cluster_type, self.slurm_config_name, self.dask_address)
        else:
            client = None
        self.run_experiment(client=client)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = BaseExperiment(parser=parser)
    experiment.run()
