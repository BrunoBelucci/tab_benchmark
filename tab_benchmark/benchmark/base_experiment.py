import argparse
from pathlib import Path
import mlflow
import os
import logging
import warnings
from tab_benchmark.benchmark.utils import treat_mlflow, get_model, load_openml_task, fit_model, evaluate_model, \
    load_own_task
from tab_benchmark.benchmark.benchmarked_models import models_dict
from tab_benchmark.utils import get_git_revision_hash, flatten_dict

warnings.simplefilter(action='ignore', category=FutureWarning)


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
            raise_on_fit_error=False, parser=None
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

        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.check_if_exists = check_if_exists
        self.retry_on_oom = retry_on_oom
        self.parser = parser
        self.models_dict = models_dict
        self.raise_on_fit_error = raise_on_fit_error

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
                            format='%(asctime)s - %(levelname)s\n%(message)s',
                            level=logging.INFO, filemode='w')
        msg = (
            f"Experiment name: {self.experiment_name}\n"
            f"Model nickname: {self.model_nickname}\n"
            f"Seeds Models: {self.seeds_model}\n"

        )
        if self.using_own_resampling:
            msg = (msg +
                   (f"Datasets names or ids: {self.datasets_names_or_ids}\n"
                    f"Seeds Datasets: {self.seeds_datasets}\n"
                    f"Resample strategy: {self.resample_strategy}\n"
                    f"K-folds: {self.k_folds}\n"
                    f"Folds: {self.folds}\n"
                    f"Percentage test: {self.pct_test}\n")
                   )
        else:
            msg = (msg +
                   (f"Tasks ids: {self.tasks_ids}\n"
                    f"Task repeats: {self.task_repeats}\n"
                    f"Task samples: {self.task_samples}\n"
                    f"Task folds: {self.task_folds}\n")
                   )
        print(msg)
        logging.info(msg)

    def get_model(self, model_nickname, seed_model, model_params=None, models_dict=models_dict, n_jobs=1,
                  logging_to_mlflow=False):
        model = get_model(model_nickname, seed_model, model_params, models_dict, n_jobs)
        if logging_to_mlflow:
            model_params = vars(model).copy()
            if hasattr(model, 'loss_fn'):
                # will be logged after
                del model_params['loss_fn']
            mlflow.log_params(model_params)
        return model

    def run_combination(self, seed_model=0, n_jobs=1, create_validation_set=False, return_to_fit=False,
                        model_params=None, is_openml=True, logging_to_mlflow=False, **kwargs):
        """

        Parameters
        ----------
        seed_model
        n_jobs
        create_validation_set
        return_to_fit
        model_params
        is_openml
        logging_to_mlflow
        kwargs:
            must contain task_id, task_repeat, task_sample, task_fold if is_openml is True
            must contain dataset_name_or_id, seed_dataset, fold if is_openml is False

        Returns
        -------

        """
        model_params = model_params if model_params is not None else {}
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
                              validation_resample_strategy=validation_resample_strategy, pct_validation=pct_validation)
            )

        # load model
        model = self.get_model(self.model_nickname, seed_model, model_params=model_params, models_dict=self.models_dict,
                               n_jobs=n_jobs, logging_to_mlflow=logging_to_mlflow)

        # fit model
        # data here is already preprocessed
        model, X_train, y_train, X_test, y_test, X_validation, y_validation = fit_model(
            model, X, y, cat_ind, att_names, task_name, train_indices, test_indices, validation_indices,
            logging_to_mlflow, return_to_fit)

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
        return results

    def run_combination_with_mlflow(self, seed_model=0, n_jobs=1, create_validation_set=False, return_to_fit=False,
                                    model_params=None,
                                    parent_run_uuid=None, is_openml=True, **kwargs):
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
            msg = f"Experiment already exists on MLflow. Skipping..."
            print(msg)
            logging.info(msg)
            return None

        if logging_to_mlflow:
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
                                            return_to_fit=return_to_fit, model_params=model_params,
                                            parent_run_uuid=parent_run_uuid, is_openml=is_openml,
                                            logging_to_mlflow=logging_to_mlflow, **kwargs)
        else:
            return self.run_combination(seed_model=seed_model, n_jobs=n_jobs,
                                        create_validation_set=create_validation_set,
                                        return_to_fit=return_to_fit, model_params=model_params,
                                        parent_run_uuid=parent_run_uuid, is_openml=is_openml,
                                        logging_to_mlflow=logging_to_mlflow, **kwargs)

    def run_experiment(self):
        if self.using_own_resampling:
            for seed_dataset in self.seeds_datasets:
                for seed_model in self.seeds_model:
                    for fold in self.folds:
                        for dataset_name_or_id in self.datasets_names_or_ids:
                            try:
                                msg = (
                                    f"Running...\n"
                                    f"Model nickname: {self.model_nickname}\n"
                                    f"Dataset name or id: {dataset_name_or_id}\n"
                                    f"Fold: {fold}\n"
                                    f"Seed model: {seed_model}\n"
                                    f"Seed dataset: {seed_dataset}\n"
                                )
                                print(msg)
                                logging.info(msg)
                                self.run_combination_with_mlflow(
                                    seed_model=seed_model, dataset_name_or_id=dataset_name_or_id,
                                    seed_dataset=seed_dataset, fold=fold, n_jobs=self.n_jobs, is_openml=False)
                            except Exception as exception:
                                if self.raise_on_fit_error:
                                    raise exception
                                msg = (
                                    f"Error\n"
                                    f"Model nickname: {self.model_nickname}\n"
                                    f"Dataset name or id: {dataset_name_or_id}\n"
                                    f"Fold: {fold}\n"
                                    f"Seed model: {seed_model}\n"
                                    f"Seed dataset: {seed_dataset}\n"
                                    f"Exception: {exception}\n"
                                )
                                print(msg)
                                logging.error(msg)
                            else:
                                msg = 'Finished!'
                                print(msg)
                                logging.info(msg)
        else:
            for task_repeat in self.task_repeats:
                for task_sample in self.task_samples:
                    for seed_model in self.seeds_model:
                        for task_fold in self.task_folds:
                            for task_id in self.tasks_ids:
                                try:
                                    msg = (
                                        f"Running...\n"
                                        f"Model nickname: {self.model_nickname}\n"
                                        f"Task id: {task_id}\n"
                                        f"Fold: {task_fold}\n"
                                        f"Seed model: {seed_model}\n"
                                        f"Task sample: {task_sample}\n"
                                        f"Task repeat: {task_repeat}\n"
                                    )
                                    print(msg)
                                    logging.info(msg)
                                    self.run_combination_with_mlflow(seed_model=seed_model,
                                                                     task_id=task_id, task_repeat=task_repeat,
                                                                     task_sample=task_sample,
                                                                     task_fold=task_fold,
                                                                     n_jobs=self.n_jobs, is_openml=True)
                                except Exception as exception:
                                    if self.raise_on_fit_error:
                                        raise exception
                                    msg = (
                                        f"Error\n"
                                        f"Model nickname: {self.model_nickname}\n"
                                        f"Task id: {task_id}\n"
                                        f"Fold: {task_fold}\n"
                                        f"Seed model: {seed_model}\n"
                                        f"Task sample: {task_sample}\n"
                                        f"Task repeat: {task_repeat}\n"
                                        f"Exception: {exception}\n"
                                    )
                                    print(msg)
                                    logging.error(msg)
                                else:
                                    msg = 'Finished!'
                                    print(msg)
                                    logging.info(msg)

    def run(self):
        self.treat_parser()
        if self.datasets_names_or_ids is not None and self.tasks_ids is None:
            self.using_own_resampling = True
        elif self.datasets_names_or_ids is None and self.tasks_ids is not None:
            self.using_own_resampling = False
        else:
            raise ValueError("You must provide either datasets_names_or_ids or tasks_ids, but not both.")
        self.create_logger()
        self.run_experiment()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = BaseExperiment(parser=parser)
    experiment.run()
