import argparse
from pathlib import Path
import mlflow
import os
import logging
import warnings
from tab_benchmark.benchmark.utils import run_openml_combination, run_own_combination
from tab_benchmark.benchmark.benchmarked_models import models_dict

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_model(model_nickname, models_dict):
    model_class, model_kwargs = models_dict[model_nickname]
    if callable(model_kwargs):
        model_kwargs = model_kwargs(model_class)
    return model_class(**model_kwargs)


class BaseExperiment:
    def __init__(
            self,
            # model specific
            model_nickname=None, seeds_models=None, n_jobs=1,
            # when performing our own resampling
            datasets_names_or_ids=None, seeds_datasets=None,
            resample_strategy='k-fold_cv', k_folds=10, folds=None, pct_test=0.2,
            # when using openml tasks
            tasks_ids=None,
            task_repeats=None, task_folds=None, task_samples=None,
            # parameters of experiment
            experiment_name='base_experiment',
            models_dict=models_dict,
            log_dir=Path.cwd() / 'logs',
            output_dir=Path.cwd() / 'output',
            mlflow_tracking_uri='sqlite:///' + str(Path.cwd().resolve()) + '/ts_benchmark.db', check_if_exists=True,
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

    def run_own_combination(self, model_nickname, model_params,
                            seed_model, dataset_name_or_id, seed_dataset, fold, resample_strategy, n_folds, pct_test,
                            n_jobs, create_validation_set, experiment_name, mlflow_tracking_uri, check_if_exists):
        run_own_combination(model_nickname=model_nickname, model_params=model_params, seed_model=seed_model,
                            dataset_name_or_id=dataset_name_or_id, seed_dataset=seed_dataset,
                            resample_strategy=resample_strategy, fold=fold, n_folds=n_folds, pct_test=pct_test,
                            create_validation_set=create_validation_set, n_jobs=n_jobs, experiment_name=experiment_name,
                            mlflow_tracking_uri=mlflow_tracking_uri, check_if_exists=check_if_exists)

    def run_openml_combination(self, model_nickname, model_params, seed_model, task_id, task_repeat, task_sample,
                               task_fold, create_validation_set, n_jobs, experiment_name, mlflow_tracking_uri,
                               check_if_exists):
        run_openml_combination(model_nickname=model_nickname, model_params=model_params, seed_model=seed_model,
                               task_id=task_id, task_repeat=task_repeat, task_sample=task_sample, task_fold=task_fold,
                               create_validation_set=create_validation_set, n_jobs=n_jobs,
                               experiment_name=experiment_name, mlflow_tracking_uri=mlflow_tracking_uri,
                               check_if_exists=check_if_exists)

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
                                self.run_own_combination(model_nickname=self.model_nickname, model_params={},
                                                         seed_model=seed_model,
                                                         dataset_name_or_id=dataset_name_or_id,
                                                         seed_dataset=seed_dataset, fold=fold,
                                                         resample_strategy=self.resample_strategy,
                                                         n_folds=self.k_folds, pct_test=self.pct_test,
                                                         n_jobs=self.n_jobs,
                                                         create_validation_set=False,
                                                         experiment_name=self.experiment_name,
                                                         mlflow_tracking_uri=self.mlflow_tracking_uri,
                                                         check_if_exists=self.check_if_exists)
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
                                    self.run_openml_combination(model_nickname=self.model_nickname, model_params={},
                                                                seed_model=seed_model,
                                                                task_id=task_id, task_repeat=task_repeat,
                                                                task_sample=task_sample, task_fold=task_fold,
                                                                create_validation_set=False, n_jobs=self.n_jobs,
                                                                experiment_name=self.experiment_name,
                                                                mlflow_tracking_uri=self.mlflow_tracking_uri,
                                                                check_if_exists=self.check_if_exists)
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
