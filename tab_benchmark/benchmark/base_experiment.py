import argparse
from pathlib import Path
import mlflow
import os
import logging
import warnings
from openml.tasks import get_task
from sklearn.model_selection import StratifiedKFold, KFold
from tab_benchmark.datasets import get_dataset
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.utils import get_git_revision_hash, set_seeds, train_test_split_forced, evaluate_set
from tab_benchmark.benchmark.benchmarked_models import models_dict

warnings.simplefilter(action='ignore', category=FutureWarning)


# We basically only need those 3 functions if we want to run an experiment iteratively,
# for example using jupyter notebook.
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
            local_tmp_dir=Path('/tmp'),
            mlflow_tracking_uri='sqlite:///ts_benchmark.db', check_if_exists=True, retry_on_oom=True,
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
        self.local_tmp_dir = local_tmp_dir
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
        self.parser.add_argument('--local_tmp_dir', type=Path, default=self.local_tmp_dir)
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
        # self.pct_valid_es = args.pct_valid_es

        self.tasks_ids = args.tasks_ids
        self.task_repeats = args.task_repeats
        self.task_folds = args.task_folds
        self.task_samples = args.task_samples

        self.log_dir = args.log_dir
        self.local_tmp_dir = args.local_tmp_dir
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
        name = self.experiment_name
        if (self.log_dir / f'{name}.log').exists():
            file_names = sorted(self.log_dir.glob(f'{name}_????.log'))
            if file_names:
                file_name = file_names[-1].name
                id = int(file_name.split('_')[-1].split('.')[0])
                name = f'{name}_{id + 1:04d}'
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

    def check_if_own_exists_on_mlflow(self, dataset_name_or_id, seed_dataset, seed_model, fold):
        filter_string = (f"params.seed_dataset = '{seed_dataset}' "
                         f"AND params.dataset_name_or_id = '{dataset_name_or_id}' "
                         f"AND params.seed_model = '{seed_model}' "
                         f"AND params.fold = '{fold}' "
                         f"AND params.model_nickname = '{self.model_nickname}' "
                         f"AND params.k_folds = '{self.k_folds}' "
                         f"AND params.resample_strategy = '{self.resample_strategy}' "
                         f"AND params.pct_test = '{self.pct_test}' ")
        runs = mlflow.search_runs(experiment_names=[self.experiment_name], filter_string=filter_string)
        runs = runs.loc[runs['status'] == 'FINISHED']
        if not runs.empty:
            msg = f"Experiment already exists on MLflow. Skipping..."
            print(msg)
            logging.info(msg)
            return True
        return False

    def check_if_openml_exists_on_mlflow(self, task_id, task_repeat, task_sample, seed_model, task_fold):
        filter_string = (f"params.task_id = '{task_id}' "
                         f"AND params.seed_model = '{seed_model}' AND params.task_fold = '{task_fold}' "
                         f"AND params.model_nickname = '{self.model_nickname}' "
                         f"AND params.task_repeat = '{task_repeat}' "
                         f"AND params.task_sample = '{task_sample}' ")
        runs = mlflow.search_runs(experiment_names=[self.experiment_name], filter_string=filter_string)
        runs = runs.loc[runs['status'] == 'FINISHED']
        if not runs.empty:
            msg = f"Experiment already exists on MLflow. Skipping..."
            print(msg)
            logging.info(msg)
            return True
        return False

    def start_mlflow_own(self, dataset_name_or_id, seed_dataset, seed_model, fold):
        mlflow.set_experiment(self.experiment_name)
        run_name = f"{self.model_nickname}_{seed_model:04d}_{dataset_name_or_id}_{seed_dataset:04d}_{fold:02d}"
        mlflow.start_run(run_name=run_name)
        dataset, task, target = get_dataset(dataset_name_or_id)
        mlflow.log_param('model_nickname', self.model_nickname)
        mlflow.log_param('git_hash', get_git_revision_hash())
        mlflow.log_param('dataset_name_or_id', dataset_name_or_id)
        mlflow.log_param('seed_dataset', seed_dataset)
        mlflow.log_param('seed_model', seed_model)
        mlflow.log_param('fold', fold)
        mlflow.log_param('dataset_id', dataset.id)
        mlflow.log_param('dataset_name', dataset.name)
        mlflow.log_param('resample_strategy', self.resample_strategy)
        mlflow.log_param('k_folds', self.k_folds)
        mlflow.log_param('pct_test', self.pct_test)
        mlflow.log_param('task', task)
        mlflow.log_param('target', target)

    def run_own_combination(self, dataset_name_or_id, seed_dataset, seed_model, fold):
        if self.check_if_exists:
            if self.check_if_own_exists_on_mlflow(dataset_name_or_id, seed_dataset, seed_model, fold):
                return
        self.start_mlflow_own(dataset_name_or_id, seed_dataset, seed_model, fold)
        dataset, task, target = get_dataset(dataset_name_or_id)
        if target is None:
            target = dataset.default_target_attribute
        X, y, cat_ind, att_names = dataset.get_data(target=target)
        if self.resample_strategy == 'hold_out':
            test_size = int(self.pct_test * len(dataset.qualities['NumberOfInstances']))
            if task in ('classification', 'binary_classification'):
                stratify = y
            elif task in ('regression', 'multi_regression'):
                stratify = None
            else:
                raise NotImplementedError
            X_train, X_test, y_train, y_test = train_test_split_forced(X, y, test_size_pct=test_size,
                                                                       random_state=seed_dataset, stratify=stratify)
            split_train = X_train.index
            split_test = X_test.index
        elif self.resample_strategy == 'k-fold_cv':
            if task == 'classification':
                kf = StratifiedKFold(n_splits=self.k_folds, random_state=seed_dataset, shuffle=True)
            elif task == 'regression':
                kf = KFold(n_splits=self.k_folds, random_state=seed_dataset, shuffle=True)
            else:
                raise NotImplementedError
            folds = list(kf.split(X, y))
            split_train, split_test = folds[fold]
        else:
            raise NotImplementedError
        self.run_combination(seed_model, dataset, split_train, split_test, target, task)

    def start_mlflow_openml(self, task_id, task_repeat, task_sample, seed_model, task_fold):
        mlflow.set_experiment(self.experiment_name)
        run_name = (f"{self.model_nickname}_{seed_model:04d}_{task_id}_{task_repeat:02d}_{task_sample:02d}_"
                    f"{task_fold:02d}")
        mlflow.start_run(run_name=run_name)
        task = get_task(task_id)
        dataset = task.get_dataset()
        mlflow.log_param('model_nickname', self.model_nickname)
        mlflow.log_param('git_hash', get_git_revision_hash())
        mlflow.log_param('task_id', task.id)
        mlflow.log_param('task_repeat', task_repeat)
        mlflow.log_param('task_sample', task_sample)
        mlflow.log_param('task_fold', task_fold)
        mlflow.log_param('seed_model', seed_model)
        mlflow.log_param('dataset_id', dataset.id)
        mlflow.log_param('dataset_name', dataset.name)
        task_type = task.task_type
        if task_type == 'Supervised Classification':
            task_name = 'classification'
        elif task_type == 'Supervised Regression':
            task_name = 'regression'
        else:
            raise NotImplementedError
        mlflow.log_param('task_type', task_type)
        mlflow.log_param('task', task_name)
        mlflow.log_param('target', task.target_name)

    def run_openml_combination(self, task_id, task_repeat, task_sample, seed_model, task_fold):
        if self.check_if_exists:
            if self.check_if_openml_exists_on_mlflow(task_id, task_repeat, task_sample, seed_model, task_fold):
                return
        self.start_mlflow_openml(task_id, task_repeat, task_sample, seed_model, task_fold)
        task = get_task(task_id)
        dataset = task.get_dataset()
        split = task.get_train_test_split_indices(task_fold, task_repeat, task_sample)
        task_type = task.task_type
        if task_type == 'Supervised Classification':
            task_name = 'classification'
        elif task_type == 'Supervised Regression':
            task_name = 'regression'
        else:
            raise NotImplementedError
        self.run_combination(seed_model, dataset, split.train, split.test, task.target_name, task_name)

    def get_model(self):
        model_class, model_kwargs = self.models_dict[self.model_nickname]
        if callable(model_kwargs):
            model_kwargs = model_kwargs(model_class)
        model = model_class(**model_kwargs)
        if hasattr(model, 'n_jobs'):
            if isinstance(model, DNNModel) and self.n_jobs == 1:
                # set n_jobs to 0 for DNNModel (no parallelism)
                setattr(model, 'n_jobs', 0)
            setattr(model, 'n_jobs', self.n_jobs)
        model_params = vars(model).copy()
        if issubclass(model_class, DNNModel):
            # will be logged after
            del model_params['loss_fn']
        mlflow.log_params(model_params)
        return model

    def run_combination(self, seed_model, dataset, split_train, split_test, target, task):
        X, y, cat_ind, att_names = dataset.get_data(target=target)
        X_train = X.iloc[split_train]
        y_train = y.iloc[split_train]
        X_test = X.iloc[split_test]
        y_test = y.iloc[split_test]
        cat_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is True]
        cont_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is False]
        model, X_train, y_train, X_test, y_test = self.preprocess_and_fit_model(seed_model, X_train, y_train, X_test,
                                                                                y_test, cat_features_names,
                                                                                cont_features_names, att_names, task)
        if task in ('classification', 'binary_classification'):
            metrics = ['logloss', 'auc']
            n_classes = len(y_train.iloc[:, 0].unique())
        elif task == 'regression':
            metrics = ['rmse', 'r2_score']
            n_classes = None
        else:
            raise NotImplementedError
        self.evaluate_model(model, X_test, y_test, metrics, n_classes)

    # can be overridden
    def evaluate_model(self, model, X_test_preprocessed, y_test_preprocessed, metrics, n_classes):
        test_results = evaluate_set(model, [X_test_preprocessed, y_test_preprocessed], metrics, n_classes)
        for metric, test_result in zip(metrics, test_results):
            mlflow.log_metric(f'test_{metric}', test_result)

    # can be overridden
    def preprocess_and_fit_model(self, seed_model, X_train, y_train, X_test, y_test, cat_features_names,
                                 cont_features_names, att_names, task):
        set_seeds(seed_model)
        model = self.get_model()
        if not (('classifier' in model._estimator_type and task in ('classification', 'binary_classification')) or (
                'regressor' in model._estimator_type and task in ('regression', 'multi_regression'))):
            raise ValueError('Model class and task do not match')
        # safer to preprocess and fit the model separately
        model.create_preprocess_pipeline(task, cat_features_names, cont_features_names, att_names)
        model.create_model_pipeline()
        data_preprocess_pipeline_ = model.data_preprocess_pipeline_
        target_preprocess_pipeline_ = model.target_preprocess_pipeline_
        X_train = data_preprocess_pipeline_.fit_transform(X_train)
        y_train = target_preprocess_pipeline_.fit_transform(y_train.to_frame())
        X_test = data_preprocess_pipeline_.transform(X_test)
        y_test = target_preprocess_pipeline_.transform(y_test.to_frame())
        model.fit(X_train, y_train, task=task, cat_features=cat_features_names)
        return model, X_train, y_train, X_test, y_test

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
                                self.run_own_combination(dataset_name_or_id, seed_dataset, seed_model, fold)
                            except Exception as exception:
                                if self.raise_on_fit_error:
                                    mlflow.end_run('FAILED')
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
                                mlflow.end_run('FAILED')
                            else:
                                msg = 'Finished!'
                                print(msg)
                                logging.info(msg)
                                mlflow.end_run('FINISHED')
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
                                    self.run_openml_combination(task_id, task_repeat, task_sample, seed_model,
                                                                task_fold)
                                except Exception as exception:
                                    if self.raise_on_fit_error:
                                        mlflow.end_run('FAILED')
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
                                    mlflow.end_run('FAILED')
                                else:
                                    msg = 'Finished!'
                                    print(msg)
                                    logging.info(msg)
                                    mlflow.end_run('FINISHED')

    def run(self):
        self.treat_parser()
        if self.datasets_names_or_ids is not None and self.tasks_ids is None:
            self.using_own_resampling = True
        elif self.datasets_names_or_ids is None and self.tasks_ids is not None:
            self.using_own_resampling = False
        else:
            raise ValueError("You must provide either datasets_names_or_ids or tasks_ids, but not both.")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.create_logger()
        self.run_experiment()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = BaseExperiment(parser=parser)
    experiment.run()
