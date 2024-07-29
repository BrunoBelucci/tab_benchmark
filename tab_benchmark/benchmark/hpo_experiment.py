import argparse
import logging

from base_experiment import BaseExperiment
from tab_benchmark.benchmark.utils import run_openml_combination_hpo, run_own_combination_hpo, run_openml_combination, \
    check_if_exists_mlflow
import mlflow
from tab_benchmark.utils import get_git_revision_hash


class HPOExperiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_algorithm = 'random_search'
        self.n_trials = 30
        self.timeout_experiment = 10 * 60 * 60  # 10 hours
        self.timeout_trial = 2 * 60 * 60  # 2 hours

    def add_arguments_to_parser(self):
        super().add_arguments_to_parser()
        self.parser.add_argument('--search_algorithm', type=str, default=self.search_algorithm)
        self.parser.add_argument('--n_trials', type=int, default=self.n_trials)
        self.parser.add_argument('--timeout_experiment', type=int, default=self.timeout_experiment)
        self.parser.add_argument('--timeout_trial', type=int, default=self.timeout_trial)

    def unpack_parser(self):
        args = super().unpack_parser()
        self.search_algorithm = args.search_algorithm
        self.n_trials = args.n_trials
        self.timeout_experiment = args.timeout_experiment
        self.timeout_trial = args.timeout_trial

    def run_openml_combination(self, model_nickname, model_params, seed_model, task_id, task_repeat, task_sample,
                               task_fold, create_validation_set, n_jobs, experiment_name, mlflow_tracking_uri,
                               check_if_exists):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        run_args = dict(model_nickname=model_nickname, seed_model=seed_model, task_id=task_id, task_repeat=task_repeat,
                        task_sample=task_sample, task_fold=task_fold, search_algorithm=self.search_algorithm,
                        n_trials=self.n_trials, timeout_experiment=self.timeout_experiment,
                        timeout_trial=self.timeout_trial, n_jobs=n_jobs)
        if self.check_if_exists:
            if check_if_exists_mlflow(experiment_name, **run_args):
                msg = f"Experiment already exists on MLflow. Skipping..."
                print(msg)
                logging.info(msg)
                return
        run_name = '_'.join([f'{k}={v}' for k, v in run_args.items()])
        with mlflow.start_run(run_name=run_name) as run:
            run_uuid = run.info.run_uuid
            mlflow.log_params(run_args)
            mlflow.log_param('git_hash', get_git_revision_hash())
            result = run_openml_combination_hpo(search_algorithm_str=self.search_algorithm, n_trials=self.n_trials,
                                                timeout_experiment=self.timeout_experiment,
                                                timeout_trial=self.timeout_trial, storage_path=self.output_dir,
                                                model_nickname=model_nickname, seed_model=seed_model, n_jobs=n_jobs,
                                                task_id=task_id, task_repeat=task_repeat, task_sample=task_sample,
                                                task_fold=task_fold, mlflow_tracking_uri=mlflow_tracking_uri,
                                                experiment_name=experiment_name, parent_run_uuid=run_uuid,
                                                models_dict=self.models_dict)
            best_result = result.get_best_result()
            mlflow.log_params(best_result.metrics['config']['model_params'])
            for metric_name, metric_value in best_result.metrics.items():
                if metric_name.startswith('validation_') or metric_name.startswith('test_'):
                    mlflow.log_metric('best_' + metric_name, metric_value)

    def run_own_combination(self, model_nickname, model_params,
                            seed_model, dataset_name_or_id, seed_dataset, fold, resample_strategy, n_folds, pct_test,
                            n_jobs, create_validation_set, experiment_name, mlflow_tracking_uri, check_if_exists):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        run_args = dict(model_nickname=model_nickname, seed_model=seed_model, dataset_name_or_id=dataset_name_or_id,
                        seed_dataset=seed_dataset, fold=fold, resample_strategy=resample_strategy, n_folds=n_folds,
                        pct_test=pct_test, search_algorithm=self.search_algorithm, n_trials=self.n_trials,
                        timeout_experiment=self.timeout_experiment, timeout_trial=self.timeout_trial, n_jobs=n_jobs)
        if self.check_if_exists:
            if check_if_exists_mlflow(experiment_name, **run_args):
                msg = f"Experiment already exists on MLflow. Skipping..."
                print(msg)
                logging.info(msg)
                return
        run_name = '_'.join([f'{k}={v}' for k, v in run_args.items()])
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(run_args)
            mlflow.log_param('git_hash', get_git_revision_hash())
            result = run_own_combination_hpo(search_algorithm_str=self.search_algorithm, n_trials=self.n_trials,
                                             timeout_experiment=self.timeout_experiment,
                                             timeout_trial=self.timeout_trial, storage_path=self.output_dir,
                                             model_nickname=model_nickname, seed_model=seed_model, n_jobs=n_jobs,
                                             dataset_name_or_id=dataset_name_or_id, seed_dataset=seed_dataset,
                                             resample_strategy=resample_strategy, fold=fold, n_folds=n_folds,
                                             pct_test=pct_test, mlflow_tracking_uri=mlflow_tracking_uri,
                                             experiment_name=experiment_name, parent_run_uuid=run.info.run_uuid,
                                             models_dict=self.models_dict)
            best_result = result.get_best_result()
            mlflow.log_params(best_result.metrics['config']['model_params'])
            for metric_name, metric_value in best_result.metrics.items():
                if metric_name.startswith('validation_') or metric_name.startswith('test_'):
                    mlflow.log_metric(metric_name, metric_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOExperiment(parser=parser)
    experiment.run()
