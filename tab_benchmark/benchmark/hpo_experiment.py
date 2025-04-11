import argparse
from shutil import copytree, rmtree
from typing import Optional
import mlflow
import optuna
from pathlib import Path
from ml_experiments.base_experiment import BaseExperiment
from ml_experiments.hpo_experiment import HPOExperiment
from tab_benchmark.benchmark.tabular_experiment import TabularExperiment
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.models.dnn_models import max_epochs_dnn
from tab_benchmark.models.xgboost import n_estimators_gbdt


class TabularHPOExperiment(HPOExperiment, TabularExperiment):
    def _load_data(self, combination: dict, unique_params: Optional[dict] = None,
                   extra_params: Optional[dict] = None, **kwargs):
        result = super()._load_data(combination=combination, unique_params=unique_params,
                                    extra_params=extra_params, **kwargs)
        # we do not need to keep the data, we will only keep the dataset_name and task_name for logging
        keep_results = {'dataset_name': result['dataset_name'], 'task_name': result['task_name']}
        return keep_results

    def get_hyperband_max_resources(self, combination: dict, unique_params: Optional[dict] = None,
                                    extra_params: Optional[dict] = None, **kwargs):
        model_nickname = combination['model_nickname']
        model_cls = self.models_dict[model_nickname][0]
        if issubclass(model_cls, DNNModel):
            return max_epochs_dnn
        else:
            return n_estimators_gbdt

    def _load_single_experiment(self, combination: dict, unique_params: Optional[dict] = None,
                                extra_params: Optional[dict] = None, **kwargs):
        tabular_experiment = TabularExperiment(
            resample_strategy=self.resample_strategy, k_folds=self.k_folds,
            pct_test=self.pct_test, validation_resample_strategy=self.validation_resample_strategy,
            pct_validation=self.pct_validation, experiment_name=self.experiment_name,
            create_validation_set=self.create_validation_set, log_dir=self.log_dir,
            log_file_name=self.log_file_name, work_root_dir=self.work_root_dir,
            save_root_dir=self.save_root_dir, clean_work_dir=self.clean_work_dir,
            raise_on_fit_error=self.raise_on_fit_error, error_score=self.error_score,
            log_to_mlflow=self.log_to_mlflow, mlflow_tracking_uri=self.mlflow_tracking_uri,
            check_if_exists=self.check_if_exists, max_time=self.max_time, timeout_combination=self.timeout_combination,
            verbose=0,
        )
        return tabular_experiment

    def _training_fn(self, single_experiment: BaseExperiment, trial_combination: dict, optuna_trial: optuna.Trial,
                     unique_params: Optional[dict] = None, extra_params: Optional[dict] = None, **kwargs):
        trial_combination['fit_params']['optuna_trial'] = optuna_trial
        return super()._training_fn(single_experiment=single_experiment, trial_combination=trial_combination,
                                    optuna_trial=optuna_trial, unique_params=unique_params, extra_params=extra_params,
                                    **kwargs)

    def _on_exception_or_train_end(self, combination: dict, unique_params: Optional[dict] = None,
                                   extra_params: Optional[dict] = None, **kwargs):
        mlflow_run_id = extra_params.get('mlflow_run_id', None)
        self._log_run_results(combination=combination, unique_params=unique_params, extra_params=extra_params,
                              mlflow_run_id=mlflow_run_id, **kwargs)

        # save and/or clean work_dir
        load_model_return = kwargs.get('load_model_return', dict())
        study = load_model_return.get('study', None)
        if study is not None:
            best_trial_result = study.best_trial.user_attrs.get('result', dict())
            best_trial_combination = best_trial_result.get('trial_combination', dict())
            best_child_run_id = best_trial_combination.get('mlflow_run_id', None)
            best_work_dir = best_trial_result.get('work_dir', None)
            work_dir = self.get_local_work_dir(combination, mlflow_run_id, unique_params)
            if self.save_root_dir:
                if best_child_run_id is not None:
                    runs = mlflow.search_runs(experiment_names=[self.experiment_name],
                                              filter_string=f'attributes.run_id = "{best_child_run_id}"', output_format='list')
                    if len(runs) != 1:
                        raise ValueError(f'Found {len(runs)} runs with run_id {best_child_run_id}.')
                    run = runs[0]
                    artifact_uri = run.info.artifact_uri
                    model_dir = Path(artifact_uri) / 'model'
                    if model_dir.exists():
                        mlflow.log_artifacts(model_dir, artifact_path='best_model', run_id=mlflow_run_id)
                else:
                    save_dir = self.save_root_dir / work_dir.name
                    best_save_dir = self.save_root_dir / best_work_dir.name
                    if best_save_dir.exists():
                        copytree(best_save_dir, save_dir, dirs_exist_ok=True)
            trials = study.trials
            for trial in trials:
                trial_result = trial.user_attrs.get('result', None)
                trial_combination = trial_result.get('trial_combination', dict())
                trial_child_run_id = trial_combination.get('mlflow_run_id', None)
                trial_work_dir = trial_result.get('work_dir', None)
                if trial_child_run_id is not None:
                    runs = mlflow.search_runs(experiment_names=[self.experiment_name],
                                              filter_string=f'attributes.run_id = "{trial_child_run_id}"',
                                              output_format='list')
                    if len(runs) != 1:
                        raise ValueError(f'Found {len(runs)} runs with run_id {trial_child_run_id}.')
                    run = runs[0]
                    artifact_uri = run.info.artifact_uri
                    run_path = Path(artifact_uri).parent
                    if run_path.exists():
                        rmtree(run_path)
                else:
                    trial_save_dir = self.save_root_dir / trial_work_dir.name
                    if trial_save_dir.exists():
                        rmtree(trial_save_dir)

        if self.clean_work_dir:
            if work_dir.exists():
                rmtree(work_dir)

        return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = TabularHPOExperiment(parser=parser)
    experiment.run()
