import argparse
from typing import Optional
import optuna
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = TabularHPOExperiment(parser=parser)
    experiment.run()
