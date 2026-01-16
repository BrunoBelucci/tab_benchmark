from typing import Optional
import mlflow
from optuna import Trial
from ml_experiments.base_experiment import BaseExperiment
from ml_experiments.hpo_experiment import HPOExperiment
from ml_experiments.utils import flatten_any, unflatten_any, update_recursively
from tab_benchmark.benchmark.tabular_experiment import TabularExperiment
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.models.dnn_models import max_epochs_dnn
from tab_benchmark.models.xgboost import n_estimators_gbdt
import numpy as np
from copy import deepcopy


class TabularHPOExperiment(HPOExperiment, TabularExperiment):

    def __init__(
        self,
        *args,
        search_space: Optional[dict] = None,
        default_values: Optional[list] = None,
        **kwargs,
    ):
        """
        Initialize the TabularHPOExperiment.

        Args:
            *args: Variable length argument list passed to parent class.
            search_space (Optional[dict], optional): Custom hyperparameter search space definition
                using Optuna distributions (e.g., FloatDistribution, IntDistribution).
                Required when using custom models not in the models dictionary.
                Defaults to None (uses model-specific search space).
            default_values (Optional[list], optional): List of default parameter configurations
                to try before starting the optimization search. Useful for providing good
                starting points or baseline configurations. Required when using custom models.
                Defaults to None (uses model-specific defaults).
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        self.search_space = search_space if search_space is not None else {}
        self.default_values = default_values if default_values is not None else []

    def get_search_space(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ) -> dict:
        model = combination["model"]
        if isinstance(model, str):
            models_dict = deepcopy(self.models_dict)
            search_space = models_dict[model]["search_space"]
        else:
            search_space = self.search_space
            if search_space is None:
                raise ValueError("Search space must be defined if model is not defined as a string.")
        return search_space

    def get_default_values(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ) -> list:
        model = combination["model"]
        if isinstance(model, str):
            models_dict = deepcopy(self.models_dict)
            default_values = models_dict[model]["default_values"]
        else:
            default_values = self.default_values
            if default_values is None:
                raise ValueError("Default values must be defined if model is not defined as a string.")
        return default_values

    def get_hyperband_max_resources(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ) -> Optional[int]:
        model = combination["model"]
        if isinstance(model, str):
            models_dict = deepcopy(self.models_dict)
            model_class = models_dict[model]["model_class"]
        elif isinstance(model, type):
            model_class = model
        else:
            model_class = type(model)
        if issubclass(model_class, DNNModel):
            return max_epochs_dnn
        else:
            return n_estimators_gbdt

    def _load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        result = super()._load_data(combination=combination, unique_params=unique_params,
                                    extra_params=extra_params, **kwargs)
        # we do not need to keep the data, we will only keep the dataset_name and task_name for logging
        keep_results = {'dataset_name': result['dataset_name'], 'task_name': result['task_name']}
        return keep_results

    def _before_fit_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        hpo_seed = unique_params["hpo_seed"]
        ret = super()._before_fit_model(combination, unique_params, extra_params, mlflow_run_id, **kwargs)
        simple_experiment = TabularExperiment(
            # experiment parameters
            experiment_name=self.experiment_name,
            log_dir=self.log_dir,
            log_file_name=self.log_file_name,
            work_root_dir=self.work_root_dir,
            save_root_dir=self.save_root_dir,
            clean_work_dir=self.clean_work_dir,
            raise_on_error=self.raise_on_error,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            check_if_exists=self.check_if_exists,
            profile_memory=self.profile_memory,
            profile_time=self.profile_time,
            verbose=0,
            # other parameters will be directly passed to _run_mlflow_and_train_model or _train_model
        )
        random_generator = np.random.default_rng(hpo_seed)
        ret["simple_experiment"] = simple_experiment
        ret["random_generator"] = random_generator
        return ret

    def training_fn(
        self,
        trial_dict: dict,
        combination: dict,
        unique_params: dict,
        extra_params: dict,
        mlflow_run_id: Optional[str] = None,
        **kwargs,
    ) -> dict:
        trial: Trial = trial_dict["trial"]
        child_run_id = trial_dict["child_run_id"]
        seed_model = trial_dict["random_seed"]
        simple_experiment: BaseExperiment = kwargs["before_fit_model_return"]["simple_experiment"]

        # update the model parameters in unique_params
        trial_params = deepcopy(trial.params)
        unique_params = deepcopy(unique_params)
        model_params = unique_params["model_params"]
        model_params = flatten_any(model_params)
        model_params = update_recursively(model_params, trial_params)
        model_params = unflatten_any(model_params)
        unique_params["model_params"] = model_params

        # update the fit_params in unique_params
        fit_params = unique_params["fit_params"]
        fit_params["optuna_trial"] = trial

        # update the seed_model in combination
        combination = deepcopy(combination)
        combination["seed_model"] = seed_model

        if mlflow_run_id is not None:
            results = simple_experiment._run_mlflow_and_train_model(
                combination=combination,
                unique_params=unique_params,
                extra_params=extra_params,
                mlflow_run_id=child_run_id,
                return_results=True,
            )
        else:
            results = simple_experiment._train_model(
                combination=combination,
                unique_params=unique_params,
                extra_params=extra_params,
                mlflow_run_id=child_run_id,
                return_results=True,
            )

        if not isinstance(results, dict):
            results = dict()

        if (
            "evaluate_model_return" not in results
        ):  # maybe we already have the run this run and we are getting the stored run result
            keep_results = {
                metric[len("metrics.") :]: value for metric, value in results.items() if metric.startswith("metrics.")
            }
        else:
            keep_results = results.get("evaluate_model_return", {})
        if "fit_model_return" not in results:
            fit_model_return_elapsed_time = results.get("metrics.fit_model_return_elapsed_time", 0)
        else:
            fit_model_return_elapsed_time = results.get("fit_model_return", {}).get("elapsed_time", 0)
        keep_results["elapsed_time"] = fit_model_return_elapsed_time
        keep_results["seed_model"] = seed_model

        if mlflow_run_id is not None:
            log_metrics = keep_results.copy()
            log_metrics.pop("elapsed_time", None)
            log_metrics.pop("max_memory_used", None)
            mlflow.log_metrics(log_metrics, run_id=mlflow_run_id, step=trial.number)
        return keep_results


if __name__ == '__main__':
    experiment = TabularHPOExperiment()
    experiment.run_from_cli()
