from __future__ import annotations
import inspect
import json
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence
from warnings import warn
import optuna
import pandas as pd
import joblib
from tab_benchmark.preprocess import create_data_preprocess_pipeline, create_target_preprocess_pipeline
from tab_benchmark.utils import sequence_to_list, get_default_tag, get_formated_file_path, get_most_recent_file_path, \
    train_test_split_forced


def merge_signatures(*signatures, repeated_parameters='keep_last'):
    parameters = {}
    for signature in signatures:
        for name, parameter in signature.parameters.items():
            if name in parameters:
                if repeated_parameters == 'keep_last':
                    parameters[name] = parameter
                else:  # repeated_parameters == 'keep_first':
                    pass
            else:
                parameters[name] = parameter
    parameters_cp = deepcopy(parameters)
    # reorder parameters if needed
    positional_only = [parameters.pop(name) for name, parameter in parameters_cp.items()
                       if parameter.kind == parameter.POSITIONAL_ONLY]
    parameters_cp = deepcopy(parameters)  # update parameters_cp
    positional_or_keyword_without_default = [parameters.pop(name) for name, parameter in parameters_cp.items()
                                             if parameter.kind == parameter.POSITIONAL_OR_KEYWORD and
                                             parameter.default == parameter.empty]
    parameters_cp = deepcopy(parameters)  # update parameters_cp
    positional_or_keyword_with_default = [parameters.pop(name) for name, parameter in parameters_cp.items()
                                          if parameter.kind == parameter.POSITIONAL_OR_KEYWORD and
                                          parameter.default != parameter.empty]
    parameters_cp = deepcopy(parameters)  # update parameters_cp
    var_positional = [parameters.pop(name) for name, parameter in parameters_cp.items()
                      if parameter.kind == parameter.VAR_POSITIONAL]
    parameters_cp = deepcopy(parameters)  # update parameters_cp
    keyword_only = [parameters.pop(name) for name, parameter in parameters_cp.items()
                    if parameter.kind == parameter.KEYWORD_ONLY]
    parameters_cp = deepcopy(parameters)  # update parameters_cp
    var_keyword = [parameters.pop(name) for name, parameter in parameters_cp.items()
                   if parameter.kind == parameter.VAR_KEYWORD]
    if repeated_parameters == 'keep_first':
        var_positional = var_positional[0] if var_positional else []
        var_keyword = [var_keyword[0]] if var_keyword else []
    else:
        var_positional = var_positional[-1] if var_positional else []
        var_keyword = [var_keyword[-1]] if var_keyword else []
        var_keyword = [var_keyword[-1]] if var_keyword else []
    orderly_parameters = (positional_only + var_positional + positional_or_keyword_without_default
                          + positional_or_keyword_with_default + keyword_only + var_keyword)
    return inspect.Signature(orderly_parameters)


def merge_and_apply_signature(signature_to_merge):
    def decorator(function):
        signature = merge_signatures(signature_to_merge, inspect.signature(function), repeated_parameters='keep_last')

        def wrapper(*args, **kwargs):
            bound_arguments = signature.bind_partial(*args, **kwargs)
            param_names = list(signature.parameters.keys())
            arguments = bound_arguments.kwargs
            for i, arg in enumerate(bound_arguments.args):
                arguments[param_names[i]] = arg
            return function(**arguments)

        wrapper.__signature__ = signature
        return wrapper

    return decorator


class TaskDependentParametersMixin:
    @property
    @abstractmethod
    def map_task_to_default_values(self):
        pass

    def fit(self, X, y, task=None, **kwargs):
        if task is not None:
            if task in self.map_task_to_default_values:
                for key, value in self.map_task_to_default_values[task].items():
                    param = self.get_params().get(key, None)
                    if param == 'default' or param is None:
                        self.set_params(**{key: value})
        else:
            raise (
                ValueError('This model has map_task_to_default_values, which means it has some values that are '
                           'task dependent. You must provide the task when calling fit.'))
        return super().fit(X, y, task=task, **kwargs)


class EarlyStoppingMixin:
    """
    Parameters
    ----------
    auto_early_stopping:
        Whether to use early stopping automatically, i.e., split the training data into training and validation sets
        and stop training when the validation score does not improve anymore.
    auto_early_stopping_validation_size:
        Size of the validation set when using auto early stopping.
    early_stopping_patience:
        Patience for early stopping.
    mlflow_run_id:
        MLFlow run ID, if using MLFlow. If None, MLFlow will not be used.
    log_interval:
        Interval for logging.
    save_checkpoint:
        Whether to save a checkpoint.
    checkpoint_interval:
        Interval for saving a checkpoint.
    output_dir:
        Output directory for saving the model checkpoint.
    es_eval_metric:
        Evaluation metric. If None, the default metric of the model will be used. This metric can be any defined
        in get_metric_fn, and if there is an equivalent metric in the model, it will be used.
    max_time:
        Maximum time for training. If it is an integer, it is the maximum time in seconds. If it is a dictionary, it
        should contain key-value pairs compatible with datetime.timedelta.
    """
    has_early_stopping = True

    def __init__(
            self,
            *,
            auto_early_stopping: bool = True,
            auto_early_stopping_validation_size=0.1,
            early_stopping_patience: int = 0,
            mlflow_run_id=None,
            log_interval: int = 50,
            save_checkpoint: bool = False,
            checkpoint_interval: int = 100,
            output_dir: Optional[str | Path] = None,
            es_eval_metric: Optional[str] = None,
            max_time: Optional[int | dict] = None,
            **kwargs
    ):
        self.auto_early_stopping = auto_early_stopping
        self.auto_early_stopping_validation_size = auto_early_stopping_validation_size
        self.early_stopping_patience = early_stopping_patience
        self.mlflow_run_id = mlflow_run_id
        self.log_interval = log_interval
        self.save_checkpoint = save_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir
        self.es_eval_metric = es_eval_metric
        self.max_time = max_time
        super().__init__(**kwargs)

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame | pd.Series,
            task: Optional[str] = None,
            cat_features: Optional[list[str]] = None,
            cat_dims: Optional[list[int]] = None,
            n_classes: Optional[int] = None,
            eval_set: Optional[list[tuple]] = None,
            eval_name: Optional[list[str]] = None,
            init_model: Optional[str | Path] = None,
            optuna_trial: Optional[optuna.Trial] = None,
            **kwargs
    ):
        if self.auto_early_stopping:
            if task == 'classification' or task == 'binary_classification':
                stratify = y
            else:
                stratify = None
            X, X_valid, y, y_valid = train_test_split_forced(
                X, y,
                test_size_pct=self.auto_early_stopping_validation_size,
                # random_state=random_seed,  this will be ensured by set_seeds
                stratify=stratify
            )
            eval_set = eval_set if eval_set else []
            eval_set = sequence_to_list(eval_set)
            eval_set.append((X_valid, y_valid))
            eval_name = eval_name if eval_name else []
            eval_name = sequence_to_list(eval_name)
            eval_name.append('validation_es')
        return super().fit(X, y, task=task, cat_features=cat_features, cat_dims=cat_dims, n_classes=n_classes,
                           eval_set=eval_set, eval_name=eval_name, init_model=init_model, optuna_trial=optuna_trial,
                           **kwargs)


class PreprocessingMixin:
    """
    Parameters
    ----------
    categorical_imputer:
        Imputer strategy for categorical features.
    continuous_imputer:
        Imputer strategy for continuous features.
    categorical_encoder:
        Encoder strategy for categorical features.
    handle_unknown_categories:
        Whether to handle unknown categories.
    variance_threshold:
        Threshold for variance.
    data_scaler:
        Scaler strategy for data.
    categorical_type:
        Data type for categorical features.
    continuous_type:
        Data type for continuous features.
    target_imputer:
        Imputer strategy for target.
    categorical_target_encoder:
        Encoder strategy for target.
    categorical_target_min_frequency:
        Minimum frequency for target.
    continuous_target_scaler:
        Scaler strategy for target.
    categorical_target_type:
        Data type for target.
    continuous_target_type:
        Data type for target.
    """

    def __init__(
            self,
            *,
            categorical_imputer: Optional[str | int | float] = 'most_frequent',
            continuous_imputer: Optional[str | int | float] = 'median',
            categorical_encoder: Optional[str] = 'one_hot',
            handle_unknown_categories: bool = True,
            variance_threshold: Optional[float] = 0.0,
            data_scaler: Optional[str] = 'standard',
            categorical_type: Optional[type | str] = 'float32',
            continuous_type: Optional[type | str] = 'float32',
            target_imputer: Optional[str | int | float] = None,
            categorical_target_encoder: Optional[str] = 'ordinal',  # only used in classification
            categorical_target_min_frequency: Optional[int | float] = 10,  # only used in classification
            continuous_target_scaler: Optional[str] = 'standard',  # only used in regression
            categorical_target_type: Optional[type | str] = 'float32',
            continuous_target_type: Optional[type | str] = 'float32',
            **kwargs
    ):
        self.categorical_imputer = categorical_imputer
        self.continuous_imputer = continuous_imputer
        self.categorical_encoder = categorical_encoder
        self.handle_unknown_categories = handle_unknown_categories
        self.variance_threshold = variance_threshold
        self.data_scaler = data_scaler
        self.categorical_type = categorical_type
        self.continuous_type = continuous_type
        self.target_imputer = target_imputer
        self.categorical_target_encoder = categorical_target_encoder
        self.categorical_target_min_frequency = categorical_target_min_frequency
        self.continuous_target_scaler = continuous_target_scaler
        self.categorical_target_type = categorical_target_type
        self.continuous_target_type = continuous_target_type
        self.target_preprocess_pipeline_ = None
        self.data_preprocess_pipeline_ = None
        super().__init__(**kwargs)

    def create_preprocess_pipeline(
            self,
            task: str,
            categorical_features_names: Optional[Sequence[int | str]] = None,
            continuous_features_names: Optional[Sequence[int | str]] = None,
            orderly_features_names: Optional[Sequence[int | str]] = None,
    ):
        self.data_preprocess_pipeline_ = create_data_preprocess_pipeline(
            categorical_features_names=categorical_features_names,
            continuous_features_names=continuous_features_names,
            orderly_features_names=orderly_features_names,
            categorical_imputer=self.categorical_imputer,
            continuous_imputer=self.continuous_imputer,
            categorical_encoder=self.categorical_encoder,
            handle_unknown_categories=self.handle_unknown_categories,
            variance_threshold=self.variance_threshold,
            scaler=self.data_scaler,
            categorical_type=self.categorical_type,
            continuous_type=self.continuous_type,
        )
        self.target_preprocess_pipeline_ = create_target_preprocess_pipeline(
            task=task,
            imputer=self.target_imputer,
            categorical_encoder=self.categorical_target_encoder,
            categorical_min_frequency=self.categorical_target_min_frequency,
            continuous_scaler=self.continuous_target_scaler,
            categorical_type=self.categorical_target_type,
            continuous_type=self.continuous_target_type,
        )
        return self.data_preprocess_pipeline_, self.target_preprocess_pipeline_

    def save_model(self, save_dir: [Path | str] = None, tag: Optional[str] = None) -> Path:
        prefix = self.__class__.__name__ + '_data_preprocess_pipeline'
        ext = 'joblib'
        file_path_data_preprocess_pipeline = get_formated_file_path(save_dir, prefix, ext, tag)
        prefix = self.__class__.__name__ + '_target_preprocess_pipeline'
        file_path_target_preprocess_pipeline = get_formated_file_path(save_dir, prefix, ext, tag)
        self._save_preprocess_pipeline(file_path_data_preprocess_pipeline, file_path_target_preprocess_pipeline)
        return super().save_model(save_dir, tag)

    def _save_preprocess_pipeline(self, data_preprocess_pipeline_path: Path | str,
                                  target_preprocess_pipeline_path: Path | str) -> None:
        """Save the preprocess pipeline to a file in the output_dir."""
        if self.data_preprocess_pipeline_ is not None:
            with open(data_preprocess_pipeline_path, 'wb') as file:
                joblib.dump(self.data_preprocess_pipeline_, file)
        if self.target_preprocess_pipeline_ is not None:
            with open(target_preprocess_pipeline_path, 'wb') as file:
                joblib.dump(self.target_preprocess_pipeline_, file)

    def load_model(self, save_dir: Path | str = None, tag: Optional[str] = None):
        prefix = self.__class__.__name__ + '_data_preprocess_pipeline'
        ext = 'joblib'
        most_recent_data_preprocess_pipeline_path = get_most_recent_file_path(save_dir, prefix, ext, tag)
        prefix = self.__class__.__name__ + '_target_preprocess_pipeline'
        most_recent_target_preprocess_pipeline_path = get_most_recent_file_path(save_dir, prefix, ext, tag)
        self._load_preprocess_pipeline(most_recent_data_preprocess_pipeline_path,
                                       most_recent_target_preprocess_pipeline_path)
        return super().load_model(save_dir, tag)

    def _load_preprocess_pipeline(self, data_preprocess_pipeline_path: Path | str,
                                  target_preprocess_pipeline_path: Path | str) -> None:
        """Load the preprocess pipeline from a file in the output_dir."""
        if data_preprocess_pipeline_path.exists():
            with open(data_preprocess_pipeline_path, 'rb') as file:
                self.data_preprocess_pipeline = joblib.load(file)
        if target_preprocess_pipeline_path.exists():
            with open(target_preprocess_pipeline_path, 'rb') as file:
                self.target_preprocess_pipeline = joblib.load(file)


def check_y_eval_set_eval_name_cat_features(X, y, eval_set, eval_name, cat_features, cat_dims=None):
    if isinstance(y, pd.Series):
        y = y.to_frame()

    eval_set = sequence_to_list(eval_set) if eval_set is not None else []
    eval_name = sequence_to_list(eval_name) if eval_name is not None else []
    if eval_set and not eval_name:
        eval_name = [f'validation_{i}' for i in range(len(eval_set))]
    if len(eval_set) != len(eval_name):
        raise AttributeError('eval_set and eval_name should have the same length')

    if cat_features:
        # if we pass cat_features as column names, we can ensure that they are in the dataframe
        # (and not dropped during preprocessing for example)
        if isinstance(cat_features[0], str):
            cat_features_without_dropped = deepcopy(cat_features)
            if cat_dims is not None:
                cat_features_dims = dict(zip(cat_features, cat_dims))
            for i, feature in enumerate(cat_features):
                if feature not in X.columns:
                    warn(f'Categorical feature {feature} is not in the dataframe. It will be ignored.')
                    cat_features_without_dropped.remove(feature)
            cat_features = cat_features_without_dropped
            if cat_dims is not None:
                cat_dims = [cat_features_dims[feature] for feature in cat_features]
    return y, eval_set, eval_name, cat_features, cat_dims


n_estimators_gbdt = 10000
early_stopping_patience_gbdt = 100


class GBDTMixin(EarlyStoppingMixin, PreprocessingMixin, TaskDependentParametersMixin):
    @merge_and_apply_signature(merge_signatures(inspect.signature(PreprocessingMixin.__init__),
                                                inspect.signature(EarlyStoppingMixin.__init__)))
    def __init__(self, **kwargs):
        self.pruned_trial = False
        self.reached_timeout = False
        super().__init__(**kwargs)

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame | pd.Series,
            task: Optional[str] = None,
            cat_features: Optional[list[str]] = None,
            cat_dims: Optional[list[int]] = None,
            n_classes: Optional[int] = None,
            eval_set: Optional[list[tuple]] = None,
            eval_name: Optional[list[str]] = None,
            init_model: Optional[str | Path] = None,
            optuna_trial: Optional[optuna.Trial] = None,
            **kwargs
    ):
        y, eval_set, eval_name, cat_features, cat_dims = check_y_eval_set_eval_name_cat_features(
            X, y, eval_set, eval_name, cat_features, cat_dims)
        return super().fit(X, y, task=task, cat_features=cat_features, cat_dims=cat_dims, n_classes=n_classes,
                           eval_set=eval_set, eval_name=eval_name, init_model=init_model, optuna_trial=optuna_trial,
                           **kwargs)


class TabBenchmarkModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def fit(self, X, y, **kwargs):
        X, y, kwargs = self.before_fit(X, y, **kwargs)
        return super().fit(X, y, **kwargs)

    def before_fit(self, X, y, **kwargs):
        return X, y, kwargs

    @staticmethod
    @abstractmethod
    def create_search_space():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_recommended_params():
        raise NotImplementedError

    @abstractmethod
    def save_model(self, save_dir: [Path | str] = None, tag: Optional[str] = None) -> Path:
        prefix = self.__class__.__name__ + '_serializable'
        ext = 'json'
        file_path = get_formated_file_path(save_dir, prefix, ext, tag)
        self._save_serializable(file_path)
        return file_path

    @abstractmethod
    def load_model(self, save_dir: Path | str = None, tag: Optional[str] = None):
        prefix = self.__class__.__name__ + '_serializable'
        ext = 'json'
        most_recent_serializable_path = get_most_recent_file_path(save_dir, prefix, ext, tag)
        self._load_serializable(most_recent_serializable_path)
        return self

    def _save_serializable(self, serializable_path: Path | str) -> None:
        """Save every serializable attribute of the model to a file in the serializable_path."""
        serializable_att = vars(self).copy()
        for key, value in serializable_att.copy().items():
            with open(serializable_path, 'w') as file:
                try:
                    json.dump(value, file)
                except TypeError:
                    serializable_att.pop(key, None)
        with open(serializable_path, 'w') as file:
            json.dump(serializable_att, file)

    def _load_serializable(self, serializable_path: Path | str) -> None:
        """Load every serializable attribute of the model from a file in the serializable_path."""
        with open(serializable_path, 'r') as file:
            serializable_att = json.load(file)
        for key, value in serializable_att.items():
            setattr(self, key, value)
