from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence
from warnings import warn
import optuna
import pandas as pd
from tab_benchmark.preprocess import create_data_preprocess_pipeline, create_target_preprocess_pipeline
from tab_benchmark.utils import sequence_to_list


def merge_signatures(*functions, repeated_parameters='keep_last'):
    parameters = {}
    for function in functions:
        for name, parameter in inspect.signature(function).parameters.items():
            if name in parameters:
                if repeated_parameters == 'keep_last':
                    parameters[name] = parameter
                else:  # repeated_parameters == 'keep_first':
                    pass
            else:
                parameters[name] = parameter
    # reorder parameters if needed
    positional_only = [parameter for parameter in parameters.values() if parameter.kind == parameter.POSITIONAL_ONLY]
    positional_or_keyword = [parameter for parameter in parameters.values()
                             if parameter.kind == parameter.POSITIONAL_OR_KEYWORD]
    var_positional = [parameter for parameter in parameters.values() if parameter.kind == parameter.VAR_POSITIONAL]
    keyword_only = [parameter for parameter in parameters.values() if parameter.kind == parameter.KEYWORD_ONLY]
    var_keyword = [parameter for parameter in parameters.values() if parameter.kind == parameter.VAR_KEYWORD]
    if repeated_parameters == 'keep_first':
        var_positional = var_positional[0] if var_positional else []
        var_keyword = [var_keyword[0]] if var_keyword else []
    else:
        var_positional = var_positional[-1] if var_positional else []
        var_keyword = [var_keyword[-1]] if var_keyword else []
    orderly_parameters = positional_only + positional_or_keyword + var_positional + keyword_only + var_keyword
    return inspect.Signature(orderly_parameters)


def apply_signature(signature):
    def decorator(function):
        def wrapper(*args, **kwargs):
            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            return function(**bound_arguments.arguments)

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
    early_stopping_validation_size:
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
    eval_metric:
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
            early_stopping_validation_size=0.1,
            early_stopping_patience: int = 0,
            mlflow_run_id=None,
            log_interval: int = 50,
            save_checkpoint: bool = False,
            checkpoint_interval: int = 100,
            output_dir: Optional[str | Path] = None,
            eval_metric: Optional[str] = None,
            max_time: Optional[int | dict] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.auto_early_stopping = auto_early_stopping
        self.early_stopping_validation_size = early_stopping_validation_size
        self.early_stopping_patience = early_stopping_patience
        self.mlflow_run_id = mlflow_run_id
        self.log_interval = log_interval
        self.save_checkpoint = save_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir
        self.eval_metric = eval_metric
        self.max_time = max_time


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
        super().__init__(**kwargs)
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


class GBDTMixin(EarlyStoppingMixin, PreprocessingMixin, TaskDependentParametersMixin):
    @apply_signature(merge_signatures(PreprocessingMixin.__init__, EarlyStoppingMixin.__init__))
    def __init__(self, **kwargs):
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
