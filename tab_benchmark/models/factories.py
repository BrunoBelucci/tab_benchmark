from __future__ import annotations
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence
import pandas as pd

from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.preprocess import create_data_preprocess_pipeline, create_target_preprocess_pipeline
from tab_benchmark.utils import train_test_split_forced, sequence_to_list
from inspect import cleandoc, signature, Signature
from tab_benchmark.utils import check_same_keys


class TabBenchmarkModel(ABC):
    """
    Class of every model in the tab_benchmark package, it adds some preprocessing functionalities with the method
    create_preprocess_pipeline, and it standardizes the fit method. It adds the following parameters to the __init__
    method of the original class:

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
            output_dir: Optional[str | os.PathLike] = Path.cwd(),
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
        self.output_dir = output_dir
        self.target_preprocess_pipeline_ = None
        self.data_preprocess_pipeline_ = None

    @abstractmethod
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
            report_to_ray: bool = False,
            init_model: Optional[str | Path] = None,
            *args,
            **kwargs
    ):
        """
        Fit the model with a common interface. It expects the following parameters besides the ones in the original
        fit method of the model:

        Parameters
        ----------
        X:
            Features.
        y:
            Target.
        task:
            Task type. Can be 'classification', 'binary_classification', 'regression', 'multi_regression'.
        cat_features:
            Categorical features.
        cat_dims:
            Categorical dimensions.
        n_classes:
            Number of classes.
        eval_set:
            Evaluation set.
        eval_name:
            Evaluation name.
        report_to_ray:
            Whether to report to Ray for tuning.
        init_model:
            Initial model to start from.
        """
        pass

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


def early_stopping_init(self, *, auto_early_stopping: bool = True, early_stopping_validation_size=0.1,
                        early_stopping_patience: int = 0, log_to_mlflow_if_running: bool = True,
                        eval_metric: Optional[str] = None):
    """
    auto_early_stopping:
            Whether to use early stopping automatically, i.e., split the training data into training and validation sets
            and stop training when the validation score does not improve anymore.
    early_stopping_validation_size:
        Size of the validation set when using auto early stopping.
    early_stopping_patience:
        Patience for early stopping.
    log_to_mlflow_if_running:
        Whether to log intermediate results to MLflow if it is running.
    eval_metric:
        Evaluation metric. If None, the default metric of the model will be used. This metric can be any defined
        in get_metric_fn, and if there is an equivalent metric in the model, it will be used.
    """
    self.auto_early_stopping = auto_early_stopping
    self.early_stopping_validation_size = early_stopping_validation_size
    self.early_stopping_patience = early_stopping_patience
    self.log_to_mlflow_if_running = log_to_mlflow_if_running
    self.eval_metric = eval_metric


def fn_to_add_auto_early_stopping(auto_early_stopping, early_stopping_validation_size,
                                  X, y, task, eval_set, eval_name):
    if auto_early_stopping:
        if task == 'classification' or task == 'binary_classification':
            stratify = y
        else:
            stratify = None
        X, X_valid, y, y_valid = train_test_split_forced(
            X, y,
            test_size_pct=early_stopping_validation_size,
            # random_state=random_seed,  this will be ensured by set_seeds
            stratify=stratify
        )
        eval_set = eval_set if eval_set else []
        eval_set = sequence_to_list(eval_set)
        eval_set.append((X_valid, y_valid))
        eval_name = eval_name if eval_name else []
        eval_name = sequence_to_list(eval_name)
        eval_name.append('validation_es')
    return X, y, eval_set, eval_name


def dnn_architecture_init(self, dnn_architecture_cls, **kwargs):
    if hasattr(dnn_architecture_cls, 'params_defined_from_dataset'):
        architecture_params_not_from_dataset = kwargs.copy()
        architecture_params = {}
    else:
        architecture_params_not_from_dataset = {}
        architecture_params = kwargs.copy()
    for key, value in kwargs.items():
        setattr(self, key, value)
    return architecture_params, architecture_params_not_from_dataset


def dnn_model_factory(dnn_architecture_cls, dnn_model_cls=DNNModel, default_values=None,
                      map_task_to_default_values=None,
                      before_fit_method=None, extra_dct=None):
    default_values = default_values.copy() if default_values else {}

    dnn_parameters = {name: param for name, param in signature(dnn_architecture_cls.__init__).parameters.items()
                      if name != 'self'}
    if hasattr(dnn_architecture_cls, 'params_defined_from_dataset'):
        dnn_parameters_from_dataset = dnn_architecture_cls.params_defined_from_dataset
    else:
        dnn_parameters_from_dataset = []

    for param in dnn_parameters_from_dataset:
        dnn_parameters.pop(param, None)

    dnn_architecture_class = default_values.get('dnn_architecture_class', dnn_architecture_cls)
    default_values['dnn_architecture_class'] = dnn_architecture_class

    TabBenchmarkSklearn = sklearn_factory(dnn_model_cls, has_early_stopping=True, default_values=default_values,
                                          map_task_to_default_values=map_task_to_default_values,
                                          before_fit_method=before_fit_method, extra_dct=extra_dct)

    class TabBenchmarkDNN(TabBenchmarkSklearn):
        def __init__(self, *args, **kwargs):
            bind_args = signature(TabBenchmarkDNN.__init__).bind(self, *args, **kwargs)
            bind_args.apply_defaults()
            arguments = bind_args.arguments
            arguments.pop('self', None)

            dnn_arguments = {name: arguments.pop(name) for name in list(arguments.keys())
                             if name in dnn_parameters}
            architecture_params, architecture_params_not_from_dataset = dnn_architecture_init(self,
                                                                                              dnn_architecture_cls,
                                                                                              **dnn_arguments)
            arguments['architecture_params'] = architecture_params
            arguments['architecture_params_not_from_dataset'] = architecture_params_not_from_dataset
            TabBenchmarkSklearn.__init__(self, **arguments)

    # TabBenchmarkSklearn parameters without 'architecture_params', 'architecture_params_not_from_dataset'
    tab_benchmark_dnn_parameters = {name: param for name, param in
                                    signature(TabBenchmarkSklearn.__init__).parameters.items()
                                    if name not in ('architecture_params', 'architecture_params_not_from_dataset')}
    self_parameter = tab_benchmark_dnn_parameters.pop('self')
    tab_benchmark_dnn_parameters = {**{'self': self_parameter}, **dnn_parameters, **tab_benchmark_dnn_parameters}
    TabBenchmarkDNN.__init__.__signature__ = Signature(parameters=list(tab_benchmark_dnn_parameters.values()))
    TabBenchmarkDNN.__doc__ = (
            TabBenchmarkSklearn.__doc__ + '\n\nArchitecture Documentation:\n\n'
            + f'Parameters that are automatically defined from the dataset are:{dnn_parameters_from_dataset}\n\n'
            + cleandoc(dnn_architecture_cls.__doc__))
    name = f'TabBenchmark{dnn_architecture_cls.__name__}'
    return type(name, (TabBenchmarkDNN,), {'__doc__': TabBenchmarkDNN.__doc__})


def sklearn_factory(sklearn_cls, has_early_stopping=False, default_values=None,
                    map_task_to_default_values=None, before_fit_method=None, extra_dct=None):
    default_values = default_values.copy() if default_values else {}
    map_task_to_default_values = map_task_to_default_values.copy() if map_task_to_default_values else {}
    extra_dct = extra_dct.copy() if extra_dct else {}

    if map_task_to_default_values:
        dicts = list(map_task_to_default_values.values())
        if not check_same_keys(*dicts):
            raise ValueError('All dictionaries in map_task_to_default_values must have the same keys.')
        keys = dicts[0].keys()
        for key in keys:
            default_values[key] = 'default'

    # init parameters and doc
    sklearn_parameters = {name: param for name, param in signature(sklearn_cls.__init__).parameters.items()}
    sklearn_var_keyword_parameter = {name: param for name, param in sklearn_parameters.items()
                                     if param.kind == param.VAR_KEYWORD}
    if sklearn_var_keyword_parameter:
        for key in sklearn_var_keyword_parameter:
            del sklearn_parameters[key]
    tab_benchmark_model_parameters = {name: param for name, param in
                                      signature(TabBenchmarkModel.__init__).parameters.items() if name != 'self'}
    early_stopping_parameters = {name: param for name, param in signature(early_stopping_init).parameters.items()
                                 if name != 'self'}
    extra_parameters = tab_benchmark_model_parameters

    sklearn_doc = cleandoc(sklearn_cls.__doc__)
    tab_benchmark_doc = cleandoc(TabBenchmarkModel.__doc__)
    early_stopping_doc = cleandoc(early_stopping_init.__doc__)
    init_doc = tab_benchmark_doc

    if has_early_stopping:
        extra_parameters.update(early_stopping_parameters)
        init_doc += '\n' + early_stopping_doc
    init_doc += '\n\nOriginal documentation:\n\n' + sklearn_doc

    # remove parameters from sklearn_cls that are in extra_parameters
    for name, param in extra_parameters.items():
        if name in sklearn_parameters:
            sklearn_parameters.pop(name)

    init_parameters = sklearn_parameters
    init_parameters.update(extra_parameters)
    if sklearn_var_keyword_parameter:
        init_parameters.update(sklearn_var_keyword_parameter)
    if default_values:
        for name, value in default_values.items():
            init_parameters[name] = init_parameters[name].replace(default=value)

    # fit parameters and doc
    tab_benchmark_fit_parameters = {name: param for name, param in signature(TabBenchmarkModel.fit).parameters.items()
                                    if name not in ('args', 'kwargs')}
    tab_benchmark_fit_doc = cleandoc(TabBenchmarkModel.fit.__doc__)
    sklearn_fit_parameters = {name: param for name, param in signature(sklearn_cls.fit).parameters.items()
                              if name != 'self'}
    sklearn_fit_doc = cleandoc(sklearn_cls.fit.__doc__)

    # remove parameters from sklearn_cls that are in tab_benchmark_fit_parameters
    for name, param in tab_benchmark_fit_parameters.items():
        if name in sklearn_fit_parameters:
            sklearn_fit_parameters.pop(name)

    fit_parameters = tab_benchmark_fit_parameters
    fit_parameters.update(sklearn_fit_parameters)
    fit_doc = tab_benchmark_fit_doc + '\n\nOriginal documentation:\n\n' + sklearn_fit_doc

    sklearn_fit_signature = signature(sklearn_cls.fit)

    # CLASS DEFINITION
    class TabBenchmarkSklearn(TabBenchmarkModel, sklearn_cls):
        def __init__(self, *args, **kwargs):
            bind_args = signature(TabBenchmarkSklearn.__init__).bind(self, *args, **kwargs)
            bind_args.apply_defaults()
            arguments = bind_args.arguments
            arguments.pop('self', None)
            sklearn_cls_arguments = {name: arguments.pop(name) for name in list(arguments.keys())
                                     if name not in extra_parameters}
            sklearn_cls.__init__(self, **sklearn_cls_arguments)
            if has_early_stopping:
                early_stopping_arguments = {name: arguments.pop(name) for name in list(arguments.keys())
                                            if name in early_stopping_parameters}
                early_stopping_init(self, **early_stopping_arguments)
            TabBenchmarkModel.__init__(self, **arguments)
            if map_task_to_default_values:
                self.map_task_to_default_values = map_task_to_default_values

        def fit(self, X, y, task=None, cat_features=None, cat_dims=None, n_classes=None, eval_set=None, eval_name=None,
                report_to_ray=False, init_model=None, *args, **kwargs):
            eval_set = sequence_to_list(eval_set) if eval_set is not None else []
            eval_name = sequence_to_list(eval_name) if eval_name is not None else []
            if eval_set and not eval_name:
                eval_name = [f'validation_{i}' for i in range(len(eval_set))]
            if len(eval_set) != len(eval_name):
                raise AttributeError('eval_set and eval_name should have the same length')

            if isinstance(y, pd.Series):
                y = y.to_frame()

            if cat_features:
                # if we pass cat_features as column names, we can ensure that they are in the dataframe
                # (and not dropped during preprocessing)
                if isinstance(cat_features[0], str):
                    cat_features_without_dropped = deepcopy(cat_features)
                    if cat_dims is not None:
                        cat_features_dims = dict(zip(cat_features, cat_dims))
                    for i, feature in enumerate(cat_features):
                        if feature not in X.columns:
                            cat_features_without_dropped.remove(feature)
                    cat_features = cat_features_without_dropped
                    if cat_dims is not None:
                        cat_dims = [cat_features_dims[feature] for feature in cat_features]

            if hasattr(self, 'map_task_to_default_values'):
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

            if hasattr(self, 'auto_early_stopping'):
                X, y, eval_set, eval_name = fn_to_add_auto_early_stopping(
                    self.auto_early_stopping, self.early_stopping_validation_size, X, y, task, eval_set, eval_name)

            # if we have a before_fit method, we call it here
            if hasattr(self, 'before_fit'):
                # fn takes all arguments passed to the fit function and returns fit_arguments (possibly modified)
                # we also incorporate the *args in the **kwargs
                bound_args = sklearn_fit_signature.bind_partial(self, X, y, *args, **kwargs)
                arg_and_kwargs = bound_args.arguments
                del arg_and_kwargs['self'], arg_and_kwargs['X'], arg_and_kwargs['y']
                fit_arguments = self.before_fit(X, y, task=task, cat_features=cat_features,
                                                cat_dims=cat_dims, n_classes=n_classes,
                                                eval_set=eval_set,  eval_name=eval_name, report_to_ray=report_to_ray,
                                                init_model=init_model, **arg_and_kwargs)
                return sklearn_cls.fit(self, **fit_arguments)
            # otherwise we assume that we will only call the original fit method with X, y, *args, **kwargs
            else:
                return sklearn_cls.fit(self, X, y, *args, **kwargs)
    # END OF CLASS DEFINITION
    if before_fit_method:
        TabBenchmarkSklearn.before_fit = before_fit_method

    if extra_dct:
        for key, value in extra_dct.items():
            setattr(TabBenchmarkSklearn, key, value)

    TabBenchmarkSklearn.__init__.__signature__ = Signature(parameters=list(init_parameters.values()))
    TabBenchmarkSklearn.fit.__signature__ = Signature(parameters=list(fit_parameters.values()))
    # TabBenchmarkSklearn.__doc__ = init_doc
    TabBenchmarkSklearn.fit.__doc__ = fit_doc
    name = f'TabBenchmark{sklearn_cls.__name__}'
    return type(name, (TabBenchmarkSklearn,), {'__doc__': init_doc})
