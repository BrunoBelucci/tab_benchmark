from __future__ import annotations
from typing import Optional
import numpy as np
from tab_benchmark.models.base_models import SkLearnExtension
from tab_benchmark.utils import extends
from inspect import cleandoc
from tab_benchmark.utils import check_same_keys
from catboost import CatBoost


def init_factory(
        cls,
        map_default_values_change=None,
        has_auto_early_stopping: bool = False,
        map_task_to_default_values=None,
        # preprocessing
        categorical_imputer: Optional[str | int | float] = 'most_frequent',
        continuous_imputer: Optional[str | int | float] = 'median',
        categorical_encoder: Optional[str] = 'one_hot',
        handle_unknown_categories: bool = True,
        variance_threshold: Optional[float] = 0.0,
        data_scaler: Optional[str] = 'standard',
        categorical_type: Optional[np.dtype | str] = np.float32,
        continuous_type: Optional[np.dtype] = np.float32,
        target_imputer: Optional[str | int | float] = None,
        categorical_target_encoder: Optional[str] = 'ordinal',  # only used in classification
        categorical_target_min_frequency: Optional[int | float] = 10,  # only used in classification
        continuous_target_scaler: Optional[str] = 'standard',  # only used in regression
        categorical_target_type: Optional[np.dtype] = np.float32,
        continuous_target_type: Optional[np.dtype] = np.float32,

):
    map_task_to_default_values_outer = map_task_to_default_values if map_task_to_default_values else {}

    @extends(cls.__init__, map_default_values_change=map_default_values_change)
    def init_fn_base(
            self,
            *args,
            map_task_to_default_values=None,
            categorical_imputer: Optional[str | int | float] = categorical_imputer,
            continuous_imputer: Optional[str | int | float] = continuous_imputer,
            categorical_encoder: Optional[str] = categorical_encoder,
            handle_unknown_categories: bool = handle_unknown_categories,
            variance_threshold: Optional[float] = variance_threshold,
            data_scaler: Optional[str] = data_scaler,
            categorical_type: Optional[np.dtype | str] = categorical_type,
            continuous_type: Optional[np.dtype] = continuous_type,
            target_imputer: Optional[str | int | float] = target_imputer,
            categorical_target_encoder: Optional[str] = categorical_target_encoder,  # only used in classification
            categorical_target_min_frequency: Optional[int | float] = categorical_target_min_frequency,
            # only used in classification
            continuous_target_scaler: Optional[str] = continuous_target_scaler,  # only used in regression
            categorical_target_type: Optional[np.dtype] = categorical_target_type,
            continuous_target_type: Optional[np.dtype] = continuous_target_type,
            **kwargs
    ):
        cls.__init__(self, *args, **kwargs)
        map_task_to_default_values = map_task_to_default_values if map_task_to_default_values else \
            map_task_to_default_values_outer
        self.map_task_to_default_values = map_task_to_default_values
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
        self.data_preprocess_pipeline_ = None
        self.target_preprocess_pipeline_ = None
        self.model_pipeline_ = None
        self.task_ = None
        self.cat_features_ = None

    init_doc = cleandoc("""Wrapper around scikit-learn class.

        Parameters
        ----------
        map_task_to_default_values:
            Mapping from task to default values.
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
        """)

    if has_auto_early_stopping:
        @extends(init_fn_base)
        def init_fn_with_es(self, *args, auto_early_stopping: bool = False, early_stopping_validation_size=0.1,
                            **kwargs):
            init_fn_base(self, *args, **kwargs)
            self.auto_early_stopping = auto_early_stopping
            self.early_stopping_validation_size = early_stopping_validation_size

        init_doc += "\n"
        init_doc += cleandoc("""
        auto_early_stopping:
            Whether to use early stopping automatically, i.e., split the training data into training and validation sets
            and stop training when the validation score does not improve anymore.
        """)

    else:
        init_fn_with_es = init_fn_base

    init_doc += "\n\nOriginal documentation:\n\n"
    return init_fn_with_es, init_doc


def fit_factory(cls, fn_to_run_before_fit=None):
    @extends(cls.fit)
    def fit_fn(self, X, y, *args, task=None, cat_features=None, **kwargs):
        self.cat_features_ = cat_features
        self.task_ = task
        if self.map_task_to_default_values is not None and task is not None:
            if task in self.map_task_to_default_values:
                for key, value in self.map_task_to_default_values[task].items():
                    setattr(self, key, value)
                    if isinstance(self, CatBoost):
                        # we need to cheat a bit for CatBoost, because it uses the _init_params dictionary
                        init_params = getattr(self, '_init_params', {})
                        init_params[key] = value
                        setattr(self, '_init_params', init_params)
        if fn_to_run_before_fit is not None:
            X, y, task, cat_features, args, kwargs = fn_to_run_before_fit(self, X, y, task, cat_features, *args,
                                                                          **kwargs)
        return cls.fit(self, X, y, *args, **kwargs)

    doc = cleandoc("""Wrapper around the fit method of the scikit-learn class.

        Parameters
        ----------
        task:
            Task type.
        cat_features:
            Categorical features.
        """)
    doc += "\n\nOriginal documentation:\n\n"
    fit_fn.__doc__ = doc + cls.fit.__doc__
    return fit_fn


class SimpleSkLearnFactory(type):
    @classmethod  # to be cleaner (not change the signature of __new__)
    def from_sk_cls(cls, sk_cls, extended_init_kwargs=None, map_default_values_change=None,
                    has_auto_early_stopping=False, map_task_to_default_values=None, fn_to_run_before_fit=None):
        extended_init_kwargs = extended_init_kwargs if extended_init_kwargs else {}
        if map_task_to_default_values:
            map_default_values_change = map_default_values_change if map_default_values_change else {}
            dicts = list(map_task_to_default_values.values())
            if not check_same_keys(*dicts):
                raise ValueError('All dictionaries in map_task_to_default_values must have the same keys.')
            keys = dicts[0].keys()
            for key in keys:
                map_default_values_change[key] = 'default'
        name = sk_cls.__name__
        init_fn, init_doc = init_factory(
            sk_cls,
            map_default_values_change=map_default_values_change,
            has_auto_early_stopping=has_auto_early_stopping,
            map_task_to_default_values=map_task_to_default_values,
            **extended_init_kwargs
        )
        dct = {
            '__init__': init_fn,
            'fit': fit_factory(sk_cls, fn_to_run_before_fit),
            '__doc__': init_doc + sk_cls.__doc__
        }
        return type(name, (sk_cls, SkLearnExtension), dct)

# maybe not good idea, because we lost documentation, annotation etc...
# class TaskDependentSkLearnFactory(type):
#     @classmethod
#     def from_multiple_sk_cls(cls, name, map_task_to_sk_cls, extended_init_kwargs=None):
#         extended_init_kwargs = extended_init_kwargs if extended_init_kwargs else {}
#         return cls(name, (), {'extended_init_kwargs': extended_init_kwargs, 'map_task_to_sk_cls': map_task_to_sk_cls})
#
#     @classmethod
#     def from_multiple_sk_cls_and_init_kwargs(cls, name, map_task_to_sk_cls_and_kwargs, extended_init_kwargs):
#         extended_init_kwargs = extended_init_kwargs if extended_init_kwargs else {}
#         return cls(name, (), {'extended_init_kwargs': extended_init_kwargs, 'map_task_to_sk_cls_and_kwargs': map_task_to_sk_cls_and_kwargs})
#
#     def __call__(cls, task, *args, **kwargs):
#         if hasattr(cls, 'map_task_to_sk_cls'):
#             if task not in cls.map_task_to_sk_cls:
#                 raise ValueError(f"Task '{task}' not registered. Available tasks: {list(cls.map_task_to_sk_cls.keys())}")
#             sk_cls = cls.map_task_to_sk_cls[task]
#         elif hasattr(cls, 'map_task_to_sk_cls_and_kwargs'):
#             if task not in cls.map_task_to_sk_cls_and_kwargs:
#                 raise ValueError(f"Task '{task}' not registered. Available tasks: {list(cls.map_task_to_sk_cls_and_kwargs.keys())}")
#             sk_cls, sk_cls_kwargs = cls.map_task_to_sk_cls_and_kwargs[task]
#             for key, value in sk_cls_kwargs.items():
#                 if not check_if_arg_in_args_kwargs_of_fn(sk_cls.__init__, key, *args, **kwargs):
#                     kwargs[key] = value
#         return SimpleSkLearnFactory.from_sk_cls(sk_cls, cls.extended_init_kwargs)(*args, **kwargs)
