from __future__ import annotations
from copy import deepcopy
from typing import Optional
import numpy as np
import pandas as pd
from tab_benchmark.models.sk_learn_extension import SkLearnExtension
from tab_benchmark.utils import extends, train_test_split_forced, sequence_to_list
from inspect import cleandoc, signature
from tab_benchmark.utils import check_same_keys


def fn_to_add_auto_early_stopping(self, X, y, task, eval_set, eval_name):
    if self.auto_early_stopping:
        if task == 'classification' or task == 'binary_classification':
            stratify = y
        else:
            stratify = None
        X, X_valid, y, y_valid = train_test_split_forced(
            X, y,
            test_size_pct=self.early_stopping_validation_size,
            # random_state=self.random_seed,  this will be ensured by set_seeds
            stratify=stratify
        )
        eval_set = eval_set if eval_set else []
        eval_set = sequence_to_list(eval_set)
        eval_set.append((X_valid, y_valid))
        eval_name = eval_name if eval_name else []
        eval_name = sequence_to_list(eval_name)
        eval_name.append('validation_es')
    return eval_set, eval_name


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
        categorical_type: Optional[type | str] = 'float32',
        continuous_type: Optional[type | str] = 'float32',
        target_imputer: Optional[str | int | float] = None,
        categorical_target_encoder: Optional[str] = 'ordinal',  # only used in classification
        categorical_target_min_frequency: Optional[int | float] = 10,  # only used in classification
        continuous_target_scaler: Optional[str] = 'standard',  # only used in regression
        categorical_target_type: Optional[type | str] = 'float32',
        continuous_target_type: Optional[type | str] = 'float32',
        # dnn architecture
        dnn_architecture_cls=None,
        add_lr_and_weight_decay_params=False
):
    map_task_to_default_values_outer = map_task_to_default_values

    @extends(cls.__init__, map_default_values_change=map_default_values_change)
    def init_fn_step_1(
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
        self.map_task_to_default_values = map_task_to_default_values if map_task_to_default_values else \
            map_task_to_default_values_outer
        self.categorical_imputer = categorical_imputer
        self.continuous_imputer = continuous_imputer
        self.categorical_encoder = categorical_encoder
        self.handle_unknown_categories = handle_unknown_categories
        self.variance_threshold = variance_threshold
        self.data_scaler = data_scaler
        if not hasattr(self, 'categorical_type'):
            self.categorical_type = categorical_type
        if not hasattr(self, 'continuous_type'):
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
        @extends(init_fn_step_1)
        def init_fn_step_2(self, *args, auto_early_stopping: bool = True, early_stopping_validation_size=0.1,
                           log_to_mlflow_if_running: bool = True,
                           **kwargs):
            init_fn_step_1(self, *args, **kwargs)
            self.auto_early_stopping = auto_early_stopping
            self.early_stopping_validation_size = early_stopping_validation_size
            self.log_to_mlflow_if_running = log_to_mlflow_if_running

        init_doc += "\n"
        init_doc += cleandoc("""
        auto_early_stopping:
            Whether to use early stopping automatically, i.e., split the training data into training and validation sets
            and stop training when the validation score does not improve anymore.
        early_stopping_validation_size:
            Size of the validation set when using auto early stopping.
        log_to_mlflow_if_running:
            Whether to log intermediate results to MLflow if it is running.
        """)

    else:
        init_fn_step_2 = init_fn_step_1

    if dnn_architecture_cls is not None:
        do_not_include_params = dnn_architecture_cls.params_defined_from_dataset
        do_not_include_params = do_not_include_params + ['self']
        parameters = signature(dnn_architecture_cls.__init__).parameters
        additional_params = {name: param for name, param in parameters.items() if name not in do_not_include_params}
        exclude_params = ['architecture_params', 'architecture_params_not_from_dataset', 'dnn_architecture_class',
                          'lit_module_class', 'lit_datamodule_class']
        if add_lr_and_weight_decay_params:
            @extends(init_fn_step_2, additional_params=additional_params.values(), exclude_params=exclude_params)
            def init_fn_step_3(
                    self,
                    *args,
                    lr: Optional[float] = None,
                    weight_decay: Optional[float] = None,
                    **kwargs
            ):
                architecture_params_not_from_dataset = {}
                for key, value in kwargs.copy().items():
                    if key in additional_params:
                        setattr(self, key, value)
                        architecture_params_not_from_dataset[key] = value
                        del kwargs[key]
                init_fn_step_2(self, *args, **kwargs)
                self.lr = lr
                self.weight_decay = weight_decay
                self.architecture_params_not_from_dataset = architecture_params_not_from_dataset
                self.dnn_architecture_class = dnn_architecture_cls

            init_doc += (f"\n\nArchitecture documentation:\n\nParameters that can be defined from the dataset are "
                         f"automatically set, they are: {dnn_architecture_cls.params_defined_from_dataset}\n\n")
            init_doc += cleandoc(dnn_architecture_cls.__doc__)
            init_doc += "\n"
            init_doc += cleandoc("""
            lr:
                Learning rate.
            weight_decay:
                Weight decay.
            """)
        else:
            @extends(dnn_architecture_cls.__init__, exclude_params=exclude_params)
            def init_fn_step_3(self, *args, **kwargs):
                init_fn_step_2(self, *args, **kwargs)
                # hopefully this will only set parameters that are not defined from the dataset
                fn_being_extended_parameters = signature(dnn_architecture_cls.__init__, eval_str=True).parameters
                parameters = []
                for i, (name, param) in enumerate(fn_being_extended_parameters.items()):
                    if name in map_default_values_change:
                        param = param.replace(default=map_default_values_change[name])
                    if name not in exclude_params:
                        parameters.append(param)
                new_signature = signature(cls.__init__).replace(parameters=parameters)
                bound_args = new_signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
                self.architecture_params_not_from_dataset = bound_args.arguments
                self.set_params(**bound_args.arguments)

            init_doc += (f"\n\n Architecture documentation:\n\nParameters that can be defined from the dataset are "
                         f"automatically set, they are: {exclude_params}\n\n")
            init_doc += cleandoc(dnn_architecture_cls.__doc__)

    else:
        init_fn_step_3 = init_fn_step_2

    init_doc += "\n\nOriginal documentation:\n\n" + cleandoc(cls.__doc__)
    return init_fn_step_3, init_doc


def fit_factory(cls):
    @extends(cls.fit)
    def fit_fn(self, X, y, *args, task=None, cat_features=None, eval_set=None, eval_name=None,
               report_to_ray=False, **kwargs):

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
                for feature in cat_features:
                    if feature not in X.columns:
                        cat_features_without_dropped.remove(feature)
                cat_features = cat_features_without_dropped

        if self.map_task_to_default_values is not None:
            if task is not None:
                if task in self.map_task_to_default_values:
                    for key, value in self.map_task_to_default_values[task].items():
                        self.set_params(**{key: value})
            else:
                raise (ValueError('This model has map_task_to_default_values, which means it has some values that are '
                                  'task dependent. You must provide the task when calling fit.'))

        if hasattr(self, 'auto_early_stopping'):
            eval_set, eval_name = fn_to_add_auto_early_stopping(self, X, y, task, eval_set, eval_name)

        # if we have a before_fit method, we call it here
        # it can modify the arguments of fit that will be passed to the original fit method
        # this way we can integrate for example the modifications on eval_set, eval_name etc
        cls_signature = signature(cls.fit)
        bound_args = cls_signature.bind_partial(self, X, y, *args, **kwargs)
        cls_parameters = cls_signature.parameters
        fit_arguments = bound_args.arguments
        del fit_arguments['self']
        extra_arguments = dict(task=task, cat_features=cat_features, eval_set=eval_set, eval_name=eval_name,
                               report_to_ray=report_to_ray)
        # if any extra_arguments are in the parameters of the original fit method, we integrate them back
        for key, value in extra_arguments.copy().items():
            if key in cls_parameters:
                fit_arguments[key] = value
                del extra_arguments[key]

        if hasattr(self, 'before_fit'):
            # fn takes extra_arguments and fit_arguments and returns fit_arguments (possibly modified)
            fit_arguments = self.before_fit(extra_arguments, **fit_arguments)

        return cls.fit(self, **fit_arguments)

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


class TabBenchmarkModelFactory(type):
    @classmethod  # to be cleaner (not change the signature of __new__)
    def from_sk_cls(cls, sk_cls, extended_init_kwargs=None, map_default_values_change=None,
                    has_auto_early_stopping=False, map_task_to_default_values=None,
                    dnn_architecture_cls=None, add_lr_and_weight_decay_params=False, extra_dct=None):
        extended_init_kwargs = extended_init_kwargs if extended_init_kwargs else {}

        if map_task_to_default_values:
            map_default_values_change = map_default_values_change if map_default_values_change else {}
            dicts = list(map_task_to_default_values.values())
            if not check_same_keys(*dicts):
                raise ValueError('All dictionaries in map_task_to_default_values must have the same keys.')
            keys = dicts[0].keys()
            for key in keys:
                map_default_values_change[key] = 'default'

        init_fn, init_doc = init_factory(
            sk_cls,
            map_default_values_change=map_default_values_change,
            has_auto_early_stopping=has_auto_early_stopping,
            map_task_to_default_values=map_task_to_default_values,
            dnn_architecture_cls=dnn_architecture_cls,
            add_lr_and_weight_decay_params=add_lr_and_weight_decay_params,
            **extended_init_kwargs
        )
        if dnn_architecture_cls is not None:
            name = dnn_architecture_cls.__name__ + 'Model'
            doc = init_doc
        else:
            name = sk_cls.__name__
            doc = init_doc
        dct = {
            '__init__': init_fn,
            'fit': fit_factory(sk_cls),
            '__doc__': doc
        }
        if extra_dct:
            dct.update(extra_dct)
        return type(name, (sk_cls, SkLearnExtension), dct)
