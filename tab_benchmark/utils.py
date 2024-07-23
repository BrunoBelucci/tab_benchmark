from __future__ import annotations

import subprocess
import warnings
from functools import partial
from inspect import signature
from typing import Sequence, Optional
import numpy as np
import random
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, root_mean_squared_error, log_loss, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split


def sequence_to_list(sequence):
    if isinstance(sequence, list):
        return sequence
    else:
        # we assume that the sequence can be converted to a list
        return list(sequence)


def check_same_keys(*dicts):
    """
    Check if all dictionaries have exactly the same keys.

    Parameters:
    *dicts: Arbitrary number of dictionary arguments.

    Returns:
    bool: True if all dictionaries have the same keys, False otherwise.
    """
    # Extract the set of keys from the first dictionary
    keys_set = set(dicts[0].keys())

    # Compare the keys set with the keys of the remaining dictionaries
    for d in dicts[1:]:
        if set(d.keys()) != keys_set:
            return False
    return True


def check_if_arg_in_kwargs_of_fn(arg_name, **kwargs):
    # check if arg_name is provided in kwargs
    if arg_name in kwargs:
        return True
    else:
        return False


def check_if_arg_in_args_of_fn(fn, arg_name, *args):
    # check if arg_name is provided in args and return index of it if it is, False otherwise
    all_parameters = signature(fn).parameters
    i_target_type = list(all_parameters).index(arg_name)
    if len(args) > i_target_type:
        return i_target_type
    else:
        return False


def check_if_arg_in_args_kwargs_of_fn(fn, arg_name, *args, return_arg=False, **kwargs):
    # check if arg_name is provided in kwargs
    if arg_name in kwargs:
        if return_arg:
            return True, kwargs[arg_name]
        else:
            return True
    else:
        # check if arg_name is provided in args
        all_parameters = signature(fn).parameters
        i_target_type = list(all_parameters).index(arg_name)
        if len(args) > i_target_type:
            if return_arg:
                return True, args[i_target_type]
            else:
                return True
        else:
            if return_arg:
                arg = all_parameters[arg_name].default
                return False, arg
            else:
                return False


def evaluate_set(model, eval_set: Sequence[pd.DataFrame], metrics: str | list[str], n_classes: Optional[int] = None) \
        -> list[float]:
    """Given an eval_set, consisting of a tuple-like (X, y), evaluate the metric on the given set.

    Args:
        model:
            Model to be evaluated.
        eval_set:
            Evaluation set to be evaluated with metric.
        metrics:
            Metrics to be evaluated on evaluation set.
        n_classes:
            Number of classes in the classification problem. If None, it is inferred from the evaluation set.

    Returns:
        The value of the metric evaluated on the evaluation set.
    """
    labels = list(range(n_classes)) if n_classes is not None else None
    # map_metric_to_func[metric] = (function, need_proba)
    map_metric_to_func = {
        'mse': (mean_squared_error, False),
        'rmse': (root_mean_squared_error, False),
        'logloss': (partial(log_loss, labels=labels), True),
        'r2_score': (r2_score, False),
        'auc': (partial(roc_auc_score, multi_class='ovr', labels=labels), True)
    }
    X = eval_set[0]
    y = eval_set[1]
    y_pred = None
    y_pred_proba = None
    scores = []
    for metric in metrics:
        if metric not in map_metric_to_func:
            msg = f'metric {metric} not implemented'
            raise NotImplementedError(msg)
        func, need_proba = map_metric_to_func[metric]
        if need_proba:
            if y_pred_proba is None:
                y_pred_proba = model.predict_proba(X)
        else:
            if y_pred is None:
                y_pred = model.predict(X)
        if metric == 'auc':
            if y.shape[1] == 1:
                y_ = y.to_numpy().reshape(-1)
            else:
                y_ = y
            if n_classes == 2:
                y_pred_proba_ = y_pred_proba[:, 1]
            else:
                y_pred_proba_ = y_pred_proba
        else:
            y_ = y
            y_pred_proba_ = y_pred_proba
        try:
            score = func(y_, y_pred_proba_ if need_proba else y_pred)
        except ValueError as e:
            score = np.nan
        scores.append(score)
    return scores


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        return 'Not a git repository'


def extends(fn_being_extended, map_default_values_change=None, additional_params=None, exclude_params=None):

    exclude_params = exclude_params if exclude_params is not None else []
    additional_params = additional_params if additional_params is not None else []

    def decorator(fn):
        fn_parameters = signature(fn, eval_str=True).parameters
        fn_being_extended_parameters = signature(fn_being_extended, eval_str=True).parameters

        # If we want to change the default values of some parameters, this takes care of documentation
        if map_default_values_change is not None:
            parameters = []
            for i, (name, param) in enumerate(fn_being_extended_parameters.items()):
                if name in map_default_values_change:
                    param = param.replace(default=map_default_values_change[name])
                if name not in exclude_params:
                    parameters.append(param)
        else:
            parameters = [param for name, param in fn_being_extended_parameters.items() if name not in exclude_params]

        # Order the parameters of the function being extended to respect python arguments order
        # Order will be: *args_fn_being_extended_parameters, *args_fn_parameters, *args_additional_params,
        # **kwargs_fn_being_extended_parameters, **kwargs_fn_parameters, **kwargs_additional_params
        parameters_being_added = [param for name, param in fn_parameters.items()
                                  if name not in fn_being_extended_parameters and name != 'kwargs' and name != 'args']

        parameters_pos_only = [param for param in parameters if param.kind == param.POSITIONAL_ONLY]
        parameters_pos_or_kw_without_default = [param for param in parameters if
                                                param.kind == param.POSITIONAL_OR_KEYWORD and
                                                param.default == param.empty]
        parameters_pos_or_kw_with_default = [param for param in parameters if
                                             param.kind == param.POSITIONAL_OR_KEYWORD and
                                             param.default != param.empty]
        parameters_var_pos = [param for param in parameters if param.kind == param.VAR_POSITIONAL]
        parameters_kw_only = [param for param in parameters if param.kind == param.KEYWORD_ONLY]
        parameters_var_kw = [param for param in parameters if param.kind == param.VAR_KEYWORD]

        parameters_being_added_pos_only = [param for param in parameters_being_added
                                           if param.kind == param.POSITIONAL_ONLY]
        parameters_being_added_pos_or_kw_without_default = [param for param in parameters_being_added if
                                                            param.kind == param.POSITIONAL_OR_KEYWORD and
                                                            param.default == param.empty]
        parameters_being_added_var_pos = [param for param in parameters_being_added
                                          if param.kind == param.VAR_POSITIONAL]
        parameters_being_added_kw_only = []
        for param in parameters_being_added:
            if param.kind == param.POSITIONAL_OR_KEYWORD and param.default != param.empty:
                param = param.replace(kind=param.KEYWORD_ONLY)
                parameters_being_added_kw_only.append(param)
        for param in parameters_being_added:
            if param.kind == param.KEYWORD_ONLY:
                parameters_being_added_kw_only.append(param)
        parameters_being_added_var_kw = [param for param in parameters_being_added if param.kind == param.VAR_KEYWORD]

        additional_params_pos_only = [param for param in additional_params if param.kind == param.POSITIONAL_ONLY]
        additional_params_pos_or_kw_without_default = [param for param in additional_params if
                                                       param.kind == param.POSITIONAL_OR_KEYWORD
                                                       and param.default == param.empty]
        additional_params_var_pos = [param for param in additional_params if param.kind == param.VAR_POSITIONAL]
        additional_params_kw_only = []
        for param in additional_params:
            if param.kind == param.POSITIONAL_OR_KEYWORD and param.default != param.empty:
                param = param.replace(kind=param.KEYWORD_ONLY)
                additional_params_kw_only.append(param)
        for param in additional_params:
            if param.kind == param.KEYWORD_ONLY:
                additional_params_kw_only.append(param)
        additional_params_var_kw = [param for param in additional_params if param.kind == param.VAR_KEYWORD]

        if parameters_var_pos:
            var_pos = parameters_var_pos
        elif parameters_being_added_var_pos:
            var_pos = parameters_being_added_var_pos
        elif additional_params_var_pos:
            var_pos = additional_params_var_pos
        else:
            var_pos = []

        if parameters_var_kw:
            var_kw = parameters_var_kw
        elif parameters_being_added_var_kw:
            var_kw = parameters_being_added_var_kw
        elif additional_params_var_kw:
            var_kw = additional_params_var_kw
        else:
            var_kw = []

        new_parameters = (parameters_pos_only + parameters_being_added_pos_only + additional_params_pos_only
                          + parameters_pos_or_kw_without_default + parameters_being_added_pos_or_kw_without_default
                          + additional_params_pos_or_kw_without_default + parameters_pos_or_kw_with_default + var_pos
                          + parameters_kw_only + parameters_being_added_kw_only + additional_params_kw_only + var_kw)

        new_signature = signature(fn_being_extended).replace(parameters=new_parameters)

        def wrapper(*args, **kwargs):
            bound_args = new_signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            return fn(**bound_args.arguments)

        wrapper.__signature__ = new_signature
        doc = ''
        if fn_being_extended.__doc__ is not None:
            doc = fn_being_extended.__doc__
        if fn.__doc__ is not None:
            doc += fn.__doc__
        wrapper.__doc__ = doc
        return wrapper
    return decorator


def train_test_split_forced(train_data, train_target, test_size_pct, random_state=None, stratify=None):
    if stratify is not None:
        number_of_classes = max(train_target.nunique())
        # If we sample less than the number of classes, we cannot have at lest one example per class in the split
        # For example, we cannot sample 10 examples if we have 11 classes, because we will have at least one
        # class with 0 examples in the validation set.
        # NOTE: apparently this does not ensure that we have at least one example per class in the validation set,
        # but it is the best we can do for now...
        if test_size_pct * len(train_data) < number_of_classes:
            test_size_pct = max(train_target.nunique())
    try:
        X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=test_size_pct,
                                                              random_state=random_state, stratify=stratify)
    except ValueError as exception:
        warnings.warn(f'Got {exception} when splitting the data, trying to fix it by artificially increasing '
                      f'the number of examples of the least frequent class.')
        # Probably the error is:
        # The least populated class in y has only 1 member, which is too few. The minimum number of groups
        # for any class cannot be less than 2.
        class_counts = train_target.value_counts()
        only_1_member_classes = class_counts[class_counts == 1]
        for only_1_member_class in only_1_member_classes.index:
            if isinstance(only_1_member_class, tuple):
                only_1_member_class = only_1_member_class[0]
            index_of_only_1_member_class = train_target[train_target.iloc[:, 0] == only_1_member_class].index[0]
            train_data = pd.concat([train_data, pd.DataFrame(train_data.loc[index_of_only_1_member_class]).T]).\
                reset_index(drop=True)
            train_target = pd.concat([train_target, pd.DataFrame(train_target.loc[index_of_only_1_member_class]).T]).\
                reset_index(drop=True)
        if stratify is not None:
            stratify = train_target
        X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=test_size_pct,
                                                              random_state=random_state, stratify=stratify)
    return X_train, X_valid, y_train, y_valid
