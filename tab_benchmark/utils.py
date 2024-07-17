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


def evaluate_set(model, eval_set: Sequence[pd.DataFrame], metric: str,
                 n_classes: Optional[int] = None) -> float:
    """Given an eval_set, consisting of a tuple-like (X, y), evaluate the metric on the given set.

    Args:
        eval_set:
            Evaluation set to be evaluated with metric.
        metric:
            Metric to be evaluated on evaluation set.

    Returns:
        The value of the metric evaluated on the evaluation set.
    """
    X = eval_set[0]
    y = eval_set[1]
    if metric == 'mse':
        y_pred = model.predict(X)
        metric = mean_squared_error
    elif metric == 'rmse':
        y_pred = model.predict(X)
        metric = root_mean_squared_error
    elif metric == 'logloss':
        y_pred = model.predict_proba(X)
        labels = list(range(n_classes))
        return log_loss(y, y_pred, labels=labels)
    elif metric == 'r2_score':
        y_pred = model.predict(X)
        metric = r2_score
    elif metric == 'auc':
        y_pred = model.predict_proba(X)
        labels = list(range(n_classes))
        if y.shape[1] == 1:
            y = y.to_numpy().reshape(-1)
        if n_classes == 2:
            y_pred = y_pred[:, 1]
        metric = partial(roc_auc_score, multi_class='ovr', labels=labels)
    else:
        msg = 'metric {} not implemented'.format(metric)
        raise NotImplementedError(msg)
    return metric(y, y_pred)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        return 'Not a git repository'


def extends(fn_being_extended):
    def decorator(fn):
        fn_parameters = signature(fn, eval_str=True).parameters
        fn_being_extended_parameters = signature(fn_being_extended, eval_str=True).parameters
        parameters = list(fn_being_extended_parameters.values())
        additional_params = [param for name, param in fn_parameters.items()
                             if name not in fn_being_extended_parameters and name != 'kwargs' and name != 'args']
        parameters_var_kw = [param for param in parameters if param.kind == param.VAR_KEYWORD]
        parameters_others = [param for param in parameters if param.kind != param.VAR_KEYWORD]
        parameters_extended = parameters_others
        parameters_extended.extend(additional_params)
        parameters_extended.extend(parameters_var_kw)

        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper.__signature__ = signature(fn_being_extended).replace(parameters=parameters_extended)
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
