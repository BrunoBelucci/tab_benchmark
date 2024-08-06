from copy import deepcopy
from xgboost import XGBClassifier as OriginalXGBClassifier, XGBRegressor as OriginalXGBRegressor, XGBModel
from catboost import CatBoost
from lightgbm import LGBMModel
from tab_benchmark.models.factories import TabBenchmarkModelFactory
from tab_benchmark.utils import (extends, train_test_split_forced, sequence_to_list, check_if_arg_in_args_of_fn,
                                 check_if_arg_in_kwargs_of_fn)
from ray import tune


n_estimators_gbdt = 10000
early_stopping_patience_gbdt = 100


def create_search_space_xgboost():
    # In Well tunned...
    # Not tunning n_estimators following discussion at
    # https://openreview.net/forum?id=Fp7__phQszn&noteId=Z7Y_qxwDjiM
    search_space = dict(
        learning_rate=tune.loguniform(1e-3, 1.0),
        reg_lambda=tune.loguniform(1e-10, 1),
        reg_alpha=tune.loguniform(1e-10, 1.0),
        gamma=tune.loguniform(1e-1, 1.0),
        colsample_bylevel=tune.uniform(0.1, 1),
        colsample_bynode=tune.uniform(0.1, 1),
        colsample_bytree=tune.uniform(0.1, 1),
        max_depth=tune.randint(1, 20),
        max_delta_step=tune.randint(0, 10),
        min_child_weight=tune.loguniform(0.1, 20),
        subsample=tune.uniform(0.01, 1),
    )
    default_values = dict(
        learning_rate=0.3,
        reg_lambda=1.0,
        reg_alpha=1e-10,
        gamma=1e-1,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        colsample_bytree=1.0,
        max_depth=6,
        max_delta_step=0,
        min_child_weight=1.0,
        subsample=1.0,
    )
    return search_space, default_values


def get_recommended_params_xgboost():
    default_values_from_search_space = create_search_space_xgboost()[1]
    default_values_from_search_space.update(dict(
        n_estimators=n_estimators_gbdt,
        auto_early_stopping=True,
        early_stopping_rounds=early_stopping_patience_gbdt,
    ))
    return default_values_from_search_space


def fn_to_run_before_fit_for_gbdt_and_dnn(self, X, y, task, cat_features, eval_set, *args, **kwargs):
    if self.auto_early_stopping:
        if self.task_ == 'classification' or self.task_ == 'binary_classification':
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
    return X, y, task, cat_features, eval_set, args, kwargs


# Just to get the parameters of the XGBModel, because XGBClassifier and XGBRegressor do not show them
# This will initialize exactly the same way as XGBClassifier and XGBRegressor.
class XGBClassifier(OriginalXGBClassifier):
    __doc__ = OriginalXGBClassifier.__doc__

    @extends(XGBModel.__init__, map_default_values_change={'objective': 'binary:logistic'})
    def __init__(self, *args, **kwargs):
        XGBModel.__init__(self, *args, **kwargs)


class XGBRegressor(OriginalXGBRegressor):
    __doc__ = OriginalXGBRegressor.__doc__

    @extends(XGBModel.__init__, map_default_values_change={'objective': 'reg:squarederror'})
    def __init__(self, *args, **kwargs):
        XGBModel.__init__(self, *args, **kwargs)


# Now we wrap them with our API
XGBClassifier = TabBenchmarkModelFactory.from_sk_cls(
    XGBClassifier,
    map_task_to_default_values={
        'classification': {'objective': 'multi:softmax', 'eval_metric': 'mlogloss'},
        'binary_classification': {'objective': 'binary:logistic', 'eval_metric': 'logloss'},
    },
    has_auto_early_stopping=True,
    fn_to_run_before_fit=fn_to_run_before_fit_for_gbdt_and_dnn,
    extended_init_kwargs={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'category',
        'data_scaler': None,
    },
    extra_dct={
        'create_search_space': staticmethod(create_search_space_xgboost),
        'get_recommended_params': staticmethod(get_recommended_params_xgboost)
    }
)

XGBRegressor = TabBenchmarkModelFactory.from_sk_cls(
    XGBRegressor,
    map_default_values_change={
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    },
    has_auto_early_stopping=True,
    fn_to_run_before_fit=fn_to_run_before_fit_for_gbdt_and_dnn,
    extended_init_kwargs={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'category',
        'data_scaler': None,
    },
    extra_dct={
        'create_search_space': staticmethod(create_search_space_xgboost),
        'get_recommended_params': staticmethod(get_recommended_params_xgboost)
    }
)
