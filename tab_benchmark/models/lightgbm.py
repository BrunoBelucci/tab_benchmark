from pathlib import Path
from typing import Optional, Dict, Any
import mlflow
from lightgbm import LGBMRegressor as OriginalLGBMRegressor, LGBMClassifier as OriginalLGBMClassifier
from lightgbm.basic import _is_numpy_1d_array, _to_string, _NUMERIC_TYPES, _is_numeric
from lightgbm.callback import CallbackEnv
from ray import tune
from ray.train import report
from tab_benchmark.models.xgboost import n_estimators_gbdt, early_stopping_patience_gbdt
from tab_benchmark.models.factories import TabBenchmarkModelFactory
import lightgbm.basic


# Monkey patching the lightgbm to support nested dictionaries
def my_param_dict_to_str(data: Optional[Dict[str, Any]]) -> str:
    """Convert Python dictionary to string, which is passed to C API."""
    if data is None or not data:
        return ""
    pairs = []
    for key, val in data.items():
        if isinstance(val, (list, tuple, set)) or _is_numpy_1d_array(val):
            pairs.append(f"{key}={','.join(map(_to_string, val))}")
        elif isinstance(val, (str, Path, _NUMERIC_TYPES)) or _is_numeric(val):
            pairs.append(f"{key}={val}")
        elif isinstance(val, dict):
            pass
        elif val is not None:
            raise TypeError(f"Unknown type of parameter:{key}, got:{type(val).__name__}")
    return " ".join(pairs)


lightgbm.basic._param_dict_to_str = my_param_dict_to_str


def create_search_space_lgbm():
    # In Well tunned... + doc
    # Not tunning n_estimators following discussion at
    # https://openreview.net/forum?id=Fp7__phQszn&noteId=Z7Y_qxwDjiM
    search_space = dict(
        learning_rate=tune.loguniform(1e-3, 1.0),
        reg_lambda=tune.loguniform(1e-10, 1),
        reg_alpha=tune.loguniform(1e-10, 1.0),
        min_split_gain=tune.loguniform(1e-10, 1.0),
        colsample_bytree=tune.uniform(0.1, 1),
        feature_fraction_bynode=tune.uniform(0.1, 1),
        max_depth=tune.randint(1, 20),
        max_delta_step=tune.randint(0, 10),
        min_child_weight=tune.loguniform(1e-3, 20),
        subsample=tune.uniform(0.01, 1),
        subsample_freq=tune.randint(1, 11),
        num_leaves=tune.randint(1, 4096),  # should be < 2^(max_depth)
        min_child_samples=tune.randint(1, 1000000)
    )
    default_values = dict(
        learning_rate=0.1,
        reg_lambda=0,
        reg_alpha=0,
        min_split_gain=0,
        colsample_bytree=1.0,
        feature_fraction_bynode=1.0,
        max_depth=6,
        max_delta_step=0,
        min_child_weight=1e-3,
        subsample=1.0,
        subsample_freq=1,
        num_leaves=31,
        min_child_samples=20
    )
    return search_space, default_values


def get_recommended_params_lgbm():
    default_values_from_search_space = create_search_space_lgbm()[1]
    default_values_from_search_space.update(dict(
        n_estimators=n_estimators_gbdt,
        auto_early_stopping=True,
        early_stopping_rounds=early_stopping_patience_gbdt,
    ))
    return default_values_from_search_space


class ReportToRayLGBM:
    def __init__(self, default_metric=None):
        super().__init__()
        self.default_metric = default_metric

    def __call__(self, env: CallbackEnv) -> None:
        result_list = env.evaluation_result_list
        report_dict = {}
        for result in result_list:
            eval_name, metric_name, eval_result, is_higher_better = result
            # this is done everywhere on the original code with the comment
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            # so we do the same here
            metric_name = metric_name.split(" ")[-1]
            report_dict[f'{eval_name}_{metric_name}'] = eval_result
            if self.default_metric:
                if metric_name == self.default_metric:
                    report_dict[f'{eval_name}_default'] = eval_result
        report(report_dict)


class LogToMLFlowLGBM:
    def __init__(self, log_every_n_steps=50, default_metric=None):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.default_metric = default_metric

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration % self.log_every_n_steps != 0:
            return
        result_list = env.evaluation_result_list
        log_dict = {}
        for result in result_list:
            eval_name, metric_name, eval_result, is_higher_better = result
            # this is done everywhere on the original code with the comment
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            # so we do the same here
            metric_name = metric_name.split(" ")[-1]
            log_dict[f'{eval_name}_{metric_name}'] = eval_result
            if self.default_metric:
                if metric_name == self.default_metric:
                    log_dict[f'{eval_name}_default'] = eval_result
        log_dict['epoch'] = env.iteration
        mlflow.log_metrics(log_dict, step=env.iteration)


def before_fit_lgbm(self, extra_arguments, **fit_arguments):
    report_to_ray = extra_arguments.get('report_to_ray')
    cat_features = extra_arguments.get('cat_features')
    eval_name = extra_arguments.get('eval_name')
    callbacks = fit_arguments.get('callbacks', [])
    callbacks = callbacks if callbacks is not None else []
    if cat_features is not None:
        fit_arguments['categorical_feature'] = cat_features
    if report_to_ray:
        callbacks.append(ReportToRayLGBM(default_metric=self.metric))
    if self.log_to_mlflow_if_running:
        if mlflow.active_run():
            callbacks.append(LogToMLFlowLGBM(default_metric=self.metric))
    fit_arguments['eval_names'] = eval_name
    fit_arguments['callbacks'] = callbacks
    return fit_arguments


LGBMRegressor = TabBenchmarkModelFactory.from_sk_cls(
    OriginalLGBMRegressor,
    map_default_values_change={
        'objective': 'regression',
        'metric': 'l2'
    },
    has_auto_early_stopping=True,
    extended_init_kwargs={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'category',
        'data_scaler': None,
    },
    extra_dct={
        'create_search_space': staticmethod(create_search_space_lgbm),
        'get_recommended_params': staticmethod(get_recommended_params_lgbm),
        'before_fit': before_fit_lgbm
    }
)


LGBMClassifier = TabBenchmarkModelFactory.from_sk_cls(
    OriginalLGBMClassifier,
    map_task_to_default_values={
        'binary_classification': {'objective': 'binary', 'metric': 'binary_logloss'},
        'classification': {'objective': 'multiclass', 'metric': 'multi_logloss'},
    },
    has_auto_early_stopping=True,
    extended_init_kwargs={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'category',
        'data_scaler': None,
    },
    extra_dct={
        'create_search_space': staticmethod(create_search_space_lgbm),
        'get_recommended_params': staticmethod(get_recommended_params_lgbm),
        'before_fit': before_fit_lgbm
    }
)
