from catboost import CatBoostRegressor as OriginalCatBoostRegressor, CatBoostClassifier as OriginalCatBoostClassifier
from tab_benchmark.models.xgboost import fn_to_run_before_fit_for_gbdt
from tab_benchmark.models.factories import TabBenchmarkModelFactory


CatBoostRegressor = TabBenchmarkModelFactory.from_sk_cls(
    OriginalCatBoostRegressor,
    map_task_to_default_values={
        'regression': {'loss_function': 'RMSE', 'eval_metric': 'RMSE'},
        'multi_regression': {'loss_function': 'MultiRMSE', 'eval_metric': 'MultiRMSE'},
    },
    has_auto_early_stopping=True,
    fn_to_run_before_fit=fn_to_run_before_fit_for_gbdt,
    extended_init_kwargs={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'int32',
        'data_scaler': None,
        'continuous_target_scaler': None,
    }
)

CatBoostClassifier = TabBenchmarkModelFactory.from_sk_cls(
    OriginalCatBoostClassifier,
    map_task_to_default_values={
        'classification': {'loss_function': 'MultiClass', 'eval_metric': 'MultiClass'},
        'binary_classification': {'loss_function': 'Logloss', 'eval_metric': 'Logloss'},
    },
    has_auto_early_stopping=True,
    fn_to_run_before_fit=fn_to_run_before_fit_for_gbdt,
    extended_init_kwargs={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'int32',
        'data_scaler': None,
        'continuous_target_scaler': None,
    }
)
