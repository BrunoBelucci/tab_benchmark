from lightgbm import LGBMRegressor as OriginalLGBMRegressor, LGBMClassifier as OriginalLGBMClassifier
from tab_benchmark.models.xgboost import fn_to_run_before_fit_for_gbdt
from tab_benchmark.models.factories import SimpleSkLearnFactory


LGBMRegressor = SimpleSkLearnFactory.from_sk_cls(
    OriginalLGBMRegressor,
    map_default_values_change={
        'objective': 'regression',
        'metric': 'l2'
    },
    has_auto_early_stopping=True,
    fn_to_run_before_fit=fn_to_run_before_fit_for_gbdt,
    extended_init_kwargs={
        'categorical_encoder': None,
        'categorical_type': 'category',
        'data_scaler': None,
        'continuous_target_scaler': None,
    }
)


LGBMClassifier = SimpleSkLearnFactory.from_sk_cls(
    OriginalLGBMClassifier,
    map_task_to_default_values={
        'binary_classification': {'objective': 'binary', 'metric': 'binary_logloss'},
        'classification': {'objective': 'multiclass', 'metric': 'multi_logloss'},
    },
    has_auto_early_stopping=True,
    fn_to_run_before_fit=fn_to_run_before_fit_for_gbdt,
    extended_init_kwargs={
        'categorical_encoder': None,
        'categorical_type': 'category',
        'data_scaler': None,
        'continuous_target_scaler': None,
    }
)
