from catboost import CatBoostRegressor as OriginalCatBoostRegressor, CatBoostClassifier as OriginalCatBoostClassifier
from tab_benchmark.models.xgboost import fn_to_run_before_fit_for_gbdt
from tab_benchmark.models.factories import TabBenchmarkModelFactory


def n_jobs_get(self):
    if not hasattr(self, '_n_jobs'):
        self._n_jobs = self.get_params().get('thread_count', 1)
    return self._n_jobs


def n_jobs_set(self, value):
    self._n_jobs = value
    self.set_params(**{'thread_count': value})


n_jobs_property = property(n_jobs_get, n_jobs_set)


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
    },
    extra_dct={
        'n_jobs': n_jobs_property
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
    },
    extra_dct={
        'n_jobs': n_jobs_property
    }
)
