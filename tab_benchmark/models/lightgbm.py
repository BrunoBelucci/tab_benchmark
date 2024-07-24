from pathlib import Path
from typing import Optional, Dict, Any
from lightgbm import LGBMRegressor as OriginalLGBMRegressor, LGBMClassifier as OriginalLGBMClassifier
from lightgbm.basic import _is_numpy_1d_array, _to_string, _NUMERIC_TYPES, _is_numeric
from tab_benchmark.models.xgboost import fn_to_run_before_fit_for_gbdt
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


LGBMRegressor = TabBenchmarkModelFactory.from_sk_cls(
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
    }
)


LGBMClassifier = TabBenchmarkModelFactory.from_sk_cls(
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
    }
)
