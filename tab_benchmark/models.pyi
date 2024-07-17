from types import NoneType
import numpy as np
import numpy
from _typeshed import Incomplete
from tabular_benchmark.TransformedTargetClassifier import TransformedTargetClassifier as TransformedTargetClassifier
from tabular_benchmark.preprocess import create_data_preprocess_pipeline as create_data_preprocess_pipeline, create_target_preprocess_pipeline as create_target_preprocess_pipeline
from tabular_benchmark.utils import extends as extends
from typing import Sequence, Union, Optional


class SkLearnExtension:
    """Wrapper around scikit-learn class.

    Parameters
    ----------
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

    Original documentation:

    """
    categorical_imputer: Incomplete
    continuous_imputer: Incomplete
    categorical_encoder: Incomplete
    handle_unknown_categories: Incomplete
    variance_threshold: Incomplete
    data_scaler: Incomplete
    categorical_type: Incomplete
    continuous_type: Incomplete
    target_imputer: Incomplete
    categorical_target_encoder: Incomplete
    categorical_target_min_frequency: Incomplete
    continuous_target_scaler: Incomplete
    categorical_target_type: Incomplete
    continuous_target_type: Incomplete
    data_preprocess_pipeline_: Incomplete
    target_preprocess_pipeline_: Incomplete
    model_pipeline_: Incomplete
    task_: Incomplete
    cat_features_: Incomplete
    def __init__(self, categorical_imputer: str | int | float | None = 'most_frequent', continuous_imputer: str | int | float | None = 'median', categorical_encoder: str | None = 'ordinal', handle_unknown_categories: bool = True, variance_threshold: float | None = 0.0, data_scaler: str | None = None, categorical_type: np.dtype | str | None = ..., continuous_type: np.dtype | None = ..., target_imputer: str | int | float | None = None, categorical_target_encoder: str | None = 'ordinal', categorical_target_min_frequency: int | float | None = 10, continuous_target_scaler: str | None = None, categorical_target_type: np.dtype | None = ..., continuous_target_type: np.dtype | None = ...) -> None: ...
    def create_preprocess_pipeline(self, task: str, categorical_features: Sequence[int | str] | None = None, continuous_features: Sequence[int | str] | None = None): ...
    def create_model_pipeline(self): ...
    def fit(self, cat_features: Incomplete | None = None) -> None:
        """Wrapper around the fit method of the scikit-learn class.

        Parameters
        ----------
        cat_features:
            Categorical features.

        Original documentation:

        """

def init_factory(cls): ...
def fit_factory(cls): ...

from sklearn.linear_model._ridge import RidgeCV
from tabular_benchmark.models import SkLearnExtension
class RidgeCV(RidgeCV, SkLearnExtension):
    def __init__(self, alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None, cv=None, gcv_mode=None, store_cv_results=None, alpha_per_target=False, store_cv_values='deprecated', categorical_imputer: Union[str, int, float, NoneType] = 'most_frequent', continuous_imputer: Union[str, int, float, NoneType] = 'median', categorical_encoder: Optional[str] = 'one_hot', handle_unknown_categories: bool = True, variance_threshold: Optional[float] = 0.0, data_scaler: Optional[str] = 'standard', categorical_type: Union[numpy.dtype, str, NoneType] = numpy.float32, continuous_type: Optional[numpy.dtype] = numpy.float32, target_imputer: Union[str, int, float, NoneType] = None, categorical_target_encoder: Optional[str] = 'ordinal', categorical_target_min_frequency: Union[int, float, NoneType] = 10, continuous_target_scaler: Optional[str] = 'standard', categorical_target_type: Optional[numpy.dtype] = numpy.float64, continuous_target_type: Optional[numpy.dtype] = numpy.float64): ...
    def fit(self, X, y, sample_weight=None, *, cat_features=None, **params): ...
