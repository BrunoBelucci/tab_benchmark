from __future__ import annotations
from typing import Optional
import numpy as np
from tab_benchmark.utils import extends

init_doc = """Wrapper around scikit-learn class.

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


def init_factory(
        cls,
        categorical_imputer: Optional[str | int | float] = 'most_frequent',
        continuous_imputer: Optional[str | int | float] = 'median',
        categorical_encoder: Optional[str] = 'one_hot',
        handle_unknown_categories: bool = True,
        variance_threshold: Optional[float] = 0.0,
        data_scaler: Optional[str] = 'standard',
        categorical_type: Optional[np.dtype | str] = np.float32,
        continuous_type: Optional[np.dtype] = np.float32,
        target_imputer: Optional[str | int | float] = None,
        categorical_target_encoder: Optional[str] = 'ordinal',  # only used in classification
        categorical_target_min_frequency: Optional[int | float] = 10,  # only used in classification
        continuous_target_scaler: Optional[str] = 'standard',  # only used in regression
        categorical_target_type: Optional[np.dtype] = np.float64,
        continuous_target_type: Optional[np.dtype] = np.float64,
):
    @extends(cls.__init__)
    def init_fn(
            self,
            *args,
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
            categorical_target_min_frequency: Optional[int | float] = categorical_target_min_frequency,  # only used in classification
            continuous_target_scaler: Optional[str] = continuous_target_scaler,  # only used in regression
            categorical_target_type: Optional[np.dtype] = categorical_target_type,
            continuous_target_type: Optional[np.dtype] = continuous_target_type,
            **kwargs
    ):
        cls.__init__(self, *args, **kwargs)
        self.categorical_imputer = categorical_imputer
        self.continuous_imputer = continuous_imputer
        self.categorical_encoder = categorical_encoder
        self.handle_unknown_categories = handle_unknown_categories
        self.variance_threshold = variance_threshold
        self.data_scaler = data_scaler
        self.categorical_type = categorical_type
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

    return init_fn


def fit_factory(cls):
    @extends(cls.fit)
    def fit_fn(self, *args, task=None, cat_features=None, **kwargs):
        self.cat_features_ = cat_features
        self.task_ = task
        return cls.fit(self, *args, **kwargs)

    doc = """Wrapper around the fit method of the scikit-learn class.

        Parameters
        ----------
        task:
            Task type.
        cat_features:
            Categorical features.

        Original documentation:

        """
    fit_fn.__doc__ = doc + cls.fit.__doc__
    return fit_fn
