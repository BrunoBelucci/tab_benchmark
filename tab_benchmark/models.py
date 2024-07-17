from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV as SKRidgeCV
from tab_benchmark.TransformedTargetClassifier import TransformedTargetClassifier
from tab_benchmark.preprocess import create_data_preprocess_pipeline, create_target_preprocess_pipeline
from tab_benchmark.utils import extends


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

    def __init__(
            self,
            # preprocessing parameters
            categorical_imputer: Optional[str | int | float] = 'most_frequent',
            continuous_imputer: Optional[str | int | float] = 'median',
            categorical_encoder: Optional[str] = 'ordinal',
            handle_unknown_categories: bool = True,
            variance_threshold: Optional[float] = 0.0,
            data_scaler: Optional[str] = None,
            categorical_type: Optional[np.dtype | str] = np.float32,
            continuous_type: Optional[np.dtype] = np.float32,
            target_imputer: Optional[str | int | float] = None,
            categorical_target_encoder: Optional[str] = 'ordinal',  # only used in classification
            categorical_target_min_frequency: Optional[int | float] = 10,  # only used in classification
            continuous_target_scaler: Optional[str] = None,  # only used in regression
            categorical_target_type: Optional[np.dtype] = np.float32,
            continuous_target_type: Optional[np.dtype] = np.float32,
    ):
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

    def create_preprocess_pipeline(
            self,
            task: str,
            categorical_features: Optional[Sequence[int | str]] = None,
            continuous_features: Optional[Sequence[int | str]] = None,
    ):
        self.task_ = task
        self.data_preprocess_pipeline_ = create_data_preprocess_pipeline(
            categorical_features=categorical_features,
            continuous_features=continuous_features,
            categorical_imputer=self.categorical_imputer,
            continuous_imputer=self.continuous_imputer,
            categorical_encoder=self.categorical_encoder,
            handle_unknown_categories=self.handle_unknown_categories,
            variance_threshold=self.variance_threshold,
            scaler=self.data_scaler,
            categorical_type=self.categorical_type,
            continuous_type=self.continuous_type,
        )
        self.target_preprocess_pipeline_ = create_target_preprocess_pipeline(
            task=task,
            imputer=self.target_imputer,
            categorical_encoder=self.categorical_target_encoder,
            categorical_min_frequency=self.categorical_target_min_frequency,
            continuous_scaler=self.continuous_target_scaler,
            categorical_type=self.categorical_target_type,
            continuous_type=self.continuous_target_type,
        )

    def create_model_pipeline(self):
        if self.data_preprocess_pipeline_ is None or self.target_preprocess_pipeline_ is None:
            raise ValueError('Please run create_preprocess_pipeline first.')
        if self.task_ == 'classification':
            target_transformer = TransformedTargetClassifier
        elif self.task_ == 'regression':
            target_transformer = TransformedTargetRegressor
        else:
            raise ValueError(f'Unknown task: {self.task_}')
        self.model_pipeline_ = Pipeline([
            ('data_preprocess', self.data_preprocess_pipeline_),
            ('target_preprocess_and_estimator', target_transformer(regressor=self,
                                                                   transformer=self.target_preprocess_pipeline_)),
        ])
        return self.model_pipeline_

    def fit(self, cat_features=None):
        """Wrapper around the fit method of the scikit-learn class.

        Parameters
        ----------
        cat_features:
            Categorical features.

        Original documentation:

        """
        self.cat_features_ = cat_features if cat_features else []


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
        SkLearnExtension.__init__(
            self,
            categorical_imputer=categorical_imputer,
            continuous_imputer=continuous_imputer,
            categorical_encoder=categorical_encoder,
            handle_unknown_categories=handle_unknown_categories,
            variance_threshold=variance_threshold,
            data_scaler=data_scaler,
            categorical_type=categorical_type,
            continuous_type=continuous_type,
            target_imputer=target_imputer,
            categorical_target_encoder=categorical_target_encoder,
            categorical_target_min_frequency=categorical_target_min_frequency,
            continuous_target_scaler=continuous_target_scaler,
            categorical_target_type=categorical_target_type,
            continuous_target_type=continuous_target_type,
        )

    return init_fn


def fit_factory(cls):
    @extends(cls.fit)
    def fit_fn(self, *args, cat_features=None, **kwargs):
        SkLearnExtension.fit(self, cat_features=cat_features)
        return cls.fit(self, *args, **kwargs)

    fit_fn.__doc__ = SkLearnExtension.fit.__doc__ + cls.fit.__doc__
    return fit_fn


# noinspection PyTypeChecker
RidgeCV = type(
    'RidgeCV',
    (SKRidgeCV, SkLearnExtension),
    {
        '__init__': init_factory(SKRidgeCV),
        'fit': fit_factory(SKRidgeCV),
        '__doc__': SkLearnExtension.__doc__ + SKRidgeCV.__doc__,
    }
)
