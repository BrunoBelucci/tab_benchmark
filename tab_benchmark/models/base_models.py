from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from tab_benchmark.TransformedTargetClassifier import TransformedTargetClassifier
from tab_benchmark.preprocess import create_data_preprocess_pipeline, create_target_preprocess_pipeline


class SkLearnExtension:
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
