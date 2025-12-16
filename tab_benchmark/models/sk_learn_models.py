from __future__ import annotations
from inspect import signature
from warnings import warn
from sklearn.linear_model import (RidgeCV, ElasticNetCV, MultiTaskElasticNetCV, LassoCV, MultiTaskLassoCV,
                                  LinearRegression, LogisticRegressionCV, RidgeClassifierCV)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.ensemble import (ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor,
                              GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import NuSVC, NuSVR
from tab_benchmark.models.mixins import (PreprocessingMixin, TabBenchmarkModel, merge_and_apply_signature,
                                         merge_signatures)
from sklearn.base import ClassifierMixin


class SkLearnMixin(PreprocessingMixin):
    pass


def sklearn_factory(sklearn_cls):

    class TabBenchmarkSklearn(SkLearnMixin, TabBenchmarkModel, sklearn_cls):
        @merge_and_apply_signature(merge_signatures(signature(sklearn_cls.__init__),
                                                    signature(PreprocessingMixin.__init__)))
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def before_fit(self, X, y, task=None, **kwargs):
            """Augment underrepresented classes for models with CV to ensure proper stratified splits."""
            # Only augment for classification tasks with CV models
            if hasattr(self, 'cv') and isinstance(self, ClassifierMixin):
                import numpy as np
                import pandas as pd
                
                # Convert y to array for analysis
                y_array = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
                if hasattr(y_array, 'ravel'):
                    y_array = y_array.ravel()
                
                # Check minimum class size
                unique_classes, class_counts = np.unique(y_array, return_counts=True)
                min_class_size = class_counts.min()
                min_samples_needed = 5  # Minimum for proper 5-fold stratified CV
                
                if min_class_size < min_samples_needed:
                    warn(f'Smallest class has only {min_class_size} samples. '
                         f'Augmenting underrepresented classes to {min_samples_needed} samples '
                         f'for proper cross-validation.')
                    
                    # Get indices to duplicate for each underrepresented class
                    augment_indices = []
                    for cls, count in zip(unique_classes, class_counts):
                        if count < min_samples_needed:
                            # Find indices of this class
                            cls_indices = np.where(y_array == cls)[0]
                            # Calculate how many samples to add
                            n_to_add = min_samples_needed - count
                            # Duplicate samples (cycle through if needed)
                            duplicates = np.tile(cls_indices, (n_to_add // count) + 1)[:n_to_add]
                            augment_indices.extend(duplicates.tolist())
                    
                    # Augment X and y with duplicated samples
                    if len(augment_indices) > 0:
                        if isinstance(X, pd.DataFrame):
                            X = pd.concat([X, X.iloc[augment_indices]], axis=0, ignore_index=True)
                        else:
                            X = np.vstack([X, X[augment_indices]])
                        
                        if isinstance(y, pd.Series):
                            y = pd.concat([y, y.iloc[augment_indices]], axis=0, ignore_index=True)
                        elif isinstance(y, pd.DataFrame):
                            y = pd.concat([y, y.iloc[augment_indices]], axis=0, ignore_index=True)
                        else:
                            if isinstance(y, np.ndarray) and y.ndim == 1:
                                y = np.concatenate([y, y[augment_indices]])
                            else:
                                y = np.vstack([y, y[augment_indices]])
            
            return X, y, kwargs

        def fit(self, X, y, ignore_extra_kwargs=True, **kwargs):
            if ignore_extra_kwargs:
                accepted_kwargs = signature(sklearn_cls.fit).parameters.keys()
                extra_kwargs = {k: v for k, v in kwargs.items() if k not in accepted_kwargs}
                if extra_kwargs:
                    warn(f'Ignoring extra kwargs: {extra_kwargs}')
                kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}
            return super().fit(X, y, **kwargs)

        @staticmethod
        def create_search_space():
            raise NotImplementedError

        @staticmethod
        def get_recommended_params():
            raise NotImplementedError

    TabBenchmarkSklearn.__name__ = f'TabBenchmark{sklearn_cls.__name__}'
    return TabBenchmarkSklearn

# Linear models
TabBenchmarkLinearRegression = sklearn_factory(LinearRegression)
TabBenchmarkLogisticRegressionCV = sklearn_factory(LogisticRegressionCV)
TabBenchmarkRidgeCV = sklearn_factory(RidgeCV)
TabBenchmarkRidgeClassifierCV = sklearn_factory(RidgeClassifierCV)
TabBenchmarkLassoCV = sklearn_factory(LassoCV)
TabBenchmarkMultiTaskLassoCV = sklearn_factory(MultiTaskLassoCV)
TabBenchmarkElasticNetCV = sklearn_factory(ElasticNetCV)
TabBenchmarkMultiTaskElasticNetCV = sklearn_factory(MultiTaskElasticNetCV)


# Tree models
TabBenchmarkDecisionTreeClassifier = sklearn_factory(DecisionTreeClassifier)
TabBenchmarkDecisionTreeRegressor = sklearn_factory(DecisionTreeRegressor)
TabBenchmarkExtraTreeClassifier = sklearn_factory(ExtraTreeClassifier)
TabBenchmarkExtraTreeRegressor = sklearn_factory(ExtraTreeRegressor)

# Ensemble models
TabBenchmarkExtraTreesRegressor = sklearn_factory(ExtraTreesRegressor)
TabBenchmarkExtraTreesClassifier = sklearn_factory(ExtraTreesClassifier)
TabBenchmarkGradientBoostingRegressor = sklearn_factory(GradientBoostingRegressor)
TabBenchmarkGradientBoostingClassifier = sklearn_factory(GradientBoostingClassifier)
TabBenchmarkRandomForestRegressor = sklearn_factory(RandomForestRegressor)
TabBenchmarkRandomForestClassifier = sklearn_factory(RandomForestClassifier)

# Kernel models
TabBenchmarkKernelRidge = sklearn_factory(KernelRidge)

# SVM models
TabBenchmarkNuSVC = sklearn_factory(NuSVC)
TabBenchmarkNuSVR = sklearn_factory(NuSVR)

all_models = [TabBenchmarkLinearRegression, TabBenchmarkLogisticRegressionCV, TabBenchmarkRidgeCV,
              TabBenchmarkRidgeClassifierCV, TabBenchmarkLassoCV, TabBenchmarkMultiTaskLassoCV,
              TabBenchmarkElasticNetCV, TabBenchmarkMultiTaskElasticNetCV, TabBenchmarkDecisionTreeClassifier,
              TabBenchmarkDecisionTreeRegressor, TabBenchmarkExtraTreeClassifier, TabBenchmarkExtraTreeRegressor,
              TabBenchmarkExtraTreesRegressor, TabBenchmarkExtraTreesClassifier, TabBenchmarkGradientBoostingRegressor,
              TabBenchmarkGradientBoostingClassifier, TabBenchmarkRandomForestRegressor,
              TabBenchmarkRandomForestClassifier, TabBenchmarkKernelRidge, TabBenchmarkNuSVC, TabBenchmarkNuSVR]
