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


class SkLearnMixin(PreprocessingMixin):
    pass


def sklearn_factory(sklearn_cls):

    class TabBenchmarkSklearn(SkLearnMixin, TabBenchmarkModel, sklearn_cls):
        @merge_and_apply_signature(merge_signatures(signature(sklearn_cls.__init__),
                                                    signature(PreprocessingMixin.__init__)))
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def fit(self, X, y, ignore_extra_kwargs=True, **kwargs):
            if ignore_extra_kwargs:
                accepted_kwargs = signature(sklearn_cls.fit).parameters.keys()
                extra_kwargs = {k: v for k, v in kwargs.items() if k not in accepted_kwargs}
                if extra_kwargs:
                    warn(f'Ignoring extra kwargs: {extra_kwargs}')
                kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}
            super().fit(X, y, **kwargs)

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
