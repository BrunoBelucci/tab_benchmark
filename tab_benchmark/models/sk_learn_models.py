from sklearn.linear_model import (RidgeCV, ElasticNetCV, MultiTaskElasticNetCV, LassoCV, MultiTaskLassoCV,
                                  LinearRegression, LogisticRegressionCV, RidgeClassifierCV)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.ensemble import (ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor,
                              GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import NuSVC, NuSVR
from tab_benchmark.models.factories import sklearn_factory

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
