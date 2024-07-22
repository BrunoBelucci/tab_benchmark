from sklearn.linear_model import (RidgeCV, ElasticNetCV, MultiTaskElasticNetCV, LassoCV, MultiTaskLassoCV,
                                  LinearRegression, LogisticRegressionCV, RidgeClassifierCV)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.ensemble import (ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor,
                              GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import NuSVC, NuSVR
from tab_benchmark.models.factories import TabBenchmarkModelFactory

# Linear models
LinearRegression = TabBenchmarkModelFactory.from_sk_cls(LinearRegression, {})
LogisticRegressionCV = TabBenchmarkModelFactory.from_sk_cls(LogisticRegressionCV, {})
RidgeCV = TabBenchmarkModelFactory.from_sk_cls(RidgeCV, {})
RidgeClassifierCV = TabBenchmarkModelFactory.from_sk_cls(RidgeClassifierCV, {})
LassoCV = TabBenchmarkModelFactory.from_sk_cls(LassoCV, {})
MultiTaskLassoCV = TabBenchmarkModelFactory.from_sk_cls(MultiTaskLassoCV, {})
ElasticNetCV = TabBenchmarkModelFactory.from_sk_cls(ElasticNetCV, {})
MultiTaskElasticNetCV = TabBenchmarkModelFactory.from_sk_cls(MultiTaskElasticNetCV, {})


# Tree models
DecisionTreeClassifier = TabBenchmarkModelFactory.from_sk_cls(DecisionTreeClassifier, {})
DecisionTreeRegressor = TabBenchmarkModelFactory.from_sk_cls(DecisionTreeRegressor, {})
ExtraTreeClassifier = TabBenchmarkModelFactory.from_sk_cls(ExtraTreeClassifier, {})
ExtraTreeRegressor = TabBenchmarkModelFactory.from_sk_cls(ExtraTreeRegressor, {})

# Ensemble models
ExtraTreesRegressor = TabBenchmarkModelFactory.from_sk_cls(ExtraTreesRegressor, {})
ExtraTreesClassifier = TabBenchmarkModelFactory.from_sk_cls(ExtraTreesClassifier, {})
GradientBoostingRegressor = TabBenchmarkModelFactory.from_sk_cls(GradientBoostingRegressor, {})
GradientBoostingClassifier = TabBenchmarkModelFactory.from_sk_cls(GradientBoostingClassifier, {})
RandomForestRegressor = TabBenchmarkModelFactory.from_sk_cls(RandomForestRegressor, {})
RandomForestClassifier = TabBenchmarkModelFactory.from_sk_cls(RandomForestClassifier, {})

# Kernel models
KernelRidge = TabBenchmarkModelFactory.from_sk_cls(KernelRidge, {})

# SVM models
NuSVC = TabBenchmarkModelFactory.from_sk_cls(NuSVC, {})
NuSVR = TabBenchmarkModelFactory.from_sk_cls(NuSVR, {})

all_models = [LinearRegression, LogisticRegressionCV, RidgeCV, RidgeClassifierCV, LassoCV, MultiTaskLassoCV,
              ElasticNetCV, MultiTaskElasticNetCV, DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier,
              ExtraTreeRegressor, ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor,
              GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier, KernelRidge, NuSVC, NuSVR]
