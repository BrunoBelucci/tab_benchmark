from sklearn.linear_model import (RidgeCV, ElasticNetCV, MultiTaskElasticNetCV, LassoCV, MultiTaskLassoCV,
                                  LinearRegression, LogisticRegressionCV, RidgeClassifierCV)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.ensemble import (ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor,
                              GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import NuSVC, NuSVR
from tab_benchmark.models.factories import sklearn_factory

# Linear models
LinearRegression = sklearn_factory(LinearRegression)
LogisticRegressionCV = sklearn_factory(LogisticRegressionCV)
RidgeCV = sklearn_factory(RidgeCV)
RidgeClassifierCV = sklearn_factory(RidgeClassifierCV)
LassoCV = sklearn_factory(LassoCV)
MultiTaskLassoCV = sklearn_factory(MultiTaskLassoCV)
ElasticNetCV = sklearn_factory(ElasticNetCV)
MultiTaskElasticNetCV = sklearn_factory(MultiTaskElasticNetCV)


# Tree models
DecisionTreeClassifier = sklearn_factory(DecisionTreeClassifier)
DecisionTreeRegressor = sklearn_factory(DecisionTreeRegressor)
ExtraTreeClassifier = sklearn_factory(ExtraTreeClassifier)
ExtraTreeRegressor = sklearn_factory(ExtraTreeRegressor)

# Ensemble models
ExtraTreesRegressor = sklearn_factory(ExtraTreesRegressor)
ExtraTreesClassifier = sklearn_factory(ExtraTreesClassifier)
GradientBoostingRegressor = sklearn_factory(GradientBoostingRegressor)
GradientBoostingClassifier = sklearn_factory(GradientBoostingClassifier)
RandomForestRegressor = sklearn_factory(RandomForestRegressor)
RandomForestClassifier = sklearn_factory(RandomForestClassifier)

# Kernel models
KernelRidge = sklearn_factory(KernelRidge)

# SVM models
NuSVC = sklearn_factory(NuSVC)
NuSVR = sklearn_factory(NuSVR)

all_models = [LinearRegression, LogisticRegressionCV, RidgeCV, RidgeClassifierCV, LassoCV, MultiTaskLassoCV,
              ElasticNetCV, MultiTaskElasticNetCV, DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier,
              ExtraTreeRegressor, ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor,
              GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier, KernelRidge, NuSVC, NuSVR]
