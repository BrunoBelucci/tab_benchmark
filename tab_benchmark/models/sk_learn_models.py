from sklearn.linear_model import (RidgeCV, ElasticNetCV, MultiTaskElasticNetCV, LassoCV, MultiTaskLassoCV,
                                  LinearRegression, LogisticRegressionCV, RidgeClassifierCV)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.ensemble import (ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor,
                              GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import NuSVC, NuSVR
from tab_benchmark.models.factories import SimpleSkLearnFactory

# Linear models
LinearRegression = SimpleSkLearnFactory.from_sk_cls(LinearRegression, {})
LogisticRegressionCV = SimpleSkLearnFactory.from_sk_cls(LogisticRegressionCV, {})
RidgeCV = SimpleSkLearnFactory.from_sk_cls(RidgeCV, {})
RidgeClassifierCV = SimpleSkLearnFactory.from_sk_cls(RidgeClassifierCV, {})
LassoCV = SimpleSkLearnFactory.from_sk_cls(LassoCV, {})
MultiTaskLassoCV = SimpleSkLearnFactory.from_sk_cls(MultiTaskLassoCV, {})
ElasticNetCV = SimpleSkLearnFactory.from_sk_cls(ElasticNetCV, {})
MultiTaskElasticNetCV = SimpleSkLearnFactory.from_sk_cls(MultiTaskElasticNetCV, {})


# Tree models
DecisionTreeClassifier = SimpleSkLearnFactory.from_sk_cls(DecisionTreeClassifier, {})
DecisionTreeRegressor = SimpleSkLearnFactory.from_sk_cls(DecisionTreeRegressor, {})
ExtraTreeClassifier = SimpleSkLearnFactory.from_sk_cls(ExtraTreeClassifier, {})
ExtraTreeRegressor = SimpleSkLearnFactory.from_sk_cls(ExtraTreeRegressor, {})

# Ensemble models
ExtraTreesRegressor = SimpleSkLearnFactory.from_sk_cls(ExtraTreesRegressor, {})
ExtraTreesClassifier = SimpleSkLearnFactory.from_sk_cls(ExtraTreesClassifier, {})
GradientBoostingRegressor = SimpleSkLearnFactory.from_sk_cls(GradientBoostingRegressor, {})
GradientBoostingClassifier = SimpleSkLearnFactory.from_sk_cls(GradientBoostingClassifier, {})
RandomForestRegressor = SimpleSkLearnFactory.from_sk_cls(RandomForestRegressor, {})
RandomForestClassifier = SimpleSkLearnFactory.from_sk_cls(RandomForestClassifier, {})

# Kernel models
KernelRidge = SimpleSkLearnFactory.from_sk_cls(KernelRidge, {})

# SVM models
NuSVC = SimpleSkLearnFactory.from_sk_cls(NuSVC, {})
NuSVR = SimpleSkLearnFactory.from_sk_cls(NuSVR, {})
