from sklearn.linear_model import (RidgeCV, ElasticNetCV, MultiTaskElasticNetCV, LassoCV, MultiTaskLassoCV,
                                  LinearRegression, LogisticRegressionCV, RidgeClassifierCV)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.ensemble import (ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor,
                              GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import NuSVC, NuSVR
from tab_benchmark.models.base_models import SkLearnExtension
from tab_benchmark.models.factories import init_factory, fit_factory, init_doc


class SimpleSkLearnFactory(type):
    @classmethod  # to be cleaner (not change the signature of __new__)
    def from_sk_cls(cls, sk_cls, extended_init_kwargs=None):
        extended_init_kwargs = extended_init_kwargs if extended_init_kwargs else {}
        name = sk_cls.__name__
        dct = {
            '__init__': init_factory(sk_cls, **extended_init_kwargs),
            'fit': fit_factory(sk_cls),
            '__doc__': init_doc + sk_cls.__doc__
        }
        return type(name, (sk_cls, SkLearnExtension), dct)


class TaskDependentSkLearnFactory(type):
    @classmethod
    def from_multiple_sk_cls(cls, name, map_sk_cls_to_task, extended_init_kwargs=None):
        extended_init_kwargs = extended_init_kwargs if extended_init_kwargs else {}
        return cls(name, (), {'extended_init_kwargs': extended_init_kwargs, 'map_sk_cls_to_task': map_sk_cls_to_task})

    def __call__(cls, task, *args, **kwargs):
        if task not in cls.map_sk_cls_to_task:
            raise ValueError(f"Task '{task}' not registered. Available tasks: {list(cls.map_sk_cls_to_task.keys())}")
        sk_cls = cls.map_sk_cls_to_task[task]
        return SimpleSkLearnFactory.from_sk_cls(sk_cls, cls.extended_init_kwargs)(*args, **kwargs)


# Linear models
LinearRegression = SimpleSkLearnFactory.from_sk_cls(LinearRegression, {})
LogisticRegressionCV = SimpleSkLearnFactory.from_sk_cls(LogisticRegressionCV, {})
RidgeCV = TaskDependentSkLearnFactory.from_multiple_sk_cls('RidgeCV', {
    'classification': RidgeClassifierCV,
    'binary_classification': RidgeClassifierCV,
    'regression': RidgeCV,
    'multi_regression': RidgeCV,
})
LassoCV = TaskDependentSkLearnFactory.from_multiple_sk_cls('LassoCV', {
    'classification': MultiTaskLassoCV,
    'binary_classification': LassoCV,
    'regression': LassoCV,
    'multi_regression': MultiTaskLassoCV,
})
ElasticNetCV = TaskDependentSkLearnFactory.from_multiple_sk_cls('ElasticNetCV', {
    'classification': MultiTaskElasticNetCV,
    'binary_classification': ElasticNetCV,
    'regression': ElasticNetCV,
    'multi_regression': MultiTaskElasticNetCV,
})

# Tree models
DecisionTree = TaskDependentSkLearnFactory.from_multiple_sk_cls('DecisionTree', {
    'classification': DecisionTreeClassifier,
    'binary_classification': DecisionTreeClassifier,
    'regression': DecisionTreeRegressor,
    'multi_regression': DecisionTreeRegressor,
})
ExtraTree = TaskDependentSkLearnFactory.from_multiple_sk_cls('ExtraTree', {
    'classification': ExtraTreeClassifier,
    'binary_classification': ExtraTreeClassifier,
    'regression': ExtraTreeRegressor,
    'multi_regression': ExtraTreeRegressor,
})

# Ensemble models
ExtraTrees = TaskDependentSkLearnFactory.from_multiple_sk_cls('ExtraTrees', {
    'classification': ExtraTreesClassifier,
    'binary_classification': ExtraTreesClassifier,
    'regression': ExtraTreesRegressor,
    'multi_regression': ExtraTreesRegressor,
})
GradientBoosting = TaskDependentSkLearnFactory.from_multiple_sk_cls('GradientBoosting', {
    'classification': GradientBoostingClassifier,
    'binary_classification': GradientBoostingClassifier,
    'regression': GradientBoostingRegressor,
    'multi_regression': GradientBoostingRegressor,
})
RandomForest = TaskDependentSkLearnFactory.from_multiple_sk_cls('RandomForest', {
    'classification': RandomForestClassifier,
    'binary_classification': RandomForestClassifier,
    'regression': RandomForestRegressor,
    'multi_regression': RandomForestRegressor,
})

# Kernel models
KernelRidge = SimpleSkLearnFactory.from_sk_cls(KernelRidge, {})

# SVM models
NuSVM = TaskDependentSkLearnFactory.from_multiple_sk_cls('RandomForest', {
    'classification': NuSVC,
    'binary_classification': NuSVC,
    'regression': NuSVR,
    'multi_regression': NuSVR,
})
