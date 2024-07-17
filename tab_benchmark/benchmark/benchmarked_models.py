from tab_benchmark.models.sk_learn_models import RidgeCV, RidgeClassifierCV, LinearRegression, LogisticRegressionCV, \
    LassoCV, MultiTaskLassoCV, ElasticNetCV, MultiTaskElasticNetCV, DecisionTreeClassifier, DecisionTreeRegressor, \
    ExtraTreeClassifier, ExtraTreeRegressor, ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor, \
    GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier, KernelRidge, NuSVC, NuSVR

models_dict = {
    # Linear models
    'LinearRegression': (LinearRegression, {}),
    'LogisticRegressionCV': (LogisticRegressionCV, {}),
    'RidgeCV': (RidgeCV, {}),
    'RidgeClassifierCV': (RidgeClassifierCV, {}),
    'LassoCV': (LassoCV, {}),
    'MultiTaskLassoCV': (MultiTaskLassoCV, {}),
    'ElasticNetCV': (ElasticNetCV, {}),
    'MultiTaskElasticNetCV': (MultiTaskElasticNetCV, {}),
    # Tree models
    'DecisionTreeClassifier': (DecisionTreeClassifier, {}),
    'DecisionTreeRegressor': (DecisionTreeRegressor, {}),
    'ExtraTreeClassifier': (ExtraTreeClassifier, {}),
    'ExtraTreeRegressor': (ExtraTreeRegressor, {}),
    # Ensemble models
    'ExtraTreesRegressor': (ExtraTreesRegressor, {}),
    'ExtraTreesClassifier': (ExtraTreesClassifier, {}),
    'GradientBoostingRegressor': (GradientBoostingRegressor, {}),
    'GradientBoostingClassifier': (GradientBoostingClassifier, {}),
    'RandomForestRegressor': (RandomForestRegressor, {}),
    'RandomForestClassifier': (RandomForestClassifier, {}),
    # Kernel models
    'KernelRidge': (KernelRidge, {}),
    # SVM models
    'NuSVC': (NuSVC, {}),
    'NuSVR': (NuSVR, {}),
}
