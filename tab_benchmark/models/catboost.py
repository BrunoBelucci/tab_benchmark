from catboost import CatBoostRegressor as OriginalCatBoostRegressor, CatBoostClassifier as OriginalCatBoostClassifier
from ray import tune
from tab_benchmark.models.xgboost import n_estimators_gbdt, early_stopping_patience_gbdt
from tab_benchmark.models.factories import TabBenchmarkModelFactory, fn_to_add_auto_early_stopping


def create_search_space_catboost():
    # In Well tunned...
    # Not tunning n_estimators following discussion at
    # https://openreview.net/forum?id=Fp7__phQszn&noteId=Z7Y_qxwDjiM
    search_space = dict(
        learning_rate=tune.loguniform(1e-6, 1.0),
        random_strength=tune.randint(1, 20),
        one_hot_max_size=tune.randint(0, 25),
        l2_leaf_reg=tune.uniform(1, 10),
        bagging_temperature=tune.uniform(0, 1),
        leaf_estimation_iterations=tune.randint(1, 10),
    )
    default_values = dict(
        learning_rate=0.03,
        random_strength=1,
        one_hot_max_size=2,
        l2_leaf_reg=3.0,
        bagging_temperature=1.0,
        leaf_estimation_iterations=1,
    )
    return search_space, default_values


def get_recommended_params_catboost():
    default_values_from_search_space = create_search_space_catboost()[1]
    default_values_from_search_space.update(dict(
        n_estimators=n_estimators_gbdt,
        auto_early_stopping=True,
        early_stopping_rounds=early_stopping_patience_gbdt,
    ))
    return default_values_from_search_space


def n_jobs_get(self):
    if not hasattr(self, '_n_jobs'):
        self._n_jobs = self.get_params().get('thread_count', 1)
    return self._n_jobs


def n_jobs_set(self, value):
    self._n_jobs = value
    self.set_params(**{'thread_count': value})


n_jobs_property = property(n_jobs_get, n_jobs_set)


def output_dir_get(self):
    if not hasattr(self, '_output_dir'):
        self._output_dir = self.get_params().get('train_dir', None)
    return self._output_dir


def output_dir_set(self, value):
    self._output_dir = value
    self.set_params(**{'train_dir': value})


output_dir_property = property(output_dir_get, output_dir_set)


def before_fit_catboost(self, extra_arguments, **fit_arguments):
    cat_features = extra_arguments.get('cat_features')
    self.set_params(**{'cat_features': cat_features})
    return fit_arguments


CatBoostRegressor = TabBenchmarkModelFactory.from_sk_cls(
    OriginalCatBoostRegressor,
    map_task_to_default_values={
        'regression': {'loss_function': 'RMSE', 'eval_metric': 'RMSE'},
        'multi_regression': {'loss_function': 'MultiRMSE', 'eval_metric': 'MultiRMSE'},
    },
    has_auto_early_stopping=True,
    extended_init_kwargs={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'int32',
        'data_scaler': None,
    },
    extra_dct={
        'n_jobs': n_jobs_property,
        'output_dir': output_dir_property,
        'create_search_space': staticmethod(create_search_space_catboost),
        'get_recommended_params': staticmethod(get_recommended_params_catboost),
        'before_fit': before_fit_catboost
    }
)


CatBoostClassifier = TabBenchmarkModelFactory.from_sk_cls(
    OriginalCatBoostClassifier,
    map_task_to_default_values={
        'classification': {'loss_function': 'MultiClass', 'eval_metric': 'MultiClass'},
        'binary_classification': {'loss_function': 'Logloss', 'eval_metric': 'Logloss'},
    },
    has_auto_early_stopping=True,
    extended_init_kwargs={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'int32',
        'data_scaler': None,
    },
    extra_dct={
        'n_jobs': n_jobs_property,
        'output_dir': output_dir_property,
        'create_search_space': staticmethod(create_search_space_catboost),
        'get_recommended_params': staticmethod(get_recommended_params_catboost),
        'before_fit': before_fit_catboost
    }
)
