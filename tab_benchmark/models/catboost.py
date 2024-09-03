import mlflow
from catboost import CatBoostRegressor, CatBoostClassifier
from ray import tune
from ray.train import report
from tab_benchmark.models.xgboost import n_estimators_gbdt, early_stopping_patience_gbdt
from tab_benchmark.models.factories import sklearn_factory


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
        early_stopping_patience=early_stopping_patience_gbdt,
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


class ReportToRayCatboost:
    def __init__(self, eval_name, default_metric=None):
        super().__init__()
        if len(eval_name) > 1:
            self.map_default_name_to_eval_name = {f'validation_{i}': name for i, name in enumerate(eval_name)}
        else:
            self.map_default_name_to_eval_name = {'validation': eval_name[0]}
        self.map_default_name_to_eval_name['learn'] = 'train'
        self.default_metric = default_metric

    def after_iteration(self, info):
        dict_to_report = {}
        for default_name, metrics in info.metrics.items():
            our_name = self.map_default_name_to_eval_name[default_name]
            dict_to_report.update({f'{our_name}_{metric}': value[-1] for metric, value in metrics.items()})
            if self.default_metric:
                dict_to_report[f'{our_name}_default'] = metrics[self.default_metric][-1]
        report(dict_to_report)
        return True


class LogToMLFlowCatboost:
    def __init__(self, eval_name, log_every_n_steps=50, default_metric=None):
        if len(eval_name) > 1:
            self.map_default_name_to_eval_name = {f'validation_{i}': name for i, name in enumerate(eval_name)}
        else:
            self.map_default_name_to_eval_name = {'validation': eval_name[0]}
        self.map_default_name_to_eval_name['learn'] = 'train'
        self.log_every_n_steps = log_every_n_steps
        self.default_metric = default_metric

    def after_iteration(self, info):
        if info.iteration % self.log_every_n_steps != 0:
            return True
        dict_to_log = {}
        for default_name, metrics in info.metrics.items():
            our_name = self.map_default_name_to_eval_name[default_name]
            dict_to_log.update({f'{our_name}_{metric}': value[-1] for metric, value in metrics.items()})
            if self.default_metric:
                dict_to_log[f'{our_name}_default'] = metrics[self.default_metric][-1]
        dict_to_log['iteration'] = info.iteration
        mlflow.log_metrics(dict_to_log, step=info.iteration)
        return True


map_our_metric_to_catboost_metric = {
    ('logloss', 'binary_classification'): 'Logloss',
    ('logloss', 'classification'): 'MultiClass',
    ('rmse', 'regression'): 'RMSE',
    ('rmse', 'multi_regression'): 'MultiRMSE',
}


def before_fit_catboost(self, X, y, task=None, cat_features=None, cat_dims=None, n_classes=None, eval_set=None,
                        eval_name=None, report_to_ray=False, init_model=None, **args_and_kwargs):
    callbacks = args_and_kwargs.get('callbacks', [])
    fit_arguments = args_and_kwargs.copy() if args_and_kwargs else {}

    if n_classes is not None:
        if n_classes > 2:
            self.set_params(**{'classes_count': n_classes})

    eval_metric = self.get_params().get('eval_metric', None)
    if eval_metric is not None:
        eval_metric = map_our_metric_to_catboost_metric[(eval_metric, task)]
        self.set_params(**{'eval_metric': eval_metric})

    callbacks = callbacks if callbacks else []

    self.set_params(**{'cat_features': cat_features})

    if self.early_stopping_patience > 0:
        # for the moment we will leave the default early stopping callback
        self.set_params(**{'early_stopping_rounds': self.early_stopping_patience})
        # we will add a parameters to allow saving snapshots (if they are not already being used)
        save_snapshot = fit_arguments.get('save_snapshot', True)
        snapshot_file = fit_arguments.get('snapshot_file', 'model-snapshot_0.cbm')
        snapshot_interval = fit_arguments.get('snapshot_interval', 300)
        fit_arguments['save_snapshot'] = save_snapshot
        fit_arguments['snapshot_file'] = snapshot_file
        fit_arguments['snapshot_interval'] = snapshot_interval

    if self.log_to_mlflow_if_running:
        if mlflow.active_run():
            callbacks.append(LogToMLFlowCatboost(eval_name, default_metric=self.get_param('eval_metric')))

    if report_to_ray:
        callbacks.append(ReportToRayCatboost(eval_name, default_metric=self.get_param('eval_metric')))

    fit_arguments['callbacks'] = callbacks
    fit_arguments.update(dict(X=X, y=y, eval_set=eval_set, init_model=init_model))
    return fit_arguments

# catboost loosely follow the sklearn pattern, it does not inherit from BaseEstimator or the Mixin classes
TabBenchmarkCatBoostRegressor = sklearn_factory(
    CatBoostRegressor,
    has_early_stopping=True,
    default_values={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'int32',
        'data_scaler': None,
    },
    map_task_to_default_values={
        'regression': {'loss_function': 'RMSE', 'eval_metric': 'rmse'},
        'multi_regression': {'loss_function': 'MultiRMSE', 'eval_metric': 'rmse'},
    },
    before_fit_method=before_fit_catboost,
    extra_dct={
        'n_jobs': n_jobs_property,
        'output_dir': output_dir_property,
        'create_search_space': staticmethod(create_search_space_catboost),
        'get_recommended_params': staticmethod(get_recommended_params_catboost),
    }
)

TabBenchmarkCatBoostClassifier = sklearn_factory(
    CatBoostClassifier,
    has_early_stopping=True,
    default_values={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'int32',
        'data_scaler': None,
    },
    map_task_to_default_values={
        'classification': {'loss_function': 'MultiClass', 'eval_metric': 'logloss'},
        'binary_classification': {'loss_function': 'Logloss', 'eval_metric': 'logloss'},
    },
    before_fit_method=before_fit_catboost,
    extra_dct={
        'n_jobs': n_jobs_property,
        'output_dir': output_dir_property,
        'create_search_space': staticmethod(create_search_space_catboost),
        'get_recommended_params': staticmethod(get_recommended_params_catboost),
    }
)
