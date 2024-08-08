import mlflow
from xgboost import XGBClassifier as OriginalXGBClassifier, XGBRegressor as OriginalXGBRegressor, XGBModel
from xgboost.callback import TrainingCallback, _Model
from tab_benchmark.models.factories import TabBenchmarkModelFactory, fn_to_add_auto_early_stopping
from tab_benchmark.utils import extends
from ray import tune
from ray.train import report


n_estimators_gbdt = 10000
early_stopping_patience_gbdt = 100


def create_search_space_xgboost():
    # In Well tunned...
    # Not tunning n_estimators following discussion at
    # https://openreview.net/forum?id=Fp7__phQszn&noteId=Z7Y_qxwDjiM
    search_space = dict(
        learning_rate=tune.loguniform(1e-3, 1.0),
        reg_lambda=tune.loguniform(1e-10, 1),
        reg_alpha=tune.loguniform(1e-10, 1.0),
        gamma=tune.loguniform(1e-1, 1.0),
        colsample_bylevel=tune.uniform(0.1, 1),
        colsample_bynode=tune.uniform(0.1, 1),
        colsample_bytree=tune.uniform(0.1, 1),
        max_depth=tune.randint(1, 20),
        max_delta_step=tune.randint(0, 10),
        min_child_weight=tune.loguniform(0.1, 20),
        subsample=tune.uniform(0.01, 1),
    )
    default_values = dict(
        learning_rate=0.3,
        reg_lambda=1.0,
        reg_alpha=1e-10,
        gamma=1e-1,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        colsample_bytree=1.0,
        max_depth=6,
        max_delta_step=0,
        min_child_weight=1.0,
        subsample=1.0,
    )
    return search_space, default_values


def get_recommended_params_xgboost():
    default_values_from_search_space = create_search_space_xgboost()[1]
    default_values_from_search_space.update(dict(
        n_estimators=n_estimators_gbdt,
        auto_early_stopping=True,
        early_stopping_rounds=early_stopping_patience_gbdt,
    ))
    return default_values_from_search_space


class ReportToRayXGBoost(TrainingCallback):
    def __init__(self, eval_name, default_metric=None):
        super().__init__()
        self.map_default_name_to_eval_name = {f'validation_{i}': name for i, name in enumerate(eval_name)}
        self.default_metric = default_metric

    def after_iteration(self, model: _Model, epoch: int, evals_log):
        for default_name, metrics in evals_log.items():
            our_name = self.map_default_name_to_eval_name[default_name]
            dict_to_report = {f'{our_name}_{metric}': value[-1] for metric, value in metrics.items()}
            if self.default_metric:
                dict_to_report[f'{our_name}_default'] = metrics[self.default_metric][-1]
            report(dict_to_report)


class LogToMLFlowXGBoost(TrainingCallback):
    def __init__(self, eval_name, log_every_n_steps=50, default_metric=None):
        super().__init__()
        self.map_default_name_to_eval_name = {f'validation_{i}': name for i, name in enumerate(eval_name)}
        self.log_every_n_steps = log_every_n_steps
        self.default_metric = default_metric

    def after_iteration(self, model: _Model, epoch: int, evals_log):
        if epoch % self.log_every_n_steps != 0:
            return
        for default_name, metrics in evals_log.items():
            our_name = self.map_default_name_to_eval_name[default_name]
            dict_to_log = {f'{our_name}_{metric}': value[-1] for metric, value in metrics.items()}
            if self.default_metric:
                dict_to_log[f'{our_name}_default'] = metrics[self.default_metric][-1]
            dict_to_log['epoch'] = epoch
            mlflow.log_metrics(dict_to_log, step=epoch)


def before_fit_xgboost(self, extra_arguments, **fit_arguments):
    cat_features = extra_arguments.get('cat_features')
    report_to_ray = extra_arguments.get('report_to_ray')
    eval_name = extra_arguments.get('eval_name')
    eval_set = fit_arguments.get('eval_set')
    X_train = fit_arguments.get('X')
    y_train = fit_arguments.get('y')
    eval_set.insert(0, (X_train, y_train))
    eval_name.insert(0, 'train')
    fit_arguments['eval_set'] = eval_set
    if cat_features is not None:
        self.set_params(**{'enable_categorical': True})
    if self.callbacks is None:
        self.callbacks = []
    if report_to_ray:
        self.callbacks.append(ReportToRayXGBoost(eval_name, default_metric=self.eval_metric))
    if self.log_to_mlflow_if_running:
        if mlflow.active_run():
            self.callbacks.append(LogToMLFlowXGBoost(eval_name, default_metric=self.eval_metric))
    return fit_arguments


# Just to get the parameters of the XGBModel, because XGBClassifier and XGBRegressor do not show them
# This will initialize exactly the same way as XGBClassifier and XGBRegressor.
class XGBClassifier(OriginalXGBClassifier):
    __doc__ = OriginalXGBClassifier.__doc__

    @extends(XGBModel.__init__, map_default_values_change={'objective': 'binary:logistic'})
    def __init__(self, *args, **kwargs):
        XGBModel.__init__(self, *args, **kwargs)


class XGBRegressor(OriginalXGBRegressor):
    __doc__ = OriginalXGBRegressor.__doc__

    @extends(XGBModel.__init__, map_default_values_change={'objective': 'reg:squarederror'})
    def __init__(self, *args, **kwargs):
        XGBModel.__init__(self, *args, **kwargs)


# Now we wrap them with our API
XGBClassifier = TabBenchmarkModelFactory.from_sk_cls(
    XGBClassifier,
    map_task_to_default_values={
        'classification': {'objective': 'multi:softmax', 'eval_metric': 'mlogloss'},
        'binary_classification': {'objective': 'binary:logistic', 'eval_metric': 'logloss'},
    },
    has_auto_early_stopping=True,
    extended_init_kwargs={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'category',
        'data_scaler': None,
    },
    extra_dct={
        'create_search_space': staticmethod(create_search_space_xgboost),
        'get_recommended_params': staticmethod(get_recommended_params_xgboost),
        'before_fit': before_fit_xgboost,
    }
)

XGBRegressor = TabBenchmarkModelFactory.from_sk_cls(
    XGBRegressor,
    map_default_values_change={
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    },
    has_auto_early_stopping=True,
    extended_init_kwargs={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'category',
        'data_scaler': None,
    },
    extra_dct={
        'create_search_space': staticmethod(create_search_space_xgboost),
        'get_recommended_params': staticmethod(get_recommended_params_xgboost),
        'before_fit': before_fit_xgboost,
    }
)
