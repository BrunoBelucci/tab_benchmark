from __future__ import annotations
import os
import pickle
from copy import deepcopy
from typing import Optional, cast
import mlflow
import numpy
from xgboost import XGBClassifier as OriginalXGBClassifier, XGBRegressor as OriginalXGBRegressor, XGBModel, collective
from xgboost.callback import (TrainingCallback, _Model, TrainingCheckPoint as OriginalTrainingCheckPoint,
                              EarlyStopping as OriginalEarlyStopping, _Score, _ScoreList)
from tab_benchmark.models.factories import TabBenchmarkModelFactory
from tab_benchmark.utils import extends, get_metric_fn
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
        early_stopping_patience=early_stopping_patience_gbdt,
    ))
    return default_values_from_search_space


class ReportToRayXGBoost(TrainingCallback):
    def __init__(self, eval_name, default_metric=None):
        super().__init__()
        self.map_default_name_to_eval_name = {f'validation_{i}': name for i, name in enumerate(eval_name)}
        self.default_metric = default_metric

    def after_iteration(self, model: _Model, epoch: int, evals_log):
        dict_to_report = {}
        for default_name, metrics in evals_log.items():
            our_name = self.map_default_name_to_eval_name[default_name]
            dict_to_report.update({f'{our_name}_{metric}': value[-1] for metric, value in metrics.items()})
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
        dict_to_log = {'epoch': epoch}
        for default_name, metrics in evals_log.items():
            our_name = self.map_default_name_to_eval_name[default_name]
            dict_to_log.update({f'{our_name}_{metric}': value[-1] for metric, value in metrics.items()})
            if self.default_metric:
                dict_to_log[f'{our_name}_default'] = metrics[self.default_metric][-1]
        mlflow.log_metrics(dict_to_log)


def remove_old_models(path, name, extension, save_top_k):
    # get saved models
    models = [
        f
        for f in os.listdir(path)
        if f.startswith(name) and f.endswith(extension)
    ]

    # sort by iteration
    models.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))

    # remove old models
    if len(models) + 1 >= save_top_k:
        for f in models[:save_top_k + 1 or None]:  # we keep the last one (None for case 0)
            os.remove(os.path.join(path, f))


class EarlyStopping(OriginalEarlyStopping):
    # Same as original early stopping, but if the model has been trained before, we should use the best score
    # from the model.
    def _update_rounds(
            self, score: _Score, name: str, metric: str, model: _Model, epoch: int
    ) -> bool:
        def get_s(value: _Score) -> float:
            """get score if it's cross validation history."""
            return value[0] if isinstance(value, tuple) else value

        def maximize(new: _Score, best: _Score) -> bool:
            """New score should be greater than the old one."""
            return numpy.greater(get_s(new) - self._min_delta, get_s(best))

        def minimize(new: _Score, best: _Score) -> bool:
            """New score should be lesser than the old one."""
            return numpy.greater(get_s(best) - self._min_delta, get_s(new))

        if self.maximize is None:
            # Just to be compatibility with old behavior before 1.3.  We should let
            # user to decide.
            maximize_metrics = (
                "auc",
                "aucpr",
                "pre",
                "pre@",
                "map",
                "ndcg",
                "auc@",
                "aucpr@",
                "map@",
                "ndcg@",
            )
            if metric != "mape" and any(metric.startswith(x) for x in maximize_metrics):
                self.maximize = True
            else:
                self.maximize = False

        if self.maximize:
            improve_op = maximize
        else:
            improve_op = minimize

        if not self.stopping_history and hasattr(model, "best_score"):
            # If model has been trained before, we should use the best score
            # from the model.
            self.best_scores[name] = {}
            self.best_scores[name][metric] = [model.best_score]
            self.stopping_history[name] = {}
            self.stopping_history[name][metric] = cast(_ScoreList, [model.best_score])

        if not self.stopping_history:  # First round
            self.current_rounds = 0
            self.stopping_history[name] = {}
            self.stopping_history[name][metric] = cast(_ScoreList, [score])
            self.best_scores[name] = {}
            self.best_scores[name][metric] = [score]
            model.set_attr(best_score=str(score), best_iteration=str(epoch))
        elif not improve_op(score, self.best_scores[name][metric][-1]):
            # Not improved
            self.stopping_history[name][metric].append(score)  # type: ignore
            self.current_rounds += 1
        else:  # Improved
            self.stopping_history[name][metric].append(score)  # type: ignore
            self.best_scores[name][metric].append(score)
            record = self.stopping_history[name][metric][-1]
            model.set_attr(best_score=str(record), best_iteration=str(epoch))
            self.current_rounds = 0  # reset

        if self.current_rounds >= self.rounds:
            # Should stop
            return True
        return False


class TrainingCheckPoint(TrainingCallback):
    # Adapt from original TrainingCheckPoint to save only the best model

    default_format = "ubj"

    def __init__(
            self,
            path: [str | os.PathLike],
            name: str = "model",
            as_pickle: bool = False,
            interval: int = 100,
            eval_metric: Optional[str] = None,
            eval_name: Optional[str] = None,
            maximize: bool = False,
            save_top_k: int = 1,
            min_delta: float = 0.0,
    ) -> None:
        super().__init__()
        self._path = path
        self._name = name
        self._as_pickle = as_pickle
        self.interval = interval
        self.save_top_k = save_top_k
        self.eval_metric = eval_metric
        self.eval_name = eval_name
        self.maximize = maximize
        self._min_delta = min_delta
        self.epoch_counter = 0

        def get_s(value: _Score) -> float:
            """get score if it's cross validation history."""
            return value[0] if isinstance(value, tuple) else value

        def maximize(new: _Score, best: _Score) -> bool:
            """New score should be greater than the old one."""
            return numpy.greater(get_s(new) - self._min_delta, get_s(best))

        def minimize(new: _Score, best: _Score) -> bool:
            """New score should be lesser than the old one."""
            return numpy.greater(get_s(best) - self._min_delta, get_s(new))

        if self.maximize:
            self.improve_op = maximize
            self.get_best_fn = numpy.max
            self.best_score = -numpy.inf
        else:
            self.improve_op = minimize
            self.get_best_fn = numpy.min
            self.best_score = numpy.inf

    def after_iteration(self, model: _Model, epoch: int, evals_log: TrainingCallback.EvalsLog) -> bool:
        if self.epoch_counter == self.interval:
            if self.eval_name:
                metrics = evals_log[self.eval_name]
            else:
                metrics = evals_log[list(evals_log.keys())[-1]]  # get the last one
            if self.eval_metric:
                best_current_score = self.get_best_fn(metrics[self.eval_metric])
            else:
                best_current_score = self.get_best_fn(metrics[list(metrics.keys())[-1]])  # get the last one
            if self.improve_op(best_current_score, self.best_score):
                self.best_score = best_current_score
                remove_old_models(self._path, self._name, ".pkl" if self._as_pickle else f".{self.default_format}",
                                  self.save_top_k)

                best_iteration = model.best_iteration
                best_score = model.best_score
                assert best_iteration is not None and best_score is not None
                model_to_save = deepcopy(model[: best_iteration + 1])
                model_to_save.best_iteration = best_iteration
                model_to_save.best_score = best_score

                path = os.path.join(
                    self._path,
                    self._name
                    + "_"
                    + (str(best_iteration))
                    + (".pkl" if self._as_pickle else f".{self.default_format}"),
                )
                if collective.get_rank() == 0:
                    # checkpoint using the first worker
                    if self._as_pickle:
                        with open(path, "wb") as fd:
                            pickle.dump(model_to_save, fd)
                    else:
                        model_to_save.save_model(path)
            self.epoch_counter = 0  # reset counter
        self.epoch_counter += 1


map_our_metric_to_xgboost_metric = {
    ('logloss', 'binary_classification'): 'logloss',
    ('logloss', 'classification'): 'mlogloss',
    ('rmse', 'regression'): 'rmse',
    ('rmse', 'multi_regression'): 'rmse',
}


def before_fit_xgboost(self, extra_arguments, **fit_arguments):
    cat_features = extra_arguments.get('cat_features')
    report_to_ray = extra_arguments.get('report_to_ray')
    eval_name = extra_arguments.get('eval_name')
    task = extra_arguments.get('task')
    init_model = extra_arguments.get('init_model')

    eval_set = fit_arguments.get('eval_set')
    X_train = fit_arguments.get('X')
    y_train = fit_arguments.get('y')

    eval_set.insert(0, (X_train, y_train))
    eval_name.insert(0, 'train')

    if cat_features is not None:
        self.set_params(**{'enable_categorical': True})

    eval_metric = self.get_params().get('eval_metric', None)

    if eval_metric is not None:
        _, _, higher_is_better = get_metric_fn(eval_metric)
        eval_metric = map_our_metric_to_xgboost_metric[(eval_metric, task)]
        self.set_params(**{'eval_metric': eval_metric})
    else:
        higher_is_better = False

    if self.callbacks is None:
        self.callbacks = []

    # we will create our Early Stopping Callback, because we can set different metric, save the best model and
    # start from a trained model
    if self.early_stopping_patience > 0:
        # disable automatic early stopping
        self.set_params(**{'early_stopping_rounds': 0})
        # create early stopping callback
        early_stopping_callback = EarlyStopping(self.early_stopping_patience, metric_name=eval_metric,
                                                maximize=higher_is_better)
        checkpoint_callback = TrainingCheckPoint(self.output_dir, eval_metric=eval_metric,
                                                 maximize=higher_is_better, save_top_k=1)
        self.callbacks.extend([checkpoint_callback, early_stopping_callback])

    if self.log_to_mlflow_if_running:
        if mlflow.active_run():
            self.callbacks.append(LogToMLFlowXGBoost(eval_name, default_metric=self.eval_metric))

    if report_to_ray:
        self.callbacks.append(ReportToRayXGBoost(eval_name, default_metric=self.eval_metric))

    fit_arguments['eval_set'] = eval_set
    fit_arguments['xgb_model'] = init_model
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
    extended_init_kwargs={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'category',
        'data_scaler': None,
    },
    has_early_stopping=True, map_task_to_default_values={
        'classification': {'objective': 'multi:softmax', 'eval_metric': 'mlogloss'},
        'binary_classification': {'objective': 'binary:logistic', 'eval_metric': 'logloss'},
    },
    extra_dct={
        'create_search_space': staticmethod(create_search_space_xgboost),
        'get_recommended_params': staticmethod(get_recommended_params_xgboost),
        'before_fit': before_fit_xgboost,
    }
)

XGBRegressor = TabBenchmarkModelFactory.from_sk_cls(
    XGBRegressor,
    extended_init_kwargs={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'category',
        'data_scaler': None,
    },
    map_default_values_change={
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    },
    has_early_stopping=True, extra_dct={
        'create_search_space': staticmethod(create_search_space_xgboost),
        'get_recommended_params': staticmethod(get_recommended_params_xgboost),
        'before_fit': before_fit_xgboost,
    }
)
