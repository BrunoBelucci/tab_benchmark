from __future__ import annotations
import datetime
import os
import pickle
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional, cast, Dict, Any
import mlflow
import numpy
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from tab_benchmark.models.mixins import TabBenchmarkModel, GBDTMixin, apply_signature, merge_signatures
from xgboost import XGBClassifier, XGBRegressor, XGBModel, collective
from xgboost.callback import (TrainingCallback, _Model,
                              EarlyStopping as OriginalEarlyStopping, _Score, _ScoreList)
from tab_benchmark.utils import get_metric_fn, flatten_dict


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


class LogToMLFlowXGBoost(TrainingCallback):
    def __init__(self, run_id, log_every_n_steps=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.run_id = run_id

    def after_iteration(self, model: _Model, epoch: int, evals_log):
        if epoch % self.log_every_n_steps != 0:
            return
        dict_to_log = {'epoch': epoch}
        for default_name, metrics in evals_log.items():
            dict_to_log.update({f'{default_name}_{metric}': value[-1] for metric, value in metrics.items()})
        mlflow.log_metrics(dict_to_log, step=epoch, run_id=self.run_id)


class ReportToOptunaXGBoost(TrainingCallback):
    def __init__(self, optuna_trial, xgb_eval_name, xgb_metric_name=None):
        super().__init__()
        self.xgb_eval_name = xgb_eval_name
        self.xgb_metric_name = xgb_metric_name
        self.optuna_trial = optuna_trial
        self.pruned_trial = False

    def after_iteration(self, model: _Model, epoch: int, evals_log):
        if self.xgb_metric_name:
            metric_to_report = evals_log[self.xgb_eval_name][self.xgb_metric_name][-1]
        else:
            # if we do not have a metric name, we will report the last metric
            metric_to_report = evals_log[self.xgb_eval_name][list(evals_log[self.xgb_eval_name].keys())[-1]][-1]
        self.optuna_trial.report(metric_to_report, step=epoch)
        if self.optuna_trial.should_prune():
            self.pruned_trial = True
            message = f'Trial was pruned at epoch {epoch}.'
            print(message)
            return True


class TrainingCheckPoint(TrainingCallback):
    # Adapt from original TrainingCheckPoint to save only the best model

    default_format = "ubj"

    def __init__(
            self,
            path: [str | os.PathLike],
            name: str = "model",
            as_pickle: bool = False,
            interval: int = 100,
            xgb_eval_metric: Optional[str] = None,
            xgb_eval_name: Optional[str] = None,
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
        self.xgb_eval_metric = xgb_eval_metric
        self.xgb_eval_name = xgb_eval_name
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
            if self.xgb_eval_name:
                metrics = evals_log[self.xgb_eval_name]
            else:
                metrics = evals_log[list(evals_log.keys())[-1]]  # get the last one
            if self.xgb_eval_metric:
                best_current_score = self.get_best_fn(metrics[self.xgb_eval_metric])
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


class TimerXGBoost(TrainingCallback):
    def __init__(self, duration: int):
        super().__init__()
        if isinstance(duration, int):
            duration = datetime.timedelta(seconds=duration)
        elif isinstance(duration, dict):
            duration = datetime.timedelta(**duration)
        else:
            raise ValueError(f"duration must be int or dict, got {type(duration)}")
        self.duration = duration
        self.start_time = time.perf_counter()
        self.reached_timeout = False

    def after_iteration(self, model: _Model, epoch: int, evals_log):
        if (time.perf_counter() - self.start_time) > self.duration.total_seconds():
            self.reached_timeout = True
            return True


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


map_our_metric_to_xgboost_metric = {
    ('logloss', 'binary_classification'): 'logloss',
    ('logloss', 'classification'): 'mlogloss',
    ('rmse', 'regression'): 'rmse',
    ('rmse', 'multi_regression'): 'rmse',
}


class XGBMixin(GBDTMixin):
    @property
    def map_task_to_default_values(self):
        return {
            'classification': {'objective': 'multi:softmax', 'eval_metric': 'logloss'},
            'binary_classification': {'objective': 'binary:logistic', 'eval_metric': 'logloss'},
            'regression': {'objective': 'reg:squarederror', 'eval_metric': 'rmse'},
            'multi_regression': {'objective': 'reg:squarederror', 'eval_metric': 'rmse'},
        }

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame | pd.Series,
            task: Optional[str] = None,
            cat_features: Optional[list[str]] = None,
            cat_dims: Optional[list[int]] = None,
            n_classes: Optional[int] = None,
            eval_set: Optional[list[tuple]] = None,
            eval_name: Optional[list[str]] = None,
            init_model: Optional[str | Path] = None,
            optuna_trial: Optional[optuna.Trial] = None,
            **kwargs
    ):
        fit_return = super().fit(X, y, task=task, cat_features=cat_features, cat_dims=cat_dims, n_classes=n_classes,
                                 eval_set=eval_set, eval_name=eval_name, init_model=init_model,
                                 optuna_trial=optuna_trial,
                                 **kwargs)
        for callback in self.callbacks:
            if isinstance(callback, ReportToOptunaXGBoost):
                self.pruned_trial = callback.pruned_trial
                if self.mlflow_run_id:
                    log_metrics = {'pruned': int(callback.pruned_trial)}
                    mlflow.log_metrics(log_metrics, run_id=self.mlflow_run_id)
            elif isinstance(callback, TimerXGBoost):
                self.reached_timeout = callback.reached_timeout
                if self.mlflow_run_id:
                    log_metrics = {'timeout': int(callback.reached_timeout)}
                    mlflow.log_metrics(log_metrics, run_id=self.mlflow_run_id)

        return fit_return

    def before_fit(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame | pd.Series,
            task: Optional[str] = None,
            cat_features: Optional[list[str]] = None,
            cat_dims: Optional[list[int]] = None,
            n_classes: Optional[int] = None,
            eval_set: Optional[list[tuple]] = None,
            eval_name: Optional[list[str]] = None,
            init_model: Optional[str | Path] = None,
            optuna_trial: Optional[optuna.Trial] = None,
            **kwargs
    ):
        eval_set.insert(0, (X, y))
        eval_name.insert(0, 'train')

        if cat_features is not None:
            self.set_params(**{'enable_categorical': True})

        if n_classes is not None:
            if n_classes > 2:
                self.set_params(**{'num_class': n_classes})

        our_eval_metric = self.eval_metric
        if our_eval_metric is not None:
            _, _, higher_is_better = get_metric_fn(our_eval_metric)
            xgb_eval_metric = map_our_metric_to_xgboost_metric[(our_eval_metric, task)]
            self.set_params(**{'eval_metric': xgb_eval_metric})
        else:
            higher_is_better = False
            xgb_eval_metric = None

        callbacks = self.get_params().get('callbacks', None)

        if callbacks is None:
            callbacks = []

        # we will create our Early Stopping Callback, because we can set different metric, save the best model and
        # start from a trained model
        if self.early_stopping_patience > 0:
            # disable automatic early stopping
            self.set_params(**{'early_stopping_rounds': 0})
            # create early stopping callback
            early_stopping_callback = EarlyStopping(self.early_stopping_patience, metric_name=xgb_eval_metric,
                                                    maximize=higher_is_better)
            callbacks.append(early_stopping_callback)

        if self.save_checkpoint:
            checkpoint_callback = TrainingCheckPoint(self.output_dir, xgb_eval_metric=xgb_eval_metric,
                                                     maximize=higher_is_better, save_top_k=1, as_pickle=False,
                                                     interval=self.checkpoint_interval)
            callbacks.append(checkpoint_callback)

        if optuna_trial:
            reported_eval_name = f'validation_{len(eval_name) - 1}'  # last one
            reported_metric = xgb_eval_metric
            callbacks.append(ReportToOptunaXGBoost(optuna_trial=optuna_trial, xgb_eval_name=reported_eval_name,
                                                   xgb_metric_name=reported_metric))
        else:
            reported_metric = None
            reported_eval_name = None

        if self.mlflow_run_id:
            log_params = {'xgb_reported_metric': reported_metric, 'xgb_reported_eval_name': reported_eval_name}
            map_xgb_name_to_eval_name = {f'validation_{i}': name for i, name in enumerate(eval_name)}
            log_params.update({'map_xgb_name_to_eval_name': flatten_dict(map_xgb_name_to_eval_name)})
            mlflow.log_params(log_params, run_id=self.mlflow_run_id)
            callbacks.append(LogToMLFlowXGBoost(run_id=self.mlflow_run_id, log_every_n_steps=self.log_interval))

        if self.max_time:
            callbacks.append(TimerXGBoost(duration=self.max_time))

        self.callbacks = callbacks

        fit_arguments = dict(X=X, y=y, eval_set=eval_set, xgb_model=init_model)
        fit_arguments.update(kwargs)

        return super().before_fit(**fit_arguments)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        # pylint: disable=attribute-defined-outside-init
        """Get parameters."""
        # Based on: https://stackoverflow.com/questions/59248211
        # The basic flow in `get_params` is:
        # 0. Return parameters in subclass first, by using inspect.
        # 1. Return parameters in `XGBModel` (the base class).
        # 2. Return whatever in `**kwargs`.
        # 3. Merge them.
        params = BaseEstimator.get_params(self, deep)
        # params = super(self.__class__, self).get_params(deep)
        # cp = copy.copy(self)
        # cp.__class__ = cp.__class__.__bases__[0]
        # params.update(cp.__class__.get_params(cp, deep))
        # if kwargs is a dict, update params accordingly
        if hasattr(self, "kwargs") and isinstance(self.kwargs, dict):
            params.update(self.kwargs)
        if isinstance(params["random_state"], np.random.RandomState):
            params["random_state"] = params["random_state"].randint(
                np.iinfo(np.int32).max
            )
        elif isinstance(params["random_state"], np.random.Generator):
            params["random_state"] = int(
                params["random_state"].integers(np.iinfo(np.int32).max)
            )

        return params


class TabBenchmarkXGBClassifier(XGBMixin, TabBenchmarkModel, XGBClassifier):
    @apply_signature(merge_signatures(XGBModel.__init__, XGBMixin.__init__))
    def __init__(
            self,
            *,
            categorical_encoder='ordinal',
            categorical_type='category',
            data_scaler=None,
            **kwargs
    ):
        super().__init__(categorical_encoder=categorical_encoder, categorical_type=categorical_type,
                         data_scaler=data_scaler, **kwargs)


class TabBenchmarkXGBRegressor(XGBMixin, TabBenchmarkModel, XGBRegressor):
    @apply_signature(merge_signatures(XGBModel.__init__, XGBMixin.__init__))
    def __init__(
            self,
            *,
            categorical_encoder='ordinal',
            categorical_type='category',
            data_scaler=None,
            **kwargs
    ):
        super().__init__(categorical_encoder=categorical_encoder, categorical_type=categorical_type,
                         data_scaler=data_scaler, **kwargs)
