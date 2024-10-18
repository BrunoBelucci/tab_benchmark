import datetime
import time
from pathlib import Path
from typing import Optional
import mlflow
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from tab_benchmark.models.mixins import n_estimators_gbdt, early_stopping_patience_gbdt, GBDTMixin, TabBenchmarkModel, \
    merge_signatures, merge_and_apply_signature
import optuna
from tab_benchmark.utils import flatten_dict


class ReportToOptunaCatboost:
    def __init__(self, eval_names, eval_name_to_report, eval_metric_to_report, optuna_trial):
        super().__init__()
        if len(eval_names) > 1:
            self.map_default_name_to_eval_name = {f'validation_{i}': name for i, name in enumerate(eval_names)}
        else:
            self.map_default_name_to_eval_name = {'validation': eval_names[0]}
        self.map_default_name_to_eval_name['learn'] = 'train'
        self.eval_metric_to_report = eval_metric_to_report
        self.eval_name_to_report = eval_name_to_report
        self.optuna_trial = optuna_trial
        self.pruned_trial = False

    def after_iteration(self, info):
        for default_name, metrics in info.metrics.items():
            our_name = self.map_default_name_to_eval_name[default_name]
            if our_name == self.eval_name_to_report:
                for metric, value in metrics.items():
                    if metric == self.eval_metric_to_report:
                        self.optuna_trial.report(value[-1], step=info.iteration)
                        if self.optuna_trial.should_prune():
                            self.pruned_trial = True
                            message = f'Trial was pruned at epoch {info.iteration}.'
                            print(message)
                            return False
                        break
                break
        return True


class LogToMLFlowCatboost:
    def __init__(self, run_id, log_every_n_steps=50):
        self.run_id = run_id
        self.log_every_n_steps = log_every_n_steps

    def after_iteration(self, info):
        if info.iteration % self.log_every_n_steps != 0:
            return True
        dict_to_log = {}
        for default_name, metrics in info.metrics.items():
            dict_to_log.update({f'{default_name}_{metric}': value[-1] for metric, value in metrics.items()})
        dict_to_log['iteration'] = info.iteration
        mlflow.log_metrics(dict_to_log, step=info.iteration, run_id=self.run_id)
        return True


class TimerCatboost:
    def __init__(self, duration):
        if isinstance(duration, int):
            duration = datetime.timedelta(seconds=duration)
        elif isinstance(duration, dict):
            duration = datetime.timedelta(**duration)
        else:
            raise ValueError(f"duration must be int or dict, got {type(duration)}")
        self.duration = duration
        self.start_time = time.perf_counter()
        self.reached_timeout = False

    def after_iteration(self, info):
        if (time.perf_counter() - self.start_time) > self.duration.total_seconds():
            self.reached_timeout = True
            return False
        else:
            return True


map_our_metric_to_catboost_metric = {
    ('logloss', 'binary_classification'): 'Logloss',
    ('logloss', 'classification'): 'MultiClass',
    ('rmse', 'regression'): 'RMSE',
    ('rmse', 'multi_regression'): 'MultiRMSE',
}


class CatBoostMixin(GBDTMixin):
    @property
    def map_task_to_default_values(self):
        return {
            'regression': {'loss_function': 'RMSE', 'eval_metric': 'rmse'},
            'multi_regression': {'loss_function': 'MultiRMSE', 'eval_metric': 'rmse'},
            'classification': {'loss_function': 'MultiClass', 'eval_metric': 'logloss'},
            'binary_classification': {'loss_function': 'Logloss', 'eval_metric': 'logloss'},
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
                                 eval_set=eval_set, eval_name=eval_name, init_model=init_model, optuna_trial=optuna_trial,
                                 **kwargs)
        for callback in self.callbacks:
            if isinstance(callback, ReportToOptunaCatboost):
                self.pruned_trial = callback.pruned_trial
                if self.mlflow_run_id:
                    log_metrics = {'pruned': int(callback.pruned_trial)}
                    mlflow.log_metrics(log_metrics, run_id=self.mlflow_run_id)
            elif isinstance(callback, TimerCatboost):
                self.reached_timeout = callback.reached_timeout
                if self.mlflow_run_id:
                    log_metrics = {'reached_timeout': int(callback.reached_timeout)}
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
        callbacks = kwargs.get('callbacks', [])
        fit_arguments = kwargs.copy() if kwargs else {}

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
            snapshot_file = fit_arguments.get('snapshot_file', 'model-snapshot.cbm')
            snapshot_interval = fit_arguments.get('snapshot_interval', 300)
            fit_arguments['save_snapshot'] = save_snapshot
            fit_arguments['snapshot_file'] = snapshot_file
            fit_arguments['snapshot_interval'] = snapshot_interval

        if optuna_trial:
            reported_metric = self.get_param('eval_metric')
            reported_eval_name = eval_name[-1]
            callbacks.append(ReportToOptunaCatboost(eval_name, eval_name_to_report=reported_eval_name,
                                                    eval_metric_to_report=reported_metric,
                                                    optuna_trial=optuna_trial))
        else:
            reported_metric = None
            reported_eval_name = None

        if self.mlflow_run_id:
            log_params = {'catboost_reported_metric': reported_metric,
                          'catboost_reported_eval_name': reported_eval_name}
            if len(eval_name) > 1:
                map_default_name_to_eval_name = {f'validation_{i}': name for i, name in enumerate(eval_name)}
            else:
                map_default_name_to_eval_name = {'validation': eval_name[0]}
            map_default_name_to_eval_name['learn'] = 'train'
            log_params.update({'map_catboost_name_to_eval_name': flatten_dict(map_default_name_to_eval_name)})
            mlflow.log_params(log_params, run_id=self.mlflow_run_id)
            callbacks.append(LogToMLFlowCatboost(run_id=self.mlflow_run_id, log_every_n_steps=self.log_interval))

        if self.max_time:
            callbacks.append(TimerCatboost(duration=self.max_time))

        fit_arguments['callbacks'] = callbacks
        self.callbacks = callbacks
        fit_arguments.update(dict(X=X, y=y, eval_set=eval_set, init_model=init_model))
        return super().before_fit(**fit_arguments)

    @staticmethod
    def create_search_space():
        # In Well tunned...
        # Not tunning n_estimators following discussion at
        # https://openreview.net/forum?id=Fp7__phQszn&noteId=Z7Y_qxwDjiM
        search_space = dict(
            learning_rate=optuna.distributions.FloatDistribution(1e-6, 1.0, log=True),
            random_strength=optuna.distributions.IntDistribution(1, 20),
            one_hot_max_size=optuna.distributions.IntDistribution(0, 25),
            l2_leaf_reg=optuna.distributions.FloatDistribution(1, 10),
            bagging_temperature=optuna.distributions.FloatDistribution(0, 1),
            leaf_estimation_iterations=optuna.distributions.IntDistribution(1, 10),
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

    @staticmethod
    def get_recommended_params():
        default_values_from_search_space = CatBoostMixin.create_search_space()[1]
        default_values_from_search_space.update(dict(
            n_estimators=n_estimators_gbdt,
            auto_early_stopping=True,
            early_stopping_patience=early_stopping_patience_gbdt,
        ))
        return default_values_from_search_space

    @property
    def n_jobs(self):
        if not hasattr(self, '_n_jobs'):
            self._n_jobs = self.get_params().get('thread_count', 1)
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value
        self.set_params(**{'thread_count': value})

    @property
    def output_dir(self):
        if not hasattr(self, '_output_dir'):
            self._output_dir = self.get_params().get('train_dir', None)
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value
        if value is not None:
            self.set_params(**{'train_dir': value})


class TabBenchmarkCatBoostRegressor(CatBoostMixin, TabBenchmarkModel, CatBoostRegressor):
    @merge_and_apply_signature(merge_signatures(CatBoostRegressor.__init__, CatBoostMixin.__init__))
    def __init__(
            self,
            *,
            loss_function='default',
            eval_metric='default',
            categorical_encoder='ordinal',
            categorical_type='int32',
            data_scaler=None,
            **kwargs
    ):
        super().__init__(loss_function=loss_function, eval_metric=eval_metric, categorical_encoder=categorical_encoder,
                         categorical_type=categorical_type, data_scaler=data_scaler, **kwargs)


class TabBenchmarkCatBoostClassifier(CatBoostMixin, TabBenchmarkModel, CatBoostClassifier):
    @merge_and_apply_signature(merge_signatures(CatBoostClassifier.__init__, CatBoostMixin.__init__))
    def __init__(
            self,
            *,
            loss_function='default',
            eval_metric='default',
            categorical_encoder='ordinal',
            categorical_type='int32',
            data_scaler=None,
            **kwargs
    ):
        super().__init__(loss_function=loss_function, eval_metric=eval_metric, categorical_encoder=categorical_encoder,
                         categorical_type=categorical_type, data_scaler=data_scaler, **kwargs)
