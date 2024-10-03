import re
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Optional, Dict, Any
import mlflow
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from lightgbm.basic import _is_numpy_1d_array, _to_string, _NUMERIC_TYPES, _is_numeric, _log_info
from lightgbm.callback import CallbackEnv, _EarlyStoppingCallback, _format_eval_result, EarlyStopException
from tab_benchmark.models.xgboost import n_estimators_gbdt, early_stopping_patience_gbdt, remove_old_models
from tab_benchmark.models.factories import sklearn_factory
import lightgbm.basic
import optuna


# Monkey patching the lightgbm to support nested dictionaries
def my_param_dict_to_str(data: Optional[Dict[str, Any]]) -> str:
    """Convert Python dictionary to string, which is passed to C API."""
    if data is None or not data:
        return ""
    pairs = []
    for key, val in data.items():
        if isinstance(val, (list, tuple, set)) or _is_numpy_1d_array(val):
            pairs.append(f"{key}={','.join(map(_to_string, val))}")
        elif isinstance(val, (str, Path, _NUMERIC_TYPES)) or _is_numeric(val):
            pairs.append(f"{key}={val}")
        elif isinstance(val, dict):
            pass
        elif val is not None:
            raise TypeError(f"Unknown type of parameter:{key}, got:{type(val).__name__}")
    return " ".join(pairs)


lightgbm.basic._param_dict_to_str = my_param_dict_to_str


# Monkey patching the lightgbm to support early stopping with the best iteration and best score being set on booster
class EarlyStoppingLGBM(_EarlyStoppingCallback):
    # Original early stopping, but setting the best iteration and best score on booster as soon as they change
    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if not self.enabled:
            return
        if env.evaluation_result_list is None:
            raise RuntimeError(
                "early_stopping() callback enabled but no evaluation results found. This is a probably bug in LightGBM. "
                "Please report it at https://github.com/microsoft/LightGBM/issues"
            )
        # self.best_score_list is initialized to an empty list
        first_time_updating_best_score_list = self.best_score_list == []
        for i in range(len(env.evaluation_result_list)):
            score = env.evaluation_result_list[i][2]
            if first_time_updating_best_score_list or self.cmp_op[i](score, self.best_score[i]):
                self.best_score[i] = score
                self.best_iter[i] = env.iteration

                # CHANGE HERE
                env.model.best_iteration = env.iteration + 1  # same as lgbm does internally after training
                if not getattr(env.model, 'best_score', None):
                    env.model.best_score = defaultdict(OrderedDict)
                eval_name = env.evaluation_result_list[i][0]
                metric_name = env.evaluation_result_list[i][1]
                metric_name = metric_name.split(" ")[-1]
                env.model.best_score[eval_name][metric_name] = score

                if first_time_updating_best_score_list:
                    self.best_score_list.append(env.evaluation_result_list)
                else:
                    self.best_score_list[i] = env.evaluation_result_list
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            eval_name_splitted = env.evaluation_result_list[i][1].split(" ")
            if self.first_metric_only and self.first_metric != eval_name_splitted[-1]:
                continue  # use only the first metric for early stopping
            if self._is_train_set(
                    ds_name=env.evaluation_result_list[i][0],
                    eval_name=eval_name_splitted[0],
                    env=env,
            ):
                continue  # train data for lgb.cv or sklearn wrapper (underlying lgb.train)
            elif env.iteration - self.best_iter[i] >= self.stopping_rounds:
                if self.verbose:
                    eval_result_str = "\t".join(
                        [_format_eval_result(x, show_stdv=True) for x in self.best_score_list[i]]
                    )
                    _log_info(f"Early stopping, best iteration is:\n[{self.best_iter[i] + 1}]\t{eval_result_str}")
                    if self.first_metric_only:
                        _log_info(f"Evaluated only: {eval_name_splitted[-1]}")
                raise EarlyStopException(self.best_iter[i], self.best_score_list[i])
            self._final_iteration_check(env, eval_name_splitted, i)


lightgbm.callback._EarlyStoppingCallback = EarlyStoppingLGBM


def conditional_num_leaves(config):
    model_params = config.get('model_params', None)
    if model_params is None:
        return 31
    max_depth = model_params.get('max_depth', None)
    if max_depth is None:
        return 31
    return np.random.randint(1, 2 ** max_depth)


def create_search_space_lgbm():
    # In Well tunned... + doc
    # Not tunning n_estimators following discussion at
    # https://openreview.net/forum?id=Fp7__phQszn&noteId=Z7Y_qxwDjiM
    search_space = dict(
        learning_rate=optuna.distributions.FloatDistribution(1e-3, 1.0, log=True),
        reg_lambda=optuna.distributions.FloatDistribution(1e-10, 1, log=True),
        reg_alpha=optuna.distributions.FloatDistribution(1e-10, 1.0, log=True),
        min_split_gain=optuna.distributions.FloatDistribution(1e-10, 1.0, log=True),
        colsample_bytree=optuna.distributions.FloatDistribution(0.1, 1),
        feature_fraction_bynode=optuna.distributions.FloatDistribution(0.1, 1),
        max_depth=optuna.distributions.IntDistribution(1, 20),
        max_delta_step=optuna.distributions.IntDistribution(0, 10),
        min_child_weight=optuna.distributions.FloatDistribution(1e-3, 20, log=True),
        subsample=optuna.distributions.FloatDistribution(0.01, 1),
        subsample_freq=optuna.distributions.IntDistribution(1, 11),
        # num_leaves=optuna.distributions.IntDistribution(1, 4096),  # should be < 2^(max_depth)
        # num_leaves=tune.sample_from(conditional_num_leaves),
        min_child_samples=optuna.distributions.IntDistribution(1, 1000000)
    )
    default_values = dict(
        learning_rate=0.1,
        reg_lambda=1e-10,
        reg_alpha=1e-10,
        min_split_gain=1e-10,
        colsample_bytree=1.0,
        feature_fraction_bynode=1.0,
        max_depth=6,
        max_delta_step=0,
        min_child_weight=1e-3,
        subsample=1.0,
        subsample_freq=1,
        # num_leaves=31,
        min_child_samples=20
    )
    return search_space, default_values


def get_recommended_params_lgbm():
    default_values_from_search_space = create_search_space_lgbm()[1]
    default_values_from_search_space.update(dict(
        n_estimators=n_estimators_gbdt,
        auto_early_stopping=True,
        early_stopping_patience=early_stopping_patience_gbdt,
    ))
    return default_values_from_search_space


class ReportToOptunaLGBM:
    def __init__(self, metric_name, eval_name, optuna_trial):
        super().__init__()
        self.metric_name = metric_name
        self.eval_name = eval_name
        self.optuna_trial = optuna_trial
        self.pruned_trial = False

    def __call__(self, env: CallbackEnv) -> None:
        result_list = env.evaluation_result_list
        for result in result_list:
            eval_name, metric_name, eval_result, is_higher_better = result
            # this is done everywhere on the original code with the comment
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            # so we do the same here
            metric_name = metric_name.split(" ")[-1]
            if eval_name == self.eval_name and metric_name == self.metric_name:
                self.optuna_trial.report(eval_result, step=env.iteration)
                if self.optuna_trial.should_prune():
                    self.pruned_trial = True
                    message = f'Trial was pruned at epoch {env.iteration}.'
                    print(message)
                    raise EarlyStopException(env.iteration, result)
                break


class LogToMLFlowLGBM:
    def __init__(self, run_id, log_every_n_steps=50, reported_metric=None, reported_eval_name=None):
        super().__init__()
        self.run_id = run_id
        self.log_every_n_steps = log_every_n_steps
        if reported_metric:
            mlflow.log_params({f'{reported_eval_name}_report_metric': reported_metric}, run_id=self.run_id)
        self.reported_metric = reported_metric
        self.reported_eval_name = reported_eval_name

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration % self.log_every_n_steps != 0:
            return
        result_list = env.evaluation_result_list
        log_dict = {}
        for result in result_list:
            eval_name, metric_name, eval_result, is_higher_better = result
            # this is done everywhere on the original code with the comment
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            # so we do the same here
            metric_name = metric_name.split(" ")[-1]
            log_dict[f'{eval_name}_{metric_name}'] = eval_result
        log_dict['epoch'] = env.iteration
        mlflow.log_metrics(log_dict, step=env.iteration)


class TrainingCheckPointLGBM:
    def __init__(self, directory, name='model', interval=100, save_top_k=1):
        self.directory = directory
        self.name = name
        self.interval = interval
        self.save_top_k = save_top_k

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration % self.interval != 0:
            return
        remove_old_models(self.directory, self.name, '.txt', self.save_top_k)
        # None = save only best iterations
        env.model.save_model(f'{self.directory}/{self.name}_{env.model.best_iteration}.txt', num_iteration=None)


map_our_metric_to_lgbm_metric = {
    ('logloss', 'binary_classification'): 'binary_logloss',
    ('logloss', 'classification'): 'multi_logloss',
    ('rmse', 'regression'): 'rmse',
    ('rmse', 'multi_regression'): 'rmse',
}


def before_fit_lgbm(self, X, y, task=None, cat_features=None, cat_dims=None, n_classes=None, eval_set=None,
                    eval_name=None, report_to_optuna=None, optuna_trial=None,
                    init_model=None, **args_and_kwargs):
    if n_classes is not None:
        if n_classes > 2:
            self.set_params(**{'num_class': n_classes})

    callbacks = args_and_kwargs.get('callbacks', [])

    eval_metric = self.get_params().get('eval_metric', None)

    fit_arguments = args_and_kwargs.copy() if args_and_kwargs else {}

    callbacks = callbacks if callbacks is not None else []

    if self.early_stopping_patience > 0:
        # for the moment we will leave the default early stopping callback
        self.set_params(**{'early_stopping_rounds': self.early_stopping_patience})
        # we will add a training checkpoint callback
        callbacks.append(TrainingCheckPointLGBM(self.output_dir))

    if eval_metric is not None:
        eval_metric = map_our_metric_to_lgbm_metric[(eval_metric, task)]
        self.set_params(**{'metric': eval_metric})
    else:
        eval_metric = self.get_params().get('metric', None)

    if report_to_optuna:
        reported_metric = eval_metric
        reported_eval_name = eval_name[-1]
        callbacks.append(ReportToOptunaLGBM(reported_metric, reported_eval_name, optuna_trial))
    else:
        reported_metric = None
        reported_eval_name = None

    if self.log_to_mlflow_if_running:
        if mlflow.active_run():
            callbacks.append(LogToMLFlowLGBM(run_id=self.run_id, reported_metric=reported_metric,
                                             reported_eval_name=reported_eval_name,
                                             log_every_n_steps=self.log_interval))

    # we will rename the columns to avoid problems with the lightgbm
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    if eval_set:
        for i in range(len(eval_set)):
            X_eval, y_eval = eval_set[i]
            X_eval = X_eval.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
            eval_set[i] = (X_eval, y_eval)

    if cat_features is not None:
        cat_features = [re.sub('[^A-Za-z0-9_]+', '', x) for x in cat_features]
        fit_arguments['categorical_feature'] = cat_features

    fit_arguments['eval_names'] = eval_name
    fit_arguments['callbacks'] = callbacks
    self.callbacks = callbacks
    fit_arguments.update(dict(X=X, y=y, eval_set=eval_set, init_model=init_model))
    return fit_arguments


def after_fit_lgbm(self, fit_return):
    for callback in self.callbacks:
        if isinstance(callback, ReportToOptunaLGBM):
            self.pruned_trial = callback.pruned_trial
            if self.log_to_mlflow_if_running:
                log_metrics = {'pruned': int(callback.pruned_trial)}
                mlflow.log_metrics(log_metrics, run_id=self.run_id)
            break
    return fit_return


TabBenchmarkLGBMRegressor = sklearn_factory(
    LGBMRegressor,
    has_early_stopping=True,
    default_values={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'category',
        'data_scaler': None,
        'objective': 'regression',
        'eval_metric': 'rmse'
    },
    before_fit_method=before_fit_lgbm,
    after_fit_method=after_fit_lgbm,
    extra_dct={
        'create_search_space': staticmethod(create_search_space_lgbm),
        'get_recommended_params': staticmethod(get_recommended_params_lgbm),
    }
)

TabBenchmarkLGBMClassifier = sklearn_factory(
    LGBMClassifier,
    has_early_stopping=True,
    default_values={
        'categorical_encoder': 'ordinal',
        'categorical_type': 'category',
        'data_scaler': None,
    },
    map_task_to_default_values={
        'binary_classification': {'objective': 'binary', 'eval_metric': 'logloss'},
        'classification': {'objective': 'multiclass', 'eval_metric': 'logloss'},
    },
    before_fit_method=before_fit_lgbm,
    after_fit_method=after_fit_lgbm,
    extra_dct={
        'create_search_space': staticmethod(create_search_space_lgbm),
        'get_recommended_params': staticmethod(get_recommended_params_lgbm),
    }
)
