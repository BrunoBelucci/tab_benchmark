from __future__ import annotations

from copy import deepcopy
from functools import partial
from inspect import signature, Signature
from pathlib import Path
from typing import Optional
from warnings import warn
import cloudpickle
import mlflow
import pandas as pd
import torch
import optuna
from lightning.pytorch.loggers import MLFlowLogger
from tab_benchmark.dnns.architectures import Node, Saint, TabTransformer, TabNet
from tab_benchmark.dnns.architectures.mlp import MLP
from tab_benchmark.dnns.architectures.resnet import ResNet
from tab_benchmark.dnns.architectures.transformer import Transformer
from tab_benchmark.dnns.callbacks import DefaultLogs
from tab_benchmark.dnns.callbacks.evaluate_metric import EvaluateMetric
from tab_benchmark.dnns.callbacks.report_to_optuna import ReportToOptuna
from tab_benchmark.dnns.callbacks.timer import TimerDNN
from tab_benchmark.dnns.modules import TabNetModule
from tab_benchmark.dnns.utils.external.node.lib.facebook_optimizer.optimizer import QHAdam
from tab_benchmark.models.dnn_model import DNNModel, early_stopping_patience_dnn, max_epochs_dnn
from tab_benchmark.models.mixins import (EarlyStoppingMixin, PreprocessingMixin, TaskDependentParametersMixin,
                                         TabBenchmarkModel, merge_signatures, merge_and_apply_signature,
                                         check_y_eval_set_eval_name_cat_features)
from tab_benchmark.utils import (get_default_tag, get_formated_file_path, get_most_recent_file_path,
                                 get_metric_fn)
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def get_early_stopping_callback(eval_name, eval_metric, early_stopping_patience) -> list[tuple[type[Callback], dict]]:
    if early_stopping_patience > 0:
        if eval_metric == 'loss':
            higher_is_better = False
        else:
            metric_fn, need_proba, higher_is_better = get_metric_fn(eval_metric)
        if higher_is_better:
            mode = 'max'
        else:
            mode = 'min'
        return [
            (EarlyStopping, dict(monitor=f'{eval_name}_{eval_metric}', patience=early_stopping_patience,
                                 min_delta=0, mode=mode)),
            (ModelCheckpoint, dict(monitor=f'{eval_name}_{eval_metric}', every_n_epochs=1, save_last=True, mode=mode)),
        ]
    else:
        return []


class DNNMixin(EarlyStoppingMixin, PreprocessingMixin, TaskDependentParametersMixin):
    _estimator_type = ['classifier', 'regressor']

    @merge_and_apply_signature(merge_signatures(signature(PreprocessingMixin.__init__),
                                                signature(EarlyStoppingMixin.__init__)))
    def __init__(
            self,
            categorical_type='int64',
            default_categorical_type='int64',
            categorical_encoder='ordinal',
            categorical_target_type='int64',
            data_scaler='standard',
            loss_fn='default',
            log_interval=1,
            early_stopping_patience=early_stopping_patience_dnn,
            use_best_model=True,
            **kwargs
    ):
        self.pruned_trial = False
        self.reached_timeout = False
        self.use_best_model = use_best_model
        super().__init__(categorical_type=categorical_type, categorical_encoder=categorical_encoder,
                         categorical_target_type=categorical_target_type, data_scaler=data_scaler, loss_fn=loss_fn,
                         log_interval=log_interval, default_categorical_type=default_categorical_type,
                         early_stopping_patience=early_stopping_patience, **kwargs)

    @property
    def map_task_to_default_values(self):
        return {
            'classification': {'loss_fn': torch.nn.functional.cross_entropy},
            'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
            'regression': {'loss_fn': torch.nn.functional.mse_loss},
            'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
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
        self.optuna_trial = optuna_trial
        fit_return = super().fit(X, y, task=task, cat_features=cat_features, cat_dims=cat_dims, n_classes=n_classes,
                                 eval_set=eval_set, eval_name=eval_name, init_model=init_model,
                                 optuna_trial=optuna_trial,
                                 **kwargs)

        # load best model
        if self.use_best_model:
            if len(self.lit_trainer_.checkpoint_callbacks) > 1:
                warn('More than one checkpoint callback found, using the one in trainer.checkpoint_callback')
            if self.lit_trainer_.checkpoint_callback and self.lit_trainer_.checkpoint_callback.best_model_path != '':
                self.lit_module_ = self.lit_module_class.load_from_checkpoint(
                    self.lit_trainer_.checkpoint_callback.best_model_path)
                if self.lit_trainer_.early_stopping_callback:
                    self.lit_module_.best_iteration_ = (self.lit_trainer_.early_stopping_callback.stopped_epoch
                                                        - self.early_stopping_patience)
                else:
                    self.lit_module_.best_iteration_ = self.lit_trainer_.current_epoch
            else:
                warn('No checkpoint callback found, cannot load best model, using last model instead.')
                self.lit_module_.best_iteration_ = self.lit_trainer_.current_epoch

        for callback in self.lit_callbacks_:
            if isinstance(callback, ReportToOptuna):
                self.pruned_trial = callback.pruned_trial
                if self.mlflow_run_id:
                    log_metrics = {'pruned': int(callback.pruned_trial)}
                    mlflow.log_metrics(log_metrics, run_id=self.mlflow_run_id)
                    log_params = {f'{self.reported_eval_name}_report_metric': self.reported_metric}
                    mlflow.log_params(log_params, run_id=self.mlflow_run_id)
            elif isinstance(callback, TimerDNN):
                self.reached_timeout = callback.reached_timeout
                if self.mlflow_run_id:
                    log_metrics = {'reached_timeout': int(self.reached_timeout)}
                    mlflow.log_metrics(log_metrics, run_id=self.mlflow_run_id)

        if self.auto_reduce_batch_size:
            if self.mlflow_run_id:
                log_param = {'final_batch_size': self.batch_size}
                mlflow.log_params(log_param, run_id=self.mlflow_run_id)
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
        y, eval_set, eval_name, cat_features, cat_dims = check_y_eval_set_eval_name_cat_features(
            X, y, eval_set, eval_name, cat_features, cat_dims)

        fit_arguments = kwargs.copy() if kwargs else {}

        # initialize callbacks and trainer params from EarlyStoppingMixin

        # we will consider that the last set of eval_set will be used as validation
        # we will consider that the last metric of eval_metric will be used as validation
        # if no eval_metric is provided, we use only the loss function
        # if no eval_set is provided, we use only the training set

        callbacks_tuples = []
        if self.es_eval_metric:
            callbacks_tuples.append((EvaluateMetric, dict(eval_metric=self.es_eval_metric)))

        if self.log_interval > 0:
            callbacks_tuples.append((DefaultLogs, dict(log_interval_epoch=self.log_interval)))

        if optuna_trial:
            self.reported_metric = self.es_eval_metric if self.es_eval_metric else 'loss'
            self.reported_eval_name = eval_name[-1] if eval_name else 'train'
            callbacks_tuples.append((ReportToOptuna, dict(optuna_trial=optuna_trial,
                                                          reported_metric=self.reported_metric,
                                                          reported_eval_name=self.reported_eval_name)))

        last_eval_name = eval_name[-1] if eval_name else 'train'
        callbacks_tuples.extend(get_early_stopping_callback(
            last_eval_name, self.es_eval_metric if self.es_eval_metric else 'loss', self.early_stopping_patience))

        if self.max_time:
            callbacks_tuples.append((TimerDNN, dict(duration=self.max_time)))

        self.lit_callbacks_tuples = self.lit_callbacks_tuples + callbacks_tuples

        if self.mlflow_run_id:
            self.lit_trainer_params['logger'] = MLFlowLogger(run_id=self.mlflow_run_id,
                                                             tracking_uri=mlflow.get_tracking_uri())

        if self.output_dir:
            self.lit_trainer_params['default_root_dir'] = self.output_dir

        fit_arguments.update(
            dict(X=X, y=y, task=task, cat_features=cat_features, cat_dims=cat_dims, n_classes=n_classes,
                 eval_set=eval_set, eval_name=eval_name, init_model=init_model))
        return super().before_fit(**fit_arguments)


def dnn_factory(architecture_cls, create_search_space_fn, get_recommended_params_fn, dnn_mixin=DNNMixin):
    architecture_parameters = signature(architecture_cls.__init__).parameters
    architecture_parameters_not_from_dataset = {name: param for name, param in architecture_parameters.items()
                                                if name not in architecture_cls.params_defined_from_dataset + ['self']}
    signature_architecture_without_parameters_from_dataset = Signature(
        list(architecture_parameters_not_from_dataset.values()))

    dnn_model_parameters = signature(DNNModel.__init__).parameters
    dnn_model_parameters_filtered = {name: param for name, param in dnn_model_parameters.items()
                                     if name not in ['architecture_params', 'architecture_params_not_from_dataset',
                                                     'dnn_architecture_class']}
    signature_dnn_model = Signature(list(dnn_model_parameters_filtered.values()))

    class TabBenchmarkDNN(dnn_mixin, TabBenchmarkModel, DNNModel):
        @merge_and_apply_signature(merge_signatures(signature(dnn_mixin.__init__), signature_dnn_model,
                                                    signature_architecture_without_parameters_from_dataset))
        def __init__(self, **kwargs):
            # get architecture parameters that are in kwargs
            architecture_params_not_from_dataset = {name: kwargs.pop(name) for name in list(kwargs.keys())
                                                    if name in architecture_parameters_not_from_dataset}
            # complete with default values
            for name, param in architecture_parameters_not_from_dataset.items():
                if name not in architecture_params_not_from_dataset:
                    architecture_params_not_from_dataset[name] = param.default

            # set parameters to self
            for key, value in architecture_params_not_from_dataset.items():
                setattr(self, key, value)

            kwargs['architecture_params_not_from_dataset'] = architecture_params_not_from_dataset
            kwargs['dnn_architecture_class'] = architecture_cls
            super().__init__(**kwargs)

        @staticmethod
        def create_search_space():
            return create_search_space_fn()

        @staticmethod
        def get_recommended_params():
            return get_recommended_params_fn()

        def save_model(self, save_dir: [Path | str] = None, tag: Optional[str] = None) -> Path:
            prefix = self.__class__.__name__ + '_dnn'
            ext = 'pt'
            if tag is None:
                tag = get_default_tag()
            file_path = get_formated_file_path(save_dir, prefix, ext, tag)
            torch.save(self, file_path, pickle_module=cloudpickle)
            return super().save_model(save_dir, tag)

        def load_model(self, save_dir: Path | str = None, tag: Optional[str] = None):
            prefix = self.__class__.__name__ + '_dnn'
            ext = 'pt'
            if tag is None:
                tag = get_default_tag()
            file_path = get_most_recent_file_path(save_dir, prefix, ext, tag)
            self = torch.load(file_path)
            return super().load_model(save_dir, tag)

    TabBenchmarkDNN.__name__ = f'TabBenchmark{architecture_cls.__name__}'
    return TabBenchmarkDNN


def create_search_space_mlp():
    # Based on Gorishniy, Yury, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
    # “Revisiting Deep Learning Models for Tabular Data.” arXiv, November 10, 2021.
    # https://doi.org/10.48550/arXiv.2106.11959.
    search_space = dict(
        n_layers=optuna.distributions.IntDistribution(1, 16),
        hidden_dims=optuna.distributions.CategoricalDistribution([1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024]),
        dropouts=optuna.distributions.FloatDistribution(0.0, 0.5),
        lr=optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
        weight_decay=optuna.distributions.FloatDistribution(1e-6, 1e-2, log=True),
        categorical_embedding_dim=optuna.distributions.CategoricalDistribution([1, 2, 4, 8, 16, 24, 32, 64, 128, 256,
                                                                                512]),
        continuous_embedding_dim=optuna.distributions.CategoricalDistribution([1, 2, 4, 8, 16, 24, 32, 64, 128, 256,
                                                                               512]),
    )
    default_values = dict(
        n_layers=4,
        hidden_dims=256,
        dropouts=0.5,
        lr=1e-3,
        weight_decay=1e-2,
        categorical_embedding_dim=256,
        continuous_embedding_dim=1,
    )
    return search_space, default_values


def get_recommended_params_dnn(create_search_space_dnn_fn):
    default_values_from_search_space = create_search_space_dnn_fn()[1]
    default_values_from_search_space.update(dict(
        max_epochs=max_epochs_dnn,
        auto_early_stopping=True,
        early_stopping_patience=early_stopping_patience_dnn,
    ))
    return default_values_from_search_space


TabBenchmarkMLP = dnn_factory(MLP, create_search_space_mlp, partial(get_recommended_params_dnn,
                                                                    create_search_space_mlp))


def create_search_space_resnet():
    # Based on Gorishniy, Yury, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
    # “Revisiting Deep Learning Models for Tabular Data.” arXiv, November 10, 2021.
    # https://doi.org/10.48550/arXiv.2106.11959.
    search_space = dict(
        n_blocks=optuna.distributions.IntDistribution(1, 16),
        blocks_dim=optuna.distributions.CategoricalDistribution([1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024]),
        dropout_1=optuna.distributions.FloatDistribution(0.0, 0.5),
        dropout_2=optuna.distributions.FloatDistribution(0.0, 0.5),
        lr=optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
        weight_decay=optuna.distributions.FloatDistribution(1e-6, 1e-2, log=True),
        categorical_embedding_dim=optuna.distributions.CategoricalDistribution(
            [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512]),
        continuous_embedding_dim=optuna.distributions.CategoricalDistribution(
            [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512]),
    )
    default_values = dict(
        n_blocks=4,
        blocks_dims=256,
        dropouts_1=0.5,
        dropouts_2=0.5,
        lr=1e-3,
        weight_decay=1e-2,
        categorical_embedding_dim=256,
        continuous_embedding_dim=1,
    )
    return search_space, default_values


TabBenchmarkResNet = dnn_factory(ResNet, create_search_space_resnet, partial(get_recommended_params_dnn,
                                                                             create_search_space_resnet))


def create_search_space_transformer():
    # Based on Gorishniy, Yury, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
    # “Revisiting Deep Learning Models for Tabular Data.” arXiv, November 10, 2021.
    # https://doi.org/10.48550/arXiv.2106.11959.
    search_space = dict(
        n_heads=optuna.distributions.IntDistribution(1, 6),
        feedforward_dims=optuna.distributions.CategoricalDistribution([16, 24, 32, 64, 128, 256, 512, 1024, 2048]),
        dropouts_attn=optuna.distributions.FloatDistribution(0.0, 0.5),
        dropouts_ff=optuna.distributions.FloatDistribution(0.0, 0.5),
        lr=optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
        weight_decay=optuna.distributions.FloatDistribution(1e-6, 1e-2, log=True),
        embedding_dim=optuna.distributions.CategoricalDistribution([1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512]),
    )
    default_values = dict(
        n_heads=3,
        feedforward_dims=512,
        dropouts_attn=0.5,
        dropouts_ff=0.5,
        lr=1e-3,
        weight_decay=1e-2,
        embedding_dim=256,
    )
    return search_space, default_values


TabBenchmarkTransformer = dnn_factory(Transformer, create_search_space_transformer,
                                      partial(get_recommended_params_dnn, create_search_space_transformer))


class NodeMixin(DNNMixin):
    def __init__(
            self,
            categorical_type='float32',
            default_categorical_type='float32',
            torch_optimizer_tuple=deepcopy((QHAdam, dict(nus=(0.7, 1.0), betas=(0.95, 0.998)))),
            **kwargs
    ):
        super().__init__(categorical_type=categorical_type, torch_optimizer_tuple=torch_optimizer_tuple,
                         default_categorical_type=default_categorical_type, **kwargs)


def create_search_space_node():
    # Following the paper
    search_space = dict(
        num_NODE_layers=optuna.distributions.CategoricalDistribution([1, 2, 4, 8]),
        depth=optuna.distributions.CategoricalDistribution([6, 8]),
        num_trees=optuna.distributions.CategoricalDistribution([1024, 2048]),
        extra_tree_dim=optuna.distributions.CategoricalDistribution([1, 2, 3]),
    )
    default_values = dict(
        num_NODE_layers=1,
        depth=6,
        num_trees=2048,
        extra_tree_dim=1,
    )
    return search_space, default_values


TabBenchmarkNode = dnn_factory(Node, create_search_space_transformer,
                               partial(get_recommended_params_dnn, create_search_space_node), NodeMixin)


class TabTransformetSaintMixin(DNNMixin):
    def __init__(
            self,
            categorical_type='float32',
            default_categorical_type='float32',
            torch_optimizer_tuple=deepcopy((torch.optim.AdamW, {'lr': 1e-4, 'weight_decay': 1e-2})),
            **kwargs
    ):
        super().__init__(categorical_type=categorical_type, torch_optimizer_tuple=torch_optimizer_tuple,
                         default_categorical_type=default_categorical_type, **kwargs)


def create_search_space_saint():
    # From TabTransformer...there is no hyperparameter tuning policy described in Saint paper.
    search_space = dict(
        embedding_dim=optuna.distributions.CategoricalDistribution([1, 2, 4, 8, 16, 32, 64, 128, 256]),
        n_heads=optuna.distributions.CategoricalDistribution([1, 2, 4, 8]),
        n_blocks=optuna.distributions.CategoricalDistribution([1, 2, 3, 6, 12]),
        mlp_hidden_mult_1=optuna.distributions.IntDistribution(1, 8),
        mlp_hidden_mult_2=optuna.distributions.IntDistribution(1, 3),
        lr=optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
        weight_decay=optuna.distributions.FloatDistribution(1e-6, 1e-1, log=True),
        attn_dropout=optuna.distributions.CategoricalDistribution([0.1, 0.2, 0.3, 0.4, 0.5]),
        ff_dropout=optuna.distributions.CategoricalDistribution([0.1, 0.2, 0.3, 0.4, 0.5]),
    )
    default_values = dict(
        embedding_dim=32,
        n_heads=8,
        n_blocks=6,
        mlp_hidden_mult_1=4,
        mlp_hidden_mult_2=2,
        lr=1e-4,
        weight_decay=1e-2,
        attn_dropout=0.2,
        ff_dropout=0.1,
    )
    return search_space, default_values


TabBenchmarkSaint = dnn_factory(Saint, create_search_space_saint,
                                partial(get_recommended_params_dnn, create_search_space_saint),
                                TabTransformetSaintMixin)

TabBenchmarkTabTransformer = dnn_factory(TabTransformer, create_search_space_saint,
                                         partial(get_recommended_params_dnn, create_search_space_saint),
                                         TabTransformetSaintMixin)


class TabNetMixin(DNNMixin):
    def __init__(
            self,
            categorical_type='float32',
            default_categorical_type='float32',
            torch_optimizer_tuple=deepcopy((torch.optim.Adam, dict(lr=0.02))),
            torch_scheduler_tuple=deepcopy((
                    torch.optim.lr_scheduler.StepLR,
                    {'gamma': 0.4, 'step_size': 8000},
                    {'interval': 'step', 'frequency': 1}
            )),
            lit_trainer_params=deepcopy({
                'gradient_clip_val': 1.0,
                'gradient_clip_algorithm': 'norm',
            }),
            lit_module_class=TabNetModule,
            gamma_sched=None,
            step_sched=None,
            **kwargs
    ):
        super().__init__(categorical_type=categorical_type, torch_optimizer_tuple=torch_optimizer_tuple,
                         torch_scheduler_tuple=torch_scheduler_tuple, lit_trainer_params=lit_trainer_params,
                         lit_module_class=lit_module_class, default_categorical_type=default_categorical_type, **kwargs)
        self.gamma_sched = gamma_sched
        self.step_sched = step_sched

    @property
    def torch_scheduler_tuple(self):
        if self.gamma_sched is not None:
            self._torch_scheduler_tuple[1]['gamma'] = self.gamma_sched
        if self.step_sched is not None:
            self._torch_scheduler_tuple[1]['step_size'] = self.step_sched
        return self._torch_scheduler_tuple


def create_search_space_tabnet():
    # From tabnet
    search_space = dict(
        n_d=optuna.distributions.CategoricalDistribution([8, 16, 24, 32, 64, 128]),
        n_a=optuna.distributions.CategoricalDistribution([8, 16, 24, 32, 64, 128]),
        n_steps=optuna.distributions.IntDistribution(3, 10),
        gamma=optuna.distributions.FloatDistribution(1, 2),
        lambda_sparse=optuna.distributions.FloatDistribution(1e-10, 0.1, log=True),
        virtual_batch_size=optuna.distributions.CategoricalDistribution([256, 512, 1024, 2048, 4096]),
        lr=optuna.distributions.FloatDistribution(0.005, 0.025, log=True),
        momentum=optuna.distributions.FloatDistribution(0.6, 0.98),
        gamma_sched=optuna.distributions.FloatDistribution(0.4, 0.95),
        step_sched=optuna.distributions.IntDistribution(500, 20000),
    )
    default_values = dict(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=0.0001,
        virtual_batch_size=1024,
        lr=0.02,
        momentum=0.9,
        gamma_sched=0.4,
        step_sched=8000,
    )
    return search_space, default_values


TabBenchmarkTabNet = dnn_factory(TabNet, create_search_space_tabnet,
                                 partial(get_recommended_params_dnn, create_search_space_tabnet), TabNetMixin)
