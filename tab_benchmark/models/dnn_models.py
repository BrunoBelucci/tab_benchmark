from copy import deepcopy

import mlflow
import torch
from lightning.pytorch.loggers import MLFlowLogger

from tab_benchmark.dnns.architectures import Node, Saint, TabTransformer, TabNet
from tab_benchmark.dnns.architectures.mlp import MLP
from tab_benchmark.dnns.architectures.resnet import ResNet
from tab_benchmark.dnns.architectures.transformer import Transformer
from tab_benchmark.dnns.modules import TabNetModule
from tab_benchmark.dnns.utils.external.node.lib.facebook_optimizer.optimizer import QHAdam
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.models.factories import TabBenchmarkModelFactory


def before_fit_dnn(self, extra_arguments, **fit_arguments):
    if self.log_to_mlflow_if_running:
        run = mlflow.active_run()
        if run:
            self.lit_trainer_params['logger'] = MLFlowLogger(run_id=run.info.run_id,
                                                             tracking_uri=mlflow.get_tracking_uri())
    return fit_arguments

MLPModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    dnn_architecture_cls=MLP,
    add_lr_and_weight_decay_params=True,
    has_auto_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    extended_init_kwargs={
        'categorical_type': 'int64',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
)


ResNetModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    dnn_architecture_cls=ResNet,
    add_lr_and_weight_decay_params=True,
    has_auto_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    extended_init_kwargs={
        'categorical_type': 'int64',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
)


TransformerModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    dnn_architecture_cls=Transformer,
    add_lr_and_weight_decay_params=True,
    has_auto_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    extended_init_kwargs={
        'categorical_type': 'int64',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
)


NodeModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    dnn_architecture_cls=Node,
    add_lr_and_weight_decay_params=True,
    has_auto_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    extended_init_kwargs={
        'categorical_type': 'float32',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    map_default_values_change={
        'torch_optimizer_tuple': deepcopy((QHAdam, dict(nus=(0.7, 1.0), betas=(0.95, 0.998))))
    }
)


SaintModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    dnn_architecture_cls=Saint,
    add_lr_and_weight_decay_params=True,
    has_auto_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    extended_init_kwargs={
        'categorical_type': 'float32',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    map_default_values_change={
        'torch_optimizer_tuple': deepcopy((torch.optim.AdamW, {'lr': 1e-4, 'weight_decay': 1e-2}))
    }
)


TabTransformerModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    dnn_architecture_cls=TabTransformer,
    add_lr_and_weight_decay_params=True,
    has_auto_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    extended_init_kwargs={
        'categorical_type': 'float32',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    map_default_values_change={
        'torch_optimizer_tuple': deepcopy((torch.optim.AdamW, {'lr': 1e-4, 'weight_decay': 1e-2}))
    }
)


TabNetModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    dnn_architecture_cls=TabNet,
    add_lr_and_weight_decay_params=True,
    has_auto_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    extended_init_kwargs={
        'categorical_type': 'float32',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    map_default_values_change={
        'torch_optimizer_tuple': deepcopy((torch.optim.Adam, dict(lr=0.02))),
        'torch_scheduler_tuple': deepcopy(
            (torch.optim.lr_scheduler.StepLR,
             {'gamma': 0.4, 'step_size': 8000},
             {'interval': 'step', 'frequency': 1})
        ),
        'lit_trainer_params': deepcopy({
            'gradient_clip_val': 1.0,
            'gradient_clip_algorithm': 'norm',
        }),
        'lit_module_class': TabNetModule,
    }
)
