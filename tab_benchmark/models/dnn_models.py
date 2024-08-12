from copy import deepcopy
import mlflow
import torch
from lightning.pytorch.loggers import MLFlowLogger
from ray import tune
from tab_benchmark.dnns.architectures import Node, Saint, TabTransformer, TabNet
from tab_benchmark.dnns.architectures.mlp import MLP
from tab_benchmark.dnns.architectures.resnet import ResNet
from tab_benchmark.dnns.architectures.transformer import Transformer
from tab_benchmark.dnns.modules import TabNetModule
from tab_benchmark.dnns.utils.external.node.lib.facebook_optimizer.optimizer import QHAdam
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.models.factories import TabBenchmarkModelFactory


early_stopping_patience_dnn = 40
max_epochs_dnn = 300


def before_fit_dnn(self, extra_arguments, **fit_arguments):
    if self.log_to_mlflow_if_running:
        run = mlflow.active_run()
        if run:
            self.lit_trainer_params['logger'] = MLFlowLogger(run_id=run.info.run_id,
                                                             tracking_uri=mlflow.get_tracking_uri())
    return fit_arguments


def create_search_space_mlp():
    # Based on Gorishniy, Yury, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
    # “Revisiting Deep Learning Models for Tabular Data.” arXiv, November 10, 2021.
    # https://doi.org/10.48550/arXiv.2106.11959.
    search_space = dict(
        n_layers=tune.randint(1, 16),
        hidden_dims=tune.choice([1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024]),
        dropouts=tune.uniform(0.0, 0.5),
        lr=tune.loguniform(1e-5, 1e-2),
        weight_decay=tune.loguniform(1e-6, 1e-2),
        categorical_embedding_dim=tune.choice([1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512]),
        continuous_embedding_dim=tune.choice([1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512]),
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


def get_recommended_params_mlp():
    default_values_from_search_space = create_search_space_mlp()[1]
    default_values_from_search_space.update(dict(
        max_epochs=max_epochs_dnn,
        auto_early_stopping=True,
        early_stopping_patience=early_stopping_patience_dnn,
    ))
    return default_values_from_search_space


MLPModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    extended_init_kwargs={
        'categorical_type': 'int64',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    has_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    dnn_architecture_cls=MLP,
    add_lr_and_weight_decay_params=True,
    extra_dct={
        'create_search_space': staticmethod(create_search_space_mlp),
        'get_recommended_params': staticmethod(get_recommended_params_mlp),
        'before_fit': before_fit_dnn,
    }
)

ResNetModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    extended_init_kwargs={
        'categorical_type': 'int64',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    has_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    dnn_architecture_cls=ResNet,
    add_lr_and_weight_decay_params=True,
    extra_dct={
        'create_search_space': lambda X: NotImplementedError,
        'get_recommended_params': lambda X: NotImplementedError,
        'before_fit': before_fit_dnn,
    }
)

TransformerModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    extended_init_kwargs={
        'categorical_type': 'int64',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    has_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    dnn_architecture_cls=Transformer,
    add_lr_and_weight_decay_params=True, extra_dct={
        'create_search_space': lambda X: NotImplementedError,
        'get_recommended_params': lambda X: NotImplementedError,
        'before_fit': before_fit_dnn,
    }
)

NodeModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    extended_init_kwargs={
        'categorical_type': 'float32',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    map_default_values_change={
        'torch_optimizer_tuple': deepcopy((QHAdam, dict(nus=(0.7, 1.0), betas=(0.95, 0.998))))
    },
    has_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    dnn_architecture_cls=Node,
    add_lr_and_weight_decay_params=True,
    extra_dct={
        'create_search_space': lambda X: NotImplementedError,
        'get_recommended_params': lambda X: NotImplementedError,
        'before_fit': before_fit_dnn,
    }
)

SaintModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    extended_init_kwargs={
        'categorical_type': 'float32',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    map_default_values_change={
        'torch_optimizer_tuple': deepcopy((torch.optim.AdamW, {'lr': 1e-4, 'weight_decay': 1e-2}))
    },
    has_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    dnn_architecture_cls=Saint,
    add_lr_and_weight_decay_params=True, extra_dct={
        'create_search_space': lambda X: NotImplementedError,
        'get_recommended_params': lambda X: NotImplementedError,
        'before_fit': before_fit_dnn,
    }
)

TabTransformerModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
    extended_init_kwargs={
        'categorical_type': 'float32',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    map_default_values_change={
        'torch_optimizer_tuple': deepcopy((torch.optim.AdamW, {'lr': 1e-4, 'weight_decay': 1e-2}))
    },
    has_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    dnn_architecture_cls=TabTransformer,
    add_lr_and_weight_decay_params=True,
    extra_dct={
        'create_search_space': lambda X: NotImplementedError,
        'get_recommended_params': lambda X: NotImplementedError,
        'before_fit': before_fit_dnn,
    }
)

TabNetModel = TabBenchmarkModelFactory.from_sk_cls(
    DNNModel,
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
    },
    has_early_stopping=True,
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    dnn_architecture_cls=TabNet,
    add_lr_and_weight_decay_params=True,
    extra_dct={
        'create_search_space': lambda X: NotImplementedError,
        'get_recommended_params': lambda X: NotImplementedError,
        'before_fit': before_fit_dnn,
    }
)
