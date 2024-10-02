from copy import deepcopy
from functools import partial
import mlflow
import torch
from lightning.pytorch.loggers import MLFlowLogger
from ray import tune
from tab_benchmark.dnns.architectures import Node, Saint, TabTransformer, TabNet
from tab_benchmark.dnns.architectures.mlp import MLP
from tab_benchmark.dnns.architectures.resnet import ResNet
from tab_benchmark.dnns.architectures.transformer import Transformer
from tab_benchmark.dnns.callbacks import DefaultLogs
from tab_benchmark.dnns.callbacks.evaluate_metric import EvaluateMetric
from tab_benchmark.dnns.modules import TabNetModule
from tab_benchmark.dnns.utils.external.node.lib.facebook_optimizer.optimizer import QHAdam
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.models.factories import dnn_model_factory
from tab_benchmark.utils import extends

early_stopping_patience_dnn = 40
max_epochs_dnn = 300


def before_fit_dnn(self, X, y, task=None, cat_features=None, cat_dims=None, n_classes=None, eval_set=None,
                   eval_name=None, report_to_optuna=False, optuna_trial=None, init_model=None, **args_and_kwargs):
    fit_arguments = args_and_kwargs.copy() if args_and_kwargs else {}
    if self.log_to_mlflow_if_running:
        self.lit_trainer_params['logger'] = MLFlowLogger(run_id=self.run_id,
                                                         tracking_uri=mlflow.get_tracking_uri())
    fit_arguments.update(dict(X=X, y=y, task=task, cat_features=cat_features, cat_dims=cat_dims, n_classes=n_classes,
                              eval_set=eval_set, eval_name=eval_name, report_to_optuna=report_to_optuna,
                              optuna_trial=optuna_trial, init_model=init_model))
    return fit_arguments


def after_fit_dnn(self, fit_return):
    if self.report_to_optuna:
        if self.report_loss_to_optuna:
            for callback in self.lit_callbacks_:
                if isinstance(callback, DefaultLogs):
                    self.pruned_trial = callback.pruned_trial
                    if self.log_to_mlflow_if_running:
                        log_metrics = {'pruned': int(callback.pruned_trial)}
                        mlflow.log_metrics(log_metrics, run_id=self.run_id)
                    break
        else:
            for callback in self.lit_callbacks_:
                if isinstance(callback, EvaluateMetric):
                    self.pruned_trial = callback.pruned_trial
                    if self.log_to_mlflow_if_running:
                        log_metrics = {'pruned': int(callback.pruned_trial)}
                        mlflow.log_metrics(log_metrics, run_id=self.run_id)
                    break
    return fit_return


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


def get_recommended_params_dnn(create_search_space_dnn_fn):
    default_values_from_search_space = create_search_space_dnn_fn()[1]
    default_values_from_search_space.update(dict(
        max_epochs=max_epochs_dnn,
        auto_early_stopping=True,
        early_stopping_patience=early_stopping_patience_dnn,
    ))
    return default_values_from_search_space


TabBenchmarkMLP = dnn_model_factory(
    MLP,
    default_values={
        'categorical_type': 'int64',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    before_fit_method=before_fit_dnn,
    extra_dct={
        'create_search_space': staticmethod(create_search_space_mlp),
        'get_recommended_params': staticmethod(partial(get_recommended_params_dnn, create_search_space_mlp)),
    }
)

def create_search_space_resnet():
    # Based on Gorishniy, Yury, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
    # “Revisiting Deep Learning Models for Tabular Data.” arXiv, November 10, 2021.
    # https://doi.org/10.48550/arXiv.2106.11959.
    search_space = dict(
        n_blocks=tune.randint(1, 16),
        blocks_dim=tune.choice([1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024]),
        dropout_1=tune.uniform(0.0, 0.5),
        dropout_2=tune.uniform(0.0, 0.5),
        lr=tune.loguniform(1e-5, 1e-2),
        weight_decay=tune.loguniform(1e-6, 1e-2),
        categorical_embedding_dim=tune.choice([1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512]),
        continuous_embedding_dim=tune.choice([1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512]),
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


TabBenchmarkResNet = dnn_model_factory(
    ResNet,
    default_values={
        'categorical_type': 'int64',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    before_fit_method=before_fit_dnn,
    extra_dct={
        'create_search_space': staticmethod(create_search_space_resnet),
        'get_recommended_params': staticmethod(partial(get_recommended_params_dnn, create_search_space_resnet)),
    }
)


def create_search_space_transformer():
    # Based on Gorishniy, Yury, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
    # “Revisiting Deep Learning Models for Tabular Data.” arXiv, November 10, 2021.
    # https://doi.org/10.48550/arXiv.2106.11959.
    search_space = dict(
        n_heads=tune.randint(1, 6),
        feedforward_dims=tune.choice([16, 24, 32, 64, 128, 256, 512, 1024, 2048]),
        dropouts_attn=tune.uniform(0.0, 0.5),
        dropouts_ff=tune.uniform(0.0, 0.5),
        lr=tune.loguniform(1e-5, 1e-2),
        weight_decay=tune.loguniform(1e-6, 1e-2),
        embedding_dim=tune.choice([1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512]),
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


TabBenchmarkTransformer = dnn_model_factory(
    Transformer,
    default_values={
        'categorical_type': 'int64',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
    },
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    before_fit_method=before_fit_dnn,
    extra_dct={
        'create_search_space': staticmethod(create_search_space_transformer),
        'get_recommended_params': staticmethod(partial(get_recommended_params_dnn, create_search_space_transformer)),
    }
)


def create_search_space_node():
    # Following the paper
    search_space = dict(
        num_NODE_layers=tune.choice([1, 2, 4, 8]),
        depth=tune.choice([6, 8]),
        num_trees=tune.choice([1024, 2048]),
        extra_tree_dim=tune.choice([1, 2, 3]),
    )
    default_values = dict(
        num_NODE_layers=1,
        depth=6,
        num_trees=2048,
        extra_tree_dim=1,
    )
    return search_space, default_values


TabBenchmarkNode = dnn_model_factory(
    Node,
    default_values={
        'categorical_type': 'float32',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
        'torch_optimizer_tuple': deepcopy((QHAdam, dict(nus=(0.7, 1.0), betas=(0.95, 0.998))))
    },
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    before_fit_method=before_fit_dnn,
    extra_dct={
        'create_search_space': staticmethod(create_search_space_node),
        'get_recommended_params': staticmethod(partial(get_recommended_params_dnn, create_search_space_node)),
    }
)


def create_search_space_saint():
    # From TabTransformer...there is no hyperparameter tuning policy described in Saint paper.
    search_space = dict(
        embedding_dim=tune.choice([1, 2, 4, 8, 16, 32, 64, 128, 256]),
        n_heads=tune.choice([1, 2, 4, 8]),
        n_blocks=tune.choice([1, 2, 3, 6, 12]),
        mlp_hidden_mult_1=tune.randint(1, 8),
        mlp_hidden_mult_2=tune.randint(1, 3),
        lr=tune.loguniform(1e-6, 1e-3),
        weight_decay=tune.loguniform(1e-6, 1e-1),
        attn_dropout=tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        ff_dropout=tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
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


TabBenchmarkSaint = dnn_model_factory(
    Saint,
    default_values={
        'categorical_type': 'float32',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
        'torch_optimizer_tuple': deepcopy((torch.optim.AdamW, {'lr': 1e-4, 'weight_decay': 1e-2}))
    },
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    before_fit_method=before_fit_dnn,
    extra_dct={
        'create_search_space': staticmethod(create_search_space_saint),
        'get_recommended_params': staticmethod(partial(get_recommended_params_dnn, create_search_space_saint)),
    }
)

TabBenchmarkTabTransformer = dnn_model_factory(
    TabTransformer,
    default_values={
        'categorical_type': 'float32',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
        'torch_optimizer_tuple': deepcopy((torch.optim.AdamW, {'lr': 1e-4, 'weight_decay': 1e-2}))
    },
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    before_fit_method=before_fit_dnn,
    extra_dct={
        'create_search_space': staticmethod(create_search_space_saint),
        'get_recommended_params': staticmethod(partial(get_recommended_params_dnn, create_search_space_saint)),
    }
)


def create_search_space_tabnet():
    # From tabnet
    search_space = dict(
        n_d=tune.choice([8, 16, 24, 32, 64, 128]),
        n_a=tune.choice([8, 16, 24, 32, 64, 128]),
        n_steps=tune.randint(3, 10),
        gamma=tune.uniform(1, 2),
        lambda_sparse=tune.loguniform(1e-10, 0.1),
        virtual_batch_size=tune.choice([256, 512, 1024, 2048, 4096]),
        lr=tune.loguniform(0.005, 0.025),
        momentum=tune.uniform(0.6, 0.98),
        gamma_sched=tune.uniform(0.4, 0.95),
        step_sched=tune.randint(500, 20000),
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


class TabNetModel(DNNModel):
    __doc__ = DNNModel.__doc__

    @extends(DNNModel.__init__)
    def __init__(self, *args, gamma_sched=None, step_sched=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_sched = gamma_sched
        self.step_sched = step_sched

    @property
    def torch_scheduler_tuple(self):
        if self.gamma_sched is not None:
            self._torch_scheduler_tuple[1]['gamma'] = self.gamma_sched
        if self.step_sched is not None:
            self._torch_scheduler_tuple[1]['step_size'] = self.step_sched
        return self._torch_scheduler_tuple


TabBenchmarkTabNet = dnn_model_factory(
    TabNet,
    dnn_model_cls=TabNetModel,
    default_values={
        'categorical_type': 'float32',
        'categorical_encoder': 'ordinal',
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
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
    map_task_to_default_values={
        'classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'binary_classification': {'loss_fn': torch.nn.functional.cross_entropy},
        'regression': {'loss_fn': torch.nn.functional.mse_loss},
        'multi_regression': {'loss_fn': torch.nn.functional.mse_loss},
    },
    before_fit_method=before_fit_dnn,
    extra_dct={
        'create_search_space': staticmethod(create_search_space_tabnet),
        'get_recommended_params': staticmethod(partial(get_recommended_params_dnn, create_search_space_tabnet)),
    }
)
