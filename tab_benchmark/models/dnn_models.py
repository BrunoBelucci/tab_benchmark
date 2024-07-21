import torch
from tab_benchmark.dnns.architectures.mlp import MLP
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.models.factories import TabBenchmarkModelFactory
from tab_benchmark.models.xgboost import fn_to_run_before_fit_for_gbdt

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
        'categorical_target_type': 'int64',
        'data_scaler': 'standard',
        'continuous_target_scaler': 'standard',
    },
    fn_to_run_before_fit=fn_to_run_before_fit_for_gbdt,
)