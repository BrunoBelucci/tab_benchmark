from __future__ import annotations
from typing import Callable, Optional
import lightning as L
import torch.nn as nn
from tab_benchmark.dnns.datasets import TabularDataset
from tab_benchmark.dnns.architectures import BaseArchitecture
from copy import deepcopy


class TabularModule(L.LightningModule):
    """LightningModule for tabular data."""
    def __init__(
            self,
            dnn_architecture_class: type[BaseArchitecture] | type[nn.Module],
            architecture_kwargs: dict,
            torch_optimizer_fn: Callable,
            torch_optimizer_kwargs: dict,
            loss_fn: Callable,
            torch_scheduler_fn: Optional[Callable],
            torch_scheduler_kwargs: Optional[dict],
            lit_scheduler_config: Optional[dict],
            learning_rate: Optional[float] = None,
    ):
        """Initialize TabularModule.

        Args:
            dnn_architecture_class:
                The class of the architecture to use.
            architecture_kwargs:
                The arguments to initialize the architecture.
            torch_optimizer_fn:
                The optimizer function to use.
            torch_optimizer_kwargs:
                The arguments to initialize the optimizer.
            loss_fn:
                The loss function to use.
            torch_scheduler_fn:
                The scheduler function to use, if any.
            torch_scheduler_kwargs:
                The arguments to initialize the scheduler.
            lit_scheduler_config:
                The configuration of the scheduler. Format is defined on
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers.
            learning_rate:
                The learning rate to use. If None, the learning rate will be taken from torch_optimizer_kwargs, if any.
                If both are given, this argument will take precedence, if both are None, the learning rate will be the
                default one from the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = dnn_architecture_class(**architecture_kwargs)
        self.optimizer_fn = torch_optimizer_fn
        self.scheduler_fn = torch_scheduler_fn
        self.optimizer_kwargs = deepcopy(torch_optimizer_kwargs)
        self.scheduler_kwargs = deepcopy(torch_scheduler_kwargs)
        self.scheduler_config = deepcopy(lit_scheduler_config)
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

    @classmethod
    def from_tabular_dataset(cls, dnn_architecture_class, architecture_kwargs_not_from_dataset, dataset: TabularDataset,
                             torch_optimizer_fn, torch_optimizer_kwargs, loss_fn,
                             torch_scheduler_fn=None, torch_scheduler_kwargs=None, lit_scheduler_config=None):
        architecture_kwargs_from_dataset = dnn_architecture_class.tabular_dataset_to_architecture_kwargs(dataset)
        architecture_kwargs_not_from_dataset.update(architecture_kwargs_from_dataset)  # now all model kwargs
        return cls(dnn_architecture_class=dnn_architecture_class,
                   architecture_kwargs=architecture_kwargs_not_from_dataset,
                   torch_optimizer_fn=torch_optimizer_fn, torch_optimizer_kwargs=torch_optimizer_kwargs,
                   loss_fn=loss_fn,
                   torch_scheduler_fn=torch_scheduler_fn, torch_scheduler_kwargs=torch_scheduler_kwargs,
                   lit_scheduler_config=lit_scheduler_config)

    def configure_optimizers(self):
        learning_rate = self.learning_rate
        if learning_rate is not None:
            self.optimizer_kwargs['lr'] = learning_rate
        optimizer = self.optimizer_fn(self.parameters(), **self.optimizer_kwargs)
        if self.scheduler_fn is not None:
            scheduler = self.scheduler_fn(optimizer, **self.scheduler_kwargs)
            lr_scheduler_config = self.scheduler_config.copy()
            lr_scheduler_config['scheduler'] = scheduler
            return [optimizer], [lr_scheduler_config]
        return optimizer

    def forward(self, batch):
        # dimensions convention (batch_size, n_features)
        return self.model(batch)

    def base_step(self, batch, batch_idx):
        y_true = batch['y_train']
        model_output = self.model(batch)
        y_pred = model_output['y_pred']
        loss = self.loss_fn(y_pred, y_true)
        outputs = {
            'loss': loss,
            'y_pred': y_pred,
            'y_true': y_true,
        }
        return outputs

    def training_step(self, train_batch, batch_idx):
        outputs = self.base_step(train_batch, batch_idx)
        return outputs

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        outputs = self.base_step(val_batch, batch_idx)
        return outputs

    def test_step(self, test_batch, batch_idx):
        outputs = self.base_step(test_batch, batch_idx)
        return outputs

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     outputs = self.base_step(batch, batch_idx)
    #     return outputs

    # @property
    # def best_iteration_(self):
    #     if self.trainer is None:
    #         return None
    #     if self.trainer.checkpoint_callback is None:
    #         return None
    #     return self.trainer.checkpoint_callback.best_model_path  # .split('=')[-1].split('.')[0]
