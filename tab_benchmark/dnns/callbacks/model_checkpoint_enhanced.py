from typing import Optional
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor
import torch


class ModelCheckpointEnhanced(ModelCheckpoint):
    """Enhanced version of the ModelCheckpoint callback from Lightning to allow relative thresholds.

    Follow https://github.com/Lightning-AI/pytorch-lightning/issues/12094 for more information.
    """
    def __init__(self, *args, min_delta=0, threshold_mode='abs', **kwargs):
        self.min_delta = min_delta
        self.threshold_mode = threshold_mode
        super().__init__(*args, **kwargs)

    def check_monitor_top_k(self, trainer: "pl.Trainer", current: Optional[Tensor] = None) -> bool:
        if current is None:
            return False

        if self.save_top_k == -1:
            return True

        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True

        monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
        min_delta = self.min_delta
        min_delta *= 1 if monitor_op == torch.gt else -1
        if self.threshold_mode == 'abs':
            should_update_best_and_save = monitor_op(current - min_delta, self.best_k_models[self.kth_best_model_path])
        else:  # self.threshold_mode == 'rel'
            should_update_best_and_save = monitor_op(current, (self.best_k_models[self.kth_best_model_path]
                                                               + min_delta * torch.abs(self.best_k_models[
                                                                                           self.kth_best_model_path])))



        # If using multiple devices, make sure all processes are unanimous on the decision.
        should_update_best_and_save = trainer.strategy.reduce_boolean_decision(bool(should_update_best_and_save))

        return should_update_best_and_save
