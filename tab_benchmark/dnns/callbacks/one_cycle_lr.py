from typing import Optional
from lightning.pytorch.callbacks import Callback, LearningRateFinder
import lightning as L
import torch
from tab_benchmark.dnns.modules import TabularModule
from lightning.pytorch.utilities.seed import isolate_rng
from tab_benchmark.dnns.tunner.lr_finder import lr_find, LRFinderEnhanced
from lightning.pytorch.utilities.exceptions import _TunerExitException


class OneCycleLR(Callback):
    def __init__(self, max_lr: float):
        self.max_lr = max_lr

    def on_fit_start(self, trainer: "L.Trainer", pl_module: TabularModule) -> None:
        scheduler_fn = torch.optim.lr_scheduler.OneCycleLR
        scheduler_kwargs = {
            'max_lr': self.max_lr,
            'total_steps': trainer.estimated_stepping_batches,
        }
        scheduler_config = {
            'interval': 'step',
        }
        pl_module.scheduler_fn = scheduler_fn
        pl_module.scheduler_kwargs = scheduler_kwargs
        pl_module.scheduler_config = scheduler_config
        trainer.strategy.setup_optimizers(trainer)


class AutomaticOneCycleLR(LearningRateFinder, OneCycleLR):
    def __init__(self, min_lr: float = 1e-8,
                 max_lr: float = 1,
                 num_training_steps: int = 100,
                 mode: str = "exponential",
                 early_stop_threshold: Optional[float] = 4.0,
                 attr_name: str = "", suggestion_method: str = 'valley'
                 ):
        LearningRateFinder.__init__(self, min_lr, max_lr, num_training_steps, mode, early_stop_threshold,
                                    update_attr=False, attr_name=attr_name)
        self.lr_finder: Optional[LRFinderEnhanced] = None
        self.optimal_lr = None
        self.suggestion_method = suggestion_method

    def lr_find(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        with isolate_rng():
            self.optimal_lr = lr_find(
                trainer,
                pl_module,
                min_lr=self._min_lr,
                max_lr=self._max_lr,
                num_training=self._num_training_steps,
                mode=self._mode,
                early_stop_threshold=self._early_stop_threshold,
                update_attr=False,  # self._update_attr should be False, we will update it manually later
                attr_name=self._attr_name,
                suggestion_method=self.suggestion_method
            )

        if self._early_exit:
            raise _TunerExitException()

    def on_fit_start(self, trainer: "L.Trainer", pl_module: TabularModule) -> None:
        self.lr_find(trainer, pl_module)
        self.max_lr = self.optimal_lr.suggestion(suggestion_method=self.suggestion_method)
        OneCycleLR.on_fit_start(self, trainer, pl_module)
