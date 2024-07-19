from __future__ import annotations

import logging
from copy import deepcopy
from typing import Optional, Any, Dict
import os
import uuid
import numpy as np
import torch
from lightning.pytorch.tuner.lr_finder import _LRFinder, _LRCallback
import lightning as L
from lightning.pytorch.utilities.parsing import lightning_hasattr, lightning_setattr
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from lightning_utilities.core.imports import RequirementCache
from lightning.pytorch.utilities.exceptions import MisconfigurationException

_MATPLOTLIB_AVAILABLE = RequirementCache("matplotlib")
log = logging.getLogger('lightning.pytorch')


# This is all copied from the lightning library (tuner/lr_finder.py)
def __lr_finder_restore_params(trainer: "L.Trainer", params: Dict[str, Any]) -> None:
    trainer.strategy.optimizers = params["optimizers"]
    trainer.strategy.lr_scheduler_configs = params["lr_scheduler_configs"]
    trainer.callbacks = params["callbacks"]
    trainer.loggers = params["loggers"]
    loop = trainer.fit_loop
    loop.epoch_loop.max_steps = params["max_steps"]
    loop.max_epochs = params["max_epochs"]
    trainer.limit_val_batches = params["limit_val_batches"]

    loop.load_state_dict(deepcopy(params["loop_state_dict"]))
    loop.restarting = False
    trainer.should_stop = False


def _try_loop_run(trainer: "L.Trainer", params: Dict[str, Any]) -> None:
    loop = trainer.fit_loop
    loop.load_state_dict(deepcopy(params["loop_state_dict"]))
    loop.restarting = False
    loop.run()


def __lr_finder_reset_params(trainer: "L.Trainer", num_training: int, early_stop_threshold: Optional[float]) -> None:
    from lightning.pytorch.loggers.logger import DummyLogger

    trainer.strategy.lr_scheduler_configs = []
    # Use special lr logger callback
    trainer.callbacks = [_LRCallback(num_training, early_stop_threshold, progress_bar_refresh_rate=1)]
    # No logging
    trainer.logger = DummyLogger() if trainer.logger is not None else None
    # Max step set to number of iterations starting at current number of iterations
    trainer.fit_loop.epoch_loop.max_steps = num_training + trainer.global_step
    trainer.fit_loop.max_epochs = num_training + trainer.current_epoch  # ensure also epochs
    # trainer.fit_loop.epoch_loop.
    trainer.limit_val_batches = num_training


def __lr_finder_dump_params(trainer: "L.Trainer") -> Dict[str, Any]:
    return {
        "optimizers": trainer.strategy.optimizers,
        "lr_scheduler_configs": trainer.strategy.lr_scheduler_configs,
        "callbacks": trainer.callbacks,
        "loggers": trainer.loggers,
        "max_steps": trainer.fit_loop.max_steps,
        "max_epochs": trainer.fit_loop.max_epochs,
        "limit_val_batches": trainer.limit_val_batches,
        "loop_state_dict": deepcopy(trainer.fit_loop.state_dict()),
    }


def _determine_lr_attr_name(model: "L.LightningModule", attr_name: str = "") -> str:
    if attr_name:
        if not lightning_hasattr(model, attr_name):
            raise AttributeError(
                f"The attribute name for the learning rate was set to {attr_name}, but"
                " could not find this as a field in `model` or `model.hparams`."
            )
        return attr_name

    attr_options = ("lr", "learning_rate")
    for attr in attr_options:
        if lightning_hasattr(model, attr):
            return attr

    raise AttributeError(
        "When using the learning rate finder, either `model` or `model.hparams` should"
        f" have one of these fields: {attr_options}. If your model has a different name for the learning rate, set"
        f" it with `.lr_find(attr_name=...)`."
    )


# The following functions are adapted (or almost copied) from the fastai library
def valley(lrs: torch.Tensor, losses: torch.Tensor):
    """Suggests a learning rate from the longest valley and returns its index"""
    n = len(losses)
    max_start, max_end = 0, 0

    # find the longest valley
    lds = [1] * n
    for i in range(1, n):
        for j in range(0, i):
            if (losses[i] < losses[j]) and (lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]

    sections = (max_end - max_start) / 3
    idx = max_start + int(sections) + int(sections / 2)

    return float(lrs[idx]), (float(lrs[idx]), losses[idx])


def slide(lrs: torch.Tensor, losses: torch.Tensor, lr_diff: int = 15, thresh: float = .005, adjust_value: float = 1.):
    """Suggests a learning rate following an interval slide rule and returns its index"""
    losses = losses.numpy()
    loss_grad = np.gradient(losses)

    r_idx = -1
    l_idx = r_idx - lr_diff
    local_min_lr = lrs[l_idx]
    while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > thresh):
        local_min_lr = lrs[l_idx]
        r_idx -= 1
        l_idx -= 1

    suggestion = float(local_min_lr) * adjust_value
    idx = np.interp(np.log10(suggestion), np.log10(lrs), losses)
    return suggestion, (suggestion, idx)


def minimum(lrs: torch.Tensor, losses: torch.Tensor):
    """"Suggests a learning rate one-tenth the minumum before divergance and returns its index"""
    lr_min = lrs[losses.argmin()].item()
    loss_idx = losses[min(range(len(lrs)), key=lambda i: abs(lrs[i]-lr_min))]
    return lr_min/10, (lr_min, loss_idx)


def steep(lrs: torch.Tensor, losses: torch.Tensor) -> (float, tuple):
    """Suggests a learning rate when the slope is the steepest and returns its index"""
    grads = (losses[1:]-losses[:-1]) / (lrs[1:].log()-lrs[:-1].log())
    lr_steep = lrs[grads.argmin()].item()
    loss_idx = losses[min(range(len(lrs)), key=lambda i: abs(lrs[i]-lr_steep))]
    return lr_steep, (lr_steep, loss_idx)


# We use the above functions to suggest the learning rate, which seems to be better
class LRFinderEnhanced(_LRFinder):
    def plot(self, suggest: bool = False, show: bool = False, ax: Optional["Axes"] = None,
             suggestion_method: str = 'valley') -> Optional["plt.Figure"]:
        """Plot results from lr_find run
        Args:
            suggest: if True, will mark suggested lr to use with a red point

            show: if True, will show figure

            ax: Axes object to which the plot is to be drawn. If not provided, a new figure is created.

            suggestion_method: the method used to suggest the learning rate. Options are 'valley', 'slide', 'minimum',
                and 'steep'.
        """
        if not _MATPLOTLIB_AVAILABLE:
            raise MisconfigurationException(
                "To use the `plot` method, you must have Matplotlib installed."
                " Install it by running `pip install -U matplotlib`."
            )
        import matplotlib.pyplot as plt

        lrs = self.results["lr"]
        losses = self.results["loss"]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure  # type: ignore[assignment]

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)
        if self.mode == "exponential":
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")

        if suggest:
            lr, loss = self.suggestion(suggestion_method=suggestion_method, return_loss=True)
            ax.plot(lr, loss, markersize=10, marker="o", color="red")

        if show:
            plt.show()

        return fig

    def suggestion(self, skip_begin: int = 10, skip_end: int = 1, suggestion_method: str = 'valley',
                   return_loss=False) -> Optional[float] | tuple[float, float]:
        """This will propose a suggestion for an initial learning rate based on the point with the steepest negative
        gradient.

        Args:
            skip_begin: how many samples to skip in the beginning; helps to avoid too naive estimates
            skip_end: how many samples to skip in the end; helps to avoid too optimistic estimates
            suggestion_method: the method used to suggest the learning rate. Options are 'valley', 'slide', 'minimum',
                and 'steep'.
            return_loss: if True, will return the tuple (suggested_lr, loss) instead of just the suggested_lr

        Returns:
            The suggested initial learning rate to use, or `None` if a suggestion is not possible due to too few
            loss samples.

        """
        if suggestion_method == 'valley':
            suggestion_fn = valley
        elif suggestion_method == 'slide':
            suggestion_fn = slide
        elif suggestion_method == 'minimum':
            suggestion_fn = minimum
        elif suggestion_method == 'steep':
            suggestion_fn = steep
        else:
            raise ValueError(f"Unknown suggestion method: {suggestion_method}")
        lrs = torch.tensor(self.results["lr"][skip_begin:-skip_end])
        losses = torch.tensor(self.results["loss"][skip_begin:-skip_end])
        lrs = lrs[torch.isfinite(losses)]
        losses = losses[torch.isfinite(losses)]

        suggestion, (suggestion, loss) = suggestion_fn(lrs, losses)
        # self._optimal_idx = idx + skip_begin
        if return_loss:
            return suggestion, loss
        else:
            return suggestion


# We use our LRFinderEnhanced class
def lr_find(
    trainer: "L.Trainer",
    model: "L.LightningModule",
    min_lr: float = 1e-8,
    max_lr: float = 1,
    num_training: int = 100,
    mode: str = "exponential",
    early_stop_threshold: Optional[float] = 4.0,
    update_attr: bool = False,
    attr_name: str = "",
    suggestion_method: str = 'valley',
) -> Optional[_LRFinder]:
    """Enables the user to do a range test of good initial learning rates, to reduce the amount of guesswork in picking
    a good starting learning rate.

    Args:
        trainer: A Trainer instance.
        model: Model to tune.
        min_lr: minimum learning rate to investigate
        max_lr: maximum learning rate to investigate
        num_training: number of learning rates to test
        mode: Search strategy to update learning rate after each batch:

            - ``'exponential'``: Increases the learning rate exponentially.
            - ``'linear'``: Increases the learning rate linearly.

        early_stop_threshold: Threshold for stopping the search. If the
            loss at any point is larger than early_stop_threshold*best_loss
            then the search is stopped. To disable, set to None.
        update_attr: Whether to update the learning rate attribute or not.
        attr_name: Name of the attribute which stores the learning rate. The names 'learning_rate' or 'lr' get
            automatically detected. Otherwise, set the name here.
        suggestion_method: the method used to suggest the learning rate. Options are 'valley', 'slide', 'minimum',

    """
    if trainer.fast_dev_run:
        rank_zero_warn("Skipping learning rate finder since `fast_dev_run` is enabled.")
        return None

    # Determine lr attr
    if update_attr:
        attr_name = _determine_lr_attr_name(model, attr_name)

    # Save initial model, that is loaded after learning rate is found
    ckpt_path = os.path.join(trainer.default_root_dir, f".lr_find_{uuid.uuid4()}.ckpt")
    ckpt_path = trainer.strategy.broadcast(ckpt_path)
    trainer.save_checkpoint(ckpt_path)

    start_steps = trainer.global_step

    # Arguments we adjust during the lr finder, save for restoring
    params = __lr_finder_dump_params(trainer)

    # Set to values that are required by the algorithm
    __lr_finder_reset_params(trainer, num_training, early_stop_threshold)

    # Disable standard progress bar for fit
    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.disable()

    # Initialize lr finder object (stores results)
    lr_finder = LRFinderEnhanced(mode, min_lr, max_lr, num_training)

    # Configure optimizer and scheduler
    lr_finder._exchange_scheduler(trainer)

    # Fit, lr & loss logged in callback
    _try_loop_run(trainer, params)

    # Prompt if we stopped early
    if trainer.global_step != num_training + start_steps:
        log.info(f"LR finder stopped early after {trainer.global_step} steps due to diverging loss.")

    # Transfer results from callback to lr finder object
    lr_finder.results.update({"lr": trainer.callbacks[0].lrs, "loss": trainer.callbacks[0].losses})
    lr_finder._total_batch_idx = trainer.fit_loop.total_batch_idx  # for debug purpose

    __lr_finder_restore_params(trainer, params)

    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.enable()

    # Update lr attr if required
    lr_finder.results = trainer.strategy.broadcast(lr_finder.results)
    if update_attr:
        lr = lr_finder.suggestion(suggestion_method=suggestion_method)

        # TODO: log lr.results to self.logger
        if lr is not None:
            lightning_setattr(model, attr_name, lr)
            log.info(f"Learning rate set to {lr}")

    # Restore initial state of model
    trainer._checkpoint_connector.restore(ckpt_path)
    trainer.strategy.remove_checkpoint(ckpt_path)
    trainer.fit_loop.restarting = False  # reset restarting flag as checkpoint restoring sets it to True
    trainer.fit_loop.epoch_loop.val_loop._combined_loader = None

    return lr_finder
