from lightning.pytorch.callbacks import EarlyStopping
from torch import Tensor
import torch
from typing import Optional, Tuple


class EarlyStoppingEnhanced(EarlyStopping):
    """Enhanced version of the EarlyStopping callback from Lightning to allow relative thresholds.

    Follow https://github.com/Lightning-AI/pytorch-lightning/issues/12094 for more information.
    """
    def __init__(self, *args, threshold_mode='abs', **kwargs):
        self.threshold_mode = threshold_mode
        super().__init__(*args, **kwargs)

    def _evaluate_stopping_criteria(self, current: Tensor) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif (self.threshold_mode == 'abs'
              and self.monitor_op(current - self.min_delta, self.best_score.to(current.device))):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        elif (self.threshold_mode == 'rel'
              and self.monitor_op(current, (self.best_score.to(current.device)
                                            + self.min_delta * torch.abs(self.best_score.to(current.device))))):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        elif self.best_score == torch.tensor(torch.inf) or self.best_score == -torch.tensor(torch.inf):
            self.best_score = current
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason

    def _improvement_message(self, current: Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            if self.threshold_mode == 'abs':
                msg = (
                    f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                    f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
                )
            else:  # self.threshold_mode == 'rel':
                msg = (
                    f"Metric {self.monitor} improved by {abs(self.best_score - current) / self.best_score:.3f} >="
                    f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
                )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg
