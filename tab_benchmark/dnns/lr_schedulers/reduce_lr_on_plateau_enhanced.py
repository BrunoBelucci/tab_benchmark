from torch.optim.lr_scheduler import ReduceLROnPlateau


class ReduceLROnPlateauEnhanced(ReduceLROnPlateau):
    """Enhanced version of ReduceLROnPlateau that corrects the relative thresholds to account for negative values.

    For more information, follow: https://github.com/pytorch/pytorch/issues/47513
    """
    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            # rel_epsilon = 1. - self.threshold
            return a < best - self.threshold * abs(best)

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            # rel_epsilon = self.threshold + 1.
            return a > best + self.threshold * abs(best)

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold
