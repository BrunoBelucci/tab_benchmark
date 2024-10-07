import datetime
import time
from lightning.pytorch.callbacks import Callback


class TimerDNN(Callback):
    """Timer callback to stop training if the maximum time is reached."""
    def __init__(self, duration=3600):
        if isinstance(duration, int):
            duration = datetime.timedelta(seconds=duration)
        elif isinstance(duration, dict):
            duration = datetime.timedelta(**duration)
        else:
            raise ValueError(f"duration must be int or dict, got {type(duration)}")
        self.duration = duration
        self.start_time = None
        self.reached_timeout = False

    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if time.perf_counter() - self.start_time > self.duration.total_seconds():
            self.reached_timeout = True
            trainer.should_stop = True
            trainer.limit_val_batches = 0
            print(f"Timeout reached after {self.duration}")
