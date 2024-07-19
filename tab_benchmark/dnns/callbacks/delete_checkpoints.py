from lightning.pytorch import callbacks
from pathlib import Path
from shutil import rmtree


class DeleteCheckpoints(callbacks.Callback):
    """Delete checkpoints saved by checkpoint callbacks at the end of the training. (to save disk space, for example)"""
    def on_train_end(self, trainer, pl_module):
        for checkpoint_callback in trainer.checkpoint_callbacks:
            dirpath = checkpoint_callback.dirpath
            if Path(dirpath).is_dir():
                rmtree(dirpath)
