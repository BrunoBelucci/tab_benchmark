from warnings import warn
from lightning.pytorch import callbacks


class LoadBestModel(callbacks.Callback):
    """Load the best model from the checkpoint callback at the end of the training.

    Unfortunately it does not work anymore...
    """
    def on_train_end(self, trainer, pl_module):
        if len(trainer.checkpoint_callbacks) > 1:
            warn('More than one checkpoint callback found, using the one in trainer.checkpoint_callback')
        if trainer.checkpoint_callback:
            pl_module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        else:
            warn('No checkpoint callback found, cannot load best model, using last model instead.')
