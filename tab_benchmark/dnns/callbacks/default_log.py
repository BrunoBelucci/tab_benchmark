from lightning.pytorch.callbacks import Callback


class DefaultLogs(Callback):
    """Default logs callback to log losses and global step.

    This callback logs the losses and the global step at the end of each epoch for the train, validation and test sets.
    It can manage multiple dataloaders (multiple train, validation and test sets).
    """
    def __init__(self, on_epoch=True, on_step=False, log_interval_epoch=1):
        self.on_epoch = on_epoch
        self.on_step = on_step
        self.log_interval_epoch = log_interval_epoch
        self.pruned_trial = False

    def on_batch_end(self, str_set, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking and trainer.current_epoch % self.log_interval_epoch == 0:
            dataset_name = None
            if hasattr(trainer.datamodule, str_set + '_datasets'):
                dataset_name = list(getattr(trainer.datamodule, str_set + '_datasets').keys())[dataloader_idx]
            elif hasattr(trainer.datamodule, str_set + '_dataset'):
                if hasattr(getattr(trainer.datamodule, str_set + '_dataset'), 'name'):
                    dataset_name = getattr(trainer.datamodule, str_set + '_dataset').name
            if dataset_name is None:
                dataset_name = str_set + '_' + str(dataloader_idx)
            pl_module.log(dataset_name + '_loss', outputs['loss'], on_epoch=self.on_epoch, on_step=self.on_step,
                          prog_bar=True, logger=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end('train', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end('validation', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end('test', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
