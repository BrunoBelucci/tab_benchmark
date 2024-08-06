from lightning.pytorch.callbacks import Callback


class DefaultLogs(Callback):
    """Default logs callback to log losses and global step.

    This callback logs the losses and the global step at the end of each epoch for the train, validation and test sets.
    It can manage multiple dataloaders (multiple train, validation and test sets).
    """
    def __init__(self, on_epoch=True, on_step=False):
        self.on_epoch = on_epoch
        self.on_step = on_step

    def on_batch_end(self, str_set, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking:
            if 'name' in batch:
                dataset_name = batch['name']
                name = dataset_name + '_loss'
            else:
                name = str_set + '_loss_' + str(dataloader_idx)
            pl_module.log('global_step', float(trainer.global_step), on_epoch=self.on_epoch, on_step=self.on_step)
            pl_module.log(name, outputs['loss'], on_epoch=self.on_epoch, on_step=self.on_step, prog_bar=True,
                          logger=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end('train', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end('validation', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end('test', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
