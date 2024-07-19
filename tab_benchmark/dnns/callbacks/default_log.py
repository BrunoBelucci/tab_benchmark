from lightning.pytorch import callbacks
import torch


class DefaultLogs(callbacks.Callback):
    """Default logs callback to log losses and global step.

    This callback logs the losses and the global step at the end of each epoch for the train, validation and test sets.
    It also logs the y_true and y_pred for the test set.
    It can manage multiple dataloaders (multiple train, validation and test sets).
    """
    def __init__(self):
        self.train_step_losses = {}
        self.validation_step_losses = {}
        self.test_step_losses = {}
        self.test_step_y_trues = {}
        self.test_step_y_preds = {}

    def on_batch_end(self, str_set, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking:
            losses_dict = getattr(self, str_set + '_step_losses')
            if dataloader_idx not in losses_dict:
                losses_dict[dataloader_idx] = []
                if str_set == 'test':
                    self.test_step_y_trues[dataloader_idx] = []
                    self.test_step_y_preds[dataloader_idx] = []
            losses_dict[dataloader_idx].append(outputs['loss'].detach().cpu())
            if str_set == 'test':
                self.test_step_y_trues[dataloader_idx].append(outputs['y_true'].detach().cpu())
                self.test_step_y_preds[dataloader_idx].append(outputs['y_pred'].detach().cpu())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end('train', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end('validation', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end('test', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_epoch_end(self, trainer, pl_module, str_set):
        if not trainer.sanity_checking:
            if str_set == 'train':
                pl_module.log('global_step', float(trainer.global_step), on_epoch=True, on_step=False)
            losses_dict = getattr(self, str_set + '_step_losses')
            for dataloader_idx in losses_dict:
                losses = losses_dict[dataloader_idx]
                mean_loss = torch.tensor(losses).mean()
                pl_module.log(str_set + '_loss_' + str(dataloader_idx), mean_loss, on_epoch=True,
                              on_step=False, prog_bar=True)
                losses_dict[dataloader_idx].clear()

    def on_train_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module, 'train')

    def on_validation_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module, 'validation')

    def on_test_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module, 'test')
