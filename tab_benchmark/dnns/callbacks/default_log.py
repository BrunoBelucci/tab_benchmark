from lightning.pytorch.callbacks import Callback


class DefaultLogs(Callback):
    """Default logs callback to log losses and global step.

    This callback logs the losses and the global step at the end of each epoch for the train, validation and test sets.
    It can manage multiple dataloaders (multiple train, validation and test sets).
    """
    def __init__(self, on_epoch=True, on_step=False, report_to_optuna=False, optuna_trial=None, report_eval_name=None):
        self.on_epoch = on_epoch
        self.on_step = on_step
        self.report_to_optuna = report_to_optuna
        self.optuna_trial = optuna_trial
        self.report_eval_name = report_eval_name
        self.pruned_trial = False

    def on_batch_end(self, str_set, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking:
            dataset_name = None
            if hasattr(trainer.datamodule, str_set + '_datasets'):
                dataset_name = list(getattr(trainer.datamodule, str_set + '_datasets').keys())[dataloader_idx]
            elif hasattr(trainer.datamodule, str_set + '_dataset'):
                if hasattr(getattr(trainer.datamodule, str_set + '_dataset'), 'name'):
                    dataset_name = getattr(trainer.datamodule, str_set + '_dataset').name
            if dataset_name is None:
                dataset_name = str_set + '_' + str(dataloader_idx)
            pl_module.log('global_step', float(trainer.global_step), on_epoch=self.on_epoch, on_step=self.on_step)
            pl_module.log(dataset_name + '_loss', outputs['loss'], on_epoch=self.on_epoch, on_step=self.on_step,
                          prog_bar=True, logger=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end('train', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end('validation', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end('test', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            if self.report_to_optuna:
                metrics = trainer.callback_metrics
                # convert all metrics to numpy
                if self.report_eval_name:
                    metrics_name = self.report_eval_name + '_loss'
                else:
                    validations_names = [name for name in metrics.keys() if name.startswith('validation_') and name.endswith('_loss')]
                    if validations_names:
                        metrics_name = validations_names[-1]
                    else:
                        train_names = [name for name in metrics.keys() if name.startswith('train_') and name.endswith('_loss')]
                        metrics_name = train_names[-1]
                self.optuna_trial.report(metrics[metrics_name], step=trainer.current_epoch)
                if self.optuna_trial.should_prune():
                    self.pruned_trial = True
                    message = f'Trial was pruned at epoch {trainer.current_epoch}.'
                    print(message)
                    # https://github.com/Lightning-AI/pytorch-lightning/issues/1406
                    trainer.should_stop = True
                    trainer.limit_val_batches = 0
