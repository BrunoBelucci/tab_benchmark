from lightning.pytorch.callbacks import Callback


class ReportToOptuna(Callback):
    def __init__(self, optuna_trial, reported_metric, reported_eval_name):
        self.optuna_trial = optuna_trial
        self.reported_metric = reported_metric
        self.reported_eval_name = reported_eval_name
        self.pruned_trial = False

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            metrics = trainer.callback_metrics
            value_to_report = metrics[self.reported_eval_name + '_' + self.reported_metric]
            self.optuna_trial.report(value_to_report.detach().cpu().numpy(), step=trainer.current_epoch)
            if self.optuna_trial.should_prune():
                self.pruned_trial = True
                message = f'Trial was pruned at epoch {trainer.current_epoch}.'
                print(message)
                # https://github.com/Lightning-AI/pytorch-lightning/issues/1406
                trainer.should_stop = True
                trainer.limit_val_batches = 0
