from __future__ import annotations
from typing import Optional
import torch
from lightning.pytorch.callbacks import Callback
from tab_benchmark.utils import evaluate_metric, get_metric_fn


class EvaluateMetric(Callback):
    def __init__(self, eval_metric, report_to_optuna: bool = False, optuna_trial=None, report_eval_metric=None,
                 report_eval_name=None):
        if isinstance(eval_metric, str):
            eval_metric = [eval_metric]
        self.eval_metric = eval_metric
        self.validation_predictions = {}
        self.report_to_optuna = report_to_optuna
        self.optuna_trial = optuna_trial
        self.report_eval_metric = report_eval_metric
        self.report_eval_name = report_eval_name
        self.pruned_trial = False

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking:
            str_set = 'validation'
            dataset_name = None
            if hasattr(trainer.datamodule, str_set + '_datasets'):  # note the plural
                dataset_name = list(getattr(trainer.datamodule, str_set + '_datasets').keys())[dataloader_idx]
            elif hasattr(trainer.datamodule, str_set + '_dataset'):
                if hasattr(getattr(trainer.datamodule, str_set + '_dataset'), 'name'):
                    dataset_name = getattr(trainer.datamodule, str_set + '_dataset').name
            if dataset_name is None:
                dataset_name = str_set + '_' + str(dataloader_idx)
            if dataset_name not in self.validation_predictions:
                self.validation_predictions[dataset_name] = []
            self.validation_predictions[dataset_name].append(outputs['y_pred'].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        dict_to_log = {}
        for name, predictions in self.validation_predictions.items():
            y_pred = torch.vstack(predictions)
            y_true = trainer.datamodule.valid_datasets[name].y
            scores = {}
            for metric in self.eval_metric:
                if metric == 'auc':
                    n_classes = len(torch.unique(trainer.datamodule.train_dataset.y))
                else:
                    n_classes = None
                metric_fn, need_proba = get_metric_fn(metric, n_classes)
                if need_proba:
                    y_pred_ = torch.softmax(y_pred, dim=1)
                else:
                    y_pred_ = y_pred
                scores[metric] = evaluate_metric(y_true, y_pred_, metric_fn, n_classes)

            dict_to_log.update({f'{name}_{metric}': score for metric, score in scores.items()})
            predictions.clear()

        pl_module.log_dict(dict_to_log, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        if self.report_to_optuna:
            # convert to numpy to avoid serialization issues
            if self.report_eval_metric:
                report_eval_metric = self.report_eval_metric
            else:
                report_eval_metric = self.eval_metric[-1]
            if self.report_eval_name:
                report_eval_name = self.report_eval_name
            else:
                report_eval_name = list(self.validation_predictions.keys())[-1]
            self.optuna_trial.report(dict_to_log[f'{report_eval_name}_{report_eval_metric}'].detach().cpu().numpy(),
                                     step=trainer.current_epoch)
            if self.optuna_trial.should_prune():
                self.pruned_trial = True
                message = f'Trial was pruned at epoch {trainer.current_epoch}.'
                print(message)
                # https://github.com/Lightning-AI/pytorch-lightning/issues/1406
                trainer.should_stop = True
                trainer.limit_val_batches = 0
