from __future__ import annotations

from typing import Optional

import torch
from lightning.pytorch.callbacks import Callback
from ray.train import report

from tab_benchmark.utils import evaluate_metric, get_metric_fn


class EvaluateMetric(Callback):
    def __init__(self, eval_metric, eval_name: Optional[str | list[str]] = None, report_to_ray: bool = False,
                 default_metric: Optional[str] = None, last_metric_as_default: bool = True):
        if isinstance(eval_metric, str):
            eval_metric = [eval_metric]
        self.eval_metric = eval_metric
        validation_predictions = {}
        if eval_name is not None:
            if isinstance(eval_name, str):
                eval_name = [eval_name]
            for name in eval_name:
                validation_predictions[name] = []
        else:
            eval_name = []
        self.eval_name = eval_name
        self.validation_predictions = validation_predictions
        self.report_to_ray = report_to_ray
        if last_metric_as_default:
            default_metric = eval_metric[-1]
        self.default_metric = default_metric

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.eval_name:
            if 'name' in batch:
                dataset_name = batch['name']
                name = dataset_name
            else:
                name = 'validation_' + str(dataloader_idx)
            if name in self.eval_name:
                self.validation_predictions[name].append(outputs['y_pred'].detach().cpu())
        else:
            name = 'validation_' + str(dataloader_idx)
            if name not in self.validation_predictions:
                self.validation_predictions[name] = []
            self.validation_predictions[name].append(outputs['y_pred'].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
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

            if self.default_metric is not None:
                if self.default_metric != 'loss_metric':
                    scores['default'] = scores[self.default_metric]

            pl_module.log_dict({f'{name}_{metric}': score for metric, score in scores.items()}, on_epoch=True,
                               on_step=False, prog_bar=True, logger=True)
            if self.report_to_ray:
                report({f'{name}_{metric}': score for metric, score in scores.items()})

            predictions.clear()
