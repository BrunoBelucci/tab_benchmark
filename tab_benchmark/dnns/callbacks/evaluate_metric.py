from __future__ import annotations

from typing import Optional

import torch
from lightning.pytorch.callbacks import Callback
from ray.train import report

from tab_benchmark.utils import evaluate_metric, get_metric_fn


class EvaluateMetric(Callback):
    def __init__(self, eval_metric, report_to_ray: bool = False, default_metric: Optional[str] = None):
        if isinstance(eval_metric, str):
            eval_metric = [eval_metric]
        self.eval_metric = eval_metric
        self.validation_predictions = {}
        self.report_to_ray = report_to_ray
        self.default_metric = default_metric

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking:
            str_set = 'validation'
            dataset_name = None
            if hasattr(trainer.datamodule, str_set + '_datasets'):
                dataset_name = list(getattr(trainer.datamodule, str_set + '_datasets').keys())[dataloader_idx]
            elif hasattr(trainer.datamodule, str_set + '_dataset'):
                if hasattr(getattr(trainer.datamodule, str_set + '_dataset'), 'name'):
                    dataset_name = getattr(trainer.datamodule, str_set + '_dataset').name
            if dataset_name is None:
                dataset_name = str_set + str(dataloader_idx)
            if dataset_name not in self.validation_predictions:
                self.validation_predictions[dataset_name] = []
            self.validation_predictions[dataset_name].append(outputs['y_pred'].detach().cpu())

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
                scores['default'] = scores[self.default_metric]

            dict_to_log = {f'{name}_{metric}': score for metric, score in scores.items()}

            pl_module.log_dict(dict_to_log, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            if self.report_to_ray:
                report(dict_to_log)

            predictions.clear()
