from __future__ import annotations
from pathlib import Path
import copy
from shutil import rmtree
from typing import Optional, Callable, Sequence
from warnings import warn
import optuna
import pandas as pd
import torch
import lightning as L
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities.memory import is_out_of_cpu_memory
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from torch import nn
from torch.utils.data import DataLoader
from tab_benchmark.dnns.callbacks import DefaultLogs
from tab_benchmark.dnns.callbacks.evaluate_metric import EvaluateMetric
from tab_benchmark.dnns.callbacks.report_to_optuna import ReportToOptuna
from tab_benchmark.dnns.callbacks.timer import TimerDNN
from tab_benchmark.dnns.datasets import TabularDataModule, TabularDataset
from tab_benchmark.dnns.modules import TabularModule
from tab_benchmark.utils import sequence_to_list, get_metric_fn, is_oom_error, clear_memory


def get_early_stopping_callback(eval_name, eval_metric, early_stopping_patience) -> list[tuple[type[Callback], dict]]:
    if early_stopping_patience > 0:
        if eval_metric == 'loss':
            higher_is_better = False
        else:
            metric_fn, need_proba, higher_is_better = get_metric_fn(eval_metric)
        if higher_is_better:
            mode = 'max'
        else:
            mode = 'min'
        return [
            (EarlyStopping, dict(monitor=f'{eval_name}_{eval_metric}', patience=early_stopping_patience,
                                 min_delta=0, mode=mode)),
            (ModelCheckpoint, dict(monitor=f'{eval_name}_{eval_metric}', every_n_epochs=1, save_last=True, mode=mode)),
        ]
    else:
        return []


class DNNModel(BaseEstimator, ClassifierMixin, RegressorMixin):
    """A class to train a DNN model using PyTorch and PyTorch Lightning.

    Parameters
    ----------
    max_epochs:
        Number of epochs to train the model. It adds max_epochs to lit_trainer_params. If None, it will not be added.
    batch_size:
        Batch size for training. Default is 1024.
    log_losses:
        If True, it will automatically create DefaultLogs callback. Default is True.
    add_default_root_dir_to_lit_trainer_kwargs:
        If True, it will add default_root_dir to lit_trainer_params. Default is True.
    n_jobs:
        Number of workers for the DataLoader. Default is 0 (no parallelism).
    early_stopping_patience:
        Number of epochs to wait before stopping the training if the validation loss does not improve. Default is 40.
        It adds EarlyStopping callback. If 0, it will not be added.
    max_time:
        Maximum time to train the model. It adds TimerDNN callback. If None, it will not be added. The interruption will
        be done at the end of the epoch, and this will be the parameter duration of the TimerDNN callback. It accepts
        an integer representing the number of seconds or a dict containing key-value compatible with timedelta.
    use_best_model:
        If True, it will load the best model. Default is True.
    output_dir:
        Directory to save the model.
    dnn_architecture_class:
        Class of the DNN architecture.
    loss_fn:
        Loss function. Default is torch.nn.functional.mse_loss.
    architecture_params:
        Parameters for the architecture class.
    architecture_params_not_from_dataset:
        Parameters for the architecture class that are not from the dataset.
    torch_optimizer_tuple:
        Tuple with the optimizer function and its parameters. Default is (torch.optim.AdamW, {}).
    torch_scheduler_tuple:
        Tuple with the scheduler function, its parameters and configuration of the scheduler. Format of the
        configuration of the scheduler is defined on
         https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers.
         Default is (None, {}, {}).
    lit_module_class:
        LightningModule class to be used. More information about lightning modules can be found at:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    lit_datamodule_class:
        LightningDataModule class to be used. More information about lightning modules can be found at:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#
    lit_callbacks_tuples:
        List of tuples with the callback function and its arguments. More information about lightning callbacks can
        be found at: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#. A list of the entry points
        for the callbacks can be found at:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks. Defaults to an empty list.
    lit_trainer_params:
        Dictionary with the arguments to be passed to the trainer. More information about the trainer can be found
        at: https://lightning.ai/docs/pytorch/stable/common/trainer.html. Defaults to an empty dictionary.
    continuous_type:
        Type of continuous features. Default is 'float32'.
    categorical_type:
        Type of categorical features. Default is 'int64'.
    min_occurrences_to_add_category:
        We will add one more category to cat_dims if the least frequent category has less than
        min_occurrences_to_add_category occurrences and if we are inferring the dimensions from the dataset.
    """
    _estimator_type = ['classifier', 'regressor']

    def __init__(
            self,
            max_epochs: Optional[int] = 300,  # will add max_epochs to lit_trainer_params, None to disable,
            batch_size: int = 1024,
            log_losses: bool = True,  # will automatically create DefaultLogs callback, False to disable
            add_default_root_dir_to_lit_trainer_kwargs: bool = True,
            n_jobs: int = 0,  # will add num_workers to lit_datamodule_kwargs
            early_stopping_patience: int = 40,  # will add EarlyStopping callback, 0 to disable
            use_best_model: bool = True,  # will load the best model if True, False to load the last model
            max_time: Optional[str | int] = None,  # will add Timer callback, None to disable
            auto_reduce_batch_size: bool = False,
            output_dir: Optional[Path | str] = None,
            dnn_architecture_class: type[nn.Module] = None,
            loss_fn: Optional[Callable] = torch.nn.functional.mse_loss,
            eval_metric: Optional[Sequence[str]] = None,
            architecture_params: Optional[dict] = None,
            architecture_params_not_from_dataset: Optional[dict] = None,
            torch_optimizer_tuple: Optional[tuple[Callable, dict]] = None,
            torch_scheduler_tuple: Optional[tuple[Callable, dict, dict]] = None,
            lit_module_class: type[TabularModule] = TabularModule,
            lit_datamodule_class: type[TabularDataModule] = TabularDataModule,
            lit_callbacks_tuples: Optional[list[tuple[type[Callback], dict]]] = None,
            lit_trainer_params: Optional[dict] = None,
            continuous_type: Optional[type | str] = 'float32',
            categorical_type: Optional[type | str] = 'int64',
            min_occurrences_to_add_category: int = 10,
            lr: Optional[float] = None,
            weight_decay: Optional[float] = None,
    ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.log_losses = log_losses
        self.add_default_root_dir_to_lit_trainer_kwargs = add_default_root_dir_to_lit_trainer_kwargs
        self.n_jobs = n_jobs
        self.early_stopping_patience = early_stopping_patience
        self.use_best_model = use_best_model
        self.max_time = max_time
        self.auto_reduce_batch_size = auto_reduce_batch_size
        self.output_dir = output_dir if output_dir else Path.cwd() / 'dnn_model_output'
        self.dnn_architecture_class = dnn_architecture_class
        self.loss_fn = loss_fn
        self.eval_metric = eval_metric
        self.architecture_params = architecture_params if architecture_params else {}
        self.architecture_params_not_from_dataset = architecture_params_not_from_dataset if (
            architecture_params_not_from_dataset) else {}
        self._torch_optimizer_tuple = torch_optimizer_tuple if torch_optimizer_tuple else (torch.optim.AdamW, {})
        self._torch_scheduler_tuple = torch_scheduler_tuple if torch_scheduler_tuple else (None, {}, {})
        self.lit_module_class = lit_module_class
        self.lit_datamodule_class = lit_datamodule_class
        self.continuous_type = continuous_type
        self.categorical_type = categorical_type
        self.min_occurrences_to_add_category = min_occurrences_to_add_category
        self.lr = lr
        self.weight_decay = weight_decay
        self.lit_callbacks_tuples = lit_callbacks_tuples if lit_callbacks_tuples else []
        self.lit_trainer_params = lit_trainer_params if lit_trainer_params else {}
        self.lit_datamodule_ = None
        self.lit_module_ = None
        self.lit_callbacks_ = None
        self.lit_trainer_ = None
        self.task_ = None
        self.cat_features_idx_ = None
        self.cat_dims_ = None
        self.n_classes_ = None
        self.pruned_trial = False
        self.report_to_optuna = None
        self.report_loss_to_optuna = None
        self.optuna_trial = None

    def __post_init__(self):
        if self.architecture_params is None and self.architecture_params_not_from_dataset is None:
            raise ValueError(
                'Either architecture_params or architecture_params_not_from_dataset must be specified, even if is'
                'an empty dictionary.')

    @property
    def torch_optimizer_tuple(self):
        if self.lr is not None:
            self._torch_optimizer_tuple[1]['lr'] = self.lr
        if self.weight_decay is not None:
            self._torch_optimizer_tuple[1]['weight_decay'] = self.weight_decay
        return self._torch_optimizer_tuple

    @property
    def torch_scheduler_tuple(self):
        return self._torch_scheduler_tuple

    def initialize_datamodule(self, X, y, task, cat_features_idx, cat_dims, n_classes, eval_set, eval_name,
                              batch_size, new_batch_size=False):
        self.lit_datamodule_ = self.lit_datamodule_class(
            x_train=X,
            y_train=y,
            task=task,
            categorical_features_idx=cat_features_idx,
            categorical_dims=cat_dims,
            n_classes=n_classes,
            eval_sets=eval_set,
            eval_names=eval_name,
            num_workers=self.n_jobs,
            batch_size=batch_size,
            new_batch_size=new_batch_size,
            store_as_tensor=True,
            continuous_type=self.continuous_type,
            categorical_type=self.categorical_type
        )

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series | pd.DataFrame,
            task: str,
            cat_features: Optional[list[int | str]] = None,
            cat_dims: Optional[list[int]] = None,
            n_classes: Optional[int] = None,
            delete_checkpoints: bool = False,
            eval_set: Optional[Sequence[tuple[pd.DataFrame, pd.DataFrame]]] = None,
            eval_name: Optional[Sequence[str]] = None,
            optuna_trial: Optional[optuna.Trial] = None,
            init_model: Optional[Path | str] = None,
    ):
        """Fit the model.

        Parameters
        ----------
        X:
            Features.
        y:
            Target.
        task:
            Task to solve. Can be 'classification', 'binary_classification', 'regression' or 'multi_regression'.
        cat_features:
            Categorical features.
        cat_dims:
            Number of categories for each categorical feature. If None, it will be inferred from the dataset.
        n_classes:
            Number of classes for classification tasks. If None, it will be inferred from the dataset.
        delete_checkpoints:
            If True, it will delete the checkpoints after training. Default is True.
        eval_set:
            Evaluation sets. The last set will be used as validation for early stopping.
        eval_name:
            Names of the evaluation sets. If None, they will be named as 'validation_i', where i is the index of
            the set.
        optuna_trial:
            If report_to_optuna is True, it will report the metrics to this trial. If None, it will not report.
        init_model:
            Path to a lightning model checkpoint to initialize the model and resume training. Default is None.
        """
        if isinstance(y, pd.Series):
            y = y.to_frame()

        cat_features = sequence_to_list(cat_features) if cat_features is not None else []
        if cat_features:
            if isinstance(cat_features[0], str):
                cat_features_idx = [X.columns.get_loc(col) for col in cat_features]
            else:
                cat_features_idx = cat_features
        else:
            cat_features_idx = []

        if cat_dims is None:
            # we will get the dimensions including the possible eval_set
            X_all = pd.concat([X] + ([set[0] for set in eval_set] if eval_set is not None else []), axis=0)
            cat_dims = [X_all.iloc[:, idx].nunique() for idx in cat_features_idx]
            del X_all

            for i, col_index in enumerate(cat_features_idx):
                if X[X.columns[col_index]].value_counts().min() < self.min_occurrences_to_add_category:
                    cat_dims[i] += 1

        if n_classes is None:
            if task in ('classification', 'binary_classification'):
                y_all = pd.concat([y] + ([set[1] for set in eval_set] if eval_set is not None else []), axis=0)
                n_classes = y_all.nunique().iloc[0]
                del y_all
            else:
                n_classes = y.shape[1]

        # initialize model

        # initialize datamodule
        self.initialize_datamodule(X, y, task, cat_features_idx, cat_dims, n_classes, eval_set, eval_name,
                                   self.batch_size)

        # initialize module
        if self.architecture_params_not_from_dataset is not None:
            self.lit_datamodule_.setup('fit')
            train_dataset = self.lit_datamodule_.train_dataset
            self.lit_module_ = self.lit_module_class.from_tabular_dataset(
                dnn_architecture_class=self.dnn_architecture_class,
                architecture_kwargs_not_from_dataset=self.architecture_params_not_from_dataset,
                dataset=train_dataset,
                torch_optimizer_fn=self.torch_optimizer_tuple[0],
                torch_optimizer_kwargs=self.torch_optimizer_tuple[1],
                loss_fn=self.loss_fn,
                torch_scheduler_fn=self.torch_scheduler_tuple[0],
                torch_scheduler_kwargs=self.torch_scheduler_tuple[1],
                lit_scheduler_config=self.torch_scheduler_tuple[2]
            )
        else:  # we have ensured that architecture_params is not None in __post_init__
            self.lit_module_ = self.lit_module_class(
                dnn_architecture_class=self.dnn_architecture_class, architecture_kwargs=self.architecture_params,
                torch_optimizer_fn=self.torch_optimizer_tuple[0], torch_optimizer_kwargs=self.torch_optimizer_tuple[1],
                loss_fn=self.loss_fn,
                torch_scheduler_fn=self.torch_scheduler_tuple[0], torch_scheduler_kwargs=self.torch_scheduler_tuple[1],
                lit_scheduler_config=self.torch_scheduler_tuple[2])

        # initialize callbacks

        # we will consider that the last set of eval_set will be used as validation
        # we will consider that the last metric of eval_metric will be used as validation
        # if no eval_metric is provided, we use only the loss function
        # if no eval_set is provided, we use only the training set
        eval_metric = sequence_to_list(self.eval_metric) if self.eval_metric else []
        eval_name = sequence_to_list(eval_name) if eval_name else ['train']

        callbacks_tuples = []
        if eval_metric:
            callbacks_tuples.append((EvaluateMetric, dict(eval_metric=eval_metric)))

        if self.log_losses:
            callbacks_tuples.append((DefaultLogs, dict()))

        if optuna_trial:
            self.reported_metric = eval_metric[-1] if eval_metric else 'loss'
            self.reported_eval_name = eval_name[-1]
            callbacks_tuples.append((ReportToOptuna, dict(optuna_trial=optuna_trial,
                                                          reported_metric=self.reported_metric,
                                                          reported_eval_name=self.reported_eval_name)))

        callbacks_tuples.extend(get_early_stopping_callback(eval_name[-1], eval_metric[-1] if eval_metric else 'loss',
                                                            self.early_stopping_patience))

        if self.max_time:
            callbacks_tuples.append((TimerDNN, dict(duration=self.max_time)))

        callbacks_tuples.extend(self.lit_callbacks_tuples)
        self.lit_callbacks_ = [fn(**kwargs) for fn, kwargs in callbacks_tuples]

        # initialize trainer
        trainer_kwargs = self.lit_trainer_params.copy()
        if self.max_epochs is not None:
            trainer_kwargs['max_epochs'] = self.max_epochs
        if self.add_default_root_dir_to_lit_trainer_kwargs:
            trainer_kwargs['default_root_dir'] = self.output_dir
        self.lit_trainer_ = L.Trainer(**trainer_kwargs, callbacks=self.lit_callbacks_)

        if init_model:
            ckpt_path = init_model
        else:
            ckpt_path = None

        refit = True
        while refit:
            # fit
            try:
                self.lit_trainer_.fit(self.lit_module_, self.lit_datamodule_, ckpt_path=ckpt_path)
            except RuntimeError as exception:
                if is_oom_error(exception):
                    if self.auto_reduce_batch_size:
                        warn('OOM error detected, reducing batch size by half.')
                        self.batch_size = self.batch_size // 2
                        if self.batch_size <= 2 or is_out_of_cpu_memory(exception):
                            msg = (f"Batch size of {self.batch_size} is too small (<2) or cpu oom, "
                                   f"reducing the batch size will not solve the problem.")
                            raise RuntimeError(msg)
                        clear_memory()
                        # change batch size in datamodule in the next fit
                        self.initialize_datamodule(X, y, task, cat_features_idx, cat_dims, n_classes, eval_set,
                                                   eval_name, self.batch_size, new_batch_size=True)
                        refit = True
                    else:
                        raise exception
                else:
                    raise exception
            else:
                refit = False

        if optuna_trial:
            for callback in self.lit_callbacks_:
                if isinstance(callback, ReportToOptuna):
                    self.pruned_trial = callback.pruned_trial
                    break

        if self.max_time:
            for callback in self.lit_callbacks_:
                if isinstance(callback, TimerDNN):
                    self.reached_timeout = callback.reached_timeout
                    break

        # load best model
        if self.use_best_model:
            if len(self.lit_trainer_.checkpoint_callbacks) > 1:
                warn('More than one checkpoint callback found, using the one in trainer.checkpoint_callback')
            if self.lit_trainer_.checkpoint_callback and self.lit_trainer_.checkpoint_callback.best_model_path != '':
                self.lit_module_ = self.lit_module_class.load_from_checkpoint(
                    self.lit_trainer_.checkpoint_callback.best_model_path)
                if self.lit_trainer_.early_stopping_callback:
                    self.lit_module_.best_iteration_ = (self.lit_trainer_.early_stopping_callback.stopped_epoch
                                                        - self.early_stopping_patience)
                else:
                    self.lit_module_.best_iteration_ = self.lit_trainer_.current_epoch
            else:
                warn('No checkpoint callback found, cannot load best model, using last model instead.')
                self.lit_module_.best_iteration_ = self.lit_trainer_.current_epoch

        # delete checkpoints
        if delete_checkpoints:
            for checkpoint_callback in self.lit_trainer_.checkpoint_callbacks:
                dirpath = checkpoint_callback.dirpath
                if Path(dirpath).is_dir():
                    rmtree(dirpath)

        # needed for predict
        self.task_ = task
        self.cat_features_idx_ = cat_features_idx
        self.cat_dims_ = cat_dims
        self.n_classes_ = n_classes
        self.optuna_trial = optuna_trial
        return self

    def predict(self, X: pd.DataFrame, logits: bool = False):
        """Predict the target of the data X.
        Args:
            X:
                Data to predict.
            logits:
                True if we want the logits (for the classification task), False if we want the class.

        Returns:
            Predictions.
        """
        dataset_kwargs = {
            'x': X,
            'y': None,
            'task': self.task_,
            'categorical_features_idx': self.cat_features_idx_,
            'categorical_dims': self.cat_dims_,
            'n_classes': self.n_classes_,
            'store_as_tensor': True,
            'continuous_type': self.continuous_type,
            'categorical_type': self.categorical_type,
        }
        dataset = TabularDataset(**dataset_kwargs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_jobs)
        preds = self.lit_trainer_.predict(self.lit_module_, dataloaders=dataloader)
        y_pred = [pred['y_pred'] for pred in preds]
        y_pred = torch.vstack(y_pred)
        if self.task_ in ('classification', 'binary_classification') and not logits:
            y_pred = torch.argmax(y_pred, dim=1)
        return pd.DataFrame(y_pred.numpy())

    def predict_proba(self, X: pd.DataFrame):
        """Predict the probabilities of the data X.
        Args:
            X:
                Data to predict.

        Returns:
            Predictions.
        """
        if self.task_ not in ('classification', 'binary_classification'):
            raise ValueError('Not trained for a classification task!')
        logits = self.predict(X, logits=True)
        return pd.DataFrame(softmax(logits, axis=1))

    def __sklearn_clone__(self):
        """Default implementation of clone. See :func:`sklearn.base.clone` for details."""
        estimator = self
        safe = True
        estimator_type = type(estimator)
        if estimator_type is dict:
            return {k: clone(v, safe=safe) for k, v in estimator.items()}
        elif estimator_type in (list, tuple, set, frozenset):
            return estimator_type([clone(e, safe=safe) for e in estimator])
        elif not hasattr(estimator, "get_params") or isinstance(estimator, type):
            if not safe:
                return copy.deepcopy(estimator)
            else:
                if isinstance(estimator, type):
                    raise TypeError(
                        "Cannot clone object. "
                        + "You should provide an instance of "
                        + "scikit-learn estimator instead of a class."
                    )
                else:
                    raise TypeError(
                        "Cannot clone object '%s' (type %s): "
                        "it does not seem to be a scikit-learn "
                        "estimator as it does not implement a "
                        "'get_params' method." % (repr(estimator), type(estimator))
                    )

        klass = estimator.__class__
        new_object_params = estimator.get_params(deep=False)
        for name, param in new_object_params.items():
            new_object_params[name] = clone(param, safe=False)

        new_object = klass(**new_object_params)
        try:
            new_object._metadata_request = copy.deepcopy(estimator._metadata_request)
        except AttributeError:
            pass

        params_set = new_object.get_params(deep=False)

        # quick sanity check of the parameters of the clone
        for name in new_object_params:
            param1 = new_object_params[name]
            param2 = params_set[name]
            if param1 is not param2 and param1 != param2:
                raise RuntimeError(
                    "Cannot clone object %s, as the constructor "
                    "either does not set or modifies parameter %s" % (estimator, name)
                )

        # _sklearn_output_config is used by `set_output` to configure the output
        # container of an estimator.
        if hasattr(estimator, "_sklearn_output_config"):
            new_object._sklearn_output_config = copy.deepcopy(
                estimator._sklearn_output_config
            )
        return new_object
