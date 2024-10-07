import pandas as pd
import torch
import numpy as np
from typing import Optional
import lightning as L
from torch.utils.data import DataLoader, Dataset

numpy_and_str_to_torch_type_dict = {
        np.bool_      : torch.bool,
        np.uint8      : torch.uint8,
        np.int8       : torch.int8,
        np.int16      : torch.int16,
        np.int32      : torch.int32,
        np.int64      : torch.int64,
        np.float16    : torch.float16,
        np.float32    : torch.float32,
        np.float64    : torch.float64,
        np.complex64  : torch.complex64,
        np.complex128 : torch.complex128,
        'bool'        : torch.bool,
        'uint8'       : torch.uint8,
        'int8'        : torch.int8,
        'int16'       : torch.int16,
        'int32'       : torch.int32,
        'int64'       : torch.int64,
        'float16'     : torch.float16,
        'float32'     : torch.float32,
        'float64'     : torch.float64,
        'complex64'   : torch.complex64,
        'complex128'  : torch.complex128,
    }


class TabularDataset(Dataset):
    """Dataset for tabular data.

    Pytorch Dataset for tabular data. It can store the data as numpy arrays or torch tensors.

    Attributes:
        x_continuous: torch.Tensor or np.ndarray
            Continuous features.
        x_categorical: torch.Tensor or np.ndarray
            Categorical features.
        y: torch.Tensor or np.ndarray
            Targets.
        task: str
            Task to solve. Either 'classification' or 'regression'.
        continuous_features_idx: list[int]
            List of indices of the continuous features.
        categorical_features_idx: list[int]
            List of indices of the categorical features.
        categorical_dims: list[int]
            List of the number of categories for each categorical feature.
    """
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame | None, task: str, categorical_features_idx: list[int],
                 categorical_dims: list[int], name=None,
                 store_as_tensor: bool = False, continuous_type: Optional[np.dtype] = None,
                 categorical_type: Optional[np.dtype] = None, n_classes: Optional[int] = None):
        """Initializes the TabularDataset.

        Args:
            x:
                Features.
            y:
                Targets.
            task:
                Task to solve. Either 'classification', 'binary_classification', 'regression' or 'multi_regression'.
            categorical_features_idx:
                List of indices of the categorical features.
            categorical_dims:
                List of the number of categories for each categorical feature.
            store_as_tensor:
                Whether to store the data as a tensor or not.
            continuous_type:
                Type of continuous features. Can be any numpy dtype.
            categorical_type:
                Type of categorical features. Can be any numpy dtype.
        """
        all_features = [i for i in range(x.shape[1])]
        continuous_features_idx = list(set(all_features) - set(categorical_features_idx))
        self.task = task
        self.continuous_features_idx = continuous_features_idx
        self.categorical_features_idx = categorical_features_idx
        self.categorical_dims = categorical_dims
        self.name = name
        if store_as_tensor:
            categorical_type = numpy_and_str_to_torch_type_dict[categorical_type]
            continuous_type = numpy_and_str_to_torch_type_dict[continuous_type]
            self.x_continuous = torch.as_tensor(x.iloc[:, continuous_features_idx].to_numpy(),
                                                dtype=continuous_type)
            self.x_categorical = torch.as_tensor(x.iloc[:, categorical_features_idx].to_numpy(),
                                                 dtype=categorical_type)
            if task in ('classification', 'binary_classification'):
                self.y = torch.from_numpy(y.to_numpy().squeeze()) if y is not None else None
            else:
                self.y = torch.from_numpy(y.to_numpy()) if y is not None else None
        else:
            self.x_continuous = x.iloc[:, continuous_features_idx].to_numpy(dtype=continuous_type)
            self.x_categorical = x.iloc[:, categorical_features_idx].to_numpy(dtype=categorical_type)
            if task in ('classification', 'binary_classification'):
                self.y = np.squeeze(y.to_numpy()) if y is not None else None
            else:
                self.y = y.to_numpy() if y is not None else None
        if n_classes is not None:
            self.n_classes = n_classes
        else:
            if task in ('classification', 'binary_classification'):
                if store_as_tensor:
                    self.n_classes = len(torch.unique(self.y))
                else:
                    self.n_classes = len(np.unique(self.y))
            else:
                if len(self.y.shape) == 1:
                    self.n_classes = 1
                else:
                    self.n_classes = self.y.shape[1]

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.x_continuous)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | np.ndarray]:
        """Gets the data at a given index."""
        x_continuous = self.x_continuous[idx]
        x_categorical = self.x_categorical[idx]
        data = {'x_continuous': x_continuous, 'x_categorical': x_categorical}
        if self.y is not None:
            y = self.y[idx]
            data['y'] = y
        return data

    def get_all_data(self) -> dict[str, torch.Tensor | np.ndarray]:
        """Gets all the data."""
        return {'x_continuous': self.x_continuous, 'x_categorical': self.x_categorical, 'y': self.y}


class EmptyDataset(Dataset):
    """Empty dataset for when there is no validation or test set."""
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None


class TabularDataModule(L.LightningDataModule):
    """LightningDataModule for tabular data.

    More info on LightningDataModule:
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html#using-a-datamodule
    https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/datamodules.html

    Attributes:
        x_train: pd.DataFrame
            Training features.
        y_train: pd.DataFrame
            Training targets.
        task: str
            Task to solve. Either 'classification' or 'regression'.
        categorical_features_idx: list
            List of indices of the categorical features.
        categorical_dims: list
            List of the number of categories for each categorical feature.
        eval_sets: list
            List of tuples of validation sets. Each tuple contains a pd.DataFrame of validation features and a
            pd.DataFrame of validation targets.
        num_workers: int
            Number of workers for the DataLoader.
        batch_size: int
            Batch size for the DataLoader.
        store_as_tensor: bool
            Whether to store the data as a tensor or not.
        continuous_type: np.dtype
            Type of continuous features. Can be any numpy dtype.
        categorical_type: np.dtype
            Type of categorical features. Can be any numpy dtype.
    """
    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, task: str, categorical_features_idx: list[int],
                 categorical_dims: list[int],
                 eval_sets: Optional[list[tuple[pd.DataFrame, pd.DataFrame]]] = None,
                 eval_names: Optional[list[str]] = None,
                 num_workers: int = 0, batch_size: int = 1, store_as_tensor: bool = False,
                 continuous_type: Optional[np.dtype] = None, categorical_type: Optional[np.dtype] = None,
                 n_classes: Optional[int] = None, new_batch_size: bool = False):
        """Initializes the TabularDataModule.

        Args:
            x_train:
                Training features.
            y_train:
                Training targets.
            task:
                Task to solve. Either 'classification' or 'regression'.
            categorical_features_idx:
                List of indices of the categorical features.
            categorical_dims:
                List of the number of categories for each categorical feature.
            eval_sets:
                List of tuples of validation sets. Each tuple contains a pd.DataFrame of validation features and a
                pd.DataFrame of validation targets.
            num_workers:
                Number of workers for the DataLoader.
            batch_size:
                Batch size for the DataLoader.
            store_as_tensor:
                Whether to store the data as a tensor or not.
            continuous_type:
                Type of continuous features. Can be any numpy dtype.
            categorical_type:
                Type of categorical features. Can be any numpy dtype.
        """
        super().__init__()
        ignore = ['x_train', 'y_train', 'eval_sets', 'new_batch_size']
        if new_batch_size:
            ignore.append('batch_size')
        self.save_hyperparameters(ignore=ignore)
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series)
        self.x_train = x_train
        self.y_train = y_train
        self.task = task
        self.categorical_features_idx = categorical_features_idx
        self.categorical_dims = categorical_dims
        self.eval_sets = eval_sets
        self.eval_names = eval_names
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.store_as_tensor = store_as_tensor
        self.continuous_type = continuous_type
        self.categorical_type = categorical_type
        self.n_classes = n_classes
        self.train_dataset = None
        self.validation_datasets = None
        self.test_dataset = None

    def setup(self, stage: str):
        """Sets up the data, creating the Datasets for the DataLoader.

        Args:
            stage:
                Stage of the setup. Can be 'fit' or 'test'.
        """
        if stage == 'fit':
            self.train_dataset = TabularDataset(x=self.x_train, y=self.y_train, task=self.task,
                                                categorical_features_idx=self.categorical_features_idx,
                                                categorical_dims=self.categorical_dims,
                                                store_as_tensor=self.store_as_tensor,
                                                continuous_type=self.continuous_type,
                                                categorical_type=self.categorical_type, name='train',
                                                n_classes=self.n_classes)
            if self.eval_sets:
                if self.eval_names is None:
                    self.eval_names = [f'validation_{i}' for i in range(len(self.eval_sets))]
                self.validation_datasets = {}
                for eval_set, name in zip(self.eval_sets, self.eval_names):
                    (x_valid, y_valid) = eval_set
                    self.validation_datasets[name] = (
                        TabularDataset(x=x_valid, y=y_valid, task=self.task, name=name,
                                       categorical_features_idx=self.categorical_features_idx,
                                       categorical_dims=self.categorical_dims, store_as_tensor=self.store_as_tensor,
                                       continuous_type=self.continuous_type, categorical_type=self.categorical_type,
                                       n_classes=self.n_classes)
                    )
        else:
            pass
            # raise NotImplementedError('Stage not implemented')

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        if len(self.train_dataset) >= self.batch_size:
            drop_last = True
        else:
            drop_last = False
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=True, drop_last=drop_last)

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        if self.validation_datasets:
            return [DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                               pin_memory=True, drop_last=False)
                    for name, valid_dataset in self.validation_datasets.items()]
        else:
            # TODO: choose whether to return None or an empty DataLoader
            return DataLoader(EmptyDataset(), batch_size=self.batch_size, drop_last=True)

    # def test_dataloader(self):
    #     if self.test_idx:
    #         return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
    #                           pin_memory=True, drop_last=False)
    #     else:
    #         DataLoader(EmptyDataset(), batch_size=2, drop_last=True)
