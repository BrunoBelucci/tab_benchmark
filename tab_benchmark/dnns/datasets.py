from typing import Optional
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np

numpy_to_torch_type_dict = {
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
        np.complex128 : torch.complex128
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
                 categorical_dims: list[int],
                 store_as_tensor: bool = False, continuous_type: Optional[np.dtype] = None,
                 categorical_type: Optional[np.dtype] = None):
        """Initializes the TabularDataset.

        Args:
            x:
                Features.
            y:
                Targets.
            task:
                Task to solve. Either 'classification' or 'regression'.
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
        if store_as_tensor:
            categorical_type = numpy_to_torch_type_dict[categorical_type]
            continuous_type = numpy_to_torch_type_dict[continuous_type]
            self.x_continuous = torch.as_tensor(x.iloc[:, continuous_features_idx].to_numpy(),
                                                dtype=continuous_type)
            self.x_categorical = torch.as_tensor(x.iloc[:, categorical_features_idx].to_numpy(),
                                                 dtype=categorical_type)
            if task == 'classification':
                self.y = torch.from_numpy(y.to_numpy().squeeze()) if y is not None else None
            else:
                self.y = torch.from_numpy(y.to_numpy()) if y is not None else None
        else:
            self.x_continuous = x.iloc[:, continuous_features_idx].to_numpy(dtype=continuous_type)
            self.x_categorical = x.iloc[:, categorical_features_idx].to_numpy(dtype=categorical_type)
            if task == 'classification':
                self.y = np.squeeze(y.to_numpy()) if y is not None else None
            else:
                self.y = y.to_numpy() if y is not None else None

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.x_continuous)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | np.ndarray]:
        """Gets the data at a given index."""
        x_continuous = self.x_continuous[idx]
        x_categorical = self.x_categorical[idx]
        if self.y is not None:
            y = self.y[idx]
            data = {'x_continuous': x_continuous, 'x_categorical': x_categorical, 'y_train': y}
        else:
            data = {'x_continuous': x_continuous, 'x_categorical': x_categorical}
        return data

    def get_all_data(self) -> dict[str, torch.Tensor | np.ndarray]:
        """Gets all the data."""
        return {'x_continuous': self.x_continuous, 'x_categorical': self.x_categorical, 'y_train': self.y}
