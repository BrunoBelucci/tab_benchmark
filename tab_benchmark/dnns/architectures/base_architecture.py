import torch.nn as nn
from abc import ABC, abstractmethod
from tab_benchmark.dnns.datasets import TabularDataset


class BaseArchitecture(nn.Module, ABC):
    """Base architecture for any DNN model."""
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, data_from_tabular_dataset: dict) -> dict:
        """Forward pass of the model.

        It gets the data from the dataset as a dictionary and returns the model output as a dictionary.
        """
        pass

    @staticmethod
    @abstractmethod
    def tabular_dataset_to_architecture_kwargs(dataset: TabularDataset) -> dict:
        """Returns the arguments necessary to initialize the architecture that can be inferred from the Dataset."""
        pass
