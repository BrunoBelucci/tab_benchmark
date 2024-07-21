from __future__ import annotations
from typing import Optional, Callable
import torch
from torch import nn
from tab_benchmark.dnns.architectures.base_architecture import BaseArchitecture
from tab_benchmark.dnns.datasets import TabularDataset
from tab_benchmark.dnns.architectures.utils import EmbeddingLayer, broadcast_to_list


class MLP(BaseArchitecture):
    """Multi-Layer Perceptron architecture.

    Parameters
    ----------
    categorical_features_idx:
        A list of integers representing the indices of the categorical features.
    continuous_features_idx:
        A list of integers representing the indices of the continuous features.
    output_dim:
        The dimension of the output data, typically the number of classes for classification tasks or
        the number of target variables for regression tasks.
    categorical_dims:
        A list of integers representing the number of unique values for each categorical feature.
    n_layers:
        The number of hidden layers. If None, hidden_dims must be a list of integers representing the
        dimensions of each hidden layer.
    hidden_dims:
        A list of integers representing the dimensions of the hidden layers.
    output_activation_fn:
        The activation function to use for the output layer. Defaults to nn.Identity().
    activation_fns:
        The activation functions to use for each hidden layers. If a single activation function is provided,
        it will be used for all hidden layers. Defaults to nn.ReLU().
    initialization_fns:
        The initialization functions for each hidden layer. If a single initialization function is provided,
        it will be used for all hidden layers.
    output_initialization_fn:
        The initialization function for the output layer. Defaults to None.
    norms_modules_class:
        The normalization module class for each hidden layer. If a single normalization function is provided,
        it will be used for all hidden layers. Defaults to nn.BatchNorm1d.
    dropouts:
        The dropout rate to apply to the hidden layers. If a list is provided, it should have the same length
        as hidden_dims and each value will be used as the dropout rate for the corresponding layer.
    dropouts_modules_class:
        The dropout module class for each hidden layer. If a single dropout function is provided, it will be used
        for all hidden layers. Defaults to nn.Dropout.
    categorical_embedding_dim:
        The dimension of the embedding for categorical features. Defaults to 256.
    continuous_embedding_dim:
        The dimension of the embedding for continuous features. If 1, continuous features are not embedded.
        Defaults to 1.
    embedding_initialization_fn:
        The initialization function for the embeddings. Defaults to None.
    """
    params_defined_from_dataset = ['continuous_features_idx', 'categorical_features_idx', 'categorical_dims',
                                   'output_dim']

    def __init__(
            self,
            continuous_features_idx: list[int],
            categorical_features_idx: list[int],
            output_dim: int,
            categorical_dims: list[int],
            n_layers: Optional[int] = 4,
            hidden_dims: int | list[int] = 256,
            output_activation_fn: nn.Module = nn.Identity(),
            activation_fns: nn.Module | list[nn.Module] = nn.ReLU(),
            initialization_fns: Optional[Callable | list[Callable]] = nn.init.kaiming_normal_,
            output_initialization_fn: Optional[Callable] = nn.init.xavier_normal_,
            norms_modules_class: type[nn.Module] | list[type[nn.Module]] = nn.BatchNorm1d,
            dropouts: float | list[float] = 0.5,
            dropouts_modules_class: type[nn.Module] | list[type[nn.Module]] = nn.Dropout,
            categorical_embedding_dim: int = 256,
            continuous_embedding_dim: Optional[int] = 1,
            embedding_initialization_fn: Optional[Callable] = nn.init.normal_,
    ):
        super().__init__()
        if n_layers is None:
            hidden_dims, activation_fns, dropouts, dropouts_modules_class, initialization_fns, norms_modules_class = (
                broadcast_to_list(
                    hidden_dims, activation_fns, dropouts, dropouts_modules_class, initialization_fns,
                    norms_modules_class)
            )
        else:
            if isinstance(hidden_dims, int):
                hidden_dims = [hidden_dims] * n_layers
                (hidden_dims, activation_fns, dropouts, dropouts_modules_class, initialization_fns,
                 norms_modules_class) = (
                    broadcast_to_list(
                        hidden_dims, activation_fns, dropouts, dropouts_modules_class, initialization_fns,
                        norms_modules_class)
                )
            else:
                raise ValueError('If n_layers is not None, hidden_dims must be an integer.')
        input_dim = (len(continuous_features_idx) * continuous_embedding_dim
                     + len(categorical_features_idx) * categorical_embedding_dim)
        hidden_layer_dims = [input_dim] + hidden_dims
        # create hidden layers
        hidden_modules = []
        for i, (dim, dropout, dropout_modules_class, activation_fn, initialization_fn, norm_modules_class) in enumerate(
                zip(hidden_layer_dims[:-1], dropouts, dropouts_modules_class, activation_fns, initialization_fns,
                    norms_modules_class)):
            linear = nn.Linear(dim, hidden_layer_dims[i + 1])
            if initialization_fn is not None:
                initialization_fn(linear.weight)
            layers = nn.Sequential(
                linear,
                norm_modules_class(hidden_layer_dims[i + 1]),
                activation_fn
            )
            if dropout > 0:
                layers.append(dropout_modules_class(dropout))
            hidden_modules.append(layers)
        self.hidden_layers = nn.Sequential(*hidden_modules)
        linear_output = nn.Linear(hidden_layer_dims[-1], output_dim)
        if output_initialization_fn is not None:
            output_initialization_fn(linear_output.weight)
        self.output_layer = nn.Sequential(
            linear_output,
            output_activation_fn
        )
        # create embeddings
        self.embedding_layer = EmbeddingLayer(categorical_dims, len(continuous_features_idx),
                                              categorical_embedding_dim, continuous_embedding_dim,
                                              flatten=True, embedding_initialization_fn=embedding_initialization_fn)

    def forward(self, data_from_tabular_dataset):
        x_categorical = data_from_tabular_dataset['x_categorical']
        x_continuous = data_from_tabular_dataset['x_continuous']
        model_output = {
            'y_pred':
            self.output_layer(
                self.hidden_layers(
                    self.embedding_layer(x_continuous, x_categorical)))
        }
        return model_output

    @staticmethod
    def tabular_dataset_to_architecture_kwargs(dataset: TabularDataset):
        if dataset.task == 'classification':
            output_dim = len(torch.unique(dataset.y))
        else:
            if len(dataset.y.shape) == 1:
                output_dim = 1
            else:
                output_dim = dataset.y.shape[1]
        return {
            'continuous_features_idx': dataset.continuous_features_idx,
            'categorical_features_idx': dataset.categorical_features_idx,
            'categorical_dims': dataset.categorical_dims,
            'output_dim': output_dim,
        }
