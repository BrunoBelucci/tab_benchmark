from __future__ import annotations
from typing import Optional, Callable
import torch
from torch import nn
from tab_benchmark.dnns.architectures.base_architecture import BaseArchitecture
from tab_benchmark.dnns.datasets import TabularDataset
from tab_benchmark.dnns.architectures.utils import EmbeddingLayer, broadcast_to_list


class ResNetBlock(nn.Module):
    """Residual block for ResNet architecture.

    Based on Gorishniy, Yury, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. “Revisiting Deep Learning Models
    for Tabular Data.” arXiv, November 10, 2021. https://doi.org/10.48550/arXiv.2106.11959 and
    on fastai's implementation (lesson 18 of the 2023 fastai course).

    One block consists of two linear layers with batch normalization and an activation function, followed by a dropout
    layer. The input is added to the output of the second linear layer and the result is passed through the final
    activation function.

    input -> linear(in_features, out_features) -> norm_1 -> activation_fn_1 -> dropout_1
    -> linear(out_features, out_features) -> norm_2 -> dropout_2 -> output_1

    input -> linear(in_features, out_features) -> output_2 (if in_features != out_features, otherwise identity)

    output = activation_fn_2(output_1 + output_2)

    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation_fn_1: nn.Module = nn.ReLU(),
            activation_fn_2: nn.Module = nn.ReLU(),
            dropout_1: float = 0.5,
            dropout_module_class_1: type[nn.Module] = nn.Dropout,
            dropout_2: float = 0.5,
            dropout_module_class_2: type[nn.Module] = nn.Dropout,
            norm_module_class_1: type[nn.Module] = nn.BatchNorm1d,
            norm_module_class_2: type[nn.Module] = nn.BatchNorm1d,
            initialization_fn_1: Optional[Callable] = None,
            initialization_fn_2: Optional[Callable] = None,
    ):
        super().__init__()
        linear_1 = nn.Linear(in_features, out_features)
        if initialization_fn_1 is not None:
            initialization_fn_1(linear_1.weight)
        mlp_block_1 = nn.Sequential(
            linear_1,
            norm_module_class_1(out_features),
            activation_fn_1
        )
        if dropout_1 > 0:
            mlp_block_1.append(dropout_module_class_1(dropout_1))
        linear_2 = nn.Linear(out_features, out_features)
        if initialization_fn_2 is not None:
            initialization_fn_2(linear_2.weight)
        mlp_block_2 = nn.Sequential(
            linear_2,
            norm_module_class_2(out_features)
        )
        if dropout_2 > 0:
            mlp_block_2.append(dropout_module_class_2(dropout_2))
        self.mlp_block = nn.Sequential(mlp_block_1, mlp_block_2)
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            if initialization_fn_2 is not None:
                initialization_fn_2(self.shortcut.weight)
        else:
            self.shortcut = nn.Identity()
        self.activation_fn_2 = activation_fn_2

    def forward(self, x):
        return self.activation_fn_2(self.mlp_block(x) + self.shortcut(x))


class ResNet(BaseArchitecture):
    """ResNet architecture for tabular data.

    Based on Gorishniy, Yury, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. “Revisiting Deep Learning Models
    for Tabular Data.” arXiv, November 10, 2021. https://doi.org/10.48550/arXiv.2106.11959 and
    on fastai's implementation (lesson 18 of the 2023 fastai course).

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

    blocks_dims:
        A list of integers representing the number of neurons for each residual block.

    activation_fns_1:
        The activation functions to use for the first linear layer in each block. If a single activation
        function is provided, it will be used for all blocks. Defaults to nn.ReLU().

    initialization_fns_1:
        The initialization functions for the first linear layer in each block. If a single initialization
        function is provided, it will be used for all blocks. Defaults to None.

    norms_modules_class_1:
        The normalization functions for the first linear layer in each block. If a single normalization function
        is provided, it will be used for all blocks. Defaults to nn.BatchNorm1d.

    dropouts_1:
        The dropout rates for the first linear layer in each block. If a single dropout rate is provided, it will
        be used for all blocks. Defaults to 0.5.

    dropouts_modules_class_1:
        The dropout functions for the first linear layer in each block. If a single dropout function is provided,
        it will be used for all blocks. Defaults to nn.Dropout.

    activation_fns_2:
        The activation functions to use for the second linear layer in each block. If a single activation function
        is provided, it will be used for all blocks. Defaults to nn.ReLU().

    initialization_fns_2:
        The initialization functions for the second linear layer in each block. If a single initialization
        function is provided, it will be used for all blocks. Defaults to None.

    norms_modules_class_2:
        The normalization functions for the second linear layer in each block. If a single normalization function
        is provided, it will be used for all blocks. Defaults to nn.BatchNorm1d.

    dropouts_2:
        The dropout rates for the second linear layer in each block. If a single dropout rate is provided, it will
        be used for all blocks. Defaults to 0.5.

    dropouts_modules_class_2:
        The dropout functions for the second linear layer in each block. If a single dropout function is provided,
        it will be used for all blocks. Defaults to nn.Dropout.

    output_activation_fn:
        The activation function to use for the output layer. Defaults to nn.Identity().

    output_initialization_fn:
        The initialization function for the output layer. Defaults to None.

    categorical_embedding_dim:
        The dimension of the embedding for categorical features. Defaults to 256.

    continuous_embedding_dim:
        The dimension of the embedding for continuous features. Defaults to 1.

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
            n_blocks: Optional[int] = 2,
            blocks_dims: int | list[int] = 256,

            activation_fns_1: nn.Module = nn.ReLU(),
            initialization_fns_1: Optional[Callable | list[Callable]] = nn.init.kaiming_normal_,
            norms_modules_class_1: type[nn.Module] | list[type[nn.Module]] = nn.BatchNorm1d,
            dropouts_1: float | list[float] = 0.5,
            dropouts_modules_class_1: type[nn.Module] = nn.Dropout,

            activation_fns_2: nn.Module = nn.ReLU(),
            initialization_fns_2: Optional[Callable | list[Callable]] = nn.init.kaiming_normal_,
            norms_modules_class_2: type[nn.Module] | list[type[nn.Module]] = nn.BatchNorm1d,
            dropouts_2: float | list[float] = 0.5,
            dropouts_modules_class_2: type[nn.Module] = nn.Dropout,

            output_activation_fn: nn.Module = nn.Identity(),
            output_initialization_fn: Optional[Callable] = nn.init.xavier_normal_,

            categorical_embedding_dim: int = 256,
            continuous_embedding_dim: Optional[int] = 1,
            embedding_initialization_fn: Optional[Callable] = nn.init.normal_,
    ):
        super().__init__()
        if n_blocks is None:
            (blocks_dims, activation_fns_1, dropouts_1, dropouts_modules_class_1, initialization_fns_1,
             norms_modules_class_1, activation_fns_2, dropouts_2, dropouts_modules_class_2, initialization_fns_2,
             norms_modules_class_2) = broadcast_to_list(
                blocks_dims, activation_fns_1, dropouts_1, dropouts_modules_class_1, initialization_fns_1,
                norms_modules_class_1, activation_fns_2, dropouts_2, dropouts_modules_class_2, initialization_fns_2,
                norms_modules_class_2)
        else:
            if isinstance(blocks_dims, int):
                blocks_dims = [blocks_dims] * n_blocks
                (blocks_dims, activation_fns_1, dropouts_1, dropouts_modules_class_1, initialization_fns_1,
                 norms_modules_class_1, activation_fns_2, dropouts_2, dropouts_modules_class_2, initialization_fns_2,
                 norms_modules_class_2) = broadcast_to_list(
                    blocks_dims, activation_fns_1, dropouts_1, dropouts_modules_class_1, initialization_fns_1,
                    norms_modules_class_1, activation_fns_2, dropouts_2, dropouts_modules_class_2, initialization_fns_2,
                    norms_modules_class_2)
            else:
                raise ValueError('blocks_dims must be an integer if n_blocks is not None.')
        input_dim = (len(continuous_features_idx) * continuous_embedding_dim
                     + len(categorical_features_idx) * categorical_embedding_dim)
        resnet_block_dims = [input_dim] + blocks_dims
        # create resnet blocks
        resnet_blocks = []
        for i, (dim, dropout_1, dropout_modules_class_1, activation_fn_1, initialization_fn_1, norm_modules_class_1,
                dropout_2, dropout_modules_class_2, activation_fn_2, initialization_fn_2, norm_modules_class_2,) in (
                enumerate(zip(resnet_block_dims[:-1],
                              dropouts_1, dropouts_modules_class_1, activation_fns_1, initialization_fns_1,
                              norms_modules_class_1,
                              dropouts_2, dropouts_modules_class_2, activation_fns_2, initialization_fns_2,
                              norms_modules_class_2))):
            block = ResNetBlock(dim, resnet_block_dims[i + 1], activation_fn_1, activation_fn_2, dropout_1,
                                dropout_modules_class_1, dropout_2, dropout_modules_class_2, norm_modules_class_1,
                                norm_modules_class_2, initialization_fn_1,
                                initialization_fn_2)
            resnet_blocks.append(block)
        self.resnet_blocks = nn.Sequential(*resnet_blocks)
        linear_output = nn.Linear(resnet_block_dims[-1], output_dim)
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
                    self.resnet_blocks(
                        self.embedding_layer(x_continuous, x_categorical)))
        }
        return model_output

    @staticmethod
    def tabular_dataset_to_architecture_kwargs(dataset: TabularDataset):
        return {
            'continuous_features_idx': dataset.continuous_features_idx,
            'categorical_features_idx': dataset.categorical_features_idx,
            'categorical_dims': dataset.categorical_dims,
            'output_dim': dataset.n_classes,
        }
