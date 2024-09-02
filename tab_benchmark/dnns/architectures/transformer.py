from __future__ import annotations
import math
from typing import Optional, Callable
import torch
from torch import nn
from tab_benchmark.dnns.architectures.base_architecture import BaseArchitecture
from tab_benchmark.dnns.datasets import TabularDataset
from tab_benchmark.dnns.architectures.utils import EmbeddingLayer, broadcast_to_list
from warnings import warn
from tab_benchmark.utils import check_if_arg_in_args_kwargs_of_fn


class MultiheadAttention(nn.MultiheadAttention):
    # This class is a copy of the MultiheadAttention class from PyTorch, but we replace the out_proj with a custom
    # out_proj, which will allow us to prune the weights of the out_proj layer. Before it was not possible because
    # we were not calling the forward method of the out_proj layer, now it should work if we pass out_proj_weight and
    # out_proj_bias as parameters to prune.
    class OutProj:
        def __init__(self, parent):
            self.parent = parent

        @property
        def weight(self):
            return self.parent.out_proj_weight

        @property
        def bias(self):
            return self.parent.out_proj_bias

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, device = check_if_arg_in_args_kwargs_of_fn(super().__init__, 'device', return_arg=True, *args, **kwargs)
        _, dtype = check_if_arg_in_args_kwargs_of_fn(super().__init__, 'dtype', return_arg=True, *args, **kwargs)
        _, bias = check_if_arg_in_args_kwargs_of_fn(super().__init__, 'bias', return_arg=True, *args, **kwargs)
        factory_kwargs = {'device': device, 'dtype': dtype}
        del self.out_proj
        self.out_proj_weight = nn.Parameter(torch.empty((self.embed_dim, self.embed_dim), **factory_kwargs))
        # direct copy from Linear, which was initialized by default
        nn.init.kaiming_uniform_(self.out_proj_weight, a=math.sqrt(5))
        if bias is not None:
            self.out_proj_bias = nn.Parameter(torch.empty(self.embed_dim, **factory_kwargs))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.out_proj_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.out_proj_bias, -bound, bound)
        else:
            self.out_proj_bias = None
        self.out_proj = self.OutProj(self)


class Transpose(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.transpose(*self.args, **self.kwargs)


class TabularSelfAttentionEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            n_heads,
            feedforward_dim: int = 2048,
            dropout_attn: float = 0.1,
            dropout_ff: float = 0.5,
            dropout_module_class_ff: type[nn.Module] = nn.Dropout,
            activation_fn_1: nn.Module = nn.ReLU(),
            activation_fn_2: nn.Module = nn.Identity(),
            norm_module_class_1: type[nn.Module] = nn.BatchNorm1d,
            norm_module_class_2: type[nn.Module] = nn.BatchNorm1d,
            initialization_fn_1: Optional[Callable] = None,
            initialization_fn_2: Optional[Callable] = None,
            norm_first: bool = True
    ):
        # input_dim = query_dim = "model_dim" = output_dim of MultiheadAttention
        # query_dim = key_dim = value_dim
        # input_dim is evenly distributed among the heads, so it should be divisible by n_heads
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiheadAttention(embed_dim=input_dim, num_heads=n_heads,
                                            dropout=dropout_attn, batch_first=True)
        if isinstance(norm_module_class_1, type(nn.BatchNorm1d)):
            # BatchNorm1d expects input to be of shape (batch_size, input_dim, seq_len),
            # so we need to transpose the input
            self.norm_1 = nn.Sequential(Transpose(-2, -1), norm_module_class_1(input_dim), Transpose(-2, -1))
        else:
            self.norm_1 = norm_module_class_1(input_dim)
        linear_1 = nn.Linear(input_dim, feedforward_dim)
        if initialization_fn_1 is not None:
            initialization_fn_1(linear_1.weight)
        self.feedforward = nn.Sequential(
            linear_1,
            activation_fn_1
        )
        if dropout_ff > 0:
            self.feedforward.append(dropout_module_class_ff(dropout_ff))
        linear_2 = nn.Linear(feedforward_dim, input_dim)
        if initialization_fn_2 is not None:
            initialization_fn_2(linear_2.weight)
        self.feedforward.append(linear_2)
        if isinstance(norm_module_class_2, type(nn.BatchNorm1d)):
            # BatchNorm1d expects input to be of shape (batch_size, input_dim, seq_len),
            # so we need to transpose the input
            self.norm_2 = nn.Sequential(Transpose(-2, -1), norm_module_class_2(input_dim), Transpose(-2, -1))
        else:
            self.norm_2 = norm_module_class_2(input_dim)
        self.activation_fn_2 = activation_fn_2

    def forward(self, x):
        # x = [batch_size, seq_len, input_dim]
        if self.norm_first:
            x = self.norm_1(x)
            x = (x + self.self_attn(x, x, x, need_weights=False)[0])
            x = self.activation_fn_2(x + self.feedforward(self.norm_2(x)))
        else:
            x = self.norm_1((x + self.self_attn(x, x, x, need_weights=False)[0]))
            x = self.activation_fn_2(self.norm_2(x + self.feedforward(x)))
        return x


class Transformer(BaseArchitecture):
    """Adaptation of the Transformer architecture for tabular data.

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

    n_heads:
        The number of attention heads.

    feedforward_dims:
        The dimensions of the feedforward layers.

    dropouts_attn:
        The dropout rates for the attention layers.

    dropouts_ff:
        The dropout rates for the feedforward layers.

    dropouts_module_class_ff:
        The dropout module class for the feedforward layers.

    activation_fns_1:
        The activation functions for the first feedforward layers.

    activation_fns_2:
        The activation functions for the second feedforward layers.

    initialization_fns_1:
        The initialization functions for the first feedforward layers.

    initialization_fns_2:
        The initialization functions for the second feedforward layers.

    norms_module_class_1:
        The normalization module class for the first feedforward layers.

    norms_module_class_2:
        The normalization module class for the second feedforward layers.

    output_activation_fn:
        The activation function to use for the output layer. Defaults to nn.Identity().

    output_initialization_fn:
        The initialization function for the output layer. Defaults to None.

    embedding_dim:
        The dimension of the embedding for all features. Defaults to 256.

    embedding_initialization_fn:
        The initialization function for the embeddings. Defaults to None.

    use_cls_token:
        Whether to use a cls token. Defaults to True.

    norm_first:
        Whether to normalize the input before the self attention layer. Defaults to True.
    """
    params_defined_from_dataset = ['continuous_features_idx', 'categorical_features_idx', 'categorical_dims',
                                   'output_dim']

    def __init__(
            self,
            continuous_features_idx: list[int],
            categorical_features_idx: list[int],
            output_dim: int,
            categorical_dims: list[int],
            n_encoders: Optional[int] = 3,
            n_heads: int = 8,
            feedforward_dims: int | list[int] = 512,
            dropouts_attn: float | list[float] = 0.1,
            dropouts_ff: float | list[float] = 0.5,
            dropouts_module_class_ff: type[nn.Module] | list[type[nn.Module]] = nn.Dropout,
            activation_fns_1: nn.Module | list[nn.Module] = nn.ReLU(),
            activation_fns_2: nn.Module | list[nn.Module] = nn.ReLU(),
            initialization_fns_1: Optional[Callable] = nn.init.kaiming_normal_,
            initialization_fns_2: Optional[Callable] = nn.init.kaiming_normal_,
            norms_module_class_1: type[nn.Module] | list[type[nn.Module]] = nn.BatchNorm1d,
            norms_module_class_2: type[nn.Module] | list[type[nn.Module]] = nn.BatchNorm1d,
            output_activation_fn: nn.Module = nn.Identity(),
            output_initialization_fn: Optional[Callable] = nn.init.xavier_normal_,
            embedding_dim: int = 256,
            embedding_initialization_fn: Optional[Callable] = nn.init.normal_,
            use_cls_token: bool = True,
            norm_first: bool = True,
    ):
        super().__init__()
        if n_encoders is None:
            feedforward_dims, dropouts_attn, dropouts_ff, dropouts_module_class_ff, activation_fns_1, activation_fns_2, \
                norms_module_class_1, norms_module_class_2, initialization_fns_1, initialization_fns_2 = broadcast_to_list(
                feedforward_dims, dropouts_attn, dropouts_ff, dropouts_module_class_ff,
                activation_fns_1, activation_fns_2, norms_module_class_1,
                norms_module_class_2, initialization_fns_1, initialization_fns_2)
        else:
            if isinstance(feedforward_dims, int):
                feedforward_dims = [feedforward_dims] * n_encoders
                feedforward_dims, dropouts_attn, dropouts_ff, dropouts_module_class_ff, activation_fns_1, activation_fns_2, \
                    norms_module_class_1, norms_module_class_2, initialization_fns_1, initialization_fns_2 = broadcast_to_list(
                    feedforward_dims, dropouts_attn, dropouts_ff, dropouts_module_class_ff,
                    activation_fns_1, activation_fns_2, norms_module_class_1,
                    norms_module_class_2, initialization_fns_1, initialization_fns_2)
            else:
                raise ValueError('n_encoders must be provided if feedforward_dims is not an int')
        self.use_cls_token = use_cls_token

        if embedding_dim % n_heads != 0:
            extra_embeddings = n_heads - embedding_dim % n_heads
            warn(
                f'embedding_dim = {embedding_dim} is not divisible by n_heads={n_heads}, we will add {extra_embeddings}'
                f'extra embeddings to make it divisible by n_heads')
            embedding_dim = embedding_dim + extra_embeddings
        # create embeddings
        if self.use_cls_token:
            categorical_dims = categorical_dims + [1]  # add 1 for the cls token
        self.embedding_layer = EmbeddingLayer(categorical_dims, len(continuous_features_idx),
                                              embedding_dim, embedding_dim,
                                              flatten=False, embedding_initialization_fn=embedding_initialization_fn)
        # transformer encoders
        self.transformer_encoders = nn.Sequential()
        for (feedforward_dim, dropout_attn, dropout_ff, dropout_module_class_ff,
             activation_1_fn, activation_2_fn, norm_module_class_1, norm_module_class_2, initialization_fn_1,
             initialization_fn_2) in \
                zip(feedforward_dims, dropouts_attn, dropouts_ff, dropouts_module_class_ff,
                    activation_fns_1, activation_fns_2, norms_module_class_1,
                    norms_module_class_2, initialization_fns_1, initialization_fns_2):
            self.transformer_encoders.append(
                TabularSelfAttentionEncoder(
                    embedding_dim, n_heads, feedforward_dim, dropout_attn, dropout_ff,
                    dropout_module_class_ff, activation_1_fn, activation_2_fn, norm_module_class_1, norm_module_class_2,
                    initialization_fn_1, initialization_fn_2, norm_first=norm_first)
            )
        # output layer
        if self.use_cls_token:
            output_linear = nn.Linear(embedding_dim, output_dim)
            if output_initialization_fn is not None:
                output_initialization_fn(output_linear.weight)
            self.output_layer = nn.Sequential(
                output_linear,
                output_activation_fn
            )
        else:
            output_linear = nn.Linear(embedding_dim * (len(continuous_features_idx) + len(categorical_features_idx)),
                                      output_dim)
            if output_initialization_fn is not None:
                output_initialization_fn(output_linear.weight)
            self.output_layer = nn.Sequential(
                nn.Flatten(),
                output_linear,
                output_activation_fn
            )

    def forward(self, data_from_tabular_dataset):
        x_categorical = data_from_tabular_dataset['x_categorical']
        x_continuous = data_from_tabular_dataset['x_continuous']
        if self.use_cls_token:
            shape_to_cat = list(x_categorical.shape)
            shape_to_cat[-1] = 1
            if x_categorical.numel() > 0:
                x_categorical = torch.cat([
                    x_categorical,
                    torch.zeros(shape_to_cat, device=x_categorical.device, dtype=x_categorical.dtype)
                ], dim=-1)
            else:
                x_categorical = torch.zeros(shape_to_cat, device=x_categorical.device, dtype=torch.int64)
        x_continuous_categorical = self.embedding_layer(x_continuous, x_categorical)
        y_pred = self.transformer_encoders(x_continuous_categorical)
        if self.use_cls_token:
            y_pred = self.output_layer(y_pred[..., -1, :])  # take the cls token
        else:
            y_pred = self.output_layer(y_pred)
        model_output = {'y_pred': y_pred}
        return model_output

    @staticmethod
    def tabular_dataset_to_architecture_kwargs(dataset: TabularDataset):
        return {
            'continuous_features_idx': dataset.continuous_features_idx,
            'categorical_features_idx': dataset.categorical_features_idx,
            'categorical_dims': dataset.categorical_dims,
            'output_dim': dataset.n_classes,
        }
