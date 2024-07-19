import torch
import torch.nn as nn
from copy import deepcopy


class EmbeddingLayer(nn.Module):
    def __init__(self, categorical_dims, n_continuous_features,
                 categorical_embedding_dim=256, continuous_embedding_dim=1, flatten=False,
                 embedding_initialization_fn=nn.init.normal_):
        super().__init__()
        self.flatten = flatten
        if not flatten and (categorical_embedding_dim != continuous_embedding_dim):
            if len(categorical_dims) > 0 and n_continuous_features > 0:
                raise ValueError('categorical_embedding_dim and continuous_embedding_dim must be equal if '
                                 'flatten=False')
        if len(categorical_dims) > 0:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(dim, categorical_embedding_dim) for dim in categorical_dims
            ])
            [embedding_initialization_fn(embedding.weight) for embedding in self.categorical_embeddings]
        else:
            self.categorical_embeddings = None
        if n_continuous_features > 0 and continuous_embedding_dim > 1:
            self.continuous_embeddings = nn.ModuleList([
                nn.Linear(1, continuous_embedding_dim) for _ in range(n_continuous_features)
            ])
            [embedding_initialization_fn(embedding.weight) for embedding in self.continuous_embeddings]
        else:
            self.continuous_embeddings = None

    def forward(self, x_continuous, x_categorical):
        if self.continuous_embeddings is not None:
            x_continuous = torch.stack(
                [emb(x_continuous[:, i].unsqueeze(1)) for i, emb in enumerate(self.continuous_embeddings)], dim=-2)
        else:
            x_continuous = x_continuous.unsqueeze(-1)
        if self.categorical_embeddings is not None:
            x_categorical = torch.stack(
                [emb(x_categorical[:, i]) for i, emb in enumerate(self.categorical_embeddings)], dim=-2)
        else:
            x_categorical = x_categorical.unsqueeze(-1)
        # each x has shape (batch_size, n_features, n_embedding_dim)
        if self.flatten:
            # (batch_size, n_features, n_embedding_dim) -> (batch_size,
            # n_categorical_features * categorical_embedding_dim + n_continuous_features * continuous_embedding_dim)
            x_continuous_categorical = torch.cat([x_continuous.flatten(-2), x_categorical.flatten(-2)], dim=-1)
        else:
            # (batch_size, n_features, n_embedding_dim) and categorical_embedding_dim == continuous_embedding_dim
            if x_categorical.numel() == 0:
                x_continuous_categorical = x_continuous
            elif x_continuous.numel() == 0:
                x_continuous_categorical = x_categorical
            else:
                x_continuous_categorical = torch.cat([x_continuous, x_categorical], dim=-2)
        return x_continuous_categorical


def broadcast_to_list(*args):
    len_args = 0
    for arg in args:
        if isinstance(arg, list):
            len_args = len(arg)
            break
    if len_args == 0:
        raise ValueError(f'At least one argument in {args} should be a list')
    lists = []
    for arg in args:
        if isinstance(arg, list):
            if len(arg) != len_args:
                raise ValueError('All lists should have the same length')
            lists.append(arg)
        else:
            lists.append([deepcopy(arg) for _ in range(len_args)])
    return lists
