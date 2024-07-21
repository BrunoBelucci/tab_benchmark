from typing import Optional
import torch
from tab_benchmark.dnns.datasets import TabularDataset
from tab_benchmark.dnns.architectures.base_architecture import BaseArchitecture
from tab_benchmark.dnns.utils.external.pytorch_tabnet.tab_network import TabNet_


class TabNet(TabNet_, BaseArchitecture):
    """TaNet architecture. Wrapper around original architecture of TabNet.

    Arik, Sercan O., and Tomas Pfister. “TabNet: Attentive Interpretable Tabular Learning.” arXiv, December 9,
    2020. https://doi.org/10.48550/arXiv.1908.07442.

    Parameters
    ----------
    input_dim:
        Dimension of input data (typically the number of features).
    output_dim:
        Dimension of output data (typically the number of outputs).
    n_d:
        Dimension of the prediction layer (usually between 4 and 64).
    n_a:
        Dimension of the attention mechanism (usually between 4 and 64).
    n_steps:
        Number of steps in the architecture (usually between 3 and 10).
    gamma:
        Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
    categorical_features_idx:
        List of indices of categorical features.
    categorical_dims:
        List of number of unique values for each categorical feature.
    cat_emb_dim:
        Dimension of the embedding for categorical features.
    n_independent:
        Number of independent GLU layer in each GLU block (default 2).
    n_shared:
        Number of shared GLU layer in each GLU block (default 2).
    epsilon:
        Value for numerical stability. Avoid log(0), this should be kept very low.
    virtual_batch_size:
        Size of the virtual batch (Ghost Batch Normalization).
    momentum:
        Float value between 0 and 1 which will be used for momentum in all batch norm.
    mask_type:
        Either "sparsemax" or "entmax" : this is the masking function to use.
    lambda_sparse:
        Sparse loss coefficient.
    """
    params_defined_from_dataset = ['input_dim', 'categorical_features_idx', 'categorical_dims',
                                   'output_dim']
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_d: int = 8,
            n_a: int = 8,
            n_steps: int = 3,
            gamma: float = 1.3,
            categorical_features_idx: Optional[list[int]] = None,
            categorical_dims: Optional[list[int]] = None,
            cat_emb_dim: int = 1,
            n_independent: int = 2,
            n_shared: int = 2,
            epsilon: float = 1e-15,
            virtual_batch_size: int = 1024,
            momentum: float = 0.02,
            mask_type: str = "sparsemax",
            lambda_sparse: float = 1e-3,
    ):
        if categorical_features_idx is None:
            categorical_features_idx = []
        if categorical_dims is None:
            categorical_dims = []
        self.lambda_sparse = lambda_sparse
        all_idx = set(range(input_dim))
        self.continuous_idx = list(all_idx - set(categorical_features_idx))
        super().__init__(input_dim=input_dim, output_dim=output_dim, n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
                         cat_idxs=categorical_features_idx, cat_dims=categorical_dims, cat_emb_dim=cat_emb_dim,
                         n_independent=n_independent, n_shared=n_shared, epsilon=epsilon,
                         virtual_batch_size=virtual_batch_size, momentum=momentum, mask_type=mask_type)

    def forward(self, data_from_tabular_dataset):
        x_categorical = data_from_tabular_dataset['x_categorical']
        x_continuous = data_from_tabular_dataset['x_continuous']
        x_continuous_categorical = torch.empty_like(torch.cat([x_categorical, x_continuous], dim=-1))
        x_continuous_categorical[..., self.cat_idxs] = x_categorical
        x_continuous_categorical[..., self.continuous_idx] = x_continuous
        x = super().forward(x_continuous_categorical)
        model_output = {'y_pred': x[0], 'M_loss': x[1]*self.lambda_sparse}
        return model_output

    @staticmethod
    def tabular_dataset_to_architecture_kwargs(dataset: TabularDataset):
        if dataset.task in ('classification', 'binary_classification'):
            output_dim = len(torch.unique(dataset.y))
        else:
            if len(dataset.y.shape) == 1:
                output_dim = 1
            else:
                output_dim = dataset.y.shape[1]
        return {
            'categorical_features_idx': dataset.categorical_features_idx,
            'categorical_dims': dataset.categorical_dims,
            'input_dim': len(dataset.continuous_features_idx) + len(dataset.categorical_features_idx),
            'output_dim': output_dim,
        }
