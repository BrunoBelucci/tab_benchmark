from typing import Optional
from tab_benchmark.dnns.architectures.base_architecture import BaseArchitecture
from tab_benchmark.dnns.utils.external.pytorch_widedeep.tabular.transformers.saint import SAINT
import torch
from tab_benchmark.dnns.datasets import TabularDataset


class Saint(SAINT, BaseArchitecture):
    """SAINT architecture. Wrapper around architecture of SAINT developed in the project pytorch_widedeep.

    Somepalli, Gowthami, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss, and Tom Goldstein. “SAINT: Improved
    Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training.” arXiv, June 2,
    2021. https://doi.org/10.48550/arXiv.2106.01342.

    Parameters
    ----------
    categorical_features_idx:
        List of indices of the categorical features.
    continuous_features_idx:
        List of indices of the continuous features.
    categorical_dims:
        List of the number of categories in each categorical feature.
    output_dim:
        Dimension of output data (typically the number of outputs).
    with_cls_token:
        Whether to use a cls token.
    cat_embed_dropout:
        Dropout for the categorical embeddings.
    use_cat_bias:
        Whether to use a bias in the categorical embeddings.
    cat_embed_activation:
        Activation function for the categorical embeddings, if any. _'tanh'_,
        _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    full_embed_dropout:
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        `pytorch_widedeep.models.transformers._layers.FullEmbeddingDropout`.
        If `full_embed_dropout = True`, `cat_embed_dropout` is ignored.
    shared_embed:
        The idea behind `shared_embed` is described in the Appendix A in the
        [TabTransformer paper](https://arxiv.org/abs/2012.06678): the
        goal of having column embedding is to enable the model to distinguish
        the classes in one column from those in the other columns. In other
        words, the idea is to let the model learn which column is embedded
        at the time.
    add_shared_embed:
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        `frac_shared_embed` with the shared embeddings.
        See `pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
    frac_shared_embed:
        The fraction of embeddings that will be shared (if `add_shared_embed
        = False`) by all the different categories for one particular
        column.
    cont_norm_layer:
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or None.
    cont_embed_dropout:
        Dropout for the continuous embeddings.
    use_cont_bias:
        Whether to use a bias in the continuous embeddings.
    cont_embed_activation:
        Activation function to be applied to the continuous embeddings, if
        any. _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    embedding_dim:
        Dimension of the embeddings for the categorical AND continuous columns.
    n_heads:
        Number of heads in the multi-head attention. (Number of attention heads per Transformer block).
    use_qkv_bias:
        Whether to use a bias in the query, key, and value projections.
    n_blocks:
        Number of SAINT-Transformer blocks.
    attn_dropout:
        Dropout that will be applied to the Multi-Head Attention column and
        row layers
    ff_dropout:
        Dropout for the feed-forward layer.
    ff_factor:
        Multiplicative factor applied to the first layer of the FF network in
        each Transformer block, This is normally set to 4.
    transformer_activation:
        Transformer Encoder activation function. _'tanh'_, _'relu'_,
        _'leaky_relu'_, _'gelu'_, _'geglu'_ and _'reglu'_ are supported.
    mlp_hidden_dims:
        List of hidden dimensions in the MLP.
    mlp_hidden_mult_1:
        Multiplier for the first hidden layer in the MLP.
        If mlp_hidden_dims is not provided it will default to $[l, mlp_hidden_mult_1
        \times l, mlp_hidden_mult_2 \times l]$ where $l$ is the MLP's input dimension.
    mlp_hidden_mult_2:
        Multiplier for the second hidden layer in the MLP.
         If mlp_hidden_dims is not provided it will default to $[l, mlp_hidden_mult_1
        \times l, mlp_hidden_mult_2 \times l]$ where $l$ is the MLP's input dimension.
    mlp_activation:
        MLP activation function. _'tanh'_, _'relu'_, _'leaky_relu'_ and
        _'gelu'_ are supported
    mlp_dropout:
        Dropout for the final MLP.
    mlp_batchnorm:
        Whether to use batch normalization in the MLP.
    mlp_batchnorm_last:
        Whether to use batch normalization in the last layer of the MLP.
    mlp_linear_first:
        Whether to use a linear layer first in the MLP. If `True: [LIN -> ACT -> BN -> DP]`.
        If `False: [BN -> DP -> LIN -> ACT]`
    """
    params_defined_from_dataset = ['continuous_features_idx', 'categorical_features_idx', 'categorical_dims',
                                   'output_dim']

    def __init__(
            self,
            categorical_features_idx: list[int],
            continuous_features_idx: list[int],
            categorical_dims: list[int],
            output_dim: int,
            with_cls_token: bool = True,
            cat_embed_dropout: float = 0.1,
            use_cat_bias: bool = False,
            cat_embed_activation: Optional[str] = None,
            full_embed_dropout: bool = False,
            shared_embed: bool = False,
            add_shared_embed: bool = False,
            frac_shared_embed: float = 0.25,
            cont_norm_layer: str = None,
            cont_embed_dropout: float = 0.1,
            use_cont_bias: bool = True,
            cont_embed_activation: Optional[str] = None,
            embedding_dim: int = 32,  # input_dim
            n_heads: int = 8,
            use_qkv_bias: bool = False,
            n_blocks: int = 6,
            attn_dropout: float = 0.2,
            ff_dropout: float = 0.1,
            ff_factor: int = 4,
            transformer_activation: str = 'gelu',
            mlp_hidden_mult_1: int = 4,
            mlp_hidden_mult_2: int = 2,
            mlp_hidden_dims: Optional[list[int]] = None,
            mlp_activation: str = 'relu',
            mlp_dropout: float = 0.1,
            mlp_batchnorm: bool = False,
            mlp_batchnorm_last: bool = False,
            mlp_linear_first: bool = True,
    ):
        if with_cls_token:
            categorical_dims = [1] + categorical_dims
            categorical_features_idx = [0] + [i + 1 for i in categorical_features_idx]
            continuous_features_idx = [i + 1 for i in continuous_features_idx]
        if mlp_hidden_dims is None:
            if with_cls_token:
                mlp_input_dim = embedding_dim
            else:
                mlp_input_dim = embedding_dim * (len(categorical_features_idx) + len(continuous_features_idx))
            mlp_hidden_dims = [mlp_input_dim, mlp_input_dim * mlp_hidden_mult_1, mlp_input_dim * mlp_hidden_mult_2,
                               output_dim]
        super().__init__(
            categorical_features_idx=categorical_features_idx,
            continuous_features_idx=continuous_features_idx,
            categorical_dims=categorical_dims,
            with_cls_token=with_cls_token,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            full_embed_dropout=full_embed_dropout,
            shared_embed=shared_embed,
            add_shared_embed=add_shared_embed,
            frac_shared_embed=frac_shared_embed,
            cont_norm_layer=cont_norm_layer,
            cont_embed_dropout=cont_embed_dropout,
            use_cont_bias=use_cont_bias,
            cont_embed_activation=cont_embed_activation,
            input_dim=embedding_dim,
            n_heads=n_heads,
            use_qkv_bias=use_qkv_bias,
            n_blocks=n_blocks,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_factor=ff_factor,
            transformer_activation=transformer_activation,
            mlp_hidden_dims=mlp_hidden_dims,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
            mlp_batchnorm=mlp_batchnorm,
            mlp_batchnorm_last=mlp_batchnorm_last,
            mlp_linear_first=mlp_linear_first,
        )

    def forward(self, data_from_tabular_dataset):
        x_categorical = data_from_tabular_dataset['x_categorical']
        if self.with_cls_token:
            x_categorical_shape = list(x_categorical.shape)
            x_categorical_shape[-1] = 1
            x_categorical = torch.cat(
                [torch.zeros(x_categorical_shape, dtype=x_categorical.dtype, device=x_categorical.device),
                 x_categorical],
                dim=-1)
        x_continuous = data_from_tabular_dataset['x_continuous']
        x_continuous_categorical = torch.empty_like(torch.cat([x_categorical, x_continuous], dim=-1))
        x_continuous_categorical[..., self.categorical_features_idx] = x_categorical
        x_continuous_categorical[..., self.continuous_features_idx] = x_continuous
        logits = super().forward(x_continuous_categorical)
        model_output = {'y_pred': logits}
        return model_output

    @staticmethod
    def tabular_dataset_to_architecture_kwargs(dataset: TabularDataset):
        categorical_features_idx = dataset.categorical_features_idx
        continuous_features_idx = dataset.continuous_features_idx
        if dataset.task in ('classification', 'binary_classification'):
            dim_out = len(torch.unique(dataset.y))
        else:
            if len(dataset.y.shape) == 1:
                dim_out = 1
            else:
                dim_out = dataset.y.shape[1]
        return {
            'categorical_features_idx': categorical_features_idx,
            'continuous_features_idx': continuous_features_idx,
            'categorical_dims': dataset.categorical_dims,
            'output_dim': dim_out,
        }
