from functools import partial
from typing import Callable
import torch
from tab_benchmark.dnns.datasets import TabularDataset
from tab_benchmark.dnns.architectures.base_architecture import BaseArchitecture
from tab_benchmark.dnns.utils.external.node.lib import entmax15, entmoid15
from tab_benchmark.dnns.utils.external.node.lib.arch import DenseBlock


def output_layer_classification(x, n_classes):
    return x[..., :n_classes].mean(dim=-2)


def output_layer_regression(x):
    return x[..., 0].mean(dim=-1, keepdim=True)


class Node(BaseArchitecture):
    """NODE architecture. Wrapper around original architecture of NODE.

    Popov, Sergei, Stanislav Morozov, and Artem Babenko. “Neural Oblivious Decision Ensembles for Deep Learning on
    Tabular Data.” arXiv, September 19, 2019. https://doi.org/10.48550/arXiv.1909.06312.

    Parameters
    ----------
    input_dim:
        Dimension of input data (typically the number of features).
    tree_dim:
        Number of response channels in the response of an individual tree (typically the number of outputs).
    output_layer:
        Function that aggregate response of last trees.
    num_trees:
        Number of trees by layer.
    num_NODE_layers:
        Number of NODE layers.
    depth:
        Number of splits in every tree.
    choice_function:
        Computes feature weights s.t. f(tensor, dim).sum(dim) == 1.
    bin_function:
        Computes tree leaf weights.
    flatten_output:
        Whether to flatten the outputs of the model.
    extra_tree_dim:
        Number of extra channels in the response of an individual tree.
    """
    params_defined_from_dataset = ['input_dim', 'tree_dim', 'output_layer']
    def __init__(
            self,
            input_dim: int,
            tree_dim: int,
            output_layer: Callable[[torch.Tensor], torch.Tensor],
            num_trees: int = 2048,
            num_NODE_layers: int = 1,
            depth: int = 6,
            choice_function: Callable[[torch.Tensor], torch.Tensor] = entmax15,
            bin_function: Callable[[torch.Tensor], torch.Tensor] = entmoid15,
            flatten_output: bool = False,
            extra_tree_dim: int = 1,
    ):
        super().__init__()
        kwargs = {
            'depth': depth,
            'choice_function': choice_function,
            'bin_function': bin_function,
        }
        self.dense_block = DenseBlock(input_dim=input_dim, layer_dim=num_trees, num_layers=num_NODE_layers,
                                      tree_dim=tree_dim + extra_tree_dim, flatten_output=flatten_output,
                                      **kwargs)
        self.output_layer = output_layer

    def forward(self, data_from_tabular_dataset):
        x_categorical = data_from_tabular_dataset['x_categorical']
        x_continuous = data_from_tabular_dataset['x_continuous']
        x_continuous_categorical = torch.cat([x_categorical, x_continuous], dim=-1)
        out = self.output_layer(self.dense_block(x_continuous_categorical))
        model_output = {'y_pred': out}
        return model_output

    @staticmethod
    def tabular_dataset_to_architecture_kwargs(dataset: TabularDataset):
        if dataset.task in ('classification', 'binary_classification'):
            tree_dim = len(torch.unique(dataset.y))
            output_layer = partial(output_layer_classification, n_classes=tree_dim)
        else:
            if len(dataset.y.shape) == 1:
                tree_dim = 1
            else:
                tree_dim = dataset.y.shape[1]
            output_layer = output_layer_regression
        return {
            'input_dim': len(dataset.continuous_features_idx) + len(dataset.categorical_features_idx),
            'tree_dim': tree_dim,
            'output_layer': output_layer,
        }
