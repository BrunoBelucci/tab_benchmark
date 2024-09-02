from typing import Optional
import torch
import torch.nn as nn
from tab_benchmark.dnns.datasets import TabularDataset
from tab_benchmark.dnns.architectures.base_architecture import BaseArchitecture


class RidgeSparseGD(BaseArchitecture):
    """RidgeSparseGD architecture.

    This architecture is a generalization of the Ridge, Lasso and Elastic regression.

    Lounici, Karim, Katia Meziani, and Benjamin Riu. “Muddling Labels for Regularization, a Novel Approach to
    Generalization.” arXiv, February 17, 2021. https://doi.org/10.48550/arXiv.2102.08769.
    """
    def __init__(self, task: str, input_dim: int, output_dim:int,
                 use_eigen_decomposition: bool = True, method: str = 'covariance', tol: float = 1e-10,
                 rank: Optional[int] = None,
                 add_alpha: bool = True, alpha_init: float = 1e2,
                 add_sparsity: bool = False, sparsity_init: float = -0.1, sparsity_epsilon: float = 1e-2,
                 elasticity_init: float = 0.5):
        """Initialize RidgeSparseGD architecture.

        Args:
            task:
                Task type, either 'classification' or 'regression'.
            input_dim:
                Dimension of input data (typically the number of features).
            output_dim:
                Dimension of output data (typically the number of outputs).
            use_eigen_decomposition:
                Whether to use eigen decomposition.
            method:
                Method to use, for the moment only 'covariance'.
            tol:
                Tolerance for eigenvalues to consider as 0.
            rank:
                Rank of the SVD decomposition.
            add_alpha:
                Whether to add an alpha parameter.
            alpha_init:
                Initial value of alpha.
            add_sparsity:
                Whether to add sparsity parameter.
            sparsity_init:
                Initial value of sparsity.
            sparsity_epsilon:
                Epsilon for sparsity.
            elasticity_init:
                Initial value of elasticity.
        """
        # We assume normalized data (mean 0, std 1), so we don't need to add a bias term
        # We assume that the categorical features are one-hot encoded
        super().__init__()
        self.task = task
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_eigen_decomposition = use_eigen_decomposition
        self.add_alpha = add_alpha
        self.method = method
        self.tol = tol
        self.rank = rank
        if self.add_alpha:
            self.alpha_init = alpha_init
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.alpha = torch.tensor(0.0, dtype=torch.float64)
            self._dummy_alpha = nn.Parameter(torch.tensor(alpha_init))
        self.add_sparsity = add_sparsity
        if self.add_sparsity:
            self.sparsity_init = sparsity_init
            self.sparsity_epsilon = sparsity_epsilon
            self.sparsity_coefficient = nn.Parameter(torch.tensor(sparsity_init, dtype=torch.float64))
            self.sparsity_vector = nn.Parameter(torch.zeros(self.input_dim, dtype=torch.float64))
        if self.add_alpha and self.add_sparsity:
            self.elasticity_init = elasticity_init
            self.elasticity = nn.Parameter(torch.tensor(elasticity_init, dtype=torch.float64))
        self.register_buffer('n_zero_eigenvalues_after_rank_reduction', torch.tensor(0))
        if self.method == 'covariance':
            self.register_buffer('Xt_X', torch.zeros(self.input_dim, self.input_dim, dtype=torch.float64))
            self.register_buffer('Xt_y', torch.zeros(self.input_dim, self.output_dim, dtype=torch.float64))
        if self.use_eigen_decomposition:
            # SVD of X = U @ S @ V.T
            # SVD of X.T @ X = V @ S^2 @ V.T = V @ S_2 @ V.T
            if rank is None:
                self.register_buffer('V', torch.zeros(self.input_dim, self.input_dim, dtype=torch.float64))
                self.register_buffer('S', torch.zeros(self.input_dim, dtype=torch.float64))
                self.register_buffer('Ut_y', torch.zeros(self.input_dim, self.output_dim, dtype=torch.float64))
            else:
                self.register_buffer('V', torch.zeros(self.input_dim, self.rank, dtype=torch.float64))
                self.register_buffer('S', torch.zeros(self.rank, dtype=torch.float64))
                self.register_buffer('Ut_y', torch.zeros(self.input_dim, self.output_dim, dtype=torch.float64))

    def pre_forward(self, data_from_tabular_dataset):
        with torch.no_grad():
            x_categorical = data_from_tabular_dataset['x_categorical']
            x_continuous = data_from_tabular_dataset['x_continuous']
            y = data_from_tabular_dataset['y_train']
            x_continuous_categorical = torch.cat([x_categorical, x_continuous], dim=-1)
            if self.task == 'classification':
                y = nn.functional.one_hot(y, num_classes=self.output_dim).to(x_continuous_categorical.dtype)
            if self.method == 'covariance':
                self.Xt_X += x_continuous_categorical.T @ x_continuous_categorical
                self.Xt_y += x_continuous_categorical.T @ y

    def cache_values(self):
        with torch.no_grad():
            if self.use_eigen_decomposition:
                S_2, V = torch.linalg.eigh(self.Xt_X)
                S_2 = S_2.flip(dims=[0])
                V = V.flip(dims=[1])
                if self.rank:
                    S_2 = S_2[:self.rank]
                    V = V[:, :self.rank]
                self.n_zero_eigenvalues_after_rank_reduction = torch.sum(S_2 < self.tol)
                # eigenvectors will already be multiplied by 0, but we do this for clarity
                V[:, S_2 < self.tol] = torch.tensor(0.0)
                # consider small eigenvalues as 0
                S_2[S_2 < self.tol] = torch.tensor(0.0)
                S = torch.sqrt(S_2)
                S_inv = torch.nan_to_num(1 / S, nan=0.0, posinf=0.0, neginf=0.0)
                Ut_y = S_inv[:, None] * (V.T @ self.Xt_y)  # broadcast S_inv
                self.S = S
                self.V = V
                self.Ut_y = Ut_y

    def forward(self, data_from_tabular_dataset):
        x_categorical = data_from_tabular_dataset['x_categorical']
        x_continuous = data_from_tabular_dataset['x_continuous']
        x_continuous_categorical = torch.cat([x_categorical, x_continuous], dim=-1)
        beta = self.get_beta()
        model_output = {'y_pred': x_continuous_categorical @ beta}
        return model_output

    def get_beta(self):
        if self.method == 'covariance':
            if self.add_sparsity:
                beta_sparse = self.get_beta_sparse()
            else:
                beta_sparse = torch.tensor(0.0)
            if self.add_alpha:
                beta_ridge = self.get_beta_ridge()
            else:
                beta_ridge = torch.tensor(0.0)
            if not self.add_alpha and not self.add_sparsity:
                beta = self.get_beta_ridge()  # alpha will be 0
            elif self.add_alpha and self.add_sparsity:
                beta = torch.sigmoid(self.elasticity) * beta_ridge + (1 - torch.sigmoid(self.elasticity)) * beta_sparse
            else:
                beta = beta_ridge + beta_sparse  # alpha or sparsity will be 0
            return beta

    def get_beta_sparse(self):
        sparsity_vector = self.sparsity_vector
        sparsity_vector = sparsity_vector - sparsity_vector.mean()
        sparsity_vector = sparsity_vector * (sparsity_vector.var() + self.sparsity_epsilon)
        sparsity_vector = torch.sigmoid(sparsity_vector * self.sparsity_coefficient)
        if self.use_eigen_decomposition:
            S = self.S[:-self.n_zero_eigenvalues_after_rank_reduction]
            V = self.V[:, :-self.n_zero_eigenvalues_after_rank_reduction]
            Ut_y = self.Ut_y[:-self.n_zero_eigenvalues_after_rank_reduction]
            beta_sparse = torch.diag(1 / sparsity_vector) @ V @ torch.diag(1 / S) @ Ut_y
        else:
            Xt_X = torch.diag(sparsity_vector) @ self.Xt_X @ torch.diag(sparsity_vector)
            Xt_y = torch.diag(sparsity_vector) @ self.Xt_y
            beta_sparse = torch.linalg.lstsq(Xt_X, Xt_y).solution
        return beta_sparse

    def get_beta_ridge(self):
        if self.use_eigen_decomposition:
            if self.n_zero_eigenvalues_after_rank_reduction != 0:
                S = self.S[:-self.n_zero_eigenvalues_after_rank_reduction]
                V = self.V[:, :-self.n_zero_eigenvalues_after_rank_reduction]
                Ut_y = self.Ut_y[:-self.n_zero_eigenvalues_after_rank_reduction]
            else:
                S = self.S
                V = self.V
                Ut_y = self.Ut_y
            S_2_alpha_inv = 1 / (S ** 2 + self.alpha)
            # look at ESL page 66
            beta_ridge = V @ torch.diag(S_2_alpha_inv) @ torch.diag(S) @ Ut_y
        else:
            Xt_X = self.Xt_X + self.alpha * torch.eye(self.Xt_X.shape[0], device=self.Xt_X.device)
            beta_ridge = torch.linalg.lstsq(Xt_X, self.Xt_y).solution
        return beta_ridge

    @staticmethod
    def tabular_dataset_to_architecture_kwargs(dataset: TabularDataset):
        return {
            'task': dataset.task,
            'input_dim': len(dataset.continuous_features_idx) + len(dataset.categorical_features_idx),
            'output_dim': dataset.n_classes,
        }
