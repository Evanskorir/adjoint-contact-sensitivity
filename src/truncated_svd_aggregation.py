import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD


class PCAForGradients:
    def __init__(self, n_components, n_age):
        self.n_age = n_age
        self.n_components = n_components
        self.svd_model = TruncatedSVD(n_components=self.n_components)

    def reconstruct_symmetric_grad_matrix(self, grad_mtx: torch.Tensor) -> np.ndarray:
        """
        Reconstructs a symmetric gradient matrix from the flattened upper-triangular elements.
        Args: grad_mtx (torch.Tensor): Flat upper triangular matrix to be reconstructed.
        Returns: np.ndarray: Symmetric matrix reconstructed from the upper triangular part.
        """
        # Initialize zero matrix
        mtx = torch.zeros((self.n_age, self.n_age))
        upper_tri_idx = torch.triu_indices(self.n_age, self.n_age)

        # Flatten the input tensor and assign values to the upper triangle
        mtx[upper_tri_idx[0], upper_tri_idx[1]] = grad_mtx.view(-1)

        # Make the matrix symmetric
        mtx += mtx.T - torch.diag(mtx.diag())

        return mtx.detach().cpu().numpy()  # Convert to numpy

    def fit_transform(self, grad_mtx: torch.Tensor) -> np.ndarray:
        """
        Applies PCA after reconstructing the symmetric gradient matrix and scaling it.
        Args: grad_mtx (torch.Tensor): Flat upper triangular matrix.
        Returns: np.ndarray: PCA-reduced matrix.
        """
        # Reconstruct the gradient matrix
        gradient_matrix = self.reconstruct_symmetric_grad_matrix(grad_mtx)

        # Apply PCA to reduce to the specified number of components
        reduced_data = self.svd_model.fit_transform(gradient_matrix)

        return reduced_data

    def get_singular_values(self) -> np.ndarray:
        """
        Returns the singular values from the PCA decomposition.
        """
        return self.svd_model.singular_values_

    def get_explained_variance(self) -> np.ndarray:
        """
        Returns the variance explained by each principal component.
        """
        return self.svd_model.explained_variance_

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Returns the explained variance ratio (in percentage) for each principal component.
        """
        explained_variance_ratio = self.svd_model.explained_variance_ratio_ * 100
        return explained_variance_ratio
