import torch


class EigenCalculator:
    def __init__(self, ngm_small_tensor) -> None:

        self.ngm_small_tensor = ngm_small_tensor

        self.dominant_eig_vecs = None
        self.dominant_eig_val = None

        self.compute_dominant_eigen()

    def compute_dominant_eigen(self):
        # Compute eigenvalues and eigenvectors
        eig_vals, eig_vecs = torch.linalg.eig(self.ngm_small_tensor)

        # Find the index of the eigenvalue with the largest magnitude
        max_eigval_idx = torch.abs(eig_vals).argmax()

        # Extract the dominant eigenvalue and its corresponding eigenvector
        self.dominant_eig_val = eig_vals[max_eigval_idx].real.item()
        self.dominant_eig_vecs = eig_vecs[:, max_eigval_idx].real




