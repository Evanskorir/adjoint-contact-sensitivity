import torch


class EigenCalculator:
    def __init__(self, ngm_small_tensor: torch.Tensor) -> None:
        self.ngm_small_tensor = ngm_small_tensor

        self.dominant_right_eig_vec = None
        self.dominant_left_eig_vec = None
        self.dominant_eig_val = None

    def run(self):
        # --- Compute Right Eigenvectors ---
        eig_val, eig_vec_right = torch.linalg.eig(self.ngm_small_tensor)

        # Find the index of the dominant eigenvalue
        max_eigval_idx = torch.abs(eig_val).argmax()

        # Extract the dominant eigenvalue and right eigenvector
        self.dominant_eig_val = eig_val[max_eigval_idx].real.item()
        self.dominant_right_eig_vec = eig_vec_right[:, max_eigval_idx].real

        # --- Compute Left Eigenvectors (Right eigenvectors of A^T) ---
        eig_val_left, eig_vec_left = torch.linalg.eig(self.ngm_small_tensor.T)

        # Extract the corresponding left eigenvector
        self.dominant_left_eig_vec = eig_vec_left[:, max_eigval_idx].real

