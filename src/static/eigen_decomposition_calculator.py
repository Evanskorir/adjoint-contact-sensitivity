import torch

from scipy.linalg import eig


class EigenDecompositionCalculator:
    def __init__(self, ngm_small_tensor: torch.Tensor) -> None:
        self.ngm_small_tensor = ngm_small_tensor

        self.dominant_eigen_val = None
        self.left_eigen_vec = None
        self.right_eigen_vec = None

    def run(self):
        # Compute eigenvalues and both left and right eigenvectors
        eig_vals, left_eig_vecs, right_eig_vecs = eig(
            self.ngm_small_tensor.detach().cpu().numpy(),
            left=True,
            right=True
        )

        # Convert eigenvalues and eigenvectors back to PyTorch tensors
        eig_vals = torch.from_numpy(eig_vals)
        left_eig_vecs = torch.from_numpy(left_eig_vecs)
        right_eig_vecs = torch.from_numpy(right_eig_vecs)

        # Find the dominant eigenvalue
        dominant_idx = torch.argmax(torch.abs(eig_vals))

        # Extract dominant eigenvalue and corresponding eigenvectors
        self.dominant_eigen_val = eig_vals[dominant_idx].real.item()
        self.left_eigen_vec = left_eig_vecs[:, dominant_idx].real
        self.right_eigen_vec = right_eig_vecs[:, dominant_idx].real









