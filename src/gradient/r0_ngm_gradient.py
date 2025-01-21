import torch


class R0NGMDerivativeCalculator:
    def __init__(self, left_eigen_vec: torch.Tensor,
                 right_eigen_vec: torch.Tensor) -> None:
        self.left_eigen_vec = left_eigen_vec
        self.right_eigen_vec = right_eigen_vec

        self.normalization = None
        self.r0_ngm_grad = None

    def run(self) -> torch.Tensor:
        # Calculate the normalization factor (dot product of left and right eigen_vecs)
        normalization = torch.dot(self.left_eigen_vec, self.right_eigen_vec)
        self.normalization = normalization

        # Compute the outer product of left and right eigenvectors
        self.r0_ngm_grad = torch.outer(self.left_eigen_vec, self.right_eigen_vec) / \
                           normalization

        return self.r0_ngm_grad


