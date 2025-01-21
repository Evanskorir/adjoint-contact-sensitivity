import torch


class R0CMDerivativeCalculator:
    def __init__(self, left_eig_vec: torch.Tensor, right_eig_vec: torch.Tensor,
                 normalization: torch.Tensor):
        self.left_eig_vec = left_eig_vec
        self.right_eig_vec = right_eig_vec

        self.normalization = normalization
        self.r0_cm_grad = None

    def run(self, ngm_small_grads: torch.Tensor) -> torch.Tensor:
        # Vectorized gradient calculation
        weighted_grads = (self.left_eig_vec[:, None, None] *  # Shape: (n_age, 1, 1)
                          self.right_eig_vec[None, None, :] *  # Shape: (1, 1, n_age)
                          ngm_small_grads).sum(dim=(0, 2))  # Sum over dimensions 0 and 2

        self.r0_cm_grad = weighted_grads / self.normalization

        return self.r0_cm_grad
