import torch


class EigenValueGradient:
    def __init__(self, ngm_small_tensor: torch.Tensor, left_eigen_vec: torch.Tensor,
                 right_eigen_vec: torch.Tensor):

        self.ngm_small_tensor = ngm_small_tensor
        self.left_eig_vec = left_eigen_vec
        self.right_eig_vec = right_eigen_vec

        self.r0_cm_grad = None
        self.r0_ngm_grad = None

    def run(self, ngm_small_grads: torch.Tensor):
        # Calculate the normalization factor (dot product of left and right eigen_vecs)
        normalization = torch.dot(self.left_eig_vec, self.right_eig_vec)

        # Vectorized gradient calculation with normalization before summation
        weighted_grads = (self.left_eig_vec.view(-1, 1, 1) *  # Shape: (n_age, 1, 1)
                          self.right_eig_vec.view(1, 1, -1) *  # Shape: (1, 1, n_age)
                          ngm_small_grads) / normalization

        # Sum over dimensions 0 and 2 to get r0 w.r.t cm
        self.r0_cm_grad = weighted_grads.sum(dim=(0, 2))  # sums over dim 0 & 2

        # r0 w.r.t ngm
        self.r0_ngm_grad = torch.outer(self.left_eig_vec, self.right_eig_vec) / \
                           normalization

        return self.r0_cm_grad, self.r0_ngm_grad
