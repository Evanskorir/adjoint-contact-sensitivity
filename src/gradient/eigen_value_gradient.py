import torch


class EigenValueGradient:
    def __init__(self, n_age: int, ngm_small_grads: torch.Tensor,
                 left_eigen_vec: torch.Tensor,
                 right_eigen_vec: torch.Tensor):

        self.left_eig_vec = left_eigen_vec
        self.right_eig_vec = right_eigen_vec
        self.ngm_small_grads = ngm_small_grads
        self.n_age = n_age

        self.r0_cm_grad = None

    def run(self):
        # Calculate v^T.w
        normalization = torch.dot(self.left_eig_vec, self.right_eig_vec)

        # calculate v.w^T / v^T.w
        s_ij = (
                self.left_eig_vec.view(-1, 1) * self.right_eig_vec.view(1, -1)
                ) / normalization

        # Reshape to s_ij for correct broadcasting
        s_ij = s_ij.view(self.n_age, 1, self.n_age)

        # Element-wise multiplication
        weighted_grads = self.ngm_small_grads * s_ij

        # Sum over dimensions 0 and 2 to get r0 w.r.t cm
        self.r0_cm_grad = weighted_grads.sum(dim=(0, 2))

        return self.r0_cm_grad
