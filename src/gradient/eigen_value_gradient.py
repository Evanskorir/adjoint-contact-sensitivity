import torch


class EigenValueGradient:
    def __init__(self, n_age: int, ngm_small_grads: torch.Tensor,
                 left_eigen_vec: torch.Tensor, right_eigen_vec: torch.Tensor,
                 dominant_eigen_val: torch.Tensor, ngm_small_tensor: torch.Tensor,
                 use_elasticity: bool = True):

        self.left_eig_vec = left_eigen_vec
        self.right_eig_vec = right_eigen_vec
        self.ngm_small_grads = ngm_small_grads
        self.n_age = n_age

        self.K = ngm_small_tensor
        self.r0 = dominant_eigen_val
        self.use_elasticity = use_elasticity

        self.r0_cm_grad = None

    def run(self):
        normalization = torch.dot(self.left_eig_vec, self.right_eig_vec)

        # calculate v.w^T / v^T.w
        s_ij = (
                self.left_eig_vec.view(-1, 1) * self.right_eig_vec.view(1, -1)
                ) / normalization

        # Reshape to s_ij for correct broadcasting
        s_ij = s_ij.view(self.n_age, 1, self.n_age)

        # Element-wise multiplication
        weighted_grads = self.ngm_small_grads * s_ij

        # Elasticity-based gradient
        K_b = self.K.view(self.n_age, 1, self.n_age)  # broadcast K to (i, 1, j)
        E = (K_b / self.r0) * s_ij
        elast_weighted = self.ngm_small_grads * E

        if self.use_elasticity:
            self.r0_cm_grad = elast_weighted.sum(dim=(0, 2))
        else:
            # Sum over dimensions 0 and 2 to get r0 w.r.t cm
            self.r0_cm_grad = weighted_grads.sum(dim=(0, 2))

        return self.r0_cm_grad




