import torch


class AggregationApproach:
    def __init__(self, n_age: int, ngm_small_grads: torch.Tensor,
                 left_eigen_vec: torch.Tensor,
                 right_eigen_vec: torch.Tensor):

        self.left_eig_vec = left_eigen_vec
        self.right_eig_vec = right_eigen_vec
        self.ngm_small_grads = ngm_small_grads
        self.n_age = n_age

        self.cum_sens = None

    def run(self) -> torch.Tensor:
        """
        Compute cumulative sensitivities (e_j) for each age group
        :return: Aggregated sensitivities for each age group
        """
        # Compute normalization factor (v^T w)
        normalization = torch.dot(self.left_eig_vec, self.right_eig_vec)

        # Compute S_ij = (v_i * w_j) / (v^T w)
        s_ij = (
                self.left_eig_vec.view(-1, 1) * self.right_eig_vec.view(1, -1)
              ) / normalization

        s_ij = s_ij.view(self.n_age, 1, self.n_age)

        # Apply the derivative ∂K/∂c_p
        weighted_sens = s_ij * self.ngm_small_grads

        # Sum over dimensions 1 and 2
        self.cum_sens = weighted_sens.sum(dim=(1, 2))

        return self.cum_sens
