import torch


class AggregationApproach:
    def __init__(self, n_age: int, r0_cm_grad: torch.Tensor):

        self.n_age = n_age
        self.r0_cm_grad = r0_cm_grad

        self.cum_sens = None

    def reconstruct_symmetric_matrix(self) -> torch.Tensor:
        """
        Reconstruct a symmetric matrix from the upper triangular elements.
        Returns: torch.Tensor: Symmetric matrix of shape (n_age, n_age).
        """
        mtx = torch.zeros((self.n_age, self.n_age), device=self.r0_cm_grad.device)
        upper_tri_indices = torch.triu_indices(self.n_age, self.n_age)

        mtx[upper_tri_indices[0], upper_tri_indices[1]] = self.r0_cm_grad.view(-1)
        mtx = mtx + mtx.T - torch.diag(mtx.diag())

        return mtx

    def run(self) -> torch.Tensor:
        """
        Compute cumulative sensitivities (s_j) for each age group.
        Returns: torch.Tensor: Aggregated sensitivities for each age group (size n_age).
        """
        sym_matrix = self.reconstruct_symmetric_matrix()
        self.cum_sens = sym_matrix.sum(dim=1)  # Summing over rows

        return self.cum_sens
