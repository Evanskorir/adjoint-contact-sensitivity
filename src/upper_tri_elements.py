import torch


class MatrixOperations:
    def __init__(self, matrix: torch.Tensor, n_age):
        self.n_age = n_age
        self.matrix = matrix.clone()

    def get_upper_triangle_elements(self) -> torch.Tensor:
        # Get the upper triangle elements, including the diagonal
        upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)
        upper_tri_elem = self.matrix[upper_tri_idx[0], upper_tri_idx[1]]
        return upper_tri_elem

    def upper_triangle_to_matrix(self, upper_tri_elem: torch.Tensor) -> torch.Tensor:
        new_matrix = torch.zeros((self.n_age, self.n_age), dtype=self.matrix.dtype)
        upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)
        new_matrix[upper_tri_idx[0], upper_tri_idx[1]] = upper_tri_elem
        # Fill the lower triangular part with the values from the upper triangular part
        new_matrix.T.tril_()[upper_tri_idx[0], upper_tri_idx[1]] = upper_tri_elem
        return new_matrix


