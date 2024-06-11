import torch


class MatrixOperations:
    def __init__(self, n_age, pop: torch.Tensor):

        self.n_age = n_age
        self.pop = pop

        self.upp_tri_elem = None
        self.matrix = None
        self.sym_cont_mtx = None

    def upper_triangle_to_matrix(self, matrix: torch.Tensor):
        # Get the upper triangle indices
        upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)

        # Extract the upper triangular elements and set requires_grad=True
        upper_tri_elements = matrix[upper_tri_idx[0],
                                    upper_tri_idx[1]].clone().detach().requires_grad_(True)

        # Fill the lower triangle with corresponding upper triangle values
        matrix.T.tril_()[upper_tri_idx[0], upper_tri_idx[1]] = matrix[upper_tri_idx[0],
                                                                      upper_tri_idx[1]]

        # Apply transformation to the matrix
        matrix_transform = self.transform_matrix(matrix)

        # Store the matrix and upper triangular elements
        self.matrix = matrix_transform
        self.upp_tri_elem = upper_tri_elements

        return self.matrix, self.upp_tri_elem

    def transform_matrix(self, matrix: torch.Tensor):
        # Get age vector as a column vector
        age_distribution = self.pop.reshape((-1, 1))  # (16, 1)
        # Get symmetric matrix
        output = (matrix + matrix.T) / 2
        # Get contact matrix
        output /= age_distribution  # divides and assign the result to output (16, 16)
        return output





