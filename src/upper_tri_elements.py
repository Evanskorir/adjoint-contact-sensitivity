import torch


class MatrixOperations:
    def __init__(self, n_age):
        self.n_age = n_age

    def upper_triangle_to_matrix(self, flattened_vectors):
        filled_dictionary = {}
        for key, tensor in flattened_vectors.items():
            matrix = torch.zeros(self.n_age, self.n_age, dtype=tensor.dtype)
            # Get the upper triangle indices
            upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)
            # Fill the upper triangle
            matrix[upper_tri_idx[0], upper_tri_idx[1]] = tensor
            # Fill the lower triangle
            matrix.T.tril_()[upper_tri_idx[0], upper_tri_idx[1]] = tensor
            filled_dictionary[key] = matrix
        return filled_dictionary



