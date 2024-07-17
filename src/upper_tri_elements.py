import torch


class MatrixOperations:
    def __init__(self, n_age: int, pop: torch.Tensor,
                 transformed_contact_matrix: torch.Tensor):
        """
        Initialize the MatrixOperations class with the number of age groups,
        population data, and contact matrix.
        Args:
            n_age (int): The number of age groups.
            pop (torch.Tensor): A tensor representing the population data.
            cm (torch.Tensor): The contact matrix.
        """
        self.n_age = n_age
        self.pop = pop
        self.transformed_contact_matrix = transformed_contact_matrix

    def get_upper_triangular_elements(self) -> torch.Tensor:
        """
        Extract the upper triangular elements of the contact matrix and
        create a symmetric matrix.
        Returns: torch.Tensor: A tensor containing the symmetric matrix derived from the
            upper triangular elements.
        """
        # Get the indices of the upper triangular part
        upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)

        # Extract the upper triangular elements and set requires_grad=True
        upper_tri_elements = self.transformed_contact_matrix[upper_tri_idx[0],
                                                             upper_tri_idx[1]]
        upper_tri_elements.requires_grad_(True)

        # Create a new matrix filled with zeros
        new_sym_contact_mtx = torch.zeros((self.n_age, self.n_age))

        # Fill the upper triangular part with the extracted elements
        new_sym_contact_mtx[upper_tri_idx[0], upper_tri_idx[1]] = upper_tri_elements

        # Transpose the new matrix
        new_sym_contact_mtx_transposed = new_sym_contact_mtx.T

        # Fill the lower triangular part using the upper triangular indices
        new_sym_contact_mtx_transposed[upper_tri_idx[0],
                                   upper_tri_idx[1]] = upper_tri_elements
        # divide by the pop to get the full symmetrized cm
        new_symmetric_matrix = new_sym_contact_mtx_transposed / self.pop

        return upper_tri_elements, new_symmetric_matrix



