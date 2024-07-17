import torch


class MatrixSymmetrization:
    def __init__(self, data):
        """
        Initialize the MatrixSymmetrization class with data.
        Args: data: An instance of DataLoader containing contact and age data.
        """
        self.data = data

    def calculate_full_transformed_contact_matrix(self) -> torch.Tensor:
        """
        Calculate and return the full transformed contact matrix.
        The full contact matrix is derived from the sum of various contact matrices
        (Home, School, Work, Other) and then symmetrized and transformed.
        Returns: torch.Tensor: The symmetrized and transformed full contact matrix.
        """
        full_contact_matrix = self._calculate_full_contact_matrix()
        transformed_matrix = self._transform_matrix(full_contact_matrix)
        return transformed_matrix

    def _calculate_full_contact_matrix(self) -> torch.Tensor:
        """
        Calculate the full contact matrix by summing up different contact matrices.
        Returns: torch.Tensor: The full contact matrix.
        """
        contact_data = self.data.contact_data
        full_contact_matrix = (
            contact_data["Home"] +
            contact_data["School"] +
            contact_data["Work"] +
            contact_data["Other"]
        )
        return full_contact_matrix

    def _transform_matrix(self, contact_matrix: torch.Tensor) -> torch.Tensor:
        """
        Transform and symmetrize the given contact matrix.
        The transformation involves multiplying by the age distribution and symmetrizing.
        Args: contact_matrix (torch.Tensor): The contact matrix to transform.
        Returns: torch.Tensor: The transformed and symmetrized contact matrix.
        """
        age_distribution = self.data.age_data.reshape((-1, 1))  # (16, 1) column vector
        symmetrized_matrix = ((contact_matrix * age_distribution) +
                              (contact_matrix * age_distribution).T) / 2
        return symmetrized_matrix

