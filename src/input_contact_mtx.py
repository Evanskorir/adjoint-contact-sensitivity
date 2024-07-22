import torch


class ContactMatrixInput:
    def __init__(self, n_age: int, pop: torch.Tensor,
                 transformed_orig_cm: torch.Tensor):
        """
        Initialize the class with the number of age groups, population data,
        and contact matrix.
        Args:
            n_age (int): The number of age groups.
            pop (torch.Tensor): A tensor representing the population data.
            cm (torch.Tensor): The contact matrix.
        """
        self.n_age = n_age
        self.pop = pop
        self.transformed_orig_cm = transformed_orig_cm

        # Get the indices of the upper triangular part
        self.upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)

    def get_upper_tri_elems_orig_cm(self) -> torch.Tensor:
        """
        Extract the upper triangular elements of the contact matrix and
        create a symmetric matrix.
        Returns: torch.Tensor: A tensor containing the symmetric matrix derived from the
            upper tri elements.
        """

        # Extract the upper tri elements of orig_cm and set requires_grad=True
        contact_input = self.transformed_orig_cm[self.upper_tri_idx[0],
                                                             self.upper_tri_idx[1]]
        contact_input.requires_grad_(True)
        return contact_input


