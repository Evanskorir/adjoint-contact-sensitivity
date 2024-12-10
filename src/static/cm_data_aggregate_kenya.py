import torch

from src.static.dataloader import DataLoader


class KenyaDataAggregator:
    def __init__(self, data: DataLoader):
        """
        Initialize the KenyaDataAggregator class with preloaded Kenya data.
        Args: data (DataLoader):
        """
        self.data = data
        self.contact_mtx = data.contact_data
        self.age_mapping = [(0, 2), (3, 5), (6, 11), (12, 15)]  # Define the 4 new age groups
        self.aggregated_contact_data = None
        self.age_data = None

        self.aggregate_kenyan_contact_matrices()
        self.aggregate_kenya_data()

    def _aggregate_contact_matrix(self, matrix: torch.Tensor,
                                  age_vector: torch.Tensor) -> torch.Tensor:
        """
        Helper function to aggregate a contact matrix using the age vector.
        Args:
            matrix (torch.Tensor): The contact matrix to aggregate.
            age_vector (torch.Tensor): The age vector for weighting.
        Returns: torch.Tensor: The aggregated matrix.
        """
        weighted_matrix = matrix * age_vector
        aggregated_matrix = torch.zeros((4, 4))

        # Aggregating based on age groups
        for i, group_i in enumerate(self.age_mapping):
            for j, group_j in enumerate(self.age_mapping):
                aggregated_matrix[i, j] = torch.sum(
                    weighted_matrix[group_i[0]:group_i[1] + 1, group_j[0]:group_j[1] + 1]
                )
        return aggregated_matrix

    def _normalize_matrix(self, aggregated_matrix: torch.Tensor) -> torch.Tensor:
        """
        Normalize the aggregated matrix by age group sizes.
        Args: aggregated_matrix (torch.Tensor): The aggregated contact matrix to normalize.
        Returns: torch.Tensor: The normalized aggregated matrix.
        """
        age_group_sizes = torch.tensor([torch.sum(
            self.data.age_data[group[0]:group[1] + 1]) for group in self.age_mapping
        ], dtype=torch.float32)

        return aggregated_matrix / age_group_sizes.view(-1, 1)

    def aggregate_kenyan_contact_matrices(self, u_h: float = 0.75):
        """
        Aggregate all contact matrices into 4 age groups for Kenya-specific data,
        applying a reduction factor to the home contact matrix.
        Args: u_h (float): Scaling factor for the home contact matrix (default is 0.75).
        """
        contact_mtx = self.contact_mtx

        # Apply reduction to the Home contact matrix
        contact_mtx["Home"] *= u_h  # Reduce home interactions by the given factor

        # Combine contact matrices into the "All" matrix
        contact_mtx["All"] = (
                contact_mtx["Home"] +
                contact_mtx["School"] +
                contact_mtx["Work"] +
                contact_mtx["Other"]
        )

        age_vector = self.data.age_data.view(-1, 1)
        aggregated_matrices = {}

        # Aggregating each contact matrix (Home, School, Work, Other, All)
        for matrix_name, matrix in contact_mtx.items():
            aggregated_matrix = self._aggregate_contact_matrix(matrix, age_vector)
            normalized_matrix = self._normalize_matrix(aggregated_matrix)
            aggregated_matrices[matrix_name] = normalized_matrix

        # Store the aggregated contact data
        self.aggregated_contact_data = aggregated_matrices

    def aggregate_age_data(self):
        """
        Aggregate the age distribution into 4 age groups.
        """
        age_vector = self.data.age_data.view(-1, 1)
        new_age_groups = torch.tensor([
            torch.sum(age_vector[group[0]:group[1] + 1]) for group in self.age_mapping
        ], dtype=torch.float32)

        # Store the aggregated age data
        self.age_data = new_age_groups

    def aggregate_kenya_data(self):
        """
        Perform full aggregation for Kenya-specific data (contact matrices and age data).
        """
        self.aggregate_kenyan_contact_matrices()
        self.aggregate_age_data()

    def get_aggregated_contact_data(self):
        """
        Get the aggregated contact data.
        Returns:
            dict: The aggregated contact matrices.
        """
        if self.aggregated_contact_data is None:
            self.aggregate_kenya_data()
        return self.aggregated_contact_data

    def get_aggregated_age_data(self):
        """
        Get the aggregated age data.
        Returns: torch.Tensor: The aggregated age groups.
        """
        if self.age_data is None:
            self.aggregate_kenya_data()
        return self.age_data
