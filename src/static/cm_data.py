import torch

from src.static.dataloader import DataLoader
from src.static.cm_data_aggregate_kenya import KenyaDataAggregator


class CMData:
    def __init__(self, data: DataLoader, model: str):
        """
        Initialize the class with data and the specific model type.
        Args:
            data (DataLoader): An instance of DataLoader containing contact and age data.
            model (str): The model type that determines how the contact matrix is loaded.
        """
        self.data = data
        self.model = model

        if self.model == "kenya":
            # Aggregate the Kenya-specific data only once
            self.kenyan_aggregated_data = KenyaDataAggregator(data=data)
            # Access pre-aggregated contact and age data directly
            self.aggregated_contact_data = self.kenyan_aggregated_data.get_aggregated_contact_data()
            self.aggregated_age_data = self.kenyan_aggregated_data.get_aggregated_age_data()

    def load_contact_matrix(self) -> torch.Tensor:
        contact_data = self.data.contact_data
        if self.model in ["seir", "italy", "british_columbia", "moghadas"]:
            full_orig_cm = contact_data["All"]

        elif self.model == "kenya":
            full_orig_cm = (
                    self.aggregated_contact_data["Home"] +
                    self.aggregated_contact_data["School"] +
                    self.aggregated_contact_data["Work"] +
                    self.aggregated_contact_data["Other"]
            )
        else:
            full_orig_cm = (
                    contact_data["Home"] +
                    contact_data["School"] +
                    contact_data["Work"] +
                    contact_data["Other"]
            )

        return full_orig_cm

    def calculate_full_contact_matrix(self) -> torch.Tensor:
        """
        Calculate and return the full contact matrix.

        Returns:
            torch.Tensor: The full contact matrix for the model.
        """
        full_orig_cm = self.load_contact_matrix()
        return full_orig_cm
