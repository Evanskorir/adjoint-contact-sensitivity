import torch

from src.dataloader import DataLoader
from src.r0_generator import R0Generator
from src.upper_tri_elements import MatrixOperations


class SimulationBase:
    def __init__(self, data: DataLoader):
        self.data = data

        self.r0 = R0Generator(param=self.data.model_parameters_data, n_age=16)

        self.upper_tri_elems = (
                self.data.flattened_vector_dict["Home"] +
                self.data.flattened_vector_dict["School"] +
                self.data.flattened_vector_dict["Work"] +
                self.data.flattened_vector_dict["Other"]
        )
        Matr = MatrixOperations(n_age=16)
        reconstructed_matrix = Matr.upper_triangle_to_matrix(
            self.data.flattened_vector_dict)

        # get matrix from upper tri elements
        self.contact_matrix = (reconstructed_matrix["Home"] +
                               reconstructed_matrix["School"] +
                               reconstructed_matrix["Work"] +
                               reconstructed_matrix["Other"]
                               )

        self.params = self.data.model_parameters_data
        susceptibility = torch.ones(16)
        susceptibility[:4] = 0.5

        self.data.model_parameters_data.update({"susc": susceptibility})
        self.n_ag = self.contact_matrix.shape[0]
        self.population = self.data.age_data



