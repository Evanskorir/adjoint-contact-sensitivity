import torch

from src.dataloader import DataLoader
from src.r0_generator import R0Generator
from src.upper_tri_elements import MatrixOperations


class SimulationBase:
    def __init__(self, data: DataLoader):
        self.data = data

        self.r0 = R0Generator(param=self.data.model_parameters_data, n_age=16)

        # get full contact matrix
        self.full_contact_mtx = (
                self.data.contact_data["Home"] +
                self.data.contact_data["School"] +
                self.data.contact_data["Work"] +
                self.data.contact_data["Other"]
        )

        # get the total full contact matrix
        self.total_contact_mtx = self.full_contact_mtx * \
                                 self.data.age_data.reshape((-1, 1))

        # get upper tri elements and symmetric matrix
        Matr = MatrixOperations(n_age=16, pop=data.age_data)
        self.symmetric_matrix, self.upper_tri_matrix = Matr.upper_triangle_to_matrix(
            self.full_contact_mtx)

        self.params = self.data.model_parameters_data
        susceptibility = torch.ones(16)
        susceptibility[:4] = 0.5

        self.data.model_parameters_data.update({"susc": susceptibility})
        self.n_ag = self.full_contact_mtx.shape[0]
        self.population = self.data.age_data





