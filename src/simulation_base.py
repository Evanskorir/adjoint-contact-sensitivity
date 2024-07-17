import torch
from src.dataloader import DataLoader
from src.r0_generator import R0Generator
from src.symmetrize_cm import MatrixSymmetrization
from src.upper_tri_elements import MatrixOperations
from src.eigen_calculator import EigenCalculator
from src.ngm_cm_grad import Eigen_grad
from src.r0_cm_gradients import Eigen_cm_grad


class SimulationBase:
    def __init__(self, data: DataLoader):
        self.data = data

        self.population = self.data.age_data
        self.n_age = len(self.data.age_data)
        self.upper_tri_elements, self.symmetric_matrix = self._calculate_upper_tri_elements()

        self.r0 = R0Generator(param=self.data.model_parameters_data, n_age=self.n_age)
        self.transformed_contact_matrix = self._calculate_transformed_contact_matrix()
        self._update_model_parameters_with_susceptibility()

        self.ngm_small_tensor = self.r0.compute_ngm_small(population=self.population,
                                                     contact_mtx=self.symmetric_matrix,
                                                     susceptibles=self.population)
        self.eigen_calculator = EigenCalculator(ngm_small_tensor=self.ngm_small_tensor)

        self.eigen_grad = Eigen_grad(
            ngm_small_tensor=self.ngm_small_tensor,
            dominant_eig_vecs=self.eigen_calculator.dominant_eig_vecs,
            upper_tri_elems=self.upper_tri_elements,
            n_age=self.n_age)
        self.ngm_small_grads = self.eigen_grad.ngm_small_grads

        r0_cm_grad = Eigen_cm_grad(ngm_small_tensor=self.ngm_small_tensor,
                                        dominant_eig_vecs=self.eigen_calculator.dominant_eig_vecs)

        self.r0_cm_grad = r0_cm_grad.calculate_eigen_val_cm_grad(
            ngm_small_grads=self.eigen_grad.ngm_small_grads)

    def _calculate_transformed_contact_matrix(self) -> torch.Tensor:
        matrix_symmetrization = MatrixSymmetrization(self.data)
        return matrix_symmetrization.calculate_full_transformed_contact_matrix()

    def _calculate_upper_tri_elements(self):
        matrix_symmetrization = MatrixSymmetrization(self.data)
        transformed_contact_matrix = \
            matrix_symmetrization.calculate_full_transformed_contact_matrix()
        matrix_operations = MatrixOperations(n_age=self.n_age, pop=self.population,
            transformed_contact_matrix=transformed_contact_matrix)
        return matrix_operations.get_upper_triangular_elements()

    def _update_model_parameters_with_susceptibility(self):
        susceptibility = torch.ones(self.n_age)
        susceptibility[:4] = 0.5
        self.data.model_parameters_data.update({"susc": susceptibility})





