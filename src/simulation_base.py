import torch
from src.symmetrize_original_cm import OriginalCMSymmetrization
from src.input_contact_mtx import ContactMatrixInput
from src.symmetrize_contact_input import SymmetricContactMatrixInput
from src.eigen_calculator import EigenCalculator
from src.ngm_cm_grad import Eigen_Grad
from src.r0_cm_grad import Eigen_CM_Grad
from src.r0_generator import R0Generator
from src.dataloader import DataLoader


class SimulationBase:
    def __init__(self, data: DataLoader):
        """
        Initialize the simulation with the provided data.
        Args: data (DataLoader): DataLoader object containing age data and model params.
        """
        self.data = data
        self.population = self.data.age_data
        self.susceptibles = 0.9 * self.population
        self.n_age = len(self.data.age_data)

        self.contact_input = None
        self.symmetric_contact_matrix = None
        self.ngm_small_tensor = None
        self.ngm_small_grads = None
        self.r0_cm_grad = None

        # Initialize R0 generator
        self.r0 = R0Generator(param=self.data.model_parameters_data, n_age=self.n_age)
        # call the run method
        self.run()

    def run(self):
        """
        Run the simulation to perform contact matrix manipulations,
        calculate eigenvalues and gradients.
        """
        # Update model parameters with susceptibility
        self._update_model_parameters_with_susceptibility()

        # Perform contact matrix manipulations
        self._contact_matrices_manipulation()

        # Calculate eigenvalues and gradients
        self._calculate_eigenvalues_and_gradients()

    def _update_model_parameters_with_susceptibility(self):
        """
        Update model parameters with susceptibility data.
        """
        susceptibility = torch.ones(self.n_age)
        susceptibility[:4] = 0.5
        self.data.model_parameters_data.update({"susc": susceptibility})

    def _contact_matrices_manipulation(self):
        """
        Perform manipulations on contact matrices to generate the necessary inputs.
        """
        # Original contact matrix symmetrization
        matrix_symmetrization = OriginalCMSymmetrization(self.data)
        transformed_orig_cm = matrix_symmetrization.calculate_full_transformed_cm()

        # Extract upper triangular elements of the contact matrix
        contact_matrix_input = ContactMatrixInput(
            n_age=self.n_age, pop=self.population,
            transformed_orig_cm=transformed_orig_cm
        )
        self.contact_input = contact_matrix_input.get_upper_tri_elems_orig_cm()

        # Create a new symmetric contact matrix
        symmetric_contact_matrix_input = SymmetricContactMatrixInput(
            n_age=self.n_age, pop=self.population, contact_matrix=self.contact_input
        )
        self.symmetric_contact_matrix = \
            symmetric_contact_matrix_input.create_symmetric_matrix()

    def _calculate_eigenvalues_and_gradients(self):
        """
        Calculate eigenvalues and gradients for the simulation.
        """
        # Compute the next generation matrix (NGM)
        self.ngm_small_tensor = self.r0.compute_ngm_small(
            population=self.population,
            symmetric_contact_mtx=self.symmetric_contact_matrix,
            susceptibles=self.susceptibles
        )

        # Calculate the dominant eigenvalue and eigenvector
        eigen_calculator = EigenCalculator(ngm_small_tensor=self.ngm_small_tensor)
        self.eigen_value = eigen_calculator.dominant_eig_val
        self.eigen_vector = eigen_calculator.dominant_eig_vec

        # Calculate gradients of the NGM
        eigen_grad = Eigen_Grad(
            ngm_small_tensor=self.ngm_small_tensor,
            dominant_eig_vec=eigen_calculator.dominant_eig_vec,
            contact_input=self.contact_input
        )
        self.ngm_small_grads = eigen_grad.ngm_small_grads

        # Calculate the gradient of R0 with respect to the contact matrix
        r0_cm_grad = Eigen_CM_Grad(
            ngm_small_tensor=self.ngm_small_tensor,
            dominant_eig_vec=eigen_calculator.dominant_eig_vec
        )
        self.r0_cm_grad = r0_cm_grad.calculate_eigen_val_cm_grad(
            ngm_small_grads=self.ngm_small_grads
        )






