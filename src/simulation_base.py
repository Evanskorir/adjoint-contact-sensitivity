import torch
from src.symmetrize_original_cm import OriginalCMSymmetrization
from src.input_contact_mtx import ContactMatrixInput
from src.symmetrize_contact_input import SymmetricContactMatrixInput
from src.eigen_calculator import EigenCalculator
from src.ngm_cm_grad import EigenGrad
from src.r0_cm_grad import EigenCMGrad
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
        self.n_age = len(self.data.age_data)

        self.contact_input = None
        self.symmetric_contact_matrix = None
        self.ngm_small_tensor = None
        self.ngm_small_grads = None
        self.r0_cm_grad = None

        # Update model parameters with susceptibility
        self._update_model_parameters_with_susceptibility()

        # Initialize R0 generator
        self.r0 = R0Generator(param=self.data.model_parameters_data, n_age=self.n_age)

    def run(self):
        """
        Run the simulation to perform contact matrix manipulations,
        calculate eigenvalues and gradients.
        """

        # 1. Create leaf of the computation graph
        self._create_leaf()

        # 2. Perform contact matrix manipulations
        self._contact_matrix_manipulation()

        # 3. Compute the next generation matrix (NGM)
        self.ngm_small_tensor = self.r0.compute_ngm_small(
            symmetric_contact_mtx=self.symmetric_contact_matrix
        )

        # 4.a. Calculate gradients of the NGM
        self.ngm_small_grads = EigenGrad(
            ngm_small_tensor=self.ngm_small_tensor,
            contact_input=self.contact_input
        ).ngm_small_grads

        # 4.b. Calculate eigenvectors
        self._calculate_eigenvectors()

        # 5. Calculate the gradient of R0 with respect to the contact matrix
        self.r0_cm_grad = EigenCMGrad(
            ngm_small_tensor=self.ngm_small_tensor,
            dominant_eig_vec=self.eigen_vector
        ).calculate_eigen_val_cm_grad(ngm_small_grads=self.ngm_small_grads)

    def _update_model_parameters_with_susceptibility(self):
        """
        Update model parameters with susceptibility data.
        """
        susceptibility = torch.ones(self.n_age)
        susceptibility[:4] = 0.5
        self.data.model_parameters_data.update({"susc": susceptibility})

    def _create_leaf(self):
        # Original contact matrix symmetrization
        transformed_total_orig_cm = OriginalCMSymmetrization(
            self.data
        ).calculate_full_total_transformed_cm()

        # Extract upper triangular elements of the contact matrix
        self.contact_input = ContactMatrixInput(
            n_age=self.n_age,
            transformed_total_orig_cm=transformed_total_orig_cm
        ).get_upper_tri_elems_total_full_orig_cm()
        self.contact_input.requires_grad_(True)

    def _contact_matrix_manipulation(self):
        """
        Perform manipulations on contact matrices to generate the necessary inputs.
        """
        # Create a new symmetric contact matrix
        self.symmetric_contact_matrix = SymmetricContactMatrixInput(
            n_age=self.n_age,
            pop=self.population.reshape((-1, 1))
        ).create_symmetric_matrix(contact_matrix=self.contact_input)

    def _calculate_eigenvectors(self):
        """
        Calculate eigenvalues and gradients for the simulation.
        """

        # Calculate the dominant eigenvalue and eigenvector
        eigen_calculator = EigenCalculator(ngm_small_tensor=self.ngm_small_tensor)
        self.eigen_vector = eigen_calculator.dominant_eig_vec

        # Additionally, we store the dominant eigenvalue
        self.eigen_value = eigen_calculator.dominant_eig_val
