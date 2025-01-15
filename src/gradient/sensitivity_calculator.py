from src.comp_graph.cm_creator import CMCreator
from src.comp_graph.cm_elements_cg_leaf import CMElementsCGLeaf
from src.gradient.eigen_value_gradient import EigenValueGradient
from src.gradient.ngm_gradient import NGMGradient
import src.models as models
import src.static as static
from src.static.cm.cm_leaf_preparator import CGLeafPreparator
from src.static.svd_eigen_calculator import SVDEigenCalculator


class SensitivityCalculator:
    def __init__(self, data: static.DataLoader, model: str, method):
        self.data = data
        self.model = model
        self.method = method  # 'eig' for Eigen Decomposition, 'svd' for SVD
        self.params = self.data.model_parameters_data
        self.population = self.data.age_data
        self.n_age = len(self.population)

        # Initialize placeholders for calculated values
        self.ngm_calculator = None
        self.ngm_small_tensor = None
        self.ngm_small_grads = None
        self.eigen_value_gradient = None

        self.eigen_value = None
        self.eigen_vector = None
        self.eigen_vector_left = None
        self.eigen_vector_right = None

        self.contact_input = None
        self.scale_value = None
        self.symmetric_contact_matrix = None

        self._initialize_r0_choices()
        self._select_ngm_calculator()

    def _select_ngm_calculator(self):
        """
        Select the appropriate NGMCalculator based on the model name.
        """
        self.ngm_calculator_class = models.model_calc_map.get(self.model)
        if not self.ngm_calculator_class:
            raise ValueError(f"Unknown model: {self.model}")

    def run(self, scale: str, params: dict):
        """
        Main function to calculate sensitivity using various steps.
        """
        # 1. Dynamically assign R0 choices based on model
        self._initialize_r0_choices()

        # 2. Initialize NGM calculator with parameters
        self._initialize_ngm_calculator(params)

        # 3. Create leaf of the computation graph
        self._create_leaf(scale)

        # 4. Perform contact matrix manipulations
        self._contact_matrix_manipulation(scale)

        # 5. Compute the next generation matrix (NGM)
        self._compute_ngm()

        # 6. Calculate gradients of the NGM
        self._calculate_ngm_gradients()

        # 7. Calculate eigenvalue and eigenvalue gradients for R0
        self._calculate_eigenvectors()

    def _initialize_ngm_calculator(self, params: dict):
        """
        Initialize the NGM calculator with the given parameters.
        """
        self.ngm_calculator = self.ngm_calculator_class(n_age=self.n_age, param=params)

    def _create_leaf(self, scale: str):
        """
        Create the computation graph leaf (CG leaf) from the original contact matrix.
        """
        cg_leaf_preparator = CGLeafPreparator(data=self.data, model=self.model)
        cg_leaf_preparator.run()
        transformed_total_orig_cm = cg_leaf_preparator.transformed_total_orig_cm

        cm_elements_cg_leaf = CMElementsCGLeaf(
            n_age=self.n_age,
            transformed_total_orig_cm=transformed_total_orig_cm,
            pop=self.population
        )
        cm_elements_cg_leaf.run(scale=scale)

        self.contact_input = cm_elements_cg_leaf.contact_input.requires_grad_(True)
        self.scale_value = cm_elements_cg_leaf.scale_value

    def _contact_matrix_manipulation(self, scale: str):
        """
        Perform manipulations on the contact matrix to produce necessary inputs.
        """
        cm_creator = CMCreator(
            n_age=self.n_age,
            pop=self.population.reshape((-1, 1))
        )
        cm_creator.run(
            contact_matrix=self.contact_input,
            scale_value=self.scale_value,
            scale=scale
        )
        self.symmetric_contact_matrix = cm_creator.cm

    def _compute_ngm(self):
        """
        Compute the next generation matrix (NGM).
        """
        self.ngm_calculator.run(symmetric_contact_mtx=self.symmetric_contact_matrix)
        self.ngm_small_tensor = self.ngm_calculator.ngm_small_tensor

    def _calculate_ngm_gradients(self):
        """
        Calculate the gradients of the next generation matrix (NGM).
        """
        ngm_grad = NGMGradient(
            ngm_small_tensor=self.ngm_small_tensor,
            contact_input=self.contact_input
        )
        ngm_grad.run()
        self.ngm_small_grads = ngm_grad.ngm_small_grads

    def _calculate_eigenvectors(self):
        """
        Select between Eigen Decomposition and SVD for eigenvector calculation.
        """
        method_map = {
            'eig': (static.EigenCalculator,
                    'dominant_left_eig_vec', 'dominant_right_eig_vec', 'dominant_eig_val'),
            'svd': (SVDEigenCalculator,
                    'left_singular_vec', 'right_singular_vec', 'dominant_singular_val')
        }

        CalculatorClass, left_vec_attr, right_vec_attr, value_attr = method_map[self.method]

        # Initialize and run the appropriate calculator
        eigen_calculator = CalculatorClass(self.ngm_small_tensor)
        eigen_calculator.run()

        # Extract eigenvectors and eigenvalue
        self.eigen_vector_left = getattr(eigen_calculator, left_vec_attr)
        self.eigen_vector_right = getattr(eigen_calculator, right_vec_attr)
        self.eigen_value = getattr(eigen_calculator, value_attr)

        # Compute the eigenvalue gradient
        eigen_value_grad = EigenValueGradient(
            ngm_small_tensor=self.ngm_small_tensor,
            left_eig_vec=self.eigen_vector_left,
            right_eig_vec=self.eigen_vector_right,
            method=self.method
        )
        eigen_value_grad.run(ngm_small_grads=self.ngm_small_grads)
        self.eigen_value_gradient = eigen_value_grad

    def _initialize_r0_choices(self):
        """
        Set R0 choices based on the selected model.
        """
        r0_mapping = {
            "british_columbia": [1.2],
            "kenya": [1.78],
            "rost": [1.8],
            "washington": [5.7],
            "seir": [1.8],
        }
        self.r0_choices = r0_mapping.get(self.model, [2.2])
