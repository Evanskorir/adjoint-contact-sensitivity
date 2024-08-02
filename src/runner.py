import torch

from src.eigen_value_gradient import EigenValueGradient
from src.gradient.cm_creator import CMCreator
from src.gradient.cm_elements_cg_leaf import CMElementsCGLeaf
from src.gradient.ngm_calculator import NGMCalculator
from src.gradient.ngm_gradient import NGMGradient
from src.static.cm_leaf_preparator import CGLeafPreparator
from src.static.dataloader import DataLoader
from src.static.eigen_calculator import EigenCalculator
from src.aggregation_logic import Aggregation
from src.plotter import Plotter


class Runner:
    def __init__(self, data: DataLoader):
        """
        Initialize the simulation with the provided data.
        Args: data (DataLoader): DataLoader object containing age data and model params.
        """
        self.data = data
        self.population = self.data.age_data
        self.n_age = len(self.data.age_data)

        self.contact_input = None
        self.contact_input_sum = None
        self.symmetric_contact_matrix = None
        self.ngm_small_tensor = None
        self.ngm_small_grads = None
        self.r0_cm_grad = None

        # Update model parameters with susceptibility
        self._update_model_parameters_with_susceptibility()

        # Initialize R0 generator
        self.ngm_calculator = NGMCalculator(param=self.data.model_parameters_data,
                                            n_age=self.n_age)

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
        self.ngm_calculator.run(
            symmetric_contact_mtx=self.symmetric_contact_matrix
        )
        self.ngm_small_tensor = self.ngm_calculator.ngm_small_tensor

        # 4.a. Calculate gradients of the NGM
        ngm_grad = NGMGradient(
            ngm_small_tensor=self.ngm_small_tensor,
            contact_input=self.contact_input
        )
        ngm_grad.run()
        self.ngm_small_grads = ngm_grad.ngm_small_grads

        # 4.b. Calculate eigenvectors
        self._calculate_eigenvectors()

        # 5. Calculate the gradient of R0 with respect to the contact matrix
        eigen_value_gradient = EigenValueGradient(
            ngm_small_tensor=self.ngm_small_tensor,
            dominant_eig_vec=self.eigen_vector
        )
        eigen_value_gradient.run(ngm_small_grads=self.ngm_small_grads)
        self.r0_cm_grad = eigen_value_gradient.eig_val_cm_grad

        # 6. Use aggregation logic to get age group pairs sensitivity values
        agg = Aggregation(data=self.data, n_age=self.n_age, pop=self.population,
                          calculation_approach="mean", grad_mtx=self.r0_cm_grad)

        # 7. Generate the plots from plotter to visualize the sensitivities
        plot = Plotter(n_age=self.n_age)
        plot.plot_small_ngm_mtx(ngm_matrix=agg.grads_mtx,
                                plot_title="Sensitivity values")
        plot.plot_grads_p_values_as_heatmap(grads_vector=agg.grads_mtx,
                                            p_values=agg.p_values_mtx,
                                            plot_title=None)
        plot.plot_aggregation_grads_pvalues(grads_vector=agg.agg_prcc,
                                            std_values=agg.agg_std,
                                            conf_upper=agg.confidence_upper,
                                            conf_lower=agg.confidence_lower,
                                            calculation_approach="mean")

    def _update_model_parameters_with_susceptibility(self):
        """
        Update model parameters with susceptibility data.
        """
        susceptibility = torch.ones(self.n_age)
        susceptibility[:4] = 0.5
        self.data.model_parameters_data.update({"susc": susceptibility})

    def _create_leaf(self):
        # Original contact matrix symmetrization
        cg_leaf_preparator = CGLeafPreparator(data=self.data)
        cg_leaf_preparator.run()
        transformed_total_orig_cm = cg_leaf_preparator.transformed_total_orig_cm

        # Extract upper triangular elements of the contact matrix
        cm_elements_cg_leaf = CMElementsCGLeaf(
            n_age=self.n_age,
            transformed_total_orig_cm=transformed_total_orig_cm,
            pop=self.population)
        cm_elements_cg_leaf.run()
        self.contact_input = cm_elements_cg_leaf.contact_input.requires_grad_(True)
        self.contact_input_sum = cm_elements_cg_leaf.contact_input_sum

    def _contact_matrix_manipulation(self):
        """
        Perform manipulations on contact matrices to generate the necessary inputs.
        """
        # Create a new symmetric contact matrix
        cm_creator = CMCreator(
            n_age=self.n_age,
            pop=self.population.reshape((-1, 1))
        )
        cm_creator.run(contact_matrix=self.contact_input,
                       contact_matrix_sum=self.contact_input_sum)
        self.symmetric_contact_matrix = cm_creator.cm

    def _calculate_eigenvectors(self):
        """
        Calculate eigenvalues and gradients for the simulation.
        """

        # Calculate the dominant eigenvalue and eigenvector
        eigen_calculator = EigenCalculator(
            ngm_small_tensor=self.ngm_small_tensor
        )
        eigen_calculator.run()
        self.eigen_vector = eigen_calculator.dominant_eig_vec

        # Additionally, we store the dominant eigenvalue
        self.eigen_value = eigen_calculator.dominant_eig_val
