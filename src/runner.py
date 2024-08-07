import os
import torch

from src.eigen_value_gradient import EigenValueGradient
from src.gradient.cm_creator import CMCreator
from src.gradient.cm_elements_cg_leaf import CMElementsCGLeaf
from src.gradient.ngm_calculator import NGMCalculator
from src.gradient.ngm_gradient import NGMGradient
from src.static.cm_leaf_preparator import CGLeafPreparator
from src.static.dataloader import DataLoader
from src.static.eigen_calculator import EigenCalculator

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
        self.params = self.data.model_parameters_data

        self.contact_input = None
        self.contact_input_sum = None
        self.symmetric_contact_matrix = None
        self.ngm_small_tensor = None
        self.ngm_small_grads = None
        self.r0_cm_grad = None

        # User-defined parameters
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.2, 1.8, 2.5]
        self.scales = ["pop_sum", "contact_sum", "no_scale"]

        # Initialize R0 generator
        self.ngm_calculator = NGMCalculator(param=self.params,
                                            n_age=self.n_age)

    def run(self):
        """
        Run the simulation to perform contact matrix manipulations,
        calculate eigenvalues and gradients.
        """
        for scale in self.scales:
            susceptibility = torch.ones(self.n_age)
            for susc in self.susc_choices:
                susceptibility[:4] = susc
                self.params.update({"susc": susceptibility})
                # 1. Create leaf of the computation graph
                self._create_leaf(scale)
                # 2. Perform contact matrix manipulations
                self._contact_matrix_manipulation(scale)

                # 3. Compute the next generation matrix (NGM)
                self.ngm_calculator = NGMCalculator(n_age=self.n_age,
                                                    param=self.params)
                self.ngm_calculator.run(
                    symmetric_contact_mtx=self.symmetric_contact_matrix)
                self.ngm_small_tensor = self.ngm_calculator.ngm_small_tensor

                # 4.a. Calculate gradients of the NGM
                ngm_grad = NGMGradient(ngm_small_tensor=self.ngm_small_tensor,
                                       contact_input=self.contact_input)
                ngm_grad.run()
                self.ngm_small_grads = ngm_grad.ngm_small_grads

                # 4.b. Calculate eigenvectors
                self._calculate_eigenvectors()

                # 5. Calculate the gradient of R0 with respect to the contact matrix
                eigen_value_gradient = EigenValueGradient(
                    ngm_small_tensor=self.ngm_small_tensor,
                    dominant_eig_vec=self.eigen_vector)
                eigen_value_gradient.run(ngm_small_grads=self.ngm_small_grads)

                for base_r0 in self.r0_choices:
                    beta = base_r0 / self.eigen_value
                    # Scale the gradients with beta
                    self.r0_cm_grad = beta * eigen_value_gradient.eig_val_cm_grad

                    # 6. Generate the plots from plotter to visualize the gradients with
                    # different base r0's and susc
                    # Create folder with base R0 and susc
                    folder = f"results_base_r0_{base_r0:.1f}_susc_{susc:.1f}"
                    os.makedirs(folder, exist_ok=True)

                    # Create sub folder for the scale
                    scale_folder = os.path.join(folder, scale)
                    os.makedirs(scale_folder, exist_ok=True)

                    plot = Plotter(n_age=self.n_age)
                    plot.plot_contact_input(contact_input=self.contact_input,
                                            plot_title=f"Contact input "
                                                       f"using {scale} scaling",
                                            filename="contact_input.pdf",
                                            folder=scale_folder)

                    matrices = [
                        (self.ngm_small_tensor, "NGM with Small Domain", "ngm_heatmap.pdf"),
                        (self.symmetric_contact_matrix, "Symmetrized Contact "
                                                        "Input Matrix", "CM.pdf"),
                        (plot.reconstruct_plot_symmetric_grad_matrix(self.r0_cm_grad),
                         "Gradient values", "Grads.pdf")
                    ]

                    for matrix, plot_title, filename in matrices:
                        plot.plot_small_ngm_contact_grad_mtx(matrix=matrix,
                                                plot_title=plot_title,
                                                filename=filename,
                                                folder=scale_folder)

    def _create_leaf(self, scale: str):
        # Original contact matrix symmetrization
        cg_leaf_preparator = CGLeafPreparator(data=self.data)
        cg_leaf_preparator.run()
        transformed_total_orig_cm = cg_leaf_preparator.transformed_total_orig_cm

        # Extract upper triangular elements of the contact matrix
        cm_elements_cg_leaf = CMElementsCGLeaf(
            n_age=self.n_age,
            transformed_total_orig_cm=transformed_total_orig_cm,
            pop=self.population)
        cm_elements_cg_leaf.run(scale=scale)
        self.contact_input = cm_elements_cg_leaf.contact_input.requires_grad_(True)
        self.contact_input_sum = cm_elements_cg_leaf.contact_input_sum

    def _contact_matrix_manipulation(self, scale: str):
        """
        Perform manipulations on contact matrices to generate the necessary inputs.
        """
        # Create a new symmetric contact matrix
        cm_creator = CMCreator(n_age=self.n_age,
                               pop=self.population.reshape((-1, 1)))
        cm_creator.run(contact_matrix=self.contact_input,
                       contact_matrix_sum=self.contact_input_sum,
                       scale=scale)
        self.symmetric_contact_matrix = cm_creator.cm

    def _calculate_eigenvectors(self):
        """
        Calculate eigenvalues and gradients for the simulation.
        """
        # Calculate the dominant eigenvalue and eigenvector
        eigen_calculator = EigenCalculator(ngm_small_tensor=self.ngm_small_tensor)
        eigen_calculator.run()
        self.eigen_vector = eigen_calculator.dominant_eig_vec
        self.eigen_value = eigen_calculator.dominant_eig_val

