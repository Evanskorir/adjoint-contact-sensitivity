import os
import torch

from src.gradient.sensitivity_calculator import SensitivityCalculator
from src.plotter import Plotter
from src.static.dataloader import DataLoader


class Runner:
    def __init__(self, data: DataLoader, model: str):
        """
        Initialize the simulation with the provided data.
        Args: data (DataLoader): DataLoader object containing age data and model params.
        """
        self.data = data
        self.population = self.data.age_data
        self.n_age = len(self.population)
        self.model = model
        self.labels = self.data.labels

        # Dynamically set susceptibility choices
        self.model_age_ranges = {"british_columbia": 3, "rost": 4}

        # Determine susceptibility choices based on the model
        if self.model in self.model_age_ranges:
            self.susc_choices = [0.5, 1.0]  # Reduced susceptibility for younger population
        else:
            self.susc_choices = [1.0]  # Uniform susceptibility for other models

        self.sensitivity_calc = SensitivityCalculator(data=self.data,
                                                      model=self.model)
        self.r0_cm_grad = None
        self.r0_choices = self.sensitivity_calc.r0_choices

        self.scales = ["pop_sum"]  # other options: "contact_sum", "no_scale"

    def set_susceptibility(self, susceptibility: torch.Tensor, susc: float):
        """
        Set susceptibility values based on the model type.
        """
        if self.model in self.model_age_ranges:
            susceptibility[:self.model_age_ranges[self.model]] = susc

    def run(self):
        """
        Run the simulation to perform contact matrix manipulations,
        calculate eigenvalues, and compute gradients.
        """
        for scale in self.scales:
            susceptibility = torch.ones(self.n_age)
            for susc in self.susc_choices:
                # Update susceptibility values in the model parameters
                self.set_susceptibility(susceptibility, susc)
                self.sensitivity_calc.params.update({"susc": susceptibility})

                # Run sensitivity calculation for the current scale and parameters
                self.sensitivity_calc.run(scale=scale, params=self.sensitivity_calc.params)

                # Create plots and process the results
                self.create_plots(scale=scale, susc=susc)

    def create_plots(self, scale: str, susc: float):
        """
        Generate plots for different R0 values and susceptibility choices,
        after calculating gradients and applying projection.
        """
        for base_r0 in self.r0_choices:
            # Calculate projected gradients for the current base R0
            self.calculate_projected_gradients(base_r0=base_r0)

            # Create folder structure for saving plots
            folder = f"generated/{self.model}/results_base_r0_{base_r0:.1f}_susc_{susc:.1f}"
            os.makedirs(folder, exist_ok=True)

            # Create sub_folder for each scale
            scale_folder = os.path.join(folder, scale)
            os.makedirs(scale_folder, exist_ok=True)

            # Generate plots for contact input, gradients, NGM matrix, etc.
            self.generate_plots(scale_folder=scale_folder, susc=susc, base_r0=base_r0)

    def calculate_projected_gradients(self, base_r0: float):
        """
        Calculate the projected R0 gradients using the dominant eigenvalue and
        eigenvalue gradient, then apply scaling by beta (base R0 / eigenvalue).
        """
        # Compute beta as base R0 divided by the dominant eigenvalue
        beta = base_r0 / self.sensitivity_calc.eigen_value
        self.sensitivity_calc.params.update({"beta": beta})

        # Scale the eigenvalue gradient by beta to get r0_cm_grad
        self.r0_cm_grad = beta * self.sensitivity_calc.eigen_value_gradient.eig_val_cm_grad

    def generate_plots(self, scale_folder: str, susc: float, base_r0: float):
        """
        Generate various plots such as contact matrix, gradients, NGM matrix, etc.,
        and save them in the specified folder.
        Args:
            scale_folder (str): Path to the folder where plots will be saved.
            susc (float): susc value used for the simulation.
            base_r0 (float): R0 value used for the simulation.
        """
        plot = Plotter(data=self.data, n_age=self.n_age, model=self.model)

        # Create a model folder for saving specific plots under model, not scale_folder
        model_folder = f"generated/{self.model}"
        os.makedirs(model_folder, exist_ok=True)

        # Plot contact matrices
        plot.plot_contact_matrices(contact_data=self.data.contact_data,
                                   model=self.model,
                                   filename="contact_matrices")

        # Plot contact input matrix and save it under the model folder only (not in scale folder)
        contact_input_folder = os.path.join(model_folder, "contact_input")
        os.makedirs(contact_input_folder, exist_ok=True)
        plot.plot_contact_input(
            contact_input=self.sensitivity_calc.contact_input,
            plot_title="",
            filename="contact_input.pdf",
            folder=contact_input_folder
        )

        # Generate and plot the percentage age group contribution bar plot under model folder
        bar_plot_folder = os.path.join(model_folder, "age_group_bar")
        os.makedirs(bar_plot_folder, exist_ok=True)
        plot.get_percentage_age_group_contact_list(
            symmetrized_cont_matrix=self.sensitivity_calc.symmetric_contact_matrix,
            filename="age_group_percentage_bar_plot.pdf",
            folder=bar_plot_folder
        )

        # Generate and plot the symmetric contact matrix under the model folder
        cm_folder = os.path.join(model_folder, "CM")
        os.makedirs(cm_folder, exist_ok=True)
        plot.plot_small_ngm_contact_grad_mtx(
            matrix=self.sensitivity_calc.symmetric_contact_matrix,
            filename="CM.pdf",
            plot_title="Full contact",
            folder=cm_folder,
            label_axes=False,
            show_colorbar=True
        )

        # Plot the NGM matrix twice, once for susc=0.5 and once for susc=1.0
        ngm_folder = os.path.join(model_folder, "NGM")
        os.makedirs(ngm_folder, exist_ok=True)

        # Plot for the current susc value, distinguished by the susc in the filename
        plot.plot_small_ngm_contact_grad_mtx(
            matrix=self.sensitivity_calc.ngm_small_tensor,
            filename=f"ngm_heatmap_susc_{susc}.pdf",
            plot_title=f"NGM for susc={susc}",
            folder=ngm_folder,
            label_axes=True,
            show_colorbar=True
        )

        # Plot R0 gradient matrix
        plot.plot_grads(
            grads=self.r0_cm_grad,
            plot_title=f"$\\overline{{\\mathcal{{R}}}}_0={base_r0}$",
            filename="Grads_tri.pdf",
            folder=scale_folder
        )

        # Plot cumulative sensitivities
        elasticity_folder = os.path.join(model_folder, "cum_elasticities")
        os.makedirs(elasticity_folder, exist_ok=True)

        plot.plot_cumulative_elasticities(
            cumulative_elasticities=self.sensitivity_calc.cumulative_elasticities,
            plot_title=f"$\\overline{{\\mathcal{{R}}}}_0={base_r0}$",
            filename=f"cum_elasticities_susc_{susc:.1f}_r0_{base_r0:.1f}",
            folder=elasticity_folder
        )


