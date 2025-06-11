import os
import torch

from src.gradient.sensitivity_calculator import SensitivityCalculator
from src.plotter import Plotter
from src.static.dataloader import DataLoader


class Runner:
    def __init__(self, data: DataLoader, model: str, target: str,
                 use_cm_elasticity: bool = False):
        """
        Initialize the simulation with the provided data.
        Args: data (DataLoader): DataLoader object containing age data and model params.
        """
        self.data = data
        self.population = self.data.age_data
        self.n_age = len(self.population)
        self.model = model
        self.labels = self.data.labels

        self.target = target
        self.use_cm_elasticity = use_cm_elasticity

        # Dynamically set susceptibility choices
        self.model_age_ranges = {"rost_agg": 2, "british_columbia": 3, "rost": 4}

        # Determine susceptibility choices based on the model
        if self.model in self.model_age_ranges:
            self.susc_choices = [0.5, 1.0]  # Reduced susceptibility for younger population
        else:
            self.susc_choices = [1.0]  # Uniform susceptibility for other models

        self.sensitivity_calc = SensitivityCalculator(
            data=self.data, model=self.model, target=self.target,
            use_cm_elasticity=self.use_cm_elasticity
        )
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
                self.sensitivity_calc.run(scale=scale,
                                          params=self.sensitivity_calc.params)

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
            folder = f"generated/{self.model}/{self.target}/" \
                     f"results_base_r0_{base_r0:.1f}_susc_{susc:.1f}"
            os.makedirs(folder, exist_ok=True)

            # Create sub_folder for each scale
            scale_folder = os.path.join(folder, scale)
            os.makedirs(scale_folder, exist_ok=True)

            # Generate plots for contact input, gradients, NGM matrix, etc.
            self.generate_plots(scale_folder=scale_folder, base_r0=base_r0)

    def calculate_projected_gradients(self, base_r0: float):
        """
        Calculate the projected R0 gradients using the dominant eigenvalue and
        eigenvalue gradient, then apply scaling by beta (base R0 / eigenvalue).
        """
        # Compute beta as base R0 divided by the dominant eigenvalue
        beta = base_r0 / self.sensitivity_calc.eigen_value
        self.sensitivity_calc.params.update({"beta": beta})

        # Scale the eigenvalue gradient by beta to get r0_cm_grad
        self.r0_cm_grad = beta * self.sensitivity_calc.r0_cm_grad

    def generate_plots(self, scale_folder: str, base_r0: float):
        """
        Generate various plots such as contact matrix, gradients, NGM matrix, etc.,
        and save them in the specified folder.
        Args:
            scale_folder (str): Path to the folder where plots will be saved.
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

        # Generate and plot the symmetric contact matrix under the model folder
        cm_folder = os.path.join(model_folder, "CM")
        os.makedirs(cm_folder, exist_ok=True)
        plot.plot_r0_small_ngm_grad_mtx(
            matrix=self.sensitivity_calc.symmetric_contact_matrix,
            filename="CM.pdf",
            folder=cm_folder,
            cmap_type="CM",
            label_color="darkblue"
        )

        # Plot R0 gradient matrix
        title = f"$\\overline{{\\mathcal{{R}}}}_0={base_r0}$"
        plot.plot_grads(
            grads=self.r0_cm_grad,
            plot_title=title,
            filename="Grads_tri.pdf",
            folder=scale_folder
        )

        # Plot cumulative sensitivities
        plot.plot_cumulative_sensitivities(
            cum_sensitivities=self.sensitivity_calc.cum_sens,
            plot_title=title,
            filename=f"cum_sens",
            folder=scale_folder
        )
