import os

import numpy as np
import torch

from src.agg_p_values import Aggregation_Pvalues
from src.dataloader import DataLoader
from src.plotter import Plotter
from src.r0_generator import R0Generator
from src.simulation_base import SimulationBase


class SimulationNPI(SimulationBase):
    def __init__(self, data: DataLoader) -> None:
        super().__init__(data=data)

        # User-defined parameters
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.2, 1.8, 2.5]

    def calculate_r0(self):
        # Update params by susceptibility vector
        susceptibility = torch.ones(self.n_ag)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            # Update params by calculated BASELINE beta
            for base_r0 in self.r0_choices:
                r0generator = R0Generator(param=self.params, n_age=self.n_ag)
                eig_val_eff, ngm_small, ngm_small_gradients = r0generator.get_eig_val(
                    contact_mtx=self.contact_matrix,
                    susceptibles=self.susceptibles.reshape(1, -1),
                    population=self.population
                )
                # Extract the eigenvalue from the list
                r0 = eig_val_eff[0]

                beta = base_r0 / r0
                self.params.update({"beta": beta})
                self.sim_state.update(
                    {"base_r0": base_r0,
                     "beta": beta,
                     "susc": susc,
                     "r0generator": r0generator})

                # Convert tensors to NumPy arrays
                ngm = ngm_small.detach().numpy()
                ngm_grad = ngm_small_gradients[0].detach().numpy()
                # Save ngm_small_transformed and ngm_small_gradients tensors
                save_dir = os.path.join("saved_files", f"R0_{base_r0}_susc_{susc}")
                os.makedirs(save_dir, exist_ok=True)
                # Save as CSV file
                ngm_file = os.path.join(save_dir, "ngm_small.csv")
                np.savetxt(ngm_file, ngm, delimiter=",")
                ngm_grad_file = os.path.join(save_dir, "ngm_small_grad.csv")
                np.savetxt(ngm_grad_file, ngm_grad, delimiter=",")

    def calculate_aggregated_p_values(self, calculation_approach):
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                print(susc, base_r0)
                save_dir = os.path.join("saved_files", f"R0_{base_r0}_susc_{susc}")
                if os.path.exists(save_dir):
                    ngm_small_filename = os.path.join(save_dir, "ngm_small.csv")
                    agg_pval = Aggregation_Pvalues(
                        sim_obj=self,
                        calculation_approach=calculation_approach
                    )
                    # Load the CSV file
                    ngm_small = np.loadtxt(ngm_small_filename, delimiter=",")
                    # Calculate p-values and save
                    p_values_matrix = agg_pval.calculate_p_values(ngm_small=ngm_small)
                    p_values_file = os.path.join(save_dir, "p_values.csv")
                    np.savetxt(p_values_file, p_values_matrix, delimiter=",")

                    # Calculate and aggregate sensitivity values
                    ngm_grad_filename = os.path.join(save_dir, "ngm_small_grad.csv")
                    # Load the CSV file
                    ngm_grad = np.loadtxt(ngm_grad_filename, delimiter=",")
                    agg_pval.aggregate_prcc_values(ngm_grad=ngm_grad)

                    # Set the aggregated values based on the calculation approach
                    stack_value = np.hstack(
                        [agg_pval.agg_prcc, agg_pval.confidence_lower,
                         agg_pval.confidence_upper]
                    ).reshape(-1, self.n_ag).T
                    agg = stack_value

                    # Save aggregated values
                    agg_file = os.path.join(save_dir, f"agg_{calculation_approach}.csv")
                    np.savetxt(agg_file, agg, delimiter=",")

    def load_plot_ngm_gradients(self, calculation_approach):
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                print(susc, base_r0)
                directory = os.path.join("saved_files", f"R0_{base_r0}_susc_{susc}")

                for val in ["ngm_small.csv", "ngm_small_grad.csv", "p_values.csv",
                            "agg_mean.csv"]:
                    file_path = os.path.join(directory, val)

                    if os.path.exists(file_path):
                        saved_data = np.loadtxt(file_path, delimiter=",")
                        plotter = Plotter(sim_obj=self, data=self.data)
                        filename_without_ext = os.path.splitext(val)[0]

                        if "agg_mean.csv" in val:
                            plot_title = file_path[15:18]
                            # Plot aggregation sensitivity p-values
                            plotter.plot_aggregation_sensitivity_pvalues(
                                prcc_vector=abs(saved_data[:, 0]),
                                conf_lower=abs(saved_data[:, 1]),
                                conf_upper=abs(saved_data[:, 2]),
                                directory=directory,
                                filename_without_ext="agg_mean",
                                calculation_approach=calculation_approach,
                                plot_title=plot_title)
                        else:
                            plot_title = file_path[15:18]
                            # Plot ngm gradients
                            plotter.plot_ngm_grad_matrices(saved_file=saved_data,
                                                           directory=directory,
                                                           filename_without_ext=
                                                           filename_without_ext,
                                                           plot_title=plot_title)

















