import numpy as np
import torch

from src.plotter import Plotter


class ContactManipulation:
    def __init__(self, sens_calc, susc: float, base_r0: float):
        self.susc = susc
        self.base_r0 = base_r0
        self.sens_calc = sens_calc

        # Ensure contact_matrix is a NumPy array
        if isinstance(self.sens_calc.symmetric_contact_matrix, torch.Tensor):
            self.contact_matrix = self.sens_calc.symmetric_contact_matrix.detach().cpu().numpy()
        else:
            self.contact_matrix = self.sens_calc.symmetric_contact_matrix
        if isinstance(self.sens_calc.params, torch.Tensor):
            self.params = self.sens_calc.params.detach().cpu().numpy()
        else:
            self.params = self.sens_calc.params
        self.plotter = Plotter(data=self.sens_calc.data,
                               n_age=self.sens_calc.n_age,
                               model=self.sens_calc.model)

    def run_plots(self, folder, file_name, plot_title):
        cm_list_orig = []
        legend_list_orig = []
        self.get_full_contact_matrix(cm_list_orig, legend_list_orig)

        # Define time range based on model type
        if self.sens_calc.model in ["rost", "kenya", "seir"]:
            t = np.arange(0, 800, 0.5)
        elif self.sens_calc.model in ["chikina", "british_columbia", "washington"]:
            t = np.arange(0, 500, 0.5)
        else:
            t = np.arange(0, 200, 0.5)

        # Define ratio and initialize lists
        ratio = [0.5]  # 50% element wise contact reduction
        cm_list = cm_list_orig.copy()
        legend_list = legend_list_orig.copy()

        # Specify pairs of elements (age groups) to be manipulated
        for i in range(self.sens_calc.n_age):
            for j in range(i, self.sens_calc.n_age):  # Loop through all pairs (including diagonal)
                # Apply the reduction ratio to each (i, j) pair
                self.generate_contact_matrix(
                    cm_list=cm_list,
                    legend_list=legend_list,
                    row_index=i,
                    col_index=j,
                    ratio=ratio
                )

        # Plot epidemic size and peak and get results list
        results_list = self.plotter.plot_epidemic_peak_and_size(
            time=t,
            cm_list=cm_list,
            legend_list=legend_list,
            sim_model=self.sens_calc.sim_model,
            ratio=ratio,
            susc=self.susc,
            base_r0=self.base_r0,
            filename=file_name,
            model=self.sens_calc.model,
            params=self.params,
            folder=folder
        )

        # Plot the lower triangular epidemic size matrix
        self.plotter.plot_lower_triangular_epidemic_size(
            results_list=results_list,
            folder=folder, filename=file_name,
            plot_title=plot_title
        )

    def get_full_contact_matrix(self, cm_list, legend_list):
        cm = self.contact_matrix
        cm_list.append(cm)
        legend_list.append("Total contact")

    def generate_contact_matrix(self, cm_list, legend_list, row_index, col_index, ratio):
        # Create a copy of the original contact matrix
        contact_matrix_spec = np.copy(self.contact_matrix)

        # Apply the specified ratio to the specific element and ensure symmetry
        self._apply_ratio_to_specific_element(
            contact_matrix_spec=contact_matrix_spec,
            row_index=row_index,
            col_index=col_index,
            ratio=ratio[0]
        )

        # Generate legend entry
        if row_index == col_index:
            legend = "{r}% reduction at diagonal indices ({ri})".format(
                r=int((1 - ratio[0]) * 100),
                ri=row_index
            )
        else:
            legend = "{r}% reduction between indices ({ri}, {ci})".format(
                r=int((1 - ratio[0]) * 100),
                ri=row_index,
                ci=col_index
            )

        legend_list.append(legend)
        cm_list.append(contact_matrix_spec)

    @staticmethod
    def _apply_ratio_to_specific_element(contact_matrix_spec, row_index,
                                         col_index, ratio):
        # Check if the entry is on the diagonal
        if row_index == col_index:
            # Apply the ratio once for diagonal elements
            contact_matrix_spec[row_index, col_index] *= ratio
        else:
            # Apply the ratio to both [row_index, col_index] and [col_index, row_index]
            # for off-diagonal elements
            contact_matrix_spec[row_index, col_index] *= ratio
            contact_matrix_spec[col_index, row_index] *= ratio
