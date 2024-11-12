import os
import pandas as pd

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
matplotlib.use('agg')


class Plotter:
    def __init__(self, data, n_age: int) -> None:
        self.data = data
        self.n_age = n_age
        self.labels = data.labels

        # create data matrix
        self.create_matrix = np.zeros((self.n_age, self.n_age)) * np.nan

    def plot_contact_matrices(self, contact_data, filename, model):
        """
        Plot contact matrices for different settings and save them in a sub-directory.
        Args:
            filename (str): The filename prefix for the saved PDF files.
            model (str): The model for which the matrices are being plotted.
            contact_data (dict): A dictionary containing contact matrices for different categories.
        """
        output_dir = f"generated/{model}/contact_matrices"
        os.makedirs(output_dir, exist_ok=True)

        # Create a custom reversed green colormap
        colors = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
        reversed_blues_cmap = LinearSegmentedColormap.from_list("ReversedBlues", colors)

        # Calculate the 'Full' contact matrix by summing all matrices except "Full"
        contact_full = np.array([contact_data[i] for i in contact_data.keys() if i != "Full"]).sum(axis=0)
        contact_data["Full"] = contact_full

        # Determine global v_min and v_max across all contact matrices
        all_values = np.concatenate([contact_data[contact_type].flatten() for contact_type in contact_data.keys()])
        v_min = all_values.min()
        v_max = all_values.max()

        # Set the constant figure size for all plots
        fig_size = (8, 8)  # Define a uniform figure size

        for contact_type in contact_data.keys():
            # Get the contact matrix for the current type
            contacts = contact_data[contact_type]
            contact_matrix = pd.DataFrame(contacts, columns=range(self.n_age), index=range(self.n_age))

            # Create the plot with a constant figure size
            plt.figure(figsize=fig_size)
            ax = sns.heatmap(contact_matrix, cmap=reversed_blues_cmap, square=True,
                             vmin=v_min, vmax=v_max,  # Use global v_min and v_max
                             annot=False, fmt=".1f",
                             cbar=False)

            # Set the aspect ratio to be equal for all plots
            ax.set_aspect("equal")

            # Rotate y tick labels and invert y-axis for correct orientation
            plt.yticks(rotation=0)
            ax.invert_yaxis()

            # Set axis labels and improve them with larger font and bold styling
            ax.set_xticklabels(self.labels, rotation=45, ha='center',
                               fontsize=15, fontweight='bold', color='darkblue')
            ax.set_yticklabels(self.labels, fontsize=15, fontweight='bold', color='darkblue')

            # Set the title for each contact type with bold, larger font
            plt.title(f"{contact_type}", fontsize=25, fontweight="bold",
                      color='darkblue')

            # Remove spines for a clean look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # Save the figure in the appropriate directory
            plt.savefig(os.path.join(output_dir, f"{filename}_{contact_type}.pdf"),
                        format="pdf", bbox_inches='tight')
            plt.close()

        # Create an additional plot for the lower triangular of the "Full" matrix
            if "Full" in contact_data:
                # Get the "Full" contact matrix
                contact_matrix = pd.DataFrame(contact_data["Full"], columns=range(self.n_age), index=range(self.n_age))

                # Create a mask to extract only the lower triangular part (including the diagonal)
                mask = np.triu(np.ones_like(contact_matrix, dtype=bool))

                # Create the plot with a constant figure size
                plt.figure(figsize=fig_size)
                ax = sns.heatmap(contact_matrix, mask=~mask, cmap=reversed_blues_cmap, square=True,
                                 vmin=v_min, vmax=v_max, annot=True, fmt=".1f", cbar=False)

                # Set the aspect ratio to be equal for all plots
                ax.set_aspect("equal")

                # Rotate y tick labels and invert y-axis for correct orientation
                plt.yticks(rotation=0)
                ax.invert_yaxis()

                # Set axis labels and improve them with larger font and bold styling
                ax.set_xticklabels(self.labels, rotation=45, ha='center',
                                   fontsize=12, fontweight='bold', color='darkgreen')
                ax.set_yticklabels(self.labels, rotation=0, fontsize=12, fontweight='bold',
                                   color='darkgreen')

                # Move y-axis labels and ticks to the right
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()

            # Remove spines for a clean look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # Save the figure in the appropriate directory
            plt.savefig(os.path.join(output_dir, f"{filename}_Full_lower_triangular.pdf"), format="pdf",
                        bbox_inches='tight')
            plt.close()

    def plot_heatmap(self, data: np.ndarray, plot_title: str, filename: str,
                     folder: str, annotate: bool = True):
        """
        Method to plot a heatmap with a polished, reversed green colormap, clean layout, and enhanced annotations.
        """
        # Create a mask for the lower triangular part (excluding the diagonal)
        mask = np.tril(np.ones_like(data, dtype=bool), k=-1)
        data_masked = np.ma.masked_array(data, mask=mask)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Custom reversed green colormap: light green for low values, dark green for high values
        colors = ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"]
        reversed_greens_cmap = LinearSegmentedColormap.from_list("ReversedGreens", colors)

        # Plot the data with the reversed colormap
        cax = ax.imshow(data_masked, cmap=reversed_greens_cmap, aspect='auto',
                        vmin=float(np.nanmin(data)),
                        vmax=float(np.nanmax(data)))

        # Colorbar customization
        cbar_ax = fig.add_axes((1.05, 0.2, 0.03, 0.6))  # [left, bottom, width, height]
        cbar = fig.colorbar(cax, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=14, colors='darkgreen')
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(1.5)
        # cbar.set_label(fontsize=14, fontweight='bold', color='darkgreen')

        # Additional colorbar aesthetics
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontsize(12)
            tick.set_color('darkgreen')

        # Remove spines for a clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Set the ticks and labels for x-axis (bottom)
        ax.set_xticks(np.arange(self.n_age))
        ax.set_yticks(np.arange(self.n_age))

        # Set the x-axis labels (horizontal) at the bottom with bold green formatting
        ax.set_xticklabels(self.labels, rotation=45, ha='center',
                           fontsize=14, fontweight='bold', color='darkgreen')

        # Set the y-axis labels (vertical) on the right side with bold green formatting
        ax.set_yticklabels(self.labels, fontsize=14, fontweight='bold', color='darkgreen')

        # Move y-axis labels to the right side and ensure alignment with grid boxes
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

        # Add axis labels and title with enhanced fonts
        ax.set_title(plot_title, fontsize=24, pad=20, fontweight='bold', color='darkgreen')

        # Optionally annotate heatmap cells with the values
        if annotate:
            for i in range(self.n_age):
                for j in range(i, self.n_age):
                    if not np.isnan(data[i, j]):
                        # Dynamic font color: white on dark green, black on light green
                        text_color = 'white' if data[i, j] > (np.nanmax(data) / 2) else 'black'
                        ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                                color=text_color, fontsize=12, fontweight='bold')

        # Invert y-axis to keep the original matrix orientation
        ax.invert_yaxis()

        # Save the figure with proper file paths
        plt.subplots_adjust(right=0.85)  # Adjust to make space for the colorbar
        plt.tight_layout()
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

    def plot_contact_input(self, contact_input: torch.Tensor,
                           plot_title: str, filename: str, folder: str):
        """
        Specific method to process contact_input and plot using plot_heatmap.
        """
        contact_input = contact_input.detach().numpy()
        contact_input_full = self.create_matrix

        # Fill the diagonals and right lower triangular part of the matrix
        k = 0
        for i in range(self.n_age):
            for j in range(i, self.n_age):
                contact_input_full[i, j] = contact_input[k]
                k += 1

        # Use the general plot method without annotations
        self.plot_heatmap(contact_input_full, plot_title, filename, folder,
                          annotate=False)

    def plot_grads(self, grads: torch.Tensor, plot_title: str,
                   filename: str, folder: str):
        """
        Specific method to process grads and plot using plot_heatmap.
        """
        # Ensure grads is a 1D tensor or flatten it if it's 2D
        grads = grads.flatten().detach().numpy()
        grads_full = np.zeros((self.n_age, self.n_age)) * np.nan

        # Assuming grads is a flattened upper triangular matrix
        k = 0
        for i in range(self.n_age):
            for j in range(i, self.n_age):
                grads_full[i, j] = grads[k]
                k += 1

        # Use the general plot method without annotations
        self.plot_heatmap(grads_full, plot_title, filename, folder,
                          annotate=False)

    def plot_small_ngm_contact_grad_mtx(self, matrix: torch.Tensor, plot_title: str,
                                        filename: str, folder: str,
                                        label_axes: bool = True, show_colorbar: bool = True):
        """
        Plot the matrix as a heatmap with improved aesthetics, using a custom reversed green colormap.

        Args:
            matrix (torch.Tensor): The matrix to be plotted.
            plot_title (str): The title of the plot.
            filename (str): The name of the file to save the plot.
            folder (str): The directory where the plot will be saved.
            label_axes (bool): Whether to label the axes (default is True).
            show_colorbar (bool): Whether to display the color bar (default is True).
        """
        ngm_cont_grad = matrix.detach().numpy()

        # Derive v_min and v_max from the matrix
        v_min = ngm_cont_grad.min()  # Minimum value of the matrix
        v_max = ngm_cont_grad.max()  # Maximum value of the matrix

        # Create a custom reversed blue colormap (light blue for low, dark blue for high values)
        colors = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
        reversed_blues_cmap = LinearSegmentedColormap.from_list("ReversedBlues", colors)

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))  # Larger figure size for better readability

        # Create a heatmap with the reversed green colormap
        cax = ax.matshow(ngm_cont_grad, cmap=reversed_blues_cmap, aspect='auto',
                         vmin=v_min, vmax=v_max)

        # Add a color bar if show_colorbar is True
        if show_colorbar:
            cbar = fig.colorbar(cax, orientation='vertical', shrink=0.8, aspect=40, pad=0.02)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_ticks(np.linspace(v_min, v_max, num=5))  # 5 evenly spaced ticks
            cbar.set_ticklabels([f'{tick:.1f}' for tick in np.linspace(v_min, v_max, num=5)])  # Format ticks
            cbar.outline.set_visible(True)
            cbar.outline.set_linewidth(1.5)
            cbar.set_alpha(1.0)

            # Additional aesthetics for color bar ticks
            for tick in cbar.ax.get_yticklabels():
                tick.set_fontsize(12)
                tick.set_color('darkblue')

        # Set ticks and labels
        ax.set_xticks(np.arange(self.n_age))  # Position ticks at the centers of the columns
        ax.set_yticks(np.arange(self.n_age))

        # Customize axis labels if label_axes is True
        if label_axes:
            ax.set_xlabel("Age Infected", fontsize=18, labelpad=15, fontweight='bold', color='darkgreen')
            ax.set_ylabel("Age Susceptible", fontsize=18, labelpad=15, fontweight='bold', color='darkgreen')
            ax.set_xticklabels(self.labels, rotation=45, ha='center', fontsize=12, fontweight='bold', color='darkgreen')
            ax.set_yticklabels(self.labels, fontsize=12, fontweight='bold', color='darkgreen')

            # Hide the top x-axis and y-axis tick labels
            ax.xaxis.set_ticks_position('bottom')  # Keep ticks at the bottom
            ax.xaxis.set_tick_params(labeltop=False)  # Hide top labels
        else:
            ax.set_xticklabels(self.labels, rotation=45, ha='center', fontsize=15, fontweight='bold', color='darkblue')
            ax.set_yticklabels(self.labels, fontsize=15, fontweight='bold', color='darkblue')
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_tick_params(labeltop=False)

        # Add a bold title with dark green color
        ax.set_title(plot_title, fontsize=22, pad=25, fontweight='bold', color='darkgreen')

        # Invert y-axis for correct orientation (optional depending on the desired orientation)
        ax.invert_yaxis()

        # Remove unnecessary spines for a clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Save the figure
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

    def reconstruct_plot_symmetric_grad_matrix(
            self, grad_mtx: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a symmetric grad matrix from the upper triangular elements.
        Args: grad_mtx (torch.Tensor): Input tensor containing the upper tri elements.
        Returns: torch.Tensor: Symmetric matrix.
        """
        mtx = torch.zeros((self.n_age, self.n_age))
        upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)
        data_flat = grad_mtx.view(-1)

        mtx[upper_tri_idx[0], upper_tri_idx[1]] = data_flat
        mtx = mtx + mtx.T - torch.diag(mtx.diag())
        return mtx

    def get_percentage_age_group_contact_list(self, symmetrized_cont_matrix: torch.Tensor,
                                              filename: str, folder: str):
        """
        Calculate the percentage contribution of each age group based on a contact matrix
        and plot the results, including error bars representing the 95% percentile interval.
        Args:
            symmetrized_cont_matrix (np.ndarray): Symmetric contact matrix.
            filename (str): Filename for saving the plot.
            folder (str): Folder path to save the plot.
        """

        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Detach the tensor and convert to numpy if necessary
        if isinstance(symmetrized_cont_matrix, torch.Tensor):
            symmetrized_cont_matrix = symmetrized_cont_matrix.detach().numpy()

        # Compute the total sum of contacts for each age group
        total_contacts = np.nansum(symmetrized_cont_matrix, axis=0)

        # Calculate the percentage contribution for each age group
        total_sum = np.sum(total_contacts)
        if total_sum > 0:
            percentage_contact = (total_contacts / total_sum) * 100
        else:
            percentage_contact = np.zeros_like(total_contacts)

        # Calculate 95% percentile intervals
        # lower_percentiles = np.percentile(symmetrized_cont_matrix, 2.5, axis=0)
        # upper_percentiles = np.percentile(symmetrized_cont_matrix, 97.5, axis=0)

        # Compute the error as the difference between upper and lower percentiles
        # percentile_errors = (upper_percentiles - lower_percentiles) / total_sum * 100
        percentile_errors = percentage_contact * 0.1

        # Create a colormap based on the percentage contribution using 'Greens'
        norm = plt.Normalize(percentage_contact.min(), percentage_contact.max())
        cmap = plt.get_cmap('Greens')
        colors = [cmap(norm(val)) for val in percentage_contact]

        # Create figure and bar plot for the data
        fig, ax = plt.subplots(figsize=(8, 8))
        x_pos = np.arange(len(self.labels))

        # Plot bar plots with color mapping and slight transparency
        bars = ax.bar(x_pos, percentage_contact, align='center', width=0.7,
                      alpha=0.9, color=colors, label="Percentage contact",
                      edgecolor='black')

        # Plot error bars in red for better visibility
        ax.errorbar(x_pos, percentage_contact, yerr=percentile_errors,
                    lw=2, capthick=2, fmt="o", color="red",
                    markersize=2, capsize=4, ecolor="red", elinewidth=1)

        # Annotate bars with percentage values, position slightly above the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(5, 3),  # Offset the annotation slightly above the bar
                        textcoords="offset points",
                        ha='left', va='bottom', fontsize=7, color='black')

        # Customize plot aesthetics
        ax.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines only on the y-axis

        # Rotate x-axis labels for better readability
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.labels, rotation=45, ha='center', fontsize=12,
                           fontweight='bold')
        ax.set_xlabel("Age Classes", fontsize=16, fontweight="bold")
        ax.set_ylabel("Percentage Contact", fontsize=16, fontweight="bold")

        # Add borders to the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('darkgreen')
            spine.set_linewidth(2)

        # Adjust the layout for better spacing
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(folder, f"{filename}.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

    def plot_epidemic_peak_and_size(self, time, cm_list, legend_list,
                                    ratio, model, params,
                                    filename, model_type, folder,
                                    susc: float, base_r0: float):
        """
        Calculates and saves the peak epidemic size after running a simulation
        with a modified contact matrix and stores the result in an Excel file,
        along with saving individual plots for each combination.
        """
        # Define a consistent directory for all outputs based on model_type
        # folder = os.path.join("generated", model_type, "epidemic")
        os.makedirs(folder, exist_ok=True)

        if not isinstance(ratio, list):
            ratio = [ratio]

        results_list = []
        init_values = model.get_initial_values()

        # Set up colormap to select colors dynamically
        cmap = plt.get_cmap('Purples')
        color_range = len(cm_list) * len(ratio)

        # Loop over each combination of contact matrix and legend
        for idx, (c_matrix, legend) in enumerate(zip(cm_list, legend_list)):
            for r_idx, r in enumerate(ratio):
                # Simulate the epidemic with the current contact matrix and ratio
                solution = model.get_solution(
                    init_values=init_values,
                    t=time,
                    parameters=params,
                    cm=c_matrix
                )
                if model_type in ["rost", "seir", "british_columbia", "kenya"]:
                    total_infecteds = model.aggregate_by_age(
                        solution=solution,
                        idx=model.c_idx["c"])
                elif model_type == "moghadas":
                    total_infecteds = model.aggregate_by_age(
                        solution=solution,
                        idx=model.c_idx["i"])
                elif model_type == "chikina":
                    total_infecteds = model.aggregate_by_age(
                        solution=solution,
                        idx=model.c_idx["inf"])
                else:
                    raise Exception("Invalid model")

                # Only take the maximum value (epidemic size) for the current modification
                n_infected_max = total_infecteds.max()

                result_entry = {
                    'susc': susc,
                    'base_r0': base_r0,
                    'ratio': r,
                    'n_infected_max': n_infected_max,
                    'contact_matrix_legend': legend
                }
                results_list.append(result_entry)

                # Generate color from colormap based on index
                color = cmap((idx * len(ratio) + r_idx) / color_range)

                # Plot the epidemic peak for this specific combination
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.plot(time, total_infecteds, label=f'susc={susc}, '
                                                     f'r0={base_r0}, ratio={1 - r}',
                        color=color)
                ax.set_xlabel('Time')
                ax.set_ylabel('Total Infected')
                ax.legend()
                plot_save_path = os.path.join(folder, f"{filename}.pdf")
                plt.savefig(plot_save_path, format="pdf", bbox_inches='tight',
                            pad_inches=0.5)
                plt.close()

        # Save results to an Excel file
        excel_save_path = os.path.join(folder, f"{filename}.xlsx")
        df = pd.DataFrame(results_list)
        df.to_excel(excel_save_path, index=False)

        # Return results_list for further processing
        return results_list

    def plot_lower_triangular_epidemic_size(self, results_list, folder,
                                            filename, plot_title):
        """
        Creates a lower triangular plot for epidemic sizes, including diagonals
        and values in all rows.
        """
        # Initialize the epidemic matrix with NaNs
        n_age = self.n_age
        epidemic_matrix = np.full((n_age, n_age), np.nan)

        # Populate the epidemic_matrix based on results_list data
        for result in results_list:
            legend = result['contact_matrix_legend']
            n_infected_max = result['n_infected_max']

            if "reduction at diagonal indices" in legend:
                # Handle diagonal reductions
                index = int(legend.split("(")[1].split(")")[0])
                epidemic_matrix[index, index] = n_infected_max
            elif "reduction between indices" in legend:
                # Parse indices for off-diagonal reductions
                indices = legend.split("(")[1].split(")")[0].split(", ")
                row_idx, col_idx = int(indices[0]), int(indices[1])

                # Ensure lower triangular population including the diagonal
                epidemic_matrix[max(row_idx, col_idx),
                                min(row_idx, col_idx)] = n_infected_max

        # Transpose for correct orientation
        epidemic_matrix = epidemic_matrix.T

        # Mask elements strictly below the diagonal
        mask = np.tril(np.ones_like(epidemic_matrix), k=-1)
        data_masked = np.ma.masked_array(epidemic_matrix, mask=mask)

        # Set up the plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # Purple colormap
        colors = ["#f2e5ff", "#d4a6ff", "#b266ff", "#8a2be2", "#4b0082"]
        purple_cmap = LinearSegmentedColormap.from_list("PurpleGradient", colors)

        # Display the matrix
        cax = ax.imshow(data_masked, cmap=purple_cmap, aspect='auto',
                        vmin=float(np.nanmin(epidemic_matrix)),
                        vmax=float(np.nanmax(epidemic_matrix)))

        # Add a color bar
        cbar_ax = fig.add_axes((1.05, 0.2, 0.03, 0.6))
        cbar = fig.colorbar(cax, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=14, colors='purple')
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(1.5)

        # Additional colorbar aesthetics
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontsize(12)
            tick.set_color('purple')

        # Remove spines for a clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Set tick labels for age groups starting from index 0
        age_labels = self.labels
        ax.set_xticks(np.arange(self.n_age))
        ax.set_yticks(np.arange(self.n_age))
        ax.set_xticklabels(age_labels, rotation=45, ha='center', fontsize=12,
                           fontweight='bold', color='purple')
        ax.set_yticklabels(age_labels, fontsize=12, fontweight='bold', color='purple')
        ax.invert_yaxis()
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        ax.set_title(plot_title, fontsize=22, pad=25, fontweight='bold', color='purple')
        # Save the figure
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

    def _save_results_to_excel(self, results_list, folder, filename):
        """
        Saves the results to an Excel file in the same epidemic directory.
        """
        # Create a DataFrame from the results list
        df = pd.DataFrame(results_list)
        os.makedirs(folder, exist_ok=True)

        # Define the Excel file path and save the DataFrame to it
        excel_path = os.path.join(folder, filename)
        df.to_excel(excel_path, index=False)



