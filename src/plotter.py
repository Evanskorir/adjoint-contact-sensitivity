import os

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

matplotlib.use('agg')


class Plotter:
    def __init__(self, data, n_age: int, model) -> None:
        self.data = data
        self.n_age = n_age
        self.labels = data.labels
        self.model = model

        # Create data matrix
        self.create_matrix = np.zeros((self.n_age, self.n_age)) * np.nan

        # Custom reversed blue colormap
        blue_colors = ["#f7fbff", "#c6dbef", "#6baed6",
                       "#2171b5", "#08306b"]
        self.reversed_blues_cmap = LinearSegmentedColormap.from_list(
            "ReversedBlues", blue_colors)
        # Custom reversed green colormap: light green for low values,
        # dark green for high values
        green_colors = ["#f7fcf5", "#c7e9c0", "#74c476",
                        "#238b45", "#00441b"]
        self.reversed_greens_cmap = LinearSegmentedColormap.from_list(
            "ReversedGreens", green_colors)

    @staticmethod
    def save_figure(ax, output_path):
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.savefig(output_path, format="pdf", bbox_inches='tight')
        plt.close()

    @staticmethod
    def get_tick_labels(labels, alternate=False):
        """Helper function to generate tick labels with optional alternation."""
        if alternate:
            return [label if i % 2 == 0 else "" for i, label in enumerate(labels)]
        return labels

    def style_axes(self, ax, label_axes=True):
        """Style the axes with appropriate labels and ticks."""
        ax.set_xticks(np.arange(self.n_age) + 0.5)
        ax.set_yticks(np.arange(self.n_age) + 0.5)

        # Customize axis labels
        if label_axes:
            alternate = self.model == "rost"
            xtick_labels = self.get_tick_labels(self.labels, alternate)
            y_tick_labels = self.get_tick_labels(self.labels, alternate)
            ax.set_xticklabels(xtick_labels, fontsize=20, fontweight='bold',
                               rotation=90, ha='center', color='darkblue')
            ax.set_yticklabels(y_tick_labels, fontsize=20, fontweight='bold',
                               rotation=0, va='center', color='darkblue')
        else:
            ax.set_xticklabels(self.labels, fontsize=20, fontweight='bold',
                               rotation=45, ha='center', color='darkblue')
            ax.set_yticklabels(self.labels, fontsize=20, fontweight='bold',
                               ha='center', color='darkblue')

        # Adjust tick mark size and appearance
        ax.tick_params(axis='both', which='major', length=10, width=3,
                       labelsize=20, color='darkblue')

        # Invert y-axis for better orientation
        ax.invert_yaxis()

        # Hide spines for a cleaner plot
        for spine in ax.spines.values():
            spine.set_visible(False)

    def setup_axes(self, ax, label_axes=True):
        """
        Helper method to set up axes with labels and aesthetics.
        """
        # Set ticks for the axes
        ax.set_xticks(np.arange(self.n_age))
        ax.set_yticks(np.arange(self.n_age))

        # Customize axes based on label_axes flag
        if label_axes:
            ax.set_xlabel("Age Infected", fontsize=18, labelpad=15,
                          fontweight='bold', color='darkgreen')
            ax.set_ylabel("Age Susceptible", fontsize=18, labelpad=15,
                          fontweight='bold', color='darkgreen')
        else:
            ax.set_xticklabels(self.labels, rotation=45, ha='center',
                               fontsize=15, fontweight='bold', color='darkblue')
            ax.set_yticklabels(self.labels, fontsize=15, fontweight='bold',
                               color='darkblue')

        ax.xaxis.set_ticks_position('bottom')  # Keep ticks at the bottom
        ax.xaxis.set_tick_params(labeltop=False)  # Hide top labels
        ax.invert_yaxis()  # Reverse y-axis for better orientation

        # Remove spines for a clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

    def plot_matrix(self, matrix, title, v_min, v_max, output_path, mask=None,
                    annotate=False):
        """Helper function to plot a single contact matrix."""
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(matrix, cmap=self.reversed_blues_cmap, square=True,
                         vmin=v_min, vmax=v_max,
                         annot=annotate, fmt=".1f", mask=mask, cbar=False)
        ax.set_aspect("equal")

        # Style axes and add title
        self.style_axes(ax)
        plt.title(title, fontsize=25, fontweight="bold", color='darkblue')

        # Save the figure
        self.save_figure(ax, output_path)

    def plot_contact_matrices(self, contact_data, filename, model):
        """
        Plot contact matrices for different settings and save them in a subdirectory.
        """
        output_dir = f"generated/{model}/contact_matrices"
        os.makedirs(output_dir, exist_ok=True)

        # Calculate the 'Full' contact matrix by summing all matrices except "Full"
        contact_full = np.array([contact_data[i] for i in
                                 contact_data.keys() if i != "Full"]).sum(axis=0)
        contact_data["Full"] = contact_full

        # Determine global v_min and v_max across all contact matrices
        all_values = np.concatenate([contact_data[contact_type].flatten() for
                                     contact_type in contact_data.keys()])
        v_min, v_max = all_values.min(), all_values.max()

        # Plot each contact matrix
        for contact_type, matrix in contact_data.items():
            output_path = os.path.join(output_dir, f"{filename}_{contact_type}.pdf")
            self.plot_matrix(pd.DataFrame(matrix), contact_type + " contact", v_min,
                             v_max, output_path)

        # Create a lower triangular plot for the "Full" matrix
        if "Full" in contact_data:
            full_matrix = pd.DataFrame(contact_data["Full"])
            mask = np.triu(np.ones_like(full_matrix, dtype=bool))  # Upper triangular mask
            output_path = os.path.join(output_dir, f"{filename}_Full_lower_triangular.pdf")
            self.plot_matrix(full_matrix, "Full (Lower Triangular)", v_min, v_max,
                             output_path, mask=~mask, annotate=True)

    def plot_heatmap(self, data: np.ndarray, plot_title: str, filename: str,
                     folder: str, annotate: bool = True):
        """
        Method to plot a heatmap with a polished, reversed green colormap, clean layout,
        and enhanced annotations.
        """
        # Create a mask for the lower triangular part (excluding the diagonal)
        mask = np.tril(np.ones_like(data, dtype=bool), k=-1)
        data_masked = np.ma.masked_array(data, mask=mask)
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot the data with the reversed colormap
        cax = ax.imshow(data_masked, cmap=self.reversed_greens_cmap, aspect='auto',
                        vmin=float(np.nanmin(data)),
                        vmax=float(np.nanmax(data)))
        cbar_ax = fig.add_axes((1.05, 0.2, 0.03, 0.6))  # [left, bottom, width, height]
        cbar = fig.colorbar(cax, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=14, colors="darkgreen")
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(1.5)

        # Additional colorbar aesthetics
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontsize(12)
            tick.set_color("darkgreen")

        # Remove spines for a clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Set the ticks and labels for x-axis (bottom)
        ax.set_xticks(np.arange(self.n_age))
        ax.set_yticks(np.arange(self.n_age))

        # Set the x-axis labels (horizontal) at the bottom with bold green formatting
        ax.set_xticklabels(self.labels, rotation=90, ha='center',
                           fontsize=15, fontweight='bold', color='darkgreen')

        # Set the y-axis labels (vertical) on the right side with bold green formatting
        ax.set_yticklabels(self.labels, fontsize=15, fontweight='bold', color='darkgreen')

        # Move y-axis labels to the right side and ensure alignment with grid boxes
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

        # Add axis labels and title with enhanced fonts
        ax.set_title(plot_title, fontsize=22, pad=20, fontweight='bold',
                     color='darkgreen')

        # Optionally annotate heatmap cells with the values
        if annotate:
            for i in range(self.n_age):
                for j in range(i, self.n_age):
                    if not np.isnan(data[i, j]):
                        text_color = 'white' if data[i, j] > (np.nanmax(data) / 2) else 'black'
                        ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                                color=text_color, fontsize=12, fontweight='bold')

        # Invert y-axis to keep the original matrix orientation
        ax.invert_yaxis()

        # Save the figure with proper file paths
        plt.subplots_adjust(right=0.85)
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

    def plot_r0_small_ngm_grad_mtx(self, matrix: torch.Tensor, plot_title: str,
                    filename: str, folder: str, cmap_type: str,
                                   label_color: str):
        """
        General function to plot a matrix as a heatmap with customizable colormap.

        Args:
            matrix (torch.Tensor): The matrix to be plotted.
            plot_title (str): The title of the plot.
            filename (str): The name of the file to save the plot.
            folder (str): The directory where the plot will be saved.
            cmap_type (str): Specify the colormap type ("CM" for Contact Matrix or "NGM" for NGM).
            label_color: Color of the text and labels.
        """
        # Define min and max values for the color scale
        matrix = matrix.detach().numpy()

        v_min = matrix.min()
        v_max = matrix.max()

        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

        # Define colormaps based on cmap_type
        if cmap_type == "CM":
            cmap = self.reversed_blues_cmap   # Blues
        elif cmap_type == "NGM":
            # Define a yellow colormap for Next Generation Matrix
            cmap = LinearSegmentedColormap.from_list(
                "YellowRedGradient", ["#FFFFE0", "#FFD700", "#FF0000"]
            )
            # cmap = plt.cm.viridis

        else:
            raise ValueError("Invalid cmap_type. Use 'CM' for Contact Matrix or "
                             "'NGM' for Next Generation Matrix.")

        # Create a heatmap
        cax = ax.matshow(matrix, cmap=cmap, aspect='equal', vmin=v_min, vmax=v_max)

        # Add a color bar
        cbar = fig.colorbar(cax, orientation='vertical', shrink=0.69, pad=0.1)
        cbar.ax.tick_params(labelsize=20, colors=label_color)
        cbar.set_ticks(np.linspace(v_min, v_max, num=5))  # 5 evenly spaced ticks
        cbar.set_ticklabels([f'{tick:.2f}' for tick in np.linspace(
            v_min, v_max, num=5)])
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(1.5)

        # Set ticks and labels
        xtick_labels = self.get_tick_labels(
            self.labels, alternate=self.model in ["rost", "seir"])
        ytick_labels = self.get_tick_labels(
            self.labels, alternate=self.model in ["rost", "seir"])
        ax.set_xticks(np.arange(self.n_age))
        ax.set_yticks(np.arange(self.n_age))
        ax.set_xticklabels(xtick_labels, rotation=90, ha='center',
                           fontsize=20, fontweight='bold', color=label_color)
        ax.set_yticklabels(ytick_labels, fontsize=20, fontweight='bold',
                           color=label_color)

        # Ensure labels appear only on the bottom x-axis
        ax.xaxis.set_ticks_position('bottom')  # Position ticks at the bottom
        ax.xaxis.set_tick_params(labeltop=False)  # Disable top x-axis labels
        ax.tick_params(axis='both', which='major', length=10, width=3,
                       labelsize=20, color=label_color)

        # Add a bold title
        ax.set_title(plot_title, fontsize=25, fontweight='bold', color=label_color)

        ax.invert_yaxis()

        # Remove unnecessary spines for a clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

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
