import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import matplotlib.patheffects as pe

from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from matplotlib.ticker import LogFormatter
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpzc} \usepackage{amsmath}'

matplotlib.use('agg')
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"""
        \usepackage{amsmath, amssymb, mathrsfs, textcomp}
        \DeclareMathAlphabet{\mathpzc}{OT1}{pzc}{m}{it}
    """
})


class Plotter:
    def __init__(self, data, n_age: int, model) -> None:
        self.data = data
        self.n_age = n_age

        self.model = model
        if self.model == "british_columbia":
            self.labels = [r"$<2$", "2-5", "6-17", "18-24", "25-34", "35-44",
                           "45-54", "55-64", "65-74", "75+"]
        else:
            self.labels = data.labels

        # Create data matrix
        self.create_matrix = np.zeros((self.n_age, self.n_age)) * np.nan

        # Custom reversed blue colormap
        blue_colors = ["#f7fbff", "#c6dbef", "#6baed6",
                       "#2171b5", "#08306b"]
        self.reversed_blues_cmap = LinearSegmentedColormap.from_list(
            "ReversedBlues", blue_colors)

        # Custom reversed green colormap
        green_colors = ["#f7fcf5", "#c7e9c0", "#74c476",
                        "#238b45", "#00441b"]
        self.reversed_greens_cmap = LinearSegmentedColormap.from_list(
            "ReversedGreens", green_colors)

    @staticmethod
    def save_figure(ax, output_path):
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
            ax.set_xticklabels(xtick_labels, fontsize=20,
                               rotation=90, ha='center', color='black')
            ax.set_yticklabels(y_tick_labels, fontsize=20,
                               rotation=0, va='center', color='black')
        else:
            ax.set_xticklabels(self.labels, fontsize=20,
                               rotation=45, ha='center', color='black')
            ax.set_yticklabels(self.labels, fontsize=20,
                               ha='center', color='black')

        # Adjust tick mark size and appearance
        ax.tick_params(axis='both', which='major', length=10, width=3,
                       labelsize=20, color='black')
        # Invert y-axis for better orientation
        ax.invert_yaxis()

    def setup_axes(self, ax, label_axes=True):
        """
        Helper method to set up axes with labels and aesthetics.
        """
        # Set ticks for the axes
        ax.set_xticks(np.arange(self.n_age))
        ax.set_yticks(np.arange(self.n_age))

        # Customize axes based on label_axes flag
        if label_axes:
            ax.set_xlabel("Age Infected", fontsize=18, labelpad=15, color='black')
            ax.set_ylabel("Age Susceptible", fontsize=18, labelpad=15, color='black')

        else:
            ax.set_xticklabels(self.labels, rotation=45, ha='center',
                               fontsize=15, color='black')
            ax.set_yticklabels(self.labels, fontsize=15,
                               color='black')
        ax.invert_yaxis()

        # Remove spines for a clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

    def plot_matrix(self, matrix, title, v_min, v_max, output_path,
                    mask=None, annotate=False, show_cbar=False, norm=None):
        """Plot a contact matrix with a shared log scale and optional colorbar."""
        matrix = matrix.copy()
        matrix[matrix <= 0] = np.nanmin(matrix[matrix > 0]) / 10

        fig, ax = plt.subplots(figsize=(8, 8))

        sns.heatmap(
            matrix,
            cmap=self.reversed_blues_cmap,
            square=True,
            norm=norm,
            annot=annotate,
            fmt=".1f",
            mask=mask,
            cbar=False,
            ax=ax
        )
        if show_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.2)
            sm = cm.ScalarMappable(norm=norm, cmap=self.reversed_blues_cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize=30, width=2, length=8)
            cbar.ax.yaxis.set_label_position('right')
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.set_label("Avg. num. contacts", fontsize=25,
                           labelpad=10, color="black")

        self.style_axes(ax)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["bottom"].set_color("black")

        ax.set_title(title, fontsize=25, color='black')

        plt.tight_layout()
        self.save_figure(ax, output_path)

    def plot_contact_matrices(self, contact_data, filename, model):
        """
        Plot contact matrices with shared log color scale.
        Colorbar is only displayed for 'Full' contact matrix but applies to all.
        """
        output_dir = f"generated/{model}/contact_matrices"
        os.makedirs(output_dir, exist_ok=True)

        contact_full = np.sum(
            [contact_data[key] for key in contact_data if key != "Full"], axis=0
        )
        contact_data["Full"] = contact_full
        all_values = np.concatenate([mat.flatten() for mat in contact_data.values()])
        v_min = np.nanmin(all_values[all_values > 0])
        v_max = np.nanmax(all_values)
        norm = LogNorm(vmin=max(v_min, 1e-3), vmax=v_max)
        for contact_type, matrix in contact_data.items():
            output_path = os.path.join(output_dir, f"{filename}_{contact_type}.pdf")
            self.plot_matrix(
                pd.DataFrame(matrix),
                title=f"{contact_type} contact",
                v_min=norm.vmin,
                v_max=norm.vmax,
                output_path=output_path,
                show_cbar=(contact_type == "Full"),
                norm=norm
            )

    def plot_side_by_side_age_distribution(self, kenya_pop: torch.Tensor,
                                           hungary_pop: torch.Tensor,
                                           output_path: str):

        mpl.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.left": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "font.size": 15
        })

        kenya_pop = kenya_pop.detach().cpu().numpy()
        hungary_pop = hungary_pop.detach().cpu().numpy()

        age_labels = self.labels
        n = len(age_labels)
        x = np.arange(n)

        bar_width = 0.2
        spacing = bar_width

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.bar(x - spacing / 2, kenya_pop, width=bar_width,
               color='cyan', edgecolor='black', linewidth=0.4, label="Kenya")
        ax.bar(x + spacing / 2, hungary_pop, width=bar_width,
               color='orange', edgecolor='black', linewidth=0.4, label="Hungary")

        ax.set_ylabel("Population", fontsize=20, fontweight='bold')
        ax.tick_params(axis='y', which='both', length=5, width=1.0,
                       direction='out', labelsize=15)

        ax.set_xticklabels(age_labels, rotation=45, ha='center', fontsize=15)
        ax.tick_params(axis='x', which='both', length=0)
        ax.set_xticks(x)

        ax.legend(loc="upper right", fontsize=15)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format="pdf", dpi=400, bbox_inches='tight')
        plt.close()

    def plot_heatmap(self, data: np.ndarray, plot_title: str, filename: str,
                     folder: str, annotate: bool = True):
        """
        Method to plot a heatmap
        and enhanced annotations.
        """

        # Create a mask for the lower triangular part (excluding the diagonal)
        mask = np.tril(np.ones_like(data, dtype=bool), k=-1)
        data_masked = np.ma.masked_array(data, mask=mask)
        fig, ax = plt.subplots(figsize=(8, 8))

        # Avoid log(0) or negative values by masking them (LogNorm requires > 0)
        safe_data = np.where(data_masked <= 0, np.nan, data_masked)

        # Determine log scale range (ignoring NaNs)
        log_vmin = np.nanmin(safe_data)
        log_vmax = np.nanmax(safe_data)

        # Plot using logarithmic normalization
        cax = ax.imshow(safe_data, cmap=self.reversed_greens_cmap, aspect='auto',
                        norm=LogNorm(vmin=log_vmin, vmax=log_vmax))

        # Set log ticks: nicely spaced, clean limits
        log_ticks = np.logspace(np.floor(np.log10(log_vmin)),
                                np.ceil(np.log10(log_vmax)),
                                num=5)

        # Create colorbar axis with generous height
        cbar_ax = fig.add_axes((1.05, 0.2, 0.05, 0.6))

        # Create colorbar with custom ticks
        cbar = fig.colorbar(
            cax,
            cax=cbar_ax,
            ticks=log_ticks,
            format=LogFormatter(labelOnlyBase=False)  # Show full tick labels
        )

        # Optional: math-style tick labels for elegance
        tick_labels = [rf"$10^{{{int(np.log10(t))}}}$" for t in log_ticks]
        cbar.set_ticklabels(tick_labels)

        # Style ticks and outline
        cbar.ax.tick_params(
            labelsize=30,
            width=2.5,
            length=10,
            direction='out',
            color='black'
        )
        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(0.5)

        # Add stroke effect to tick labels (glow for clarity)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight('bold')
            tick.set_color("black")
            tick.set_path_effects([pe.withStroke(linewidth=0.7,
                                                 foreground='white')])

        # Remove spines for a clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Set the ticks and labels for x-axis (bottom)
        ax.set_xticks(np.arange(self.n_age))
        ax.set_yticks(np.arange(self.n_age))

        # Set the x-axis labels (horizontal) at the bottom with bold green formatting
        ax.set_xticklabels(self.labels, rotation=90, ha='center',
                           fontsize=20, color='black', usetex=False)

        # Set the y-axis labels (vertical) on the right side with bold green formatting
        ax.set_yticklabels(self.labels, fontsize=20, color='black', usetex=False)

        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

        ax.tick_params(axis='both', which='both', bottom=True, top=False,
                       labelsize=20, width=2, length=8, color='black')

        # Add axis labels and title with enhanced fonts
        ax.set_title(plot_title, fontsize=30, pad=20, fontweight='bold',
                     color='black')

        if annotate:
            for i in range(self.n_age):
                for j in range(i, self.n_age):
                    if not np.isnan(data[i, j]):
                        text_color = 'white' if data[i, j] > (np.nanmax(data) /
                                                              2) else 'black'
                        ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                                color=text_color, fontsize=12, fontweight='bold')

        # Invert y-axis to keep the original matrix orientation
        ax.invert_yaxis()

        # Correct coordinates considering inverted y-axis
        ax.axhline(y=-0.5, color='black', linewidth=5)  # visual bottom border
        ax.axvline(x=self.n_age - 0.5, color='black', linewidth=5)  # right border

        plt.subplots_adjust(right=0.85)
        plt.tight_layout()
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

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

    def plot_r0_small_ngm_grad_mtx(self, matrix: torch.Tensor,
                                   filename: str, folder: str, cmap_type: str,
                                   label_color: str):
        """
        Plot a matrix as a heatmap with linear-scaled color and consistent ticks/labels.
        """
        matrix = matrix.detach().numpy()

        # Handle invalid or non-positive values
        matrix[matrix < 0] = 0  # Optional: treat negatives as zero
        v_min = np.nanmin(matrix)
        v_max = np.nanmax(matrix)

        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

        # Select colormap
        if cmap_type == "CM":
            cmap = self.reversed_blues_cmap
        elif cmap_type == "NGM":
            cmap = LinearSegmentedColormap.from_list(
                "YellowRedGradient", ["#FFFFE0", "#FFD700", "#FF0000"]
            )
        else:
            raise ValueError("Invalid cmap_type. Use 'CM' or 'NGM'.")

        # Plot with linear-scaled colors
        cax = ax.matshow(matrix, cmap=cmap, vmin=v_min, vmax=v_max, aspect='equal')

        # Color bar
        cbar = fig.colorbar(cax, orientation='vertical', shrink=0.63, pad=0.1)
        cbar.ax.tick_params(labelsize=25, colors="black", width=2, length=8)
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(1.0)
        cbar.set_label("Avg. num. contacts", fontsize=22, labelpad=10, color="black",
                       fontweight='normal')

        # Axis labels
        xtick_labels = self.get_tick_labels(
            self.labels, alternate=self.model in ["rostr", "seirr"])
        ytick_labels = self.get_tick_labels(
            self.labels, alternate=self.model in ["rostr", "seirr"])

        ax.set_xticks(np.arange(self.n_age))
        ax.set_yticks(np.arange(self.n_age))

        ax.set_xticklabels(xtick_labels, rotation=90, ha='center',
                           fontsize=20, fontweight='normal')
        ax.set_yticklabels(ytick_labels, fontsize=20, fontweight='normal')

        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(labeltop=False)
        ax.tick_params(axis='both', which='major', length=10, width=2,
                       labelsize=20, color="black")

        ax.invert_yaxis()
        plot_title = "Full contact"

        ax.set_title(plot_title, fontsize=40, fontweight='bold', color="black")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["bottom"].set_color("black")

        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

    def plot_cumulative_sensitivities(self, cum_sensitivities: torch.Tensor,
                                      plot_title: str, filename: str, folder: str,
                                      lower: torch.Tensor = None,
                                      upper: torch.Tensor = None):
        """
        Plot cumulative sensitivities with bars
        """

        os.makedirs(folder, exist_ok=True)

        if isinstance(cum_sensitivities, torch.Tensor):
            cum_sensitivities = cum_sensitivities.detach().cpu().numpy()

        total_sensitivities = cum_sensitivities.sum()
        normalized_sensitivities = cum_sensitivities / total_sensitivities

        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(self.labels))

        # Set up colormap and normalize for coloring (using Reds instead of Greens)
        norm = plt.Normalize(vmin=normalized_sensitivities.min(),
                             vmax=normalized_sensitivities.max())
        cmap = plt.get_cmap("YlOrRd")

        # Assign colors based on normalized elasticities
        colors = [cmap(norm(value)) for value in normalized_sensitivities]

        ax.bar(x_pos, normalized_sensitivities, align='center', alpha=0.9,
               color=colors, edgecolor='black', zorder=3)

        # color = '#5e2b91'
        # bars = ax.bar(x_pos, normalized_sensitivities, align='center', alpha=0.95,
        #               color=color, zorder=3, linewidth=0.8, width=0.85, edgecolor='white')

        if lower is not None and upper is not None:
            if isinstance(lower, torch.Tensor):
                lower = lower.detach().cpu().numpy()
            if isinstance(upper, torch.Tensor):
                upper = upper.detach().cpu().numpy()

            lower = lower / total_sensitivities
            upper = upper / total_sensitivities

            error = [normalized_sensitivities - lower,
                     upper - normalized_sensitivities]

            ax.errorbar(x_pos, normalized_sensitivities, yerr=error,
                        fmt='none', ecolor='red', elinewidth=2.5, capsize=4,
                        capthick=2, zorder=4)

        ylabel_text = {
            "rost": r"Cumulative sensitivities, $\mathcal{S}_j(\mathpzc{m}=\text{R})$",
            "seir": r"Cumulative sensitivities, $\mathcal{S}_j(\mathpzc{m}=\text{P})$"
        }.get(self.model, r"Cumulative sensitivities, "
                          r"$\mathcal{S}_j(\mathpzc{m}=\text{I})$")

        # Clean up axes
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.plot([0, 0], [0, 1], transform=ax.transAxes, color='black',
                linewidth=2, clip_on=False)

        ax.set_xticks(x_pos)
        ax.set_title(plot_title, fontsize=40, fontweight="bold",
                     color='black', pad=20)
        ax.set_xticklabels(self.labels, rotation=90, ha='center',
                           fontsize=20, color='black', usetex=False)

        ax.tick_params(axis='y', which='both',
                       right=False, left=True,
                       labelright=False, labelleft=True,
                       direction='out', length=20, width=3.0,
                       labelsize=25)
        ax.tick_params(axis='y', colors='black')
        ax.tick_params(axis='x', which='both',
                       top=False, bottom=True,
                       labelbottom=True, length=8, width=2.5)

        ax.set_ylabel("")
        ax.grid(False)

        plt.tight_layout()
        save_path = os.path.join(folder, f"{filename}.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
