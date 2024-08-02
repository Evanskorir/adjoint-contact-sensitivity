import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatterSciNotation as LogFormatter
from matplotlib.ticker import LogLocator
from matplotlib.tri import Triangulation
matplotlib.use('agg')


class Plotter:
    def __init__(self, n_age: int) -> None:
        self.n_age = n_age

    def plot_small_ngm_mtx(self, ngm_matrix: torch.Tensor,
                           plot_title='NGM with Small Domain'):
        fig, ax = plt.subplots(figsize=(12, 10))
        ngm = ngm_matrix.detach().numpy()  # Detach the tensor before converting to numpy

        # Create a heatmap with the 'Greens' colormap
        cax = ax.matshow(ngm, cmap='Greens', aspect='auto', vmin=ngm.min(),
                         vmax=ngm.max())

        # Add color bar
        cbar = fig.colorbar(cax, orientation='vertical', shrink=1.0, aspect=50)

        # Set ticks and labels dynamically based on self.n_age
        ax.set_xticks(np.arange(self.n_age))
        ax.set_yticks(np.arange(self.n_age))
        ax.set_xticklabels(np.arange(1, self.n_age + 1), fontsize=12)
        ax.set_yticklabels(np.arange(1, self.n_age + 1), fontsize=12)

        ax.xaxis.set_label_position('bottom')  # Move x-axis labels to the bottom
        ax.yaxis.set_label_position('left')  # Ensure y-axis labels are on the left
        ax.xaxis.tick_bottom()  # Ensure x-axis ticks are on the bottom
        ax.yaxis.tick_left()
        ax.set_xlabel("Age Infected", fontsize=16)
        ax.set_ylabel("Age Susceptible", fontsize=16)
        ax.set_xlabel("Age", fontsize=16)
        ax.set_ylabel("Age", fontsize=16)
        ax.set_title(plot_title, fontsize=20, pad=20)
        # Annotate heatmap cells with the values
        for i in range(self.n_age):
            for j in range(self.n_age):
                text = ax.text(j, i, f'{ngm[i, j]:.2f}', ha='center', va='center',
                               color='black')
        ax.invert_yaxis()
        plt.tight_layout()

        # Save the figure
        os.makedirs("sens_data", exist_ok=True)
        save_path = os.path.join("sens_data", "ngm_heatmap.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

    def construct_triangle_grids_grads_p_value(self):
        """
        construct_triangle_grids_prcc_p_value(grads_vector, p_values)
        :return: A list containing triangulation objects for PRCC and p-values.
        """
        # vertices of the little squares
        xv, yv = torch.meshgrid(torch.arange(-0.5, self.n_age),
                                torch.arange(-0.5, self.n_age))
        # centers of the little square
        xc, yc = torch.meshgrid(torch.arange(0, self.n_age),
                                torch.arange(0, self.n_age))
        x = torch.cat([xv.ravel(), xc.ravel()])
        y = torch.cat([yv.ravel(), yc.ravel()])
        triangles_prcc = [(i + j * (self.n_age + 1), i + 1 + j * (self.n_age + 1),
                           i + (j + 1) * (self.n_age + 1))
                          for j in range(self.n_age) for i in range(self.n_age)]
        triangles_p = [(i + 1 + j * (self.n_age + 1), i + 1 + (j + 1) *
                        (self.n_age + 1),
                        i + (j + 1) * (self.n_age + 1))
                       for j in range(self.n_age) for i in range(self.n_age)]
        triang = [Triangulation(x, y, triangles, mask=None)
                  for triangles in [triangles_prcc, triangles_p]]
        return triang

    @staticmethod
    def get_mask_and_values(grad_mtx, p_val_mtx):
        # Convert vectors to tensors
        grads_mtx = torch.tensor(grad_mtx)
        p_values_mtx = torch.tensor(p_val_mtx)

        # Masking the lower triangular part
        mask_grads = torch.tril(grads_mtx)
        mask_p_values = torch.tril(p_values_mtx)

        # Optionally, replace upper triangular part with NaN
        mask_grads[mask_grads == 0] = float('nan')
        mask_p_values[mask_p_values == 0] = float('nan')

        return mask_grads, mask_p_values

    @staticmethod
    def adjust_colormap(cmap_name):
        if cmap_name == "Greens":
            # Get the colors from the "Greens" colormap for values greater than or equal to 0
            cmap_greens = plt.cm.get_cmap("Greens")
            colors_greens = cmap_greens(torch.linspace(0, 1, 128))

            # Get the colors from the "viridis" colormap for values less than 0
            cmap_viridis = plt.cm.get_cmap("Greys")
            colors_viridis = cmap_viridis(torch.linspace(0, 1, 128))

            # Combine the two colormaps
            color = torch.vstack((colors_viridis, colors_greens))

            # Create a new colormap
            cmap = colors.LinearSegmentedColormap.from_list(cmap_name, color)
            return cmap
        else:
            return plt.get_cmap(cmap_name)

    def plot_grads_p_values_as_heatmap(self, grads_vector,
                                      p_values, plot_title):
        """
        Prepares for plotting PRCC and p-values as a heatmap.
       :param grads_vector: (torch.Tensor): The grads vector.
       :param p_values: (torch.Tensor): The p-values vector.
       :param plot_title: The title of the plot.
       :return: None
       """
        os.makedirs("sens_data", exist_ok=True)
        save_path = os.path.join("sens_data", "grads_plot.pdf")

        p_value_cmap = colors.ListedColormap(['Orange', 'red', 'darkred'])
        cmaps = ["Greens", p_value_cmap]
        log_norm = colors.LogNorm(vmin=1e-3, vmax=1e0)  # used for p_values
        norm = plt.Normalize(vmin=0, vmax=140)  # used for PRCC_values
        fig, ax = plt.subplots()
        triang = self.construct_triangle_grids_grads_p_value()
        grads_mtx_masked, p_values_mtx_masked = self.get_mask_and_values(
            grad_mtx=grads_vector, p_val_mtx=p_values)

        images = [ax.tripcolor(t, val.detach().numpy().ravel(), cmap=cmap, ec="white")
                  for t, val, cmap in zip(triang, [grads_mtx_masked, p_values_mtx_masked], cmaps)]

        cbar = fig.colorbar(images[0], ax=ax, shrink=0.7, aspect=20 * 0.7, pad=0.1)
        cbar_pval = fig.colorbar(images[1], ax=ax, shrink=0.7, aspect=20 * 0.7, pad=0.1)

        images[1].set_norm(norm=log_norm)
        images[0].set_norm(norm=norm)

        locator = LogLocator()
        formatter = LogFormatter()
        cbar_pval.locator = locator
        cbar_pval.formatter = formatter
        cbar_pval.update_normal(images[1])
        cbar.update_normal(images[0])

        ax.set_xticks(range(self.n_age))
        ax.set_yticks(range(self.n_age))
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set(frame_on=False)
        plt.gca().grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.margins(x=0, y=0)
        ax.set_aspect('equal', 'box')  # square cells
        plt.title(plot_title, y=1.03, fontsize=20)
        plt.tight_layout()
        plt.savefig(save_path, format="pdf", bbox_inches='tight')
        plt.close()

    @staticmethod
    def aggregated_grads_pvalues_plots(grads_vector, std_values,
                                      conf_lower, conf_upper, plot_title,
                                       calculation_approach):
        """
        Prepares for plotting aggregated PRCC and standard values as error bars.
        :param grads_vector: (numpy.ndarray): The aggregated gradients vector.
        :param std_values: (numpy.ndarray): standard deviation of aggregated grads vector.
        :param conf_lower: (numpy.ndarray): lower quartiles of aggregated grads vector.
        :param conf_upper: (numpy.ndarray): upper quartiles of aggregated grads vector.
        :param plot_title: The title of the plots.
        :param calculation_approach: Calculation method for the aggregated prcc:
        mean or median
        :return: None
        """
        os.makedirs("sens_data/agg_plot", exist_ok=True)
        save_path = os.path.join("sens_data", "agg_plot.pdf")

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        plt.figure(figsize=(15, 12))
        plt.tick_params(direction="in")

        labels = ["0-4", "5-9", "10-14", "15-19", "20-24",
                  "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                  "55-59", "60-64", "65-69", "70-74", "75+"]

        num_params = len(labels)
        y_pos = np.arange(num_params)
        # Convert tensors to NumPy arrays and handle shapes
        grads_vector_np = np.array([tensor.detach().numpy() for
                                    tensor in grads_vector]).flatten()[:num_params]
        conf_lower_np = np.array([tensor.detach().numpy() for
                                  tensor in conf_lower]).flatten()[:num_params]
        conf_upper_np = np.array([tensor.detach().numpy() for
                                  tensor in conf_upper]).flatten()[:num_params]
        if std_values is None:
            std_values_np = np.array([tensor.detach().numpy() for tensor in
                                      std_values]).flatten()[:num_params]

        fig, ax = plt.subplots(figsize=(10, 8))

        color = ['lightgreen' if abs(grads) < 25 else 'green' if
        25 <= abs(grads) <= 50 else '#07553d' for grads in
                 grads_vector_np]

        plt.bar(y_pos, grads_vector_np, align='center', width=0.8, alpha=0.8,
                color=color, label="grads")

        if calculation_approach == "median":
            for pos, y, cl, cu in zip(y_pos, grads_vector_np, conf_lower_np, conf_upper_np):
                yerr_lower = max(0, y - cl)
                yerr_upper = max(0, cu - y)
                plt.errorbar(x=pos, y=y, yerr=[[yerr_lower], [yerr_upper]],
                             lw=2, capthick=2, fmt="o", markersize=5,
                             capsize=4, ecolor="r", elinewidth=2)
        else:
            std_values_np = np.array([tensor.detach().numpy() for tensor in
                                      std_values]).flatten()[:num_params]
            for pos, y, err in zip(y_pos, grads_vector_np, std_values_np):
                yerr = max(0, err)
                plt.errorbar(pos, y, yerr=yerr, lw=2, capthick=2, fmt="o",
                             markersize=5, capsize=4, ecolor="r",
                             elinewidth=2)

        ax.grid(False)
        ax.set_xticks(y_pos)
        ax.set_xticklabels(labels, rotation=90, ha='center')
        ax.legend([r'$\mathrm{\textbf{P}}$', r'$\mathcal{CI}$'])

        plt.title(plot_title, y=1.03, fontsize=20)
        plt.savefig(save_path, format="pdf", bbox_inches='tight')
        plt.close()

    def plot_aggregation_grads_pvalues(self, grads_vector, std_values, conf_lower,
                                      conf_upper, calculation_approach):
        """
        Generates actual aggregated PRCC plots with std values as error bars.
        :param grads_vector: (numpy.ndarray): The gradients vector.
        :param std_values: (numpy.ndarray): standard deviation of aggregated grads vector.
        :param conf_lower: (numpy.ndarray): lower quartiles of aggregated grads vector.
        :param conf_upper: (numpy.ndarray): upper quartiles of aggregated grads vector.
        :param calculation_approach: Calculation method for the aggregated prcc;
        mean or median
        :return: bar plots with error bars
        """
        self.aggregated_grads_pvalues_plots(
                                           grads_vector=grads_vector,
                                           std_values=std_values,
                                           conf_lower=conf_lower, conf_upper=conf_upper,
                                           plot_title=None,
                                           calculation_approach=calculation_approach)
        