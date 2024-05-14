import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from src.dataloader import DataLoader
from src.simulation_base import SimulationBase
matplotlib.use('Agg')  # Use the Agg backend


class Plotter:
    def __init__(self, data: DataLoader, sim_obj: SimulationBase) -> None:

        self.data = data
        self.sim_obj = sim_obj

    def plot_ngm_grad_matrices(self, saved_file, directory, filename_without_ext,
                               plot_title):
        output_dir = os.path.join(directory, "plots")  # Create a subdirectory for plots
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, filename_without_ext + '.pdf')

        # Define a custom green colormap
        cmap = LinearSegmentedColormap.from_list(name='custom_green',
                                                 colors=['#E1FFE5', '#1B5E20'])

        # Customize heatmap appearance
        plt.figure(figsize=(12, 10))
        sns.heatmap(saved_file, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5,
                    linecolor='black', cbar=True)

        # Customize font and title
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plot_title = '$\\overline{\\mathcal{R}}_0=$' + plot_title
        ax = plt.gca()
        ax.invert_yaxis()
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.title(plot_title, y=1.03, fontsize=20)

        plt.savefig(save_path, format="pdf", bbox_inches='tight')
        plt.close()

    def aggregated_sensitivity_pvalues_plots(self, param_list, directory, prcc_vector,
                                             conf_lower, conf_upper, plot_title,
                                             filename_to_save, calculation_approach):

        output_dir = os.path.join(directory, "agg_plots")
        os.makedirs(output_dir, exist_ok=True)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.margins(0, tight=False)
        xp = range(param_list)
        plt.figure(figsize=(15, 12))
        plt.tick_params(direction="in")

        labels = ["0-4", "5-9", "10-14", "15-19", "20-24",
                      "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                      "55-59", "60-64", "65-69", "70-74", "75+"]

        num_params = len(labels)
        y_pos = np.arange(num_params)

        fig, ax = plt.subplots(figsize=(10, 8))
        # Plotting sensitivity values with different colors based on conditions
        color = ['lightgreen'
                 if abs(prcc) < 0.3 else 'green' if
        0.3 <= abs(prcc) <= 0.5 else '#07553d' for prcc in prcc_vector]

        plt.bar(xp, list(prcc_vector), align='center', width=0.8, alpha=0.8,
                color=color, label="PRCC")

        for pos, y, cl, cu in zip(xp, list(prcc_vector),
                                  list(conf_lower), list(conf_upper)):
            plt.errorbar(x=pos, y=y, yerr=[[cl], [cu]], lw=4, capthick=4, fmt="or",
                         markersize=5, capsize=4, ecolor="r", elinewidth=4)

        # Remove vertical lines
        ax.grid(False)
        ax.set_xticks(y_pos)
        ax.legend([r'$\mathrm{\textbf{P}}$', r'$\mathrm{\textbf{s}}$'])
        plot_title = '$\\overline{\\mathcal{R}}_0=$' + plot_title
        plt.title(plot_title, y=1.03, fontsize=20)
        save_path = os.path.join(output_dir, filename_to_save + '.pdf')
        plt.savefig(save_path, format="pdf", bbox_inches='tight')
        plt.close()

    def plot_aggregation_sensitivity_pvalues(self, prcc_vector, conf_lower,
                                      conf_upper, filename_without_ext, directory,
                                             plot_title, calculation_approach):
        self.aggregated_sensitivity_pvalues_plots(param_list=self.sim_obj.n_ag,
                                           prcc_vector=prcc_vector,
                                           conf_lower=conf_lower, conf_upper=conf_upper,
                                           filename_to_save=filename_without_ext,
                                           directory=directory,
                                           plot_title=plot_title,
                                           calculation_approach=calculation_approach)




