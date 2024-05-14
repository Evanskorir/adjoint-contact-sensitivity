import torch
from src.dataloader import DataLoader

from src.simulation import SimulationNPI


def main():
    data = DataLoader()
    sim = SimulationNPI(data)

    sim.calculate_r0()
    sim.calculate_aggregated_p_values(calculation_approach="mean")
    sim.load_plot_ngm_gradients(calculation_approach="mean")


if __name__ == '__main__':
    main()
