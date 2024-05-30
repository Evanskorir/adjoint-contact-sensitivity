from src.dataloader import DataLoader
from src.simulation_base import SimulationBase
from src.eigen_calculator import EigenCalculator


def main():
    data = DataLoader()
    sim = SimulationBase(data=data)
    ngm_small_tensor, ngm_small_grads = sim.r0.get_eig_val(contact_mtx=sim.contact_matrix,
                                                       population=sim.population,
                                                       susceptibles=sim.population)
    eig_calculator = EigenCalculator(ngm_small_tensor=ngm_small_tensor)


if __name__ == '__main__':
    main()
