import torch
from abc import ABC, abstractmethod
from scipy.integrate import odeint


class EpidemicModelBase(ABC):
    def __init__(self, model_data, compartments: list) -> None:
        self.population = torch.tensor(model_data.age_data.flatten(),
                                       dtype=torch.float32)
        self.compartments = compartments
        self.c_idx = {comp: idx for idx, comp in enumerate(self.compartments)}
        self.n_age = self.population.shape[0]

    def initialize(self):
        iv = {key: torch.zeros(self.n_age, dtype=torch.float32) for key in self.compartments}
        return iv

    def aggregate_by_age(self, solution, idx) -> torch.Tensor:
        return torch.sum(solution[:, idx * self.n_age:(idx + 1) * self.n_age], dim=1)

    def get_cumulative(self, solution) -> torch.Tensor:
        idx = self.c_idx["c"]
        return self.aggregate_by_age(solution, idx)

    def get_deaths(self, solution) -> torch.Tensor:
        idx = self.c_idx["d"]
        return self.aggregate_by_age(solution, idx)

    def get_solution(self, t: torch.Tensor, parameters: dict, cm: torch.Tensor):
        initial_values = self.get_initial_values()
        return torch.tensor(odeint(self.get_model, initial_values, t,
                                   args=(parameters, cm)), dtype=torch.float32)

    def get_array_from_dict(self, comp_dict) -> torch.Tensor:
        return torch.stack([comp_dict[comp] for comp in self.compartments])

    def get_initial_values(self) -> torch.Tensor:
        iv = self.initialize()
        self.update_initial_values(iv=iv)
        return self.get_array_from_dict(comp_dict=iv)

    @abstractmethod
    def update_initial_values(self, iv: dict):
        pass

    @abstractmethod
    def get_model(self, xs, ts, ps, cm):
        pass
