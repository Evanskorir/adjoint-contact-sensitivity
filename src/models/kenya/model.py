import numpy as np
import torch

from src.static.model_base import EpidemicModelBase


class KenyaModel(EpidemicModelBase):
    def __init__(self, model_data, model=None) -> None:

        # Define the compartments of the model
        compartments = ["s", "e", "d", "a", "q", "r", "c"]
        super().__init__(model_data=model_data, compartments=compartments, model=model)

    def update_initial_values(self, iv: dict):

        iv["e"][4] = 1
        iv.update({"c": iv["a"] + iv["d"] + iv["q"] + iv["r"]
                   })
        iv.update({"s": self.population - (iv["c"] + iv["e"])})

    @staticmethod
    def compute_tau(cumulative_cases, capacity, tau0):
        """
        Compute the time-dependent isolation rate tau(t).
        """
        if cumulative_cases < capacity:
            return tau0  # Active isolation
        else:
            return 0.0  # No active isolation after capacity is reached

    def get_model(self, xs: np.ndarray, _, ps: dict, cm: np.ndarray) -> np.ndarray:
        """
        Main function for computing the epidemic model based on the given
        compartments and parameters.
        """

        s, e, d, a, q, r, c = xs.reshape(-1, self.n_age)
        ps = {key: val.numpy() if isinstance(val, torch.Tensor) else
        val for key, val in ps.items()}
        cm = cm.numpy() if isinstance(cm, torch.Tensor) else cm
        actual_population = np.array(self.population)
        force_of_infection = ps["beta"] * np.array((ps["epsilon_d"] * d +
                                                    ps["epsilon_a"] * a) /
                                                   actual_population).dot(cm)

        # Compute time-dependent tau
        cumulative_cases = ps["sigma"] * e
        tau = self.compute_tau(
            cumulative_cases=cumulative_cases.sum(),  # Sum across all age groups
            capacity=ps.get("capacity", 1000),  # Default capacity
            tau0=ps["tau0"]
        )

        model_eq_dict = {
            "s": -s * force_of_infection,  # Susceptible
            "e": s * force_of_infection - ps["sigma"] * e,  # Exposed
            "d": ps["delta"] * ps["sigma"] * e - ps["gamma"] * d - tau * d,  # Symptomatic (detected)
            "a": (1 - ps["delta"]) * ps["sigma"] * e - ps["gamma"] * a,  # Asymptomatic
            "q": tau * d - ps["psi"] * q,  # Isolated
            "r": ps["gamma"] * (a + d) + ps["psi"] * q,  # Recovered
            "c": cumulative_cases  # Cumulative cases
        }

        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_hospitalized(self, solution: np.ndarray) -> np.ndarray:
        idx_d = self.c_idx["d"]
        idx_q = self.c_idx["q"]
        return self.aggregate_by_age(solution, idx_d) + \
               self.aggregate_by_age(solution, idx_q)

    def get_recovered(self, solution: np.ndarray) -> np.ndarray:
        idx_r = self.c_idx["r"]
        return self.aggregate_by_age(solution, idx_r)
