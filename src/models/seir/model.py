import numpy as np

from src.static.model_base import EpidemicModelBase


class SeirUK(EpidemicModelBase):
    def __init__(self, model_data) -> None:
        """
        SEIR model for the UK with compartments for cumulative cases.
        """
        compartments = ["s", "e", "i", "r", "c"]
        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        """
        Initialize compartment values for the model.
        """
        iv["e"][2] = 1
        iv.update({"c": iv["i"] + iv["r"]
                   })
        iv.update({"s": self.population - (iv["c"] + iv["e"])})

    def get_model(self, xs: np.ndarray, t: float, ps: dict, cm: np.ndarray) -> np.ndarray:
        """
        Define the differential equations governing the SEIR model.
        """
        # Extract compartments
        s, e, i, r, c = xs.reshape(-1, self.n_age)
        transmission = ps["beta"] * np.array(i).dot(cm) / self.population

        model_eq_dict = {
            "s": -transmission * s,  # Susceptible
            "e": transmission * s - ps["gamma"] * e,  # Exposed
            "i": ps["gamma"] * e - ps["rho"] * i,  # Infected
            "r": ps["rho"] * i,  # Recovered
            "c": ps["gamma"] * e  # Cumulative cases (transition from E to I)
        }

        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_infected(self, solution: np.ndarray) -> np.ndarray:
        """
        Return the total number of currently infected individuals.
        """
        idx = self.c_idx["i"]
        return self.aggregate_by_age(solution, idx)

    def get_recovered(self, solution: np.ndarray) -> np.ndarray:
        """
        Return the total number of recovered individuals.
        """
        idx = self.c_idx["r"]
        return self.aggregate_by_age(solution, idx)
