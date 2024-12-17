import numpy as np

from src.static.model_base import EpidemicModelBase


class WashingtonModel(EpidemicModelBase):
    def __init__(self, model_data) -> None:
        compartments = ["s", "i", "r", "h", "icu", "m", "c"]
        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        iv["i"][3] = 1
        iv.update({"c": iv["r"] + iv["h"] + iv["icu"] + iv["m"]
                   })
        iv.update({"s": self.population - (iv["c"] + iv["i"])})

    def get_model(self, xs: np.ndarray, _, ps: dict, cm: np.ndarray) -> np.ndarray:
        beta = ps["beta"]
        s, i, r, h, icu, m, c = xs.reshape(-1, self.n_age)

        transmission = beta * np.array(i).dot(cm)

        sir_eq_dict = {
            "s": - transmission * s / self.population,  # S'(t)
            "i":  transmission * s / self.population - ps["gamma"] * i,  # I'(t)
            "r": ps["gamma"] * i,  # R'(t)
            "h": ps["gamma"] * h * i,  # H'(t)
            "icu": ps["gamma"] * h * icu * i,  # ICU'(t)
            "m": ps["gamma"] * m * i,  # ICU'(t)
            "c": ps["gamma"] * i  # C'(t)
        }

        return self.get_array_from_dict(comp_dict=sir_eq_dict)

    def get_infected(self, solution) -> np.ndarray:
        idx = self.c_idx["i"]
        return self.aggregate_by_age(solution, idx)

    def get_icu_dynamics(self, solution: np.ndarray) -> np.ndarray:
        idx = self.c_idx["icu"]
        return self.aggregate_by_age(solution, idx)
