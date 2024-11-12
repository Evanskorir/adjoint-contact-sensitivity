import numpy as np
import torch
from src.static.model_base import EpidemicModelBase


class KenyaModel(EpidemicModelBase):
    def __init__(self, model_data) -> None:
        compartments = ["s", "e", "a",
                        "m", "h", "icu", "c"]
        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        iv["e"][2] = 1
        iv.update({"c": iv["e"] + iv["a"] + iv["m"] + iv["h"] + iv["icu"]
                   })

        iv.update({"s": self.population - (iv["c"])})

    def get_model(self, xs: np.ndarray, _, ps: dict, cm: np.ndarray) -> np.ndarray:
        # the same order as in self.compartments!
        s, e, a, m, h, icu, c = xs.reshape(-1, self.n_age)

        ps = {key: val.numpy() if isinstance(val, torch.Tensor) else
        val for key, val in ps.items()}
        cm = cm.numpy() if isinstance(cm, torch.Tensor) else cm

        actual_population = np.array(self.population)
        susc = np.array(ps["susc"])

        transmission_1 = ps["beta"] * np.array(a).dot(cm) / actual_population
        transmission_2 = ps["beta"] * (1 - ps["alpha"]) * np.array(a).dot(cm) / \
                         actual_population

        model_eq_dict = {
            "s": -susc * (transmission_1 + transmission_2) * s,  # S'(t)
            "e": susc * (transmission_1 + transmission_2) * s - ps["omega"] * e,  # E'(t)
            "a": ps["theta_i"] * ps["omega"] * e - ps["gamma_a"] * a,  # A'(t)
            "m": (1 - ps["theta_i"]) * ps["omega"] * e - (ps["kappa_i"] + ps["gamma_m"]) * m,  # M'(t)
            "h": ps["kappa_i"] * m + ps["phi_i"] * c - (ps["zeta_i"] + ps["gamma_h"]) * h,  # H'(t)
            "icu": ps["zeta_i"] * h - ps["phi_i"] * c - ps["lambda_i"] * c,  # ICU'(t)
            "r": ps["gamma_a"] * a + ps["gamma_m"] * m + ps["gamma_h"] * h,  # R'(t)
            "d": 2 * ps["lambda_i"] * c,  # D'(t)
            # add compartments for collecting total infected values
            "c": ps["omega"] * e  # C'(t)
        }
        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_hospitalized(self, solution: np.ndarray) -> np.ndarray:
        idx = self.c_idx["h"]
        idx_2 = self.c_idx["icu"]
        return self.aggregate_by_age(solution, idx) + self.aggregate_by_age(solution, idx_2)

    def get_recovered(self, solution: np.ndarray) -> np.ndarray:
        idx = self.c_idx["r"]
        return self.aggregate_by_age(solution, idx)
