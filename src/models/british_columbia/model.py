import numpy as np
import torch

from src.static.model_base import EpidemicModelBase


class BCModel(EpidemicModelBase):
    def __init__(self, model_data) -> None:
        compartments = ["s", "e1", "e2", "i1", "i2", "r", "c"]

        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        iv["e2"][3] = 1
        iv.update({"c": iv["e1"] + iv["i1"] + iv["i2"] + iv["r"]
                   })

        iv.update({"s": self.population - (iv["c"] + iv["e2"])})

    def get_model(self, xs: np.ndarray, t, ps: dict, cm: np.ndarray) -> np.ndarray:
        # the same order as in self.compartments!
        s, e1, e2, i1, i2, r, c = xs.reshape(-1, self.n_age)
        ps = {key: val.numpy() if isinstance(val, torch.Tensor) else
        val for key, val in ps.items()}
        cm = cm.numpy() if isinstance(cm, torch.Tensor) else cm

        transmission = ps["beta"] * np.array(e2 + i1 + i2).dot(cm)
        actual_population = np.array(self.population)

        susc = np.array(ps["susc"])
        model_eq_dict = {
            "s": -susc * transmission * s / actual_population,  # S'(t)
            "e1": susc * s / actual_population * transmission - e1 * ps["H_1"],  # E1'(t)
            "e2": e1 * ps["H_1"] - e2 * ps["H_2"],  # E2'(t)
            "i1": e2 * ps["H_2"] - 2 * ps["gamma"] * i1,  # I1'(t)
            "i2": 2 * ps["gamma"] * i1 - 2 * ps["gamma"] * i2,  # I2'(t)
            "r": 2 * ps["gamma"] * i2,  # R'(t)

            # add compartment to store total infecteds
            "c": e2 * ps["H_2"]  # C'(t)
        }
        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_recovered(self, solution: np.ndarray) -> np.ndarray:
        """
        Return the total number of recovered individuals.
        """
        idx = self.c_idx["r"]
        return self.aggregate_by_age(solution, idx)
