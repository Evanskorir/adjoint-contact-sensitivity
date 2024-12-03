import numpy as np
import torch
from src.static.model_base import EpidemicModelBase


class KenyaModel(EpidemicModelBase):
    def __init__(self, aggregated_data, model=None) -> None:
        """
        Initialize the Kenya model with aggregated data.
        Args: aggregated_data (KenyaDataAggregator): Aggregated data for Kenya,
            including contact matrices and age data.
        """
        # Define the compartments of the model
        compartments = ["s", "e", "a", "m", "h", "icu", "d", "r", "c"]
        super().__init__(model_data=aggregated_data, compartments=compartments, model=model)

    def update_initial_values(self, iv: dict):
        iv["e"][2] = 1
        iv.update({"c": iv["a"] + iv["m"] + iv["h"] + iv["icu"] + iv["d"] + iv["r"]
                   })

        iv.update({"s": self.population - (iv["c"] + iv["e"])})

    def get_model(self, xs: np.ndarray, _, ps: dict, cm: np.ndarray) -> np.ndarray:
        """
        Main function for computing the epidemic model based on the given
        compartments and parameters.
        """

        # Extract compartments
        s, e, a, m, h, icu, d, c, r = xs.reshape(-1, self.n_age)
        ps = {key: val.numpy() if isinstance(val, torch.Tensor) else val for key,
                                                                             val in ps.items()}
        cm = cm.numpy() if isinstance(cm, torch.Tensor) else cm
        actual_population = np.array(self.population)
        transmission = ps["beta"] * np.array(a).dot(cm) / actual_population

        model_eq_dict = {
            "s": -transmission * s,  # S
            "e": transmission * s - ps["omega"] * e,  # E
            "a": ps["theta_i"] * ps["omega"] * e - ps["gamma_a"] * a,  # A
            "m": (1 - ps["theta_i"]) * ps["omega"] * e - (ps["kappa_i"] +
                                                          ps["gamma_m"]) * m,  # M
            "h": ps["kappa_i"] * m - (ps["zeta_i"] + ps["gamma_h"]) * h,  # H (non-ICU)
            "icu": ps["zeta_i"] * h - ps["lambda_i"] * icu,  # ICU
            "d": ps["lambda_i"] * icu,  # Deaths
            "r": ps["gamma_a"] * a + ps["gamma_m"] * m + ps["gamma_h"] * h,  # Recovered
            "c": ps["omega"] * e  # Cumulative cases

        }

        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_hospitalized(self, solution: np.ndarray) -> np.ndarray:
        """
        Return the total number of hospitalized individuals by aggregating over all age groups.
        """
        idx_h = self.c_idx["h"]
        idx_icu = self.c_idx["icu"]
        return self.aggregate_by_age(solution, idx_h) + \
               self.aggregate_by_age(solution, idx_icu)

    def get_recovered(self, solution: np.ndarray) -> np.ndarray:
        """
        Return the total number of recovered individuals by aggregating over all age groups.
        """
        idx_r = self.c_idx["r"]
        return self.aggregate_by_age(solution, idx_r)

    def get_cumulative_hospitalizations(self, solution: np.ndarray) -> np.ndarray:
        """
        Return the total cumulative hospitalizations by aggregating over all age groups.
        """
        idx_hosp = self.c_idx["hosp"]
        return self.aggregate_by_age(solution, idx_hosp)
