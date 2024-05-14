import torch
from src.model_base import EpidemicModelBase


class RostModelHungary(EpidemicModelBase):
    def __init__(self, model_data) -> None:
        compartments = ["s", "l1", "l2", "ip", "ia1", "ia2", "ia3", "is1",
                        "is2", "is3", "ih", "ic", "icr", "r", "d", "c"]
        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv: dict):
        iv["l1"][2] = 1
        c = iv["ip"] + iv["ia1"] + iv["ia2"] + iv["ia3"] + iv["is1"] + iv["is2"] + \
            iv["is3"] + iv["r"] + iv["d"]
        iv.update({"c": c})
        iv.update({"s": self.population - (c + iv["l1"] + iv["l2"])})

    def get_model(self, xs: torch.Tensor, _, ps: dict, cm: torch.Tensor) -> torch.Tensor:
        s, l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, ih, ic, icr, r, \
        d, c = xs.view(-1, self.n_age)

        transmission = ps["beta"] * torch.dot(torch.tensor([ip + ps["inf_a"] *
                                                            (ia1 + ia2 + ia3) +
                                                            (is1 + is2 + is3)]), cm)
        actual_population = self.population

        model_eq_dict = {
            "s": -ps["susc"] * (s / actual_population) * transmission,
            "l1": ps["susc"] * (s / actual_population) * transmission - 2 *
                  ps["alpha_l"] * l1,
            "l2": 2 * ps["alpha_l"] * l1 - 2 * ps["alpha_l"] * l2,
            "ip": 2 * ps["alpha_l"] * l2 - ps["alpha_p"] * ip,
            "ia1": ps["p"] * ps["alpha_p"] * ip - 3 * ps["gamma_a"] * ia1,
            "ia2": 3 * ps["gamma_a"] * ia1 - 3 * ps["gamma_a"] * ia2,
            "ia3": 3 * ps["gamma_a"] * ia2 - 3 * ps["gamma_a"] * ia3,
            "is1": (1 - ps["p"]) * ps["alpha_p"] * ip - 3 * ps["gamma_s"] * is1,
            "is2": 3 * ps["gamma_s"] * is1 - 3 * ps["gamma_s"] * is2,
            "is3": 3 * ps["gamma_s"] * is2 - 3 * ps["gamma_s"] * is3,
            "ih": ps["h"] * (1 - ps["xi"]) * 3 * ps["gamma_s"] * is3 - ps["gamma_h"] * ih,
            "ic": ps["h"] * ps["xi"] * 3 * ps["gamma_s"] * is3 - ps["gamma_c"] * ic,
            "icr": (1 - ps["mu"]) * ps["gamma_c"] * ic - ps["gamma_cr"] * icr,
            "r": 3 * ps["gamma_a"] * ia3 + (1 - ps["h"]) * 3 * ps["gamma_s"] * is3 +
                 ps["gamma_h"] * ih +
                 ps["gamma_cr"] * icr,
            "d": ps["mu"] * ps["gamma_c"] * ic,
            "c": 2 * ps["alpha_l"] * l2
        }
        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_hospitalized(self, solution: torch.Tensor) -> torch.Tensor:
        idx = self.c_idx["ih"]
        idx_2 = self.c_idx["icr"]
        return self.aggregate_by_age(solution, idx) + self.aggregate_by_age(solution, idx_2)

    def get_ventilated(self, solution: torch.Tensor) -> torch.Tensor:
        idx = self.c_idx["ic"]
        return self.aggregate_by_age(solution, idx)
