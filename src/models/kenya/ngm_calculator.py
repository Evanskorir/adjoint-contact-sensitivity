import torch
from src.models.kenya.v_matrix_calculator import VMatrixCalculator
from src.static.ngm_calculator_base import NGMCalculatorBase


class NGMCalculator(NGMCalculatorBase):
    def __init__(self, param: dict, n_age: int) -> None:
        states = ["e", "a", "m", "h", "icu"]
        self.n_states = len(states)
        super().__init__(param=param, n_age=n_age, states=states)

        self.symmetric_contact_matrix = None
        self.v_matrix = VMatrixCalculator(param=param, n_age=n_age, states=self.states)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = torch.zeros((self.n_age * n_states, self.n_age * n_states))
        inf_a = self.parameters["inf_a"] if "inf_a" in self.parameters.keys() else 0.7
        inf_m = self.parameters["inf_m"] if "inf_m" in self.parameters.keys() else 0.7

        f[i["e"]:s_mtx:n_states, i["a"]:s_mtx:n_states] = inf_a * contact_mtx.T
        f[i["e"]:s_mtx:n_states, i["m"]:s_mtx:n_states] = inf_m * contact_mtx.T

        return f
