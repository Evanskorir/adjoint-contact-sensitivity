import torch

from src.models.kenya.v_matrix_calculator import VMatrixCalculator
from src.static.model import NGMCalculatorBase


class NGMCalculator(NGMCalculatorBase):
    def __init__(self, param: dict, n_age: int) -> None:
        states = ["e", "a", "m"]
        self.n_states = len(states)
        super().__init__(param=param, n_age=n_age, states=states)

        self.symmetric_contact_matrix = None
        self.v_matrix = VMatrixCalculator(param=param, n_age=n_age, states=self.states)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = torch.zeros((self.n_age * n_states, self.n_age * n_states))

        susc_vec = self.parameters["susc"].reshape((-1, 1))

        # Infections from asymptomatic A
        f[i["e"]:s_mtx:n_states, i["a"]:s_mtx:n_states] = \
            self.parameters["beta_a"] * contact_mtx.T * susc_vec
        # Infections from mild symptomatic M
        f[i["e"]:s_mtx:n_states, i["m"]:s_mtx:n_states] = \
            self.parameters["beta_m"] * contact_mtx.T * susc_vec

        return f
