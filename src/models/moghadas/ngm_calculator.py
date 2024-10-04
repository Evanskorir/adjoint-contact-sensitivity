import torch

from src.static.e_matrix_calculator import EMatrixCalculator
from src.models.moghadas.v_matrix_calculator import VMatrixCalculator


class NGMCalculator:
    def __init__(self, param: dict, n_age: int) -> None:
        self.ngm_small_tensor = None
        self.symmetric_contact_matrix = None
        states = ["e", "i_n", "q_n", "i_h", "q_h", "a_n", "a_q"]
        self.states = states
        self.n_age = n_age
        self.parameters = param
        self.n_states = len(self.states)
        self.i = {self.states[index]: index for index in range(0, self.n_states)}
        self.s_mtx = self.n_age * self.n_states

        self.e_matrix = EMatrixCalculator(n_states=self.n_states, n_age=n_age)
        self.v_matrix = VMatrixCalculator(param=param, n_states=self.n_states,
                                          n_age=n_age, states=self.states)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))
        susc_vec = self.parameters["susc"].reshape((-1, 1))
        inf_a = self.parameters["k"] if "k" in self.parameters.keys() else 1.0
        inf_s = self.parameters["inf_s"] if "inf_s" in self.parameters.keys() else 1.0

        f[i["e"]:s_mtx:n_states, i["a_n"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["a_q"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["i_n"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["i_h"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["q_n"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["e"]:s_mtx:n_states, i["q_h"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec

        return f

    def run(self, symmetric_contact_mtx: torch.Tensor):
        # Calculate large domain NGM
        f = self._get_f(contact_mtx=symmetric_contact_mtx)
        ngm_large = torch.matmul(f, self.v_matrix.v_inv)

        # Calculate small domain NGM
        self.ngm_small_tensor = torch.matmul(
            torch.matmul(self.e_matrix.e, ngm_large),
            self.e_matrix.e.T
        )
