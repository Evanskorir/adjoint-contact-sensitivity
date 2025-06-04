import torch

from src.static.model import VMatrixCalculatorBase


class VMatrixCalculator(VMatrixCalculatorBase):
    def __init__(self, param: dict, n_age: int, states) -> None:
        super().__init__(param=param, n_age=n_age, states=states)

        self._get_v()

    def _get_v(self):
        idx = self._idx
        v = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))

        # E -> E (leaving due to incubation)
        v[idx("e"), idx("e")] = self.parameters["sigma"]

        # E -> A (become asymptomatic infectious)
        v[idx("a"), idx("e")] = -self.parameters["q"] * self.parameters["sigma"]

        # E -> M (become mild symptomatic infectious)
        v[idx("m"), idx("e")] = -(1 - self.parameters["q"]) * self.parameters["sigma"]

        # A -> A (leaving due to recovery)
        v[idx("a"), idx("a")] = self.parameters["gamma_a"]

        # M -> M (leaving due to progression/recovery)
        v[idx("m"), idx("m")] = self.parameters["kappa"] + self.parameters["gamma_m"]

        self.v_inv = torch.linalg.inv(v)
