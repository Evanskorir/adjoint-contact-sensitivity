import torch

from src.static.model import VMatrixCalculatorBase


class VMatrixCalculator(VMatrixCalculatorBase):
    def __init__(self, param: dict, n_age: int, states) -> None:
        super().__init__(param=param, n_age=n_age, states=states)
        self._get_v()

    def _get_v(self):
        idx = self._idx
        v = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))

        # E -> E (exposed to exposed)
        v[idx("e"), idx("e")] = self.parameters["sigma"]

        # E -> A (exposed to asymptomatic)
        v[idx("a"), idx("e")] = -(1 - self.parameters["delta"]) * self.parameters["sigma"]

        # E -> D (exposed to symptomatic)
        v[idx("d"), idx("e")] = -self.parameters["delta"] * self.parameters["sigma"]

        # A -> A (asymptomatic to asymptomatic)
        v[idx("a"), idx("a")] = self.parameters["gamma"]

        # D -> D (symptomatic to symptomatic)
        v[idx("d"), idx("d")] = self.parameters["gamma"]

        self.v_inv = torch.linalg.inv(v)
