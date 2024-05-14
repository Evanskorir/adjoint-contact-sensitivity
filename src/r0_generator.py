import torch
from scipy.linalg import block_diag
import torch.nn as nn


class R0Generator(nn.Module):
    def __init__(self, param: dict, n_age: int) -> None:
        super(R0Generator, self).__init__()
        self.contact_matrix = None
        states = ["l1", "l2", "ip", "a1", "a2", "a3", "i1", "i2", "i3"]
        self.states = states
        self.n_age = n_age
        self.parameters = param
        self.n_states = len(self.states)
        self.i = {self.states[index]: index for index in range(0, self.n_states)}
        self.s_mtx = self.n_age * self.n_states

        self.n_l = 2
        self.n_a = 3
        self.n_i = 3

        # Define your linear layer
        self.linear_layer = nn.Linear(16, 16)

        self._get_e()
        self._get_v()

    def forward(self, ngm_small_tensor):
        # Apply linear transformation
        transformed_tensor = self.linear_layer(ngm_small_tensor)
        return transformed_tensor

    def _idx(self, state: str) -> torch.Tensor:
        return torch.arange(self.n_age * self.n_states) % self.n_states == self.i[state]

    def _get_v(self) -> torch.Tensor:
        idx = self._idx
        v = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))
        # L1 -> L2
        v[idx("l1"), idx("l1")] = self.n_l * self.parameters["alpha_l"]
        v[idx("l2"), idx("l1")] = -self.n_l * self.parameters["alpha_l"]
        # L2 -> Ip
        v[idx("l2"), idx("l2")] = self.n_l * self.parameters["alpha_l"]
        v[idx("ip"), idx("l2")] = -self.n_l * self.parameters["alpha_l"]
        # ip -> I1/A1
        v[idx("ip"), idx("ip")] = self.parameters["alpha_p"]
        v[idx("i1"), idx("ip")] = -self.parameters["alpha_p"] * (1 - self.parameters["p"])
        v[idx("a1"), idx("ip")] = -self.parameters["alpha_p"] * self.parameters["p"]
        # A1 -> A2
        v[idx("a1"), idx("a1")] = self.n_a * self.parameters["gamma_a"]
        v[idx("a2"), idx("a1")] = -self.n_a * self.parameters["gamma_a"]
        # A2 -> A3
        v[idx("a2"), idx("a2")] = self.n_a * self.parameters["gamma_a"]
        v[idx("a3"), idx("a2")] = -self.n_a * self.parameters["gamma_a"]
        # A3 -> R
        v[idx("a3"), idx("a3")] = self.n_a * self.parameters["gamma_a"]
        # I1 -> I2
        v[idx("i1"), idx("i1")] = self.n_i * self.parameters["gamma_s"]
        v[idx("i2"), idx("i1")] = -self.n_i * self.parameters["gamma_s"]
        # I2 -> I3
        v[idx("i2"), idx("i2")] = self.n_i * self.parameters["gamma_s"]
        v[idx("i3"), idx("i2")] = -self.n_i * self.parameters["gamma_s"]
        # I3 -> Ih/Ic (& R)
        v[idx("i3"), idx("i3")] = self.n_i * self.parameters["gamma_s"]

        self.v_inv = torch.linalg.inv(v)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = torch.zeros((self.n_age * n_states, self.n_age * n_states))
        inf_a = self.parameters["inf_a"] if "inf_a" in self.parameters.keys() else 1.0
        inf_s = self.parameters["inf_s"] if "inf_s" in self.parameters.keys() else 1.0
        inf_p = self.parameters["inf_p"] if "inf_p" in self.parameters.keys() else 1.0

        susc_vec = self.parameters["susc"].reshape((-1, 1))
        f[i["l1"]:s_mtx:n_states, i["ip"]:s_mtx:n_states] = inf_p * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["a1"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["a2"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["a3"]:s_mtx:n_states] = inf_a * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["i1"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["i2"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec
        f[i["l1"]:s_mtx:n_states, i["i3"]:s_mtx:n_states] = inf_s * contact_mtx.T * susc_vec

        return f

    def _get_e(self):
        block = torch.zeros(self.n_states)
        block[0] = 1
        self.e = block
        for _ in range(1, self.n_age):
            self.e = block_diag(self.e, block)
            self.e = torch.tensor(self.e, dtype=torch.float32)

    def get_eig_val(self, susceptibles: torch.Tensor, population: torch.Tensor,
                 contact_mtx: torch.Tensor = None):
        if contact_mtx is not None:
            self.contact_matrix = contact_mtx

        # Calculate contact matrix
        contact_matrix = self.contact_matrix / population.view(-1, 1)
        cm_tensor = contact_matrix.repeat(susceptibles.shape[0], 1, 1)
        # Calculate susceptible tensor
        susc_tensor = susceptibles.view(susceptibles.shape[0], susceptibles.shape[1], 1)
        susc_tensor = susc_tensor.view(-1, 1, susc_tensor.shape[-1])

        # Element-wise multiplication to get contact matrix tensor
        contact_matrix_tensor = cm_tensor * susc_tensor
        eig_val_eff = []
        ngm_small_tensor_grads = []

        for cm in contact_matrix_tensor:
            # Compute f
            f = self._get_f(cm)
            # Compute ngm_large
            ngm_large = f @ self.v_inv
            ngm_large_tensor = torch.tensor(ngm_large, dtype=torch.float32,
                                            requires_grad=True)
            # Compute ngm_small
            ngm_small = torch.matmul(torch.matmul(self.e, ngm_large_tensor),
                                     self.e.T)
            ngm_small_tensor = torch.tensor(ngm_small, dtype=torch.float32,
                                     requires_grad=True)
            # Linear Transform ngm_small
            ngm_small_transformed = self.forward(ngm_small_tensor)
            # Calculate MSE(loss) based on ngm_small_transformed
            mse = torch.mean((ngm_small_transformed - torch.zeros_like(
                ngm_small_transformed)) ** 2)
            # Calculate RMSE
            rmse = torch.sqrt(mse)

            # Compute gradients of loss with respect to network parameters
            self.linear_layer.zero_grad()  # Clear previous gradients
            rmse.backward()  # Backpropagation
            ngm_small_gradients = self.linear_layer.weight.grad  # Retrieve gradients
            ngm_small_tensor_grads.append(ngm_small_gradients)

            # Compute eigenvalues based on linear transformed ngm small
            eig_val = torch.sort(torch.abs(torch.linalg.eigvals(ngm_small_transformed)),
                                 descending=True)[0]
            eig_val_eff.append(float(eig_val[0]))

            return eig_val_eff, ngm_small, ngm_small_tensor_grads






