import torch

from src.e_matrix import EMatrix
from src.v_matrix import VMatrix
from src.upper_tri_elements import MatrixOperations


class R0Generator:
    def __init__(self, param: dict, n_age: int) -> None:
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

        self.e_matrix = EMatrix(n_states=self.n_states, n_age=n_age)
        self.v_matrix = VMatrix(param=param, n_states=self.n_states,
                                n_age=n_age, states=self.states)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = torch.zeros((self.n_age * n_states, self.n_age * n_states),
                        requires_grad=True)
        inf_a = self.parameters["inf_a"] if "inf_a" in self.parameters.keys() else 1.0
        inf_s = self.parameters["inf_s"] if "inf_s" in self.parameters.keys() else 1.0
        inf_p = self.parameters["inf_p"] if "inf_p" in self.parameters.keys() else 1.0

        susc_vec = self.parameters["susc"].reshape((-1, 1))

        indices_l1 = torch.arange(i["l1"], s_mtx, n_states)
        indices_ip = torch.arange(i["ip"], s_mtx, n_states)
        indices_a1 = torch.arange(i["a1"], s_mtx, n_states)
        indices_a2 = torch.arange(i["a2"], s_mtx, n_states)
        indices_a3 = torch.arange(i["a3"], s_mtx, n_states)
        indices_i1 = torch.arange(i["i1"], s_mtx, n_states)
        indices_i2 = torch.arange(i["i2"], s_mtx, n_states)
        indices_i3 = torch.arange(i["i3"], s_mtx, n_states)

        f_new = f.clone()
        f_new[indices_l1[:, None], indices_ip] = inf_p * contact_mtx.T * susc_vec
        f_new[indices_l1[:, None], indices_a1] = inf_a * contact_mtx.T * susc_vec
        f_new[indices_l1[:, None], indices_a2] = inf_a * contact_mtx.T * susc_vec
        f_new[indices_l1[:, None], indices_a3] = inf_a * contact_mtx.T * susc_vec
        f_new[indices_l1[:, None], indices_i1] = inf_s * contact_mtx.T * susc_vec
        f_new[indices_l1[:, None], indices_i2] = inf_s * contact_mtx.T * susc_vec
        f_new[indices_l1[:, None], indices_i3] = inf_s * contact_mtx.T * susc_vec

        return f_new

    def _update_contact_matrix(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        matr = MatrixOperations(matrix=contact_mtx, n_age=self.n_age)
        # Get upper triangular elements
        upper_tri_elements = matr.get_upper_triangle_elements()
        # Update upper triangle of the contact matrix
        updated_contact_mtx = matr.upper_triangle_to_matrix(upper_tri_elements)
        return upper_tri_elements, updated_contact_mtx

    def get_eig_val(self, contact_mtx: torch.Tensor = None):
        if contact_mtx is not None:
            self.contact_matrix = contact_mtx.clone().detach().requires_grad_(True)

        upper_tri_elements, updated_contact_mtx = self._update_contact_matrix(
            self.contact_matrix)
        f = self._get_f(updated_contact_mtx)
        ngm_large = torch.matmul(f, self.v_matrix.v_inv)
        # Compute ngm_small
        ngm_small_tensor = torch.matmul(torch.matmul(self.e_matrix.e, ngm_large),
                                        self.e_matrix.e.T)
        # Clear previous gradients
        self.contact_matrix.grad = None
        # Compute gradients of ngm_small_tensor with respect to contact_mtx
        ngm_small_grads = torch.autograd.grad(outputs=ngm_small_tensor.sum(),
                                              inputs=updated_contact_mtx,
                                              retain_graph=True)[0]
        # Get upper triangular elements (136) of the gradients
        matr = MatrixOperations(matrix=ngm_small_grads, n_age=self.n_age)
        ngm_small_grads_upp = matr.get_upper_triangle_elements()

        # Compute eigenvalues
        eig_vals = torch.linalg.eigvals(ngm_small_grads)
        eig_vals_sorted = torch.sort(torch.abs(eig_vals), descending=True)[0]

        # Extract the dominant eigenvalue
        dominant_eig_val = eig_vals_sorted[0].item()
        return dominant_eig_val, ngm_small_tensor, ngm_small_grads_upp











