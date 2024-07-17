import torch


class Eigen_grad:
    def __init__(self, n_age: int, ngm_small_tensor: torch.Tensor,
                 dominant_eig_vecs: torch.Tensor, upper_tri_elems: torch.Tensor):
        self.n_age = n_age
        self.ngm_small_tensor = ngm_small_tensor
        self.dominant_eig_vecs = dominant_eig_vecs
        self.upper_tri_elems = upper_tri_elems

        self.ngm_small_grads = self._compute_ngm_small_grads()

    def _compute_ngm_small_grads(self):
        ngm_small_grads = torch.zeros((self.n_age, self.n_age,
                                       len(self.upper_tri_elems)), dtype=torch.float32)

        for i in range(self.n_age):
            for j in range(self.n_age):
                grad = torch.autograd.grad(outputs=self.ngm_small_tensor[i, j],
                                           inputs=self.upper_tri_elems,
                                           retain_graph=True,
                                           create_graph=True)[0]
                ngm_small_grads[i, j] = grad
        return ngm_small_grads
