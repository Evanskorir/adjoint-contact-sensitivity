import torch


class Eigen_Grad:
    def __init__(self, ngm_small_tensor: torch.Tensor,
                 dominant_eig_vec: torch.Tensor, contact_input: torch.Tensor):
        self.ngm_small_tensor = ngm_small_tensor
        self.dominant_eig_vec = dominant_eig_vec
        self.contact_input = contact_input

        self.ngm_small_grads = self._compute_ngm_small_grads()

    def _compute_ngm_small_grads(self):
        ngm_small_grads = torch.zeros((self.ngm_small_tensor.size(0),
                                       self.ngm_small_tensor.size(1),
                                       self.contact_input.size(0)), dtype=torch.float32)
        for i in range(self.ngm_small_tensor.size(0)):
            for j in range(self.ngm_small_tensor.size(1)):
                grad = torch.autograd.grad(outputs=self.ngm_small_tensor[i, j],
                                           inputs=self.contact_input,
                                           retain_graph=True,
                                           create_graph=True)[0]
                ngm_small_grads[i, j, ] = grad
        return ngm_small_grads
