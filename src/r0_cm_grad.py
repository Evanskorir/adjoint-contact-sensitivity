import torch


class EigenCMGrad:
    def __init__(self, ngm_small_tensor, dominant_eig_vec):
        self.ngm_small_tensor = ngm_small_tensor
        self.dominant_eig_vec = dominant_eig_vec

    def calculate_eigen_val_cm_grad(self, ngm_small_grads):
        # Ensure eigenvector is a column vector
        eig_vec = self.dominant_eig_vec.view(-1, 1)

        # Perform matrix multiplications and reshape
        eig_val_cm_grad = torch.matmul(ngm_small_grads, eig_vec).view(1, 136, 16)
        eig_val_cm_grad = torch.matmul(eig_val_cm_grad, eig_vec).squeeze(dim=0)

        return eig_val_cm_grad
