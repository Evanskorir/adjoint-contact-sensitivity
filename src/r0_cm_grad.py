import torch


class EigenCMGrad:
    def __init__(self, ngm_small_tensor, dominant_eig_vec):
        self.ngm_small_tensor = ngm_small_tensor
        self.dominant_eig_vec = dominant_eig_vec

    def calculate_eigen_val_cm_grad(self, ngm_small_grads):
        # Reshape self.dominant_eig_vec to be a column vector
        eig_vec = self.dominant_eig_vec.view(-1, 1)  # Shape: [16, 1]

        # Compute Ax: ngm_small_grads @ eig_vec for each slice
        Ax = torch.matmul(ngm_small_grads, eig_vec).squeeze(dim=2)  # Shape: [16, 136]

        # Compute eig_vec.T @ Ax
        # Reshape eig_vec.T to match the multiplication with B
        eig_vec_T = self.dominant_eig_vec.view(1, -1)  # Shape: [1, 16]
        eig_val_cm_grad = torch.matmul(eig_vec_T, Ax)

        return eig_val_cm_grad
