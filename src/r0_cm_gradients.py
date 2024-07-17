import torch


class Eigen_cm_grad:
    def __init__(self, ngm_small_tensor, dominant_eig_vecs):
        self.ngm_small_tensor = ngm_small_tensor
        self.dominant_eig_vecs = dominant_eig_vecs

    def calculate_eigen_val_cm_grad(self, ngm_small_grads):
        # ensure eigenvector is a column vector (16, 1)
        eig_vecs = self.dominant_eig_vecs.view(-1, 1)

        # Perform first matrix multiplication
        first_result = torch.matmul(eig_vecs.T, ngm_small_grads)  # Shape: (1, 16, 136)

        # Reshape first_result to (1, 136, 16)
        first_result = first_result.view(1, 136, 16)

        # Perform second matrix multiplication
        second_result = torch.matmul(first_result, eig_vecs)  # Shape: (1, 136, 1)
        # Squeeze along the first dimension (1) to get the final result vector
        eig_val_cm_grad = second_result.squeeze(dim=0)

        return eig_val_cm_grad

