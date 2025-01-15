import torch


class EigenValueGradient:
    def __init__(self, ngm_small_tensor: torch.Tensor, method: str = 'eig',
                 right_eig_vec: torch.Tensor = None,
                 left_eig_vec: torch.Tensor = None):

        self.ngm_small_tensor = ngm_small_tensor
        self.method = method
        self.right_eig_vec = right_eig_vec
        self.left_eig_vec = left_eig_vec

        self.eig_val_cm_grad = None

    def run(self, ngm_small_grads: torch.Tensor):
        if self.method == "eig":
            self._compute_eig_gradient(ngm_small_grads)
        elif self.method == "svd":
            self._compute_svd_gradient(ngm_small_grads)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'eig' or 'svd'.")

    def _compute_eig_gradient(self, ngm_small_grads: torch.Tensor):
        """
        Computes the gradient for the Eigen Decomposition method.
        """
        right_vec = self.right_eig_vec.view(-1, 1)  # [N, 1]
        right_vec_transpose = right_vec.T           # [1, N]

        # ∇K @ v
        grad_K_v = torch.matmul(ngm_small_grads, right_vec).squeeze(dim=2)  # [N, M]

        # v.T @ ∇K @ v
        self.eig_val_cm_grad = torch.matmul(right_vec_transpose, grad_K_v).squeeze()  # [M]

    def _compute_svd_gradient(self, ngm_small_grads: torch.Tensor):
        """
        Computes the gradient for the SVD method: v_left.T @ ∇K @ v
        """
        left_vec_transpose = self.left_eig_vec.view(1, -1)  # transpose it; [1, N]
        right_vec = self.right_eig_vec.view(-1, 1)          # remains; [N, 1]

        # Compute ∇K @ v
        grad_K_v = torch.matmul(ngm_small_grads, right_vec).squeeze(dim=2)  # [N, M]

        # Compute v_left.T @ ∇K @ v
        self.eig_val_cm_grad = torch.matmul(left_vec_transpose, grad_K_v).squeeze()  # [M]


