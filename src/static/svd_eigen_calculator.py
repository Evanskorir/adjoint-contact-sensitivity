import torch


class SVDEigenCalculator:
    def __init__(self, ngm_small_tensor: torch.Tensor) -> None:
        self.ngm_small_tensor = ngm_small_tensor

        self.dominant_singular_val = None
        self.left_singular_vec = None
        self.right_singular_vec = None

    def run(self):
        # Compute the Singular Value Decomposition
        V_left, sigma, V_right = torch.linalg.svd(self.ngm_small_tensor)

        # Extract the dominant singular value and its corresponding singular vectors
        self.dominant_singular_val = sigma[0].item()
        self.left_singular_vec = V_left[:, 0]
        self.right_singular_vec = V_right[0, :]




