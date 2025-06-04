import torch
import numpy as np
from scipy.linalg import eig


class ContactMatrixEigenCalculator:
    def __init__(self, contact_matrix: torch.Tensor):
        self.contact_matrix = contact_matrix
        self.dominant_eigenvalue = None

    def run(self):
        mat_np = self.contact_matrix.detach().cpu().numpy()
        # Compute eigenvalues and eigenvectors
        eig_vals, left_eig_vecs, right_eig_vecs = eig(mat_np, left=True, right=True)
        self.dominant_eigenvalue = np.abs(eig_vals).max().real.item()

        return self.dominant_eigenvalue
