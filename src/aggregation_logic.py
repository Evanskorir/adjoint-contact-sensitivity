import torch
import scipy.stats as ss


class Aggregation:
    def __init__(self, data, n_age: int, pop: torch.Tensor,
                 grad_mtx: torch.Tensor, calculation_approach: str):

        self.data = data
        self.n_age = n_age
        self.population = pop
        self.grads = grad_mtx
        self.calculation_approach = calculation_approach

        # Initialize output attributes
        self.agg_prcc = None
        self.confidence_lower = None
        self.confidence_upper = None
        self.agg_std = None

        # Process gradients and p-values
        self.grads_mtx = self._reconstruct_symmetric_grad_matrix(self.grads)
        self.p_values = self._calculate_p_values().detach().numpy()
        self.p_values_mtx = self._reconstruct_symmetric_grad_matrix(
            torch.tensor(self.p_values, dtype=torch.float32)
        )

        # Aggregate grads values based on the selected approach
        self._aggregate_grad_values()

    def _reconstruct_symmetric_grad_matrix(self, sym_mtx: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a symmetric matrix from the upper triangular elements.
        Args: sym_mtx (torch.Tensor): Input tensor containing the upper tri elements.
        Returns: torch.Tensor: Symmetric matrix.
        """
        mtx = torch.zeros((self.n_age, self.n_age))
        upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)
        data_flat = sym_mtx.view(-1)

        mtx[upper_tri_idx[0], upper_tri_idx[1]] = data_flat
        mtx = mtx + mtx.T - torch.diag(mtx.diag())
        return mtx

    def _calculate_p_values(self) -> torch.Tensor:
        """
        Calculate p-values for the gradients using t-scores.
        Returns: torch.Tensor: Tensor containing p-values.
        """
        dof = torch.tensor(136 - 2, dtype=torch.float32)
        t = self.grads * torch.sqrt(
            dof / torch.abs(1 - self.grads ** 2)
        )
        # p-value for 2-sided test
        p_values = 2 * (1 - torch.tensor(ss.t.cdf(
            x=torch.abs(t.detach()),
            df=dof))).detach()

        return p_values

    def _calculate_distribution_grads_p_val(self) -> torch.Tensor:
        """
        Calculate distribution of grads p-values based on the selected logic.
        Returns: torch.Tensor: Distribution of grads p-values.
        """
        distribution_prcc_p_val = (1 - self.p_values_mtx) / \
                                  torch.sum(1 - self.p_values_mtx, axis=1, keepdims=True)

        return distribution_prcc_p_val

    def _aggregate_grads_values_mean(self):
        """
        Aggregate grad values using the mean approach and calculate confidence intervals.
        """
        distribution_grad_p_val = self._calculate_distribution_grads_p_val()
        agg = torch.sum(self.grads_mtx * distribution_grad_p_val, axis=1)
        agg_square = torch.sum(self.grads_mtx ** 2 * distribution_grad_p_val, axis=1)
        agg_std = torch.sqrt(agg_square - agg ** 2)

        self.confidence_lower = agg_std
        self.confidence_upper = agg_std
        self.agg_std = agg_std
        self.agg_prcc = agg
        return agg.flatten(), agg_std.flatten()

    def _aggregate_grads_values_median(self):
        """
        Aggregate grad values using the median approach and calculate confidence intervals.
        """
        median_values = []
        conf_lower = []
        conf_upper = []

        distribution_grad_p_val = self._calculate_distribution_grads_p_val()

        for i in range(self.n_age):
            grads_column = self.grads_mtx[i, :]
            prob_value_column = distribution_grad_p_val[i, :]

            combined_matrix = torch.column_stack((grads_column, prob_value_column))
            combined_matrix[:, 0] = torch.abs(combined_matrix[:, 0])
            sorted_indices = torch.argsort(combined_matrix[:, 0])
            combined_matrix_sorted = combined_matrix[sorted_indices]

            cumul_distr = torch.cumsum(combined_matrix_sorted[:, 1], dim=0)
            median_value = combined_matrix_sorted[cumul_distr >= 0.5, 0][0]
            q1_value = combined_matrix_sorted[cumul_distr >= 0.25, 0][0]
            q3_value = combined_matrix_sorted[cumul_distr >= 0.75, 0][0]

            median_values.append(median_value)
            conf_lower.append(median_value - q1_value)
            conf_upper.append(q3_value - median_value)

        self.agg_grad = median_values
        self.confidence_lower = conf_lower
        self.confidence_upper = conf_upper

    def _aggregate_grad_values(self):
        """
        Aggregates grad values based on the specified approach.
        """
        if self.calculation_approach == 'mean':
            return self._aggregate_grads_values_mean()
        else:
            return self._aggregate_grads_values_median()
