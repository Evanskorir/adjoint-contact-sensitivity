import numpy as np
from scipy.stats import ttest_ind
import scipy.stats as ss
from scipy.stats import pearsonr

import pandas as pd


class Aggregation_Pvalues:
    def __init__(self, sim_obj, calculation_approach):
        self.sim_obj = sim_obj
        self.calculation_approach = calculation_approach

        self.agg_prcc = None
        self.confidence_lower = None
        self.confidence_upper = None
        self.p_value = None
        self.p_value_mtx = None

        self.prcc_list = None
        self.agg_std = None

        self.complex_logic = True
        self.pop_logic = False

    def calculate_p_values(self, ngm_small):
        n = ngm_small.shape[0]
        p_values_matrix = np.zeros((n, n))

        # Calculate p-values for each pair of elements
        for i in range(n):
            for j in range(n):
                r, p = pearsonr(ngm_small[i], ngm_small[j])
                p_values_matrix[i, j] = p

        self.p_value_mtx = p_values_matrix
        return p_values_matrix

    def _calculate_distribution_prcc_p_val(self, ngm_grad):
        distribution_p_val = ngm_grad * (1 - self.p_value_mtx)
        if self.complex_logic:
            distribution_prcc_p_val = distribution_p_val / \
                                      np.sum(ngm_grad * (1 - self.p_value_mtx), axis=1,
                                             keepdims=True)
        elif self.pop_logic:
            distribution_prcc_p_val = ((1 - self.p_value_mtx) *
                                       self.sim_obj.population.reshape((1, -1)) /
                                       np.sum((1 - self.p_value_mtx) *
                                              self.sim_obj.population.reshape((1, -1)),
                                              axis=1, keepdims=True))
        else:
            distribution_prcc_p_val = (1 - self.p_value_mtx) / np.sum(1 - self.p_value_mtx,
                                                                      axis=1, keepdims=True)
        return distribution_prcc_p_val

    def aggregate_prcc_values_mean(self, ngm_grad):
        distribution_prcc_p_val = self._calculate_distribution_prcc_p_val(ngm_grad=ngm_grad)
        agg = np.sum(ngm_grad * distribution_prcc_p_val, axis=1)
        agg_square = np.sum(ngm_grad ** 2 * distribution_prcc_p_val, axis=1)
        agg_std = np.sqrt(abs(agg_square - agg ** 2))

        self.confidence_lower = agg_std
        self.confidence_upper = agg_std
        self.agg_std = agg_std
        self.agg_prcc = agg
        return agg.flatten(), agg_std.flatten()

    def aggregate_prcc_values_median(self, ngm_grad):
        median_values = []
        conf_lower = []
        conf_upper = []
        # prob using complex logic
        distribution_prcc_p_val = self._calculate_distribution_prcc_p_val(ngm_grad=ngm_grad)

        # Iterate over the columns of prcc and p_value
        for i in range(self.sim_obj.n_ag):
            # Take the ith column from prcc
            prcc_column = ngm_grad[i, :]

            # Take the ith column from distribution_prcc_p_val
            prob_value_column = distribution_prcc_p_val[i, :]

            # Combine prcc_column and prob_value_column into a single matrix (16 * 2)
            combined_matrix = np.column_stack((prcc_column, prob_value_column))
            # Take the absolute values of the first column to avoid -ve median values
            combined_matrix[:, 0] = np.abs(combined_matrix[:, 0])
            # Sort the rows of combined_matrix by the first column
            sorted_indices = np.argsort(combined_matrix[:, 0])
            combined_matrix_sorted = combined_matrix[sorted_indices]
            # Calculate the cumulative sum of the second column
            cumul_distr = np.cumsum(combined_matrix_sorted[:, 1])
            # Find the median value
            median_value = combined_matrix_sorted[cumul_distr >= 0.5, 0][0]
            # Find the first quartile
            q1_value = combined_matrix_sorted[cumul_distr >= 0.25, 0][0]
            # Find the third quartile
            q3_value = combined_matrix_sorted[cumul_distr >= 0.75, 0][0]

            # Append the median, Q1, and Q3 values to their respective lists
            median_values.append(median_value)
            conf_lower.append(median_value - q1_value)
            conf_upper.append(q3_value - median_value)

        self.agg_prcc = median_values
        self.confidence_lower = conf_lower
        self.confidence_upper = conf_upper

    def aggregate_prcc_values(self, ngm_grad):
        if self.calculation_approach == 'mean':
            return self.aggregate_prcc_values_mean(ngm_grad=ngm_grad)
        else:
            return self.aggregate_prcc_values_median(ngm_grad=ngm_grad)
