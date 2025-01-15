import torch


class AggregationApproach:
    def __init__(self, eig_val_cm_grad: torch.Tensor,
                 contact_input: torch.Tensor, r0: float, n_age: int):

        self.eig_val_cm_grad = eig_val_cm_grad
        self.contact_input = contact_input
        self.r0 = r0
        self.n_age = n_age

        self.cumulative_elasticities = None

    def run(self) -> torch.Tensor:
        """
        Compute aggregated elasticities (e_j) for each age group.
        :return: Aggregated elasticities for each age group
        """
        if self.eig_val_cm_grad is None:
            raise ValueError("You must run the gradient computation before"
                             " computing elasticities.")

        # Compute pairwise elasticities e_k
        pairwise_elasticities = (
            self.eig_val_cm_grad.squeeze() * self.contact_input
        ) / self.r0

        # Map the pairwise elasticities back to age groups
        # Create a mapping from the upper tri elements to the age groups
        age_group_contributions = torch.zeros(self.n_age,
                                              device=pairwise_elasticities.device)

        index = 0
        for i in range(self.n_age):
            for j in range(i, self.n_age):
                # Add contributions for age group i and j
                age_group_contributions[i] += pairwise_elasticities[index]
                if i != j:
                    age_group_contributions[j] += pairwise_elasticities[index]
                index += 1

        self.cumulative_elasticities = age_group_contributions

        return self.cumulative_elasticities
