import torch


class OutcomeTransitionParameters:
    def __init__(self, n_age: int, params: dict, model: str):
        self.n_age = n_age
        self.model = model

        if model in ["rost", "rost_agg"]:
            self.p = params["p"]  # Asymptomatic prob
            self.h = params["h"]  # Hospitalization prob
            self.xi = params["xi"]  # ICU prob given hosp
            self.mu = params["mu"]  # Death prob given ICU
        else:
            pass

    def apply(self, ngm_small_tensor: torch.Tensor, outcome: str) -> torch.Tensor:
        if outcome == "r0" or outcome == "infected":
            return ngm_small_tensor
        if self.model in ["rost", "rost_agg"]:
            p_symptomatic = 1.0 - self.p
            if outcome == "hospitalized":
                return ngm_small_tensor * (p_symptomatic * self.h *
                                           (1.0 * self.xi)).view(-1, 1)
            elif outcome == "icu":
                return ngm_small_tensor * (p_symptomatic * self.h * self.xi).view(-1, 1)
            elif outcome == "death":
                return ngm_small_tensor * (p_symptomatic * self.h *
                                           self.xi * self.mu).view(-1, 1)

        else:
            raise ValueError(f"Unsupported outcome target: {outcome}")
