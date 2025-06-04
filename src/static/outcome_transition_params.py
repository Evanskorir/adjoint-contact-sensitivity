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

        elif model in ["kenya", "kenya_agg"]:
            self.q = params["q"]
            self.kappa = params["kappa"]
            self.gamma_m = params["gamma_m"]
            self.phi = params["phi"]
            self.gamma_h = params["gamma_h"]
            self.delta = params["delta_c"]
            self.xi = params["xi"]

        else:
            pass

    def apply(self, ngm_small_tensor: torch.Tensor, outcome: str) -> torch.Tensor:
        if outcome == "r0" or outcome == "infected":
            return ngm_small_tensor
        if self.model in ["rost", "rost_agg"]:
            p_symptomatic = 1.0 - self.p
            if outcome == "hospitalized":
                return ngm_small_tensor * (p_symptomatic * self.h).view(-1, 1)
            elif outcome == "icu":
                return ngm_small_tensor * (p_symptomatic * self.h * self.xi).view(-1, 1)
            elif outcome == "death":
                return ngm_small_tensor * (p_symptomatic * self.h *
                                           self.xi * self.mu).view(-1, 1)
        elif self.model in ["kenya", "kenya_agg"]:
            # Calculate probabilities from transitions
            p_hosp = (1.0 - self.q) * (self.kappa / (self.kappa + self.gamma_m))
            p_icu = p_hosp * (self.phi / (self.phi + self.gamma_h))
            p_death = p_icu * (self.delta / (self.delta + self.xi))

            if outcome == "hospitalized":
                return ngm_small_tensor * p_hosp.view(-1, 1)
            elif outcome == "icu":
                return ngm_small_tensor * p_icu.view(-1, 1)
            elif outcome == "death":
                return ngm_small_tensor * p_death.view(-1, 1)

        else:
            raise ValueError(f"Unsupported outcome target: {outcome}")

