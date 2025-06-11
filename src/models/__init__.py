from src.models.british_columbia.ngm_calculator import NGMCalculator as BCNGMCalculator
from src.models.rost.ngm_calculator import NGMCalculator as RostNGMCalculator
from src.models.seir.ngm_calculator import NGMCalculator as SeirNGMCalculator

model_calc_map = {
    "british_columbia": BCNGMCalculator,
    "rost": RostNGMCalculator,
    "rost_agg": RostNGMCalculator,
    "seir": SeirNGMCalculator,
    "seir_agg": SeirNGMCalculator
}
