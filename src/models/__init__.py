from src.models.british_columbia.ngm_calculator import NGMCalculator as BCNGMCalculator
from src.models.kenya.ngm_calculator import NGMCalculator as KenyaNGMCalculator
from src.models.rost.ngm_calculator import NGMCalculator as RostNGMCalculator
from src.models.seir.ngm_calculator import NGMCalculator as SeirNGMCalculator

model_calc_map = {
    "british_columbia": BCNGMCalculator,
    "kenya": KenyaNGMCalculator,
    "kenya_agg": KenyaNGMCalculator,
    "rost": RostNGMCalculator,
    "rost_agg": RostNGMCalculator,
    "seir": SeirNGMCalculator,
    "seir_agg": SeirNGMCalculator
}
