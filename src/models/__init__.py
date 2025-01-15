from src.models.british_columbia.ngm_calculator import NGMCalculator as BCNGMCalculator
from src.models.chikina.ngm_calculator import NGMCalculator as ChikinaNGMCalculator
from src.models.italy.ngm_calculator import NGMCalculator as ItalyNGMCalculator
from src.models.kenya.ngm_calculator import NGMCalculator as KenyaNGMCalculator
from src.models.moghadas.ngm_calculator import NGMCalculator as MoghadasNGMCalculator
from src.models.rost.ngm_calculator import NGMCalculator as RostNGMCalculator
from src.models.seir.ngm_calculator import NGMCalculator as SeirNGMCalculator
from src.models.washington.ngm_calculator import NGMCalculator as WashingtonGMCalculator

model_calc_map = {
    "british_columbia": BCNGMCalculator,
    "chikina": ChikinaNGMCalculator,
    "italy": ItalyNGMCalculator,
    "kenya": KenyaNGMCalculator,
    "moghadas": MoghadasNGMCalculator,
    "rost": RostNGMCalculator,
    "seir": SeirNGMCalculator,
    "washington": WashingtonGMCalculator
}
