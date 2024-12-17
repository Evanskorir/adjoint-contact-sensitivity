from src.models.british_columbia.ngm_calculator import NGMCalculator as BCNGMCalculator
from src.models.chikina.ngm_calculator import NGMCalculator as ChikinaNGMCalculator
from src.models.italy.ngm_calculator import NGMCalculator as ItalyNGMCalculator
from src.models.kenya.ngm_calculator import NGMCalculator as KenyaNGMCalculator
from src.models.moghadas.ngm_calculator import NGMCalculator as MoghadasNGMCalculator
from src.models.rost.ngm_calculator import NGMCalculator as RostNGMCalculator
from src.models.seir.ngm_calculator import NGMCalculator as SeirNGMCalculator
from src.models.washington.ngm_calculator import NGMCalculator as WashingtonGMCalculator

from src.models.british_columbia.model import BCModel
from src.models.chikina.model import ChikinaModel
from src.models.kenya.model import KenyaModel
from src.models.moghadas.model import MoghadasModel
from src.models.rost.model import RostModelHungary
from src.models.seir.model import SeirUK
from src.models.washington.model import WashingtonModel

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

model_map = {
    "british_columbia": BCModel,
    "chikina": ChikinaModel,
    "kenya": KenyaModel,
    "moghadas": MoghadasModel,
    "rost": RostModelHungary,
    "seir": SeirUK,
    "washington": WashingtonModel
}
