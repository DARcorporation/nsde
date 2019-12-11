__version__ = '1.13.0'

from .nsde import DifferentialEvolution
from .evolution_strategy import EvolutionStrategy

try:
    from .openmdao_driver import DifferentialEvolutionDriver
except ModuleNotFoundError:
    pass
