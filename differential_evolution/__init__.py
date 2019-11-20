__version__ = '1.10.0'

from .differential_evolution import DifferentialEvolution
from .evolution_strategy import EvolutionStrategy

try:
    from .openmdao_driver import DifferentialEvolutionDriver
except ModuleNotFoundError:
    pass
