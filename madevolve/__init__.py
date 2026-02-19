"""
MadEvolve: A General-Purpose LLM-Driven Evolution Framework for Code Optimization

This framework provides a flexible and extensible platform for evolving code
using Large Language Models with evolutionary algorithms.
"""

__version__ = "0.1.0"
__author__ = "MadEvolve Team"

from madevolve.engine.orchestrator import EvolutionOrchestrator
from madevolve.engine.configuration import (
    EvolutionConfig,
    PopulationConfig,
    ModelConfig,
    ExecutorConfig,
    StorageConfig,
)

__all__ = [
    "EvolutionOrchestrator",
    "EvolutionConfig",
    "PopulationConfig",
    "ModelConfig",
    "ExecutorConfig",
    "StorageConfig",
    "__version__",
]
