"""
MadEvolve Templates Module - Prompt templates for LLM interactions.
"""

from madevolve.templates.bootstrap import build_initial_prompt
from madevolve.templates.differential import build_differential_prompt
from madevolve.templates.holistic import build_holistic_prompt
from madevolve.templates.hybrid import build_synthesis_prompt
from madevolve.templates.insight import build_analysis_prompt

__all__ = [
    "build_initial_prompt",
    "build_differential_prompt",
    "build_holistic_prompt",
    "build_synthesis_prompt",
    "build_analysis_prompt",
]
