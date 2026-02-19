"""
MadEvolve Transformer Module - Code transformation and editing.
"""

from madevolve.transformer.patcher import apply_differential_patch
from madevolve.transformer.rewriter import apply_holistic_rewrite
from madevolve.transformer.changeset import (
    extract_code_block,
    apply_synthesis_patch,
    summarize_changes,
)
from madevolve.transformer.parallel import ParameterOptimizer
from madevolve.transformer.blocks import (
    has_evolve_blocks,
    extract_mutable_content,
    split_code_regions,
    replace_mutable_content,
    format_code_for_prompt,
)

__all__ = [
    "apply_differential_patch",
    "apply_holistic_rewrite",
    "extract_code_block",
    "apply_synthesis_patch",
    "summarize_changes",
    "ParameterOptimizer",
    "has_evolve_blocks",
    "extract_mutable_content",
    "split_code_regions",
    "replace_mutable_content",
    "format_code_for_prompt",
]
