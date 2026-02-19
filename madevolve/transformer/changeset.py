"""
Changeset utilities for code transformation.

This module provides utilities for extracting code from LLM output,
computing diffs, and handling synthesis patches.
"""

import difflib
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ChangeStats:
    """Statistics about code changes."""
    lines_added: int
    lines_removed: int
    lines_modified: int
    total_changes: int


def extract_code_block(
    text: str,
    language: Optional[str] = None,
) -> Optional[str]:
    """
    Extract code from markdown code blocks.

    Args:
        text: Text containing code blocks
        language: Optional language identifier

    Returns:
        Extracted code or None
    """
    if language:
        escaped = re.escape(language)
        pattern = rf"```{escaped}[23]?\s*(.*?)\s*```"
    else:
        pattern = r"```(?:\w*)\s*(.*?)\s*```"

    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try without language identifier
    if language:
        return extract_code_block(text, None)

    return None


def apply_synthesis_patch(
    base_code: str,
    llm_output: str,
) -> Optional[str]:
    """
    Apply a synthesis (crossover) patch combining ideas from multiple sources.

    Args:
        base_code: Base code to start from
        llm_output: LLM output with synthesized code

    Returns:
        Synthesized code or None if extraction fails
    """
    # For synthesis, we expect a complete code output
    code = extract_code_block(llm_output, "python")

    if code:
        return code

    # Try fallback extraction
    from madevolve.transformer.rewriter import apply_holistic_rewrite
    return apply_holistic_rewrite(llm_output)


def summarize_changes(
    original: str,
    modified: str,
) -> Tuple[ChangeStats, str]:
    """
    Summarize changes between original and modified code.

    Args:
        original: Original code
        modified: Modified code

    Returns:
        Tuple of (ChangeStats, unified diff string)
    """
    original_lines = original.split("\n")
    modified_lines = modified.split("\n")

    # Generate unified diff
    diff = list(difflib.unified_diff(
        original_lines,
        modified_lines,
        lineterm="",
    ))

    # Count changes
    added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

    # Estimate modifications (lines that changed but weren't just added/removed)
    matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)
    modified_count = sum(
        1 for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag == "replace"
    )

    stats = ChangeStats(
        lines_added=added,
        lines_removed=removed,
        lines_modified=modified_count,
        total_changes=added + removed,
    )

    diff_str = "\n".join(diff)

    return stats, diff_str


def redact_static_regions(
    code: str,
    marker_start: str = "# === STATIC-START ===",
    marker_end: str = "# === STATIC-END ===",
) -> str:
    """
    Remove static (non-evolvable) regions from code.

    Args:
        code: Source code
        marker_start: Start marker for static regions
        marker_end: End marker for static regions

    Returns:
        Code with static regions removed
    """
    lines = code.split("\n")
    result = []
    in_static = False

    for line in lines:
        if marker_start in line:
            in_static = True
            continue
        elif marker_end in line:
            in_static = False
            continue

        if not in_static:
            result.append(line)

    return "\n".join(result)


def extract_evolve_blocks(
    code: str,
    block_start: Optional[str] = None,
    block_end: Optional[str] = None,
) -> List[Tuple[int, int, str]]:
    """
    Extract evolvable blocks from code.

    Args:
        code: Source code
        block_start: Start marker pattern (uses regex default if None)
        block_end: End marker pattern (uses regex default if None)

    Returns:
        List of (start_line, end_line, content) tuples
    """
    from madevolve.common.constants import EVOLVE_BLOCK_START_PATTERN, EVOLVE_BLOCK_END_PATTERN

    start_pattern = block_start or EVOLVE_BLOCK_START_PATTERN
    end_pattern = block_end or EVOLVE_BLOCK_END_PATTERN

    lines = code.split("\n")
    blocks = []
    current_start = None
    current_lines = []

    for i, line in enumerate(lines):
        if re.search(start_pattern, line):
            current_start = i
            current_lines = []
        elif re.search(end_pattern, line):
            if current_start is not None:
                blocks.append((
                    current_start,
                    i,
                    "\n".join(current_lines),
                ))
            current_start = None
            current_lines = []
        elif current_start is not None:
            current_lines.append(line)

    return blocks


def compute_edit_distance(code1: str, code2: str) -> int:
    """
    Compute edit distance between two code strings.

    Args:
        code1: First code
        code2: Second code

    Returns:
        Edit distance
    """
    # Use line-based edit distance
    lines1 = code1.split("\n")
    lines2 = code2.split("\n")

    m, n = len(lines1), len(lines2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if lines1[i - 1] == lines2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[m][n]


def normalize_code(code: str) -> str:
    """
    Normalize code for comparison.

    Removes extra whitespace and standardizes formatting.

    Args:
        code: Source code

    Returns:
        Normalized code
    """
    lines = code.split("\n")

    # Remove trailing whitespace
    lines = [line.rstrip() for line in lines]

    # Remove trailing blank lines
    while lines and not lines[-1]:
        lines.pop()

    # Remove leading blank lines
    while lines and not lines[0]:
        lines.pop(0)

    return "\n".join(lines)
