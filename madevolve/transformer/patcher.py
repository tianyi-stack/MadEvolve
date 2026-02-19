"""
Differential Patch Application - SEARCH/REPLACE style edits.

This module handles application of diff-style patches using
the SEARCH/REPLACE format commonly used in LLM code editing.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from madevolve.common.constants import SEARCH_MARKER, REPLACE_MARKER, END_MARKER  # noqa: F401 — kept for external consumers

logger = logging.getLogger(__name__)


@dataclass
class PatchBlock:
    """A single SEARCH/REPLACE block."""
    search: str
    replace: str
    line_number: int = 0


@dataclass
class PatchResult:
    """Result of patch application."""
    success: bool
    code: Optional[str]
    blocks_applied: int
    blocks_failed: int
    error_message: Optional[str] = None


def apply_differential_patch(
    original_code: str,
    llm_output: str,
    strict: bool = False,
) -> Optional[str]:
    """
    Apply SEARCH/REPLACE patches to code.

    Args:
        original_code: Original source code
        llm_output: LLM output containing patch blocks
        strict: If True, fail on any failed block

    Returns:
        Patched code or None if all blocks failed
    """
    result = _apply_patches(original_code, llm_output, strict)

    if result.success:
        return result.code
    else:
        logger.warning(f"Patch application failed: {result.error_message}")
        return None


def _apply_patches(
    original_code: str,
    llm_output: str,
    strict: bool = False,
) -> PatchResult:
    """
    Internal function to apply patches with detailed result.

    Args:
        original_code: Original source code
        llm_output: LLM output containing patch blocks
        strict: If True, fail on any failed block

    Returns:
        PatchResult with details
    """
    # Extract patch blocks
    blocks = _extract_patch_blocks(llm_output)

    if not blocks:
        # Try to extract code block as fallback
        from madevolve.transformer.changeset import extract_code_block
        code = extract_code_block(llm_output, "python")
        if code:
            return PatchResult(
                success=True,
                code=code,
                blocks_applied=0,
                blocks_failed=0,
            )

        return PatchResult(
            success=False,
            code=None,
            blocks_applied=0,
            blocks_failed=0,
            error_message="No patch blocks found in output",
        )

    # Apply each block
    current_code = original_code
    applied = 0
    failed = 0
    errors = []

    for i, block in enumerate(blocks):
        success, new_code, error = _apply_single_block(current_code, block)

        if success:
            current_code = new_code
            applied += 1
        else:
            failed += 1
            errors.append(f"Block {i + 1}: {error}")

            if strict:
                return PatchResult(
                    success=False,
                    code=None,
                    blocks_applied=applied,
                    blocks_failed=failed,
                    error_message=error,
                )

    if applied == 0:
        return PatchResult(
            success=False,
            code=None,
            blocks_applied=0,
            blocks_failed=failed,
            error_message="; ".join(errors) if errors else "No blocks applied",
        )

    return PatchResult(
        success=True,
        code=current_code,
        blocks_applied=applied,
        blocks_failed=failed,
        error_message="; ".join(errors) if errors else None,
    )


def _extract_patch_blocks(llm_output: str) -> List[PatchBlock]:
    """
    Extract SEARCH/REPLACE blocks from LLM output.

    Args:
        llm_output: Raw LLM output

    Returns:
        List of PatchBlock instances
    """
    blocks = []

    # Lenient pattern: accept 3+ marker characters (e.g. <<<<<<< or <<<<<< SEARCH)
    pattern = r"<{3,}\s*SEARCH\s*\n(.*?)\n\s*={3,}\s*\n(.*?)\n\s*>{3,}\s*REPLACE"

    matches = re.finditer(pattern, llm_output, re.DOTALL)

    for match in matches:
        search_text = match.group(1)
        replace_text = match.group(2)

        # Clean up trailing whitespace but preserve structure
        search_text = search_text.rstrip("\n")
        replace_text = replace_text.rstrip("\n")

        blocks.append(PatchBlock(
            search=search_text,
            replace=replace_text,
        ))

    return blocks


def _apply_single_block(
    code: str,
    block: PatchBlock,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Apply a single patch block to code.

    Args:
        code: Current code
        block: Patch block to apply

    Returns:
        Tuple of (success, new_code, error_message)
    """
    search = block.search
    replace = block.replace

    # Try exact match first
    if search in code:
        new_code = code.replace(search, replace, 1)
        return True, new_code, None

    # Try with normalized whitespace
    search_normalized = _normalize_whitespace(search)
    lines = code.split("\n")
    match_start, match_end = _find_fuzzy_match(lines, search_normalized)

    if match_start is not None:
        # Preserve original indentation
        original_indent = _get_indentation(lines[match_start])
        replace_lines = replace.split("\n")
        indented_replace = _apply_indentation(replace_lines, original_indent)

        new_lines = lines[:match_start] + indented_replace + lines[match_end + 1:]
        return True, "\n".join(new_lines), None

    return False, None, f"Search text not found"


def _normalize_whitespace(text: str) -> List[str]:
    """Normalize whitespace for fuzzy matching."""
    lines = text.split("\n")
    return [line.strip() for line in lines if line.strip()]


def _find_fuzzy_match(
    code_lines: List[str],
    search_normalized: List[str],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find fuzzy match for normalized search text.

    Args:
        code_lines: Lines of code
        search_normalized: Normalized search lines

    Returns:
        Tuple of (start_line, end_line) or (None, None)
    """
    if not search_normalized:
        return None, None

    code_normalized = [line.strip() for line in code_lines]
    search_len = len(search_normalized)

    for i in range(len(code_normalized) - search_len + 1):
        # Check if this position matches
        match = True
        for j in range(search_len):
            if code_normalized[i + j] != search_normalized[j]:
                match = False
                break

        if match:
            # Find actual line range (accounting for blank lines)
            start = i
            end = i + search_len - 1

            return start, end

    return None, None


def _get_indentation(line: str) -> str:
    """Extract leading whitespace from a line."""
    stripped = line.lstrip()
    return line[:len(line) - len(stripped)]


def _apply_indentation(lines: List[str], base_indent: str) -> List[str]:
    """Apply base indentation to lines while preserving relative indent."""
    if not lines:
        return lines

    # Find minimum indentation in replacement
    min_indent = float("inf")
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)

    if min_indent == float("inf"):
        min_indent = 0

    # Apply new base indentation
    result = []
    for line in lines:
        if line.strip():
            current_indent = len(line) - len(line.lstrip())
            relative_indent = current_indent - min_indent
            new_line = base_indent + " " * relative_indent + line.lstrip()
            result.append(new_line)
        else:
            result.append("")

    return result


def validate_patch_syntax(llm_output: str) -> Tuple[bool, List[str]]:
    """
    Validate patch syntax without applying.

    Args:
        llm_output: LLM output to validate

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Count markers using lenient regex (same tolerance as _extract_patch_blocks)
    search_count = len(re.findall(r"<{3,}\s*SEARCH", llm_output))
    end_count = len(re.findall(r">{3,}\s*REPLACE", llm_output))

    if search_count != end_count:
        issues.append(
            f"Marker count mismatch: SEARCH openers={search_count}, "
            f"REPLACE closers={end_count}"
        )

    # Check for properly formed blocks
    blocks = _extract_patch_blocks(llm_output)

    if search_count > 0 and len(blocks) == 0:
        issues.append("Markers present but no valid blocks could be extracted")

    for i, block in enumerate(blocks):
        if not block.search.strip():
            issues.append(f"Block {i + 1}: Empty search text")

    return len(issues) == 0, issues
