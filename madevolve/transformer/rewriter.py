"""
Holistic Code Rewriter - Full code replacement from LLM output.

This module handles extraction and validation of complete code
rewrites from LLM responses.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def apply_holistic_rewrite(llm_output: str, has_evolve_block: bool = False) -> Optional[str]:
    """
    Extract complete code rewrite from LLM output.

    Args:
        llm_output: Raw LLM output containing code
        has_evolve_block: If True, the parent code uses evolve-block markers.
            Relaxes validation (no def/class requirement) and strips echoed markers.

    Returns:
        Extracted code or None if extraction fails
    """
    require_definition = not has_evolve_block

    # Try to find code block
    code = _extract_fenced_code(llm_output)

    if code:
        if has_evolve_block:
            code = _clean_evolve_markers(code)
        if _is_valid_code(code, require_definition=require_definition):
            return code

    # Fallback: try to extract any code-like content
    code = _extract_code_heuristic(llm_output)

    if code:
        if has_evolve_block:
            code = _clean_evolve_markers(code)
        if _is_valid_code(code, require_definition=require_definition):
            return code

    logger.warning("Failed to extract valid code from holistic rewrite")
    return None


def _extract_fenced_code(text: str) -> Optional[str]:
    """
    Extract code from markdown fenced blocks.

    Args:
        text: Text containing code blocks

    Returns:
        Extracted code or None
    """
    # Try python-specific blocks first (case-insensitive, allow python2/python3 suffix)
    python_pattern = r"```[Pp]ython[23]?\s*(.*?)\s*```"
    match = re.search(python_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code blocks (any language tag or none)
    generic_pattern = r"```\w*\s*(.*?)\s*```"
    match = re.search(generic_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try alternative fence markers
    alt_pattern = r"~~~\w*\s*(.*?)\s*~~~"
    match = re.search(alt_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def _clean_evolve_markers(code: str) -> str:
    """
    Strip EVOLVE-BLOCK marker lines that LLMs sometimes echo from the prompt.

    Removes lines matching both formats:
    - ``# EVOLVE-BLOCK-START``
    - ``# === EVOLVE-BLOCK-START ===``
    """
    marker_pattern = re.compile(
        r"^\s*#\s*(?:===\s*)?EVOLVE-BLOCK-(START|END)(?:\s*===)?\s*$"
    )
    lines = code.split("\n")
    cleaned = [line for line in lines if not marker_pattern.match(line)]
    return "\n".join(cleaned)


def _extract_code_heuristic(text: str) -> Optional[str]:
    """
    Extract code using heuristics when fenced blocks fail.

    Args:
        text: Text possibly containing code

    Returns:
        Extracted code or None
    """
    lines = text.split("\n")

    # Find first line that looks like code
    start_idx = None
    end_idx = None

    code_indicators = [
        "def ", "class ", "import ", "from ", "if ", "for ", "while ",
        "return ", "yield ", "async ", "await ", "with ", "#",
    ]

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip obvious non-code lines
        if not stripped:
            continue

        is_code = any(stripped.startswith(ind) for ind in code_indicators)
        is_code = is_code or (stripped.startswith("@") and not stripped.startswith("@@"))
        is_code = is_code or re.match(r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=", stripped)

        if is_code and start_idx is None:
            start_idx = i

        if start_idx is not None:
            end_idx = i

    if start_idx is not None and end_idx is not None:
        code_lines = lines[start_idx:end_idx + 1]
        return "\n".join(code_lines)

    return None


def _is_valid_code(code: str, require_definition: bool = True) -> bool:
    """
    Check if text is valid, non-degenerate Python code.

    Validates that the code:
    1. Is syntactically valid Python (compile() must pass)
    2. Optionally contains at least one function or class definition

    Args:
        code: Source code to validate.
        require_definition: When True (default), reject code that lacks
            ``def`` or ``class`` definitions.  Set to False for evolve-block
            fragments that may be pure expressions or TUNABLE param blocks.
    """
    if not code or len(code.strip()) < 10:
        return False

    # Hard requirement: must be syntactically valid Python
    try:
        compile(code, "<string>", "exec")
    except SyntaxError:
        return False

    if require_definition:
        # Must contain at least one function or class definition.
        # Reject comment-only or assignment-only outputs (degenerate LLM responses).
        has_definition = re.search(r"\bdef\s+\w+\s*\(", code, re.MULTILINE)
        if not has_definition:
            has_definition = re.search(r"\bclass\s+\w+", code, re.MULTILINE)

        if not has_definition:
            logger.warning("Code has no function/class definitions, rejecting as degenerate output")
            return False

    return True


def rewrite_with_preservation(
    original_code: str,
    llm_output: str,
    preserve_imports: bool = True,
    preserve_docstrings: bool = True,
    has_evolve_block: bool = False,
) -> Optional[str]:
    """
    Apply holistic rewrite while preserving specified elements.

    Args:
        original_code: Original source code
        llm_output: LLM output with new code
        preserve_imports: Keep original imports
        preserve_docstrings: Keep original docstrings
        has_evolve_block: Forward to apply_holistic_rewrite for relaxed validation

    Returns:
        Rewritten code with preserved elements
    """
    new_code = apply_holistic_rewrite(llm_output, has_evolve_block=has_evolve_block)

    if not new_code:
        return None

    result_parts = []

    if preserve_imports:
        # Extract imports from original
        original_imports = _extract_imports(original_code)
        new_imports = _extract_imports(new_code)

        # Merge imports
        all_imports = set(original_imports) | set(new_imports)
        result_parts.extend(sorted(all_imports))

        # Remove imports from new_code for rest
        new_code = _remove_imports(new_code)

    if preserve_docstrings:
        # Extract module docstring from original if present
        module_doc = _extract_module_docstring(original_code)
        if module_doc:
            result_parts.insert(0, module_doc)

    result_parts.append(new_code)

    return "\n\n".join(result_parts)


def _extract_imports(code: str) -> list:
    """Extract import statements from code."""
    import_pattern = r"^(?:from\s+[\w.]+\s+)?import\s+.+$"
    return re.findall(import_pattern, code, re.MULTILINE)


def _remove_imports(code: str) -> str:
    """Remove import statements from code."""
    lines = code.split("\n")
    result = []
    for line in lines:
        if not re.match(r"^(?:from\s+[\w.]+\s+)?import\s+", line.strip()):
            result.append(line)

    # Remove leading blank lines
    while result and not result[0].strip():
        result.pop(0)

    return "\n".join(result)


def _extract_module_docstring(code: str) -> Optional[str]:
    """Extract module-level docstring."""
    match = re.match(r'^(""".*?"""|\'\'\'.*?\'\'\')', code, re.DOTALL)
    if match:
        return match.group(1)
    return None
