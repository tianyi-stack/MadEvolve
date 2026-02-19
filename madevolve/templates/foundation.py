"""
Foundation templates and base prompts for MadEvolve.
"""

BASE_SYSTEM_PROMPT = """You are an expert programmer specializing in algorithm optimization.

Your task is to improve code to maximize a performance metric while maintaining correctness.

Guidelines:
- Focus on algorithmic improvements over cosmetic changes
- Consider computational efficiency and numerical stability
- Maintain code readability and structure
- Test your changes mentally before proposing them
- Be specific about what you're changing and why"""

EVOLUTION_CONTEXT = """You are participating in an evolutionary code optimization process.
Programs that achieve higher scores on the evaluation metric will be selected for further improvement.
Your modifications should aim to improve the score while maintaining program correctness."""

CODE_QUALITY_GUIDELINES = """Code Quality Guidelines:
- Use clear, descriptive variable names
- Add comments for complex logic
- Handle edge cases appropriately
- Avoid unnecessary complexity
- Follow Python best practices"""

PERFORMANCE_FOCUS = """Performance Optimization Focus:
- Minimize computational complexity where possible
- Use efficient data structures
- Avoid redundant calculations
- Consider numerical precision
- Profile-guided optimization when applicable"""


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format metrics dictionary as readable string."""
    if not metrics:
        return f"{prefix}No metrics available"

    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.6f}")
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)


def format_code_block(code: str, language: str = "python") -> str:
    """Format code as markdown code block."""
    return f"```{language}\n{code}\n```"


def truncate_code(code: str, max_lines: int = 100) -> str:
    """Truncate code to maximum lines."""
    lines = code.split("\n")
    if len(lines) <= max_lines:
        return code

    return "\n".join(lines[:max_lines]) + "\n# ... (truncated)"


def format_code_block_for_evolve(code: str, show_full: bool = True) -> str:
    """Format code for prompt, optionally extracting only the mutable block."""
    from madevolve.transformer.blocks import format_code_for_prompt
    return format_code_block(format_code_for_prompt(code, show_full=show_full))
