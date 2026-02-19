"""
Bootstrap templates for initial program generation.
"""

from madevolve.templates.foundation import (
    BASE_SYSTEM_PROMPT,
    EVOLUTION_CONTEXT,
    format_code_block,
)


def build_initial_prompt(
    task_description: str,
    template_code: str = None,
    constraints: str = None,
) -> str:
    """
    Build prompt for initial program generation.

    Args:
        task_description: Description of the optimization task
        template_code: Optional template/skeleton code
        constraints: Optional constraints or requirements

    Returns:
        Complete prompt string
    """
    sections = [
        "# Task Description",
        "",
        task_description,
        "",
    ]

    if constraints:
        sections.extend([
            "# Constraints",
            "",
            constraints,
            "",
        ])

    if template_code:
        sections.extend([
            "# Template Code",
            "",
            "Use this as a starting point:",
            "",
            format_code_block(template_code),
            "",
        ])

    sections.extend([
        "# Instructions",
        "",
        "Generate a complete, working implementation that solves the described task.",
        "Focus on:",
        "- Correctness: The code must produce valid output",
        "- Efficiency: Optimize for performance where possible",
        "- Clarity: Write readable, maintainable code",
        "",
        "Provide your implementation in a Python code block.",
    ])

    return "\n".join(sections)


def get_initial_system_prompt(domain: str = None) -> str:
    """
    Get system prompt for initial generation.

    Args:
        domain: Optional domain-specific context

    Returns:
        System prompt string
    """
    prompt = BASE_SYSTEM_PROMPT + "\n\n" + EVOLUTION_CONTEXT

    if domain:
        prompt += f"\n\nDomain context: {domain}"

    return prompt


INITIAL_GENERATION_TEMPLATE = """# Initial Program Generation

## Task
{task_description}

## Requirements
Generate a complete Python program that:
1. Implements the core algorithm for the task
2. Is syntactically correct and can be executed
3. Produces output that can be evaluated

## Output Format
Provide your implementation as a complete Python code block:

```python
# Your implementation here
```

Make sure the code is self-contained and ready to run."""
