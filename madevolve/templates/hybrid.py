"""
Hybrid/synthesis templates for combining multiple approaches.
"""

from madevolve.templates.foundation import format_code_block, format_metrics
from madevolve.transformer.blocks import has_evolve_blocks, extract_mutable_content


def build_synthesis_prompt(
    parent_code: str,
    parent_score: float,
    inspirations: list,
    task_description: str = None,
) -> str:
    """
    Build prompt for synthesis (crossover) of multiple implementations.

    Args:
        parent_code: Base code to build upon
        parent_score: Base score
        inspirations: List of inspiration programs to combine
        task_description: Optional task description

    Returns:
        Complete prompt string
    """
    sections = []

    if task_description:
        sections.extend([
            "# Task",
            "",
            task_description,
            "",
        ])

    sections.extend([
        "# Base Implementation",
        "",
        f"**Score:** {parent_score:.6f}",
        "",
        format_code_block(parent_code),
        "",
        "# Implementations to Synthesize",
        "",
        "Combine the best ideas from these successful implementations:",
        "",
    ])

    for i, insp in enumerate(inspirations[:4], 1):
        insp_code = insp.get('code', '')
        insp_code = extract_mutable_content(insp_code)[:1500]
        sections.extend([
            f"## Implementation {i} (Score: {insp.get('score', 'N/A')})",
            "",
            format_code_block(insp_code),
            "",
        ])

    if has_evolve_blocks(parent_code):
        sections.extend([
            "# Instructions",
            "",
            "Create a new implementation that synthesizes the best ideas from all sources.",
            "",
            "Provide only the new content for the evolve block region (the code between",
            "the `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers).",
            "Do NOT include the markers themselves or any code outside the markers.",
            "",
            "Approach:",
            "1. Identify the key strengths of each implementation",
            "2. Find ways to combine these strengths",
            "3. Avoid incorporating weaknesses",
            "4. Create something better than any individual source",
            "",
            "Provide your synthesized implementation in a Python code block.",
        ])
    else:
        sections.extend([
            "# Instructions",
            "",
            "Create a new implementation that synthesizes the best ideas from all sources.",
            "",
            "Approach:",
            "1. Identify the key strengths of each implementation",
            "2. Find ways to combine these strengths",
            "3. Avoid incorporating weaknesses",
            "4. Create something better than any individual source",
            "",
            "The result should be a complete, working implementation.",
            "",
            "Provide your synthesized implementation in a Python code block.",
        ])

    return "\n".join(sections)


SYNTHESIS_SYSTEM_PROMPT = """You are an expert at combining the best ideas from multiple implementations.

Your task is to synthesize a superior implementation by:
1. Identifying the strengths of each source
2. Finding compatible ways to combine them
3. Avoiding weaknesses and conflicts
4. Creating emergent improvements

The result should outperform any individual source implementation.

Output your synthesized code in a Python code block."""


def build_crossover_prompt(
    parent_a: dict,
    parent_b: dict,
    crossover_points: list = None,
) -> str:
    """
    Build prompt for crossover between two parents.

    Args:
        parent_a: First parent (code, score)
        parent_b: Second parent (code, score)
        crossover_points: Optional suggested crossover points

    Returns:
        Complete prompt string
    """
    code_a = extract_mutable_content(parent_a.get('code', ''))
    code_b = extract_mutable_content(parent_b.get('code', ''))
    sections = [
        "# Crossover Operation",
        "",
        "Combine these two implementations to create an improved offspring.",
        "",
        f"## Parent A (Score: {parent_a.get('score', 'N/A')})",
        "",
        format_code_block(code_a),
        "",
        f"## Parent B (Score: {parent_b.get('score', 'N/A')})",
        "",
        format_code_block(code_b),
        "",
    ]

    if crossover_points:
        sections.extend([
            "## Suggested Crossover Points",
            "",
        ])
        for point in crossover_points:
            sections.append(f"- {point}")
        sections.append("")

    sections.extend([
        "# Instructions",
        "",
        "Create a child implementation that inherits beneficial traits from both parents.",
        "",
        "The child should:",
        "- Combine the strengths of both parents",
        "- Be a complete, working implementation",
        "- Potentially improve upon both parents",
        "",
        "Provide your implementation in a Python code block.",
    ])

    return "\n".join(sections)
