"""
Holistic rewrite templates for complete code regeneration.
"""

from madevolve.templates.foundation import format_code_block, format_metrics
from madevolve.transformer.blocks import has_evolve_blocks, extract_mutable_content


def build_holistic_prompt(
    parent_code: str,
    parent_score: float,
    parent_metrics: dict,
    task_description: str = None,
    inspirations: list = None,
    feedback: str = None,
) -> str:
    """
    Build prompt for holistic (complete) code rewrite.

    Args:
        parent_code: Current code to improve
        parent_score: Current score
        parent_metrics: Current metrics
        task_description: Optional task description
        inspirations: Optional inspiration programs
        feedback: Optional evaluator feedback

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
        "# Current Implementation",
        "",
        f"**Score:** {parent_score:.6f}",
        "",
        "**Metrics:**",
        format_metrics(parent_metrics, prefix="  "),
        "",
        format_code_block(parent_code),
        "",
    ])

    if feedback:
        sections.extend([
            "# Evaluator Feedback",
            "",
            feedback,
            "",
        ])

    if inspirations:
        sections.extend([
            "# High-Performing Alternatives",
            "",
            "These implementations achieved higher scores:",
            "",
        ])
        for i, insp in enumerate(inspirations[:2], 1):
            insp_code = insp.get('code', '')
            insp_code = extract_mutable_content(insp_code)[:2000]
            sections.extend([
                f"## Alternative {i} (Score: {insp.get('score', 'N/A')})",
                "",
                format_code_block(insp_code),
                "",
            ])

    if has_evolve_blocks(parent_code):
        sections.extend([
            "# Instructions",
            "",
            "Provide only the new content for the evolve block region (the code between",
            "the `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers).",
            "Do NOT include the markers themselves or any code outside the markers.",
            "",
            "You may:",
            "- Restructure the algorithm within the block entirely",
            "- Change data structures and approaches",
            "- Combine ideas from the alternatives",
            "- Try novel approaches not seen before",
            "",
            "Requirements:",
            "- Only rewrite the mutable block content",
            "- The code must be complete and executable when reassembled",
            "- Focus on maximizing the score",
            "",
            "Provide your implementation in a Python code block.",
        ])
    else:
        sections.extend([
            "# Instructions",
            "",
            "Provide a complete rewrite of the implementation that improves the score.",
            "",
            "You may:",
            "- Restructure the algorithm entirely",
            "- Change data structures and approaches",
            "- Combine ideas from the alternatives",
            "- Try novel approaches not seen before",
            "",
            "Requirements:",
            "- Maintain the same function signature/interface",
            "- The code must be complete and executable",
            "- Focus on maximizing the score",
            "",
            "Provide your complete implementation in a Python code block.",
        ])

    return "\n".join(sections)


HOLISTIC_SYSTEM_PROMPT = """You are an expert algorithm designer tasked with reimagining implementations.

Your goal is to create a completely new approach that outperforms the current implementation.

Guidelines:
- You have freedom to completely restructure the code
- Focus on algorithmic innovation, not just code style
- The function signature must remain compatible
- Provide complete, working code

Output your improved implementation in a Python code block."""


HOLISTIC_VARIANTS = {
    "conservative": """
Focus on incremental improvements while maintaining the overall structure.
Make changes that are likely to improve performance without introducing risk.""",

    "aggressive": """
Consider radical restructuring of the algorithm.
Don't be afraid to try completely different approaches.""",

    "hybrid": """
Combine the best elements from multiple approaches.
Look for synergies between different techniques.""",
}
