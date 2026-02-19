"""
Differential patch templates for incremental code changes.
"""

from madevolve.templates.foundation import format_code_block, format_metrics
from madevolve.transformer.blocks import has_evolve_blocks


def build_differential_prompt(
    parent_code: str,
    parent_score: float,
    parent_metrics: dict,
    inspirations: list = None,
    feedback: str = None,
    recommendations: str = None,
) -> str:
    """
    Build prompt for differential (SEARCH/REPLACE) patches.

    Args:
        parent_code: Current code to modify
        parent_score: Current score
        parent_metrics: Current metrics
        inspirations: Optional inspiration programs
        feedback: Optional evaluator feedback
        recommendations: Optional strategic recommendations

    Returns:
        Complete prompt string
    """
    sections = [
        "# Current Implementation",
        "",
        f"**Score:** {parent_score:.6f}",
        "",
        "**Metrics:**",
        format_metrics(parent_metrics, prefix="  "),
        "",
        format_code_block(parent_code),
        "",
    ]

    if feedback:
        sections.extend([
            "# Evaluator Feedback",
            "",
            feedback,
            "",
        ])

    if inspirations:
        sections.extend([
            "# Reference Implementations",
            "",
            "Consider these successful implementations for inspiration:",
            "",
        ])
        for i, insp in enumerate(inspirations[:3], 1):
            sections.extend([
                f"## Reference {i} (Score: {insp.get('score', 'N/A')})",
                "",
                format_code_block(insp.get('code', '')[:2000]),
                "",
            ])

    if recommendations:
        sections.extend([
            "# Strategic Recommendations",
            "",
            recommendations,
            "",
        ])

    sections.extend([
        "# Instructions",
        "",
        "Make targeted improvements using SEARCH/REPLACE blocks.",
        "Use this exact format (note: 7 angle-brackets, 7 equals signs):",
        "",
        "<DIFF>",
        "<<<<<<< SEARCH",
        "[exact code to find — must match the original verbatim]",
        "=======",
        "[replacement code]",
        ">>>>>>> REPLACE",
        "</DIFF>",
        "",
        "Guidelines:",
        "- The SEARCH text must match the original code exactly (including indentation)",
        "- Make specific, targeted changes — do not rewrite the entire program",
        "- Focus on improving the score",
        "- You can use multiple SEARCH/REPLACE blocks, each wrapped in <DIFF></DIFF>",
    ])

    if has_evolve_blocks(parent_code):
        sections.extend([
            "",
            "**Important:** Only make changes within the EVOLVE-BLOCK markers. "
            "Do not modify code outside the markers.",
        ])

    return "\n".join(sections)


DIFFERENTIAL_SYSTEM_PROMPT = """You are an expert code optimizer using differential patches.

Your task is to make targeted improvements to code using SEARCH/REPLACE blocks.

Rules:
1. The SEARCH text must match the original code EXACTLY (including whitespace and indentation)
2. Make focused changes that address specific improvements
3. You can provide multiple SEARCH/REPLACE blocks for different changes
4. Each change should be independent and targeted
5. Do NOT add any text between SEARCH/REPLACE blocks

You MUST use this exact format — note the 7 angle-brackets and 7 equals signs:

<DIFF>
<<<<<<< SEARCH
for i in range(n):
    result += data[i]
=======
result = sum(data[:n])
>>>>>>> REPLACE
</DIFF>"""
