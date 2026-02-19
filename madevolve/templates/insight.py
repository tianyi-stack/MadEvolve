"""
Insight/analysis templates for strategic evolution guidance.
"""


def build_analysis_prompt(
    programs: list,
    global_context: str = None,
    analysis_stage: int = 1,
) -> str:
    """
    Build prompt for program analysis (stage 1 of reflection).

    Args:
        programs: List of programs to analyze
        global_context: Optional accumulated global context
        analysis_stage: Which stage of analysis (1, 2, or 3)

    Returns:
        Complete prompt string
    """
    if analysis_stage == 1:
        return _build_individual_analysis_prompt(programs)
    elif analysis_stage == 2:
        return _build_pattern_extraction_prompt(programs, global_context)
    else:
        return _build_recommendation_prompt(global_context)


def _build_individual_analysis_prompt(programs: list) -> str:
    """Build prompt for individual program analysis."""
    sections = [
        "# Individual Program Analysis",
        "",
        "Analyze each of these evolved programs briefly.",
        "Focus on identifying what makes each notable (good or bad).",
        "",
    ]

    for i, prog in enumerate(programs, 1):
        sections.extend([
            f"## Program {i}",
            f"Generation: {prog.get('generation', 'N/A')}",
            f"Score: {prog.get('score', 'N/A')}",
            "",
            f"```python\n{prog.get('code', '')[:1000]}\n```",
            "",
        ])

    sections.extend([
        "# Instructions",
        "",
        "For each program, provide a 1-2 sentence summary of what makes it notable.",
        "Focus on algorithmic choices, patterns, and potential issues.",
    ])

    return "\n".join(sections)


def _build_pattern_extraction_prompt(summaries: list, global_context: str) -> str:
    """Build prompt for pattern extraction."""
    sections = [
        "# Pattern Extraction",
        "",
    ]

    if global_context:
        sections.extend([
            "## Previous Insights",
            "",
            global_context,
            "",
        ])

    sections.extend([
        "## Recent Program Summaries",
        "",
    ])

    for summary in summaries:
        sections.append(f"- {summary}")

    sections.extend([
        "",
        "# Instructions",
        "",
        "Identify 2-3 key patterns or trends from these programs.",
        "",
        "For each pattern, provide:",
        "- Category (convergence, diversity, stagnation, breakthrough, etc.)",
        "- Description (1-2 sentences)",
        "- Confidence (high/medium/low)",
        "",
        "Format:",
        "PATTERN: [category]",
        "[description]",
        "CONFIDENCE: [level]",
    ])

    return "\n".join(sections)


def _build_recommendation_prompt(insights: str) -> str:
    """Build prompt for recommendation generation."""
    sections = [
        "# Strategic Recommendation Generation",
        "",
        "## Current Insights",
        "",
        insights if insights else "No previous insights available.",
        "",
        "# Instructions",
        "",
        "Based on these insights, generate 3-5 specific, actionable recommendations",
        "for improving the next generation of evolved programs.",
        "",
        "Each recommendation should be:",
        "- Specific and implementable",
        "- Directly addressing identified patterns",
        "- Focused on improving evolution outcomes",
        "",
        "Format as a numbered list.",
    ])

    return "\n".join(sections)


ANALYSIS_SYSTEM_PROMPT = """You are an evolution analyst studying patterns in evolved programs.

Your role is to:
1. Identify what makes programs successful or unsuccessful
2. Spot patterns and trends across generations
3. Generate actionable insights for improvement

Be concise and focus on actionable observations."""
