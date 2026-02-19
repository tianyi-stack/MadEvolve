"""
Prompt Composer - Builds context-rich prompts for LLM code generation.

This module constructs prompts that include parent code, inspirations,
and performance context.
"""

import logging
from typing import Any, Dict, List, Optional

from madevolve.repository.storage.artifact_store import ProgramRecord
from madevolve.transformer.blocks import extract_mutable_content, has_evolve_blocks

logger = logging.getLogger(__name__)


class PromptComposer:
    """
    Composes prompts for LLM-based code evolution.

    Builds context-rich prompts incorporating:
    - Parent program code and metrics
    - Inspiration programs from archive
    - Performance trends and feedback
    """

    def __init__(self, container):
        """
        Initialize the prompt composer.

        Args:
            container: ServiceContainer with configuration
        """
        self.config = container.config
        self._system_messages = {}
        self._init_system_messages()

    def _init_system_messages(self):
        """Initialize system messages for different patch modes."""
        base_system = """You are an expert programmer tasked with improving algorithms through evolutionary optimization.

Your role is to make targeted modifications to code that improve its performance while maintaining correctness.

Guidelines:
- Focus on algorithmic improvements, not just code style
- Consider computational efficiency and numerical stability
- Preserve the overall structure unless specifically asked to change it
- Make changes that can be validated through metrics"""

        self._system_messages["differential"] = base_system + """

You will provide changes in SEARCH/REPLACE format, wrapped in <DIFF> tags.
Use exactly 7 angle-brackets and 7 equals signs for the markers:

<DIFF>
<<<<<<< SEARCH
[exact code to find]
=======
[replacement code]
>>>>>>> REPLACE
</DIFF>"""

        self._system_messages["holistic"] = base_system + """

You will provide a complete rewrite of the code, keeping the same function signature."""

        self._system_messages["synthesis"] = base_system + """

You will synthesize ideas from multiple programs to create an improved version."""

    def get_system_message(self, patch_mode: str) -> str:
        """Get system message for a patch mode."""
        if self.config.system_prompt_override:
            return self.config.system_prompt_override
        return self._system_messages.get(patch_mode, self._system_messages["differential"])

    def compose(
        self,
        parent: ProgramRecord,
        archive_inspirations: List[ProgramRecord],
        top_k_inspirations: List[ProgramRecord],
        patch_mode: str,
        task_description: Optional[str] = None,
        diverse_inspirations: Optional[List[ProgramRecord]] = None,
    ) -> str:
        """
        Compose a complete prompt for code evolution.

        Args:
            parent: Parent program to evolve
            archive_inspirations: Programs from elite archive
            top_k_inspirations: Top-performing programs
            patch_mode: Type of patch to generate
            task_description: Description of the optimization task
            diverse_inspirations: Structurally diverse programs from MAP-Elites grid

        Returns:
            Complete prompt string
        """
        diverse_inspirations = diverse_inspirations or []
        sections = []

        # Task description
        if task_description:
            sections.append(self._format_task_section(task_description))

        # Parent program
        sections.append(self._format_parent_section(parent, patch_mode))

        # Inspirations
        if archive_inspirations or top_k_inspirations or diverse_inspirations:
            sections.append(self._format_inspirations_section(
                archive_inspirations,
                top_k_inspirations,
                diverse_inspirations,
            ))

        # Performance feedback
        if self.config.include_text_feedback and parent.text_feedback:
            sections.append(self._format_feedback_section(parent.text_feedback))

        # Instructions
        sections.append(self._format_instructions_section(
            patch_mode,
            has_blocks=has_evolve_blocks(parent.code),
        ))

        return "\n\n".join(sections)

    def _format_task_section(self, task_description: str) -> str:
        """Format the task description section."""
        return f"""## Task

{task_description}"""

    def _format_parent_section(self, parent: ProgramRecord, patch_mode: str) -> str:
        """Format the parent program section."""
        metrics_str = self._format_metrics(parent.public_metrics)

        section = f"""## Current Program (Generation {parent.generation})

**Score:** {parent.combined_score:.4f}
{metrics_str}

```python
{parent.code}
```"""

        if has_evolve_blocks(parent.code):
            section += "\n\n**Note:** Only modify code within the `EVOLVE-BLOCK` markers."

        return section

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics as a bullet list."""
        if not metrics:
            return ""

        lines = ["**Metrics:**"]
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    def _format_inspirations_section(
        self,
        archive: List[ProgramRecord],
        top_k: List[ProgramRecord],
        diverse: Optional[List[ProgramRecord]] = None,
    ) -> str:
        """Format the inspirations section."""
        diverse = diverse or []
        lines = ["## Reference Programs", ""]

        if archive:
            lines.append("### From Elite Archive")
            for i, prog in enumerate(archive[:3], 1):
                lines.append(f"\n**Reference {i}** (Score: {prog.combined_score:.4f})")
                code = extract_mutable_content(prog.code)
                lines.append(f"```python\n{self._truncate_code(code)}\n```")

        if top_k:
            lines.append("\n### Top Performers")
            for i, prog in enumerate(top_k[:3], 1):
                lines.append(f"\n**Top {i}** (Score: {prog.combined_score:.4f})")
                code = extract_mutable_content(prog.code)
                lines.append(f"```python\n{self._truncate_code(code)}\n```")

        if diverse:
            lines.append("\n### Structurally Diverse Neighbors")
            for i, prog in enumerate(diverse[:3], 1):
                lines.append(f"\n**Diverse {i}** (Score: {prog.combined_score:.4f})")
                code = extract_mutable_content(prog.code)
                lines.append(f"```python\n{self._truncate_code(code)}\n```")

        return "\n".join(lines)

    def _truncate_code(self, code: str, max_lines: int = 50) -> str:
        """Truncate code to maximum lines."""
        lines = code.split("\n")
        if len(lines) <= max_lines:
            return code

        return "\n".join(lines[:max_lines]) + "\n# ... (truncated)"

    def _format_feedback_section(self, feedback: str) -> str:
        """Format the feedback section."""
        return f"""## Evaluator Feedback

{feedback}"""

    def _format_instructions_section(self, patch_mode: str, has_blocks: bool = False) -> str:
        """Format the instructions section based on patch mode."""
        block_note = ""
        if has_blocks:
            block_note = "\n\n**Important:** Only modify code between the `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers. Do not modify code outside the markers."

        if patch_mode == "differential":
            return """## Instructions

Provide specific improvements using SEARCH/REPLACE blocks.
Wrap each block in <DIFF> tags:

<DIFF>
<<<<<<< SEARCH
[exact code to find and replace]
=======
[improved code]
>>>>>>> REPLACE
</DIFF>

Make targeted changes that improve the score while maintaining correctness.""" + block_note

        elif patch_mode == "holistic":
            if has_blocks:
                return """## Instructions

Provide only the new content for the evolve block region (the code between the
`EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers). Do NOT include the markers
themselves or any code outside the markers.

Reimagine the implementation to achieve better performance.

Wrap your code in ```python``` blocks."""
            return """## Instructions

Provide a complete rewrite of the program that improves upon the current version.

Keep the same function signature and interface, but reimagine the implementation
to achieve better performance.

Wrap your code in ```python``` blocks."""

        elif patch_mode == "synthesis":
            if has_blocks:
                return """## Instructions

Synthesize the best ideas from the reference programs to create an improved version.

Provide only the new content for the evolve block region (the code between the
`EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers). Do NOT include the markers
themselves or any code outside the markers.

Combine successful patterns and techniques while avoiding weaknesses.

Wrap your code in ```python``` blocks."""
            return """## Instructions

Synthesize the best ideas from the reference programs to create an improved version.

Combine successful patterns and techniques while avoiding weaknesses.

Provide the complete improved code in ```python``` blocks."""

        else:
            return """## Instructions

Provide improved code that achieves a better score.""" + block_note


class PerformanceContextBuilder:
    """
    Builds performance context from evolution history.

    Analyzes trends and identifies areas for improvement.
    """

    def __init__(self, window_size: int = 10):
        """
        Initialize the context builder.

        Args:
            window_size: Number of recent programs to analyze
        """
        self.window_size = window_size
        self._history: List[Dict[str, float]] = []

    def record(self, metrics: Dict[str, float]):
        """Record metrics from a program."""
        self._history.append(metrics.copy())
        if len(self._history) > self.window_size * 2:
            self._history = self._history[-self.window_size * 2:]

    def get_trend_context(self) -> str:
        """
        Get formatted context about metric trends.

        Returns:
            Trend analysis string
        """
        if len(self._history) < 2:
            return ""

        recent = self._history[-self.window_size:]
        earlier = self._history[-self.window_size * 2:-self.window_size] if len(self._history) > self.window_size else []

        if not earlier:
            return ""

        lines = ["Performance Trends:"]

        # Compute average metrics
        for key in recent[0].keys():
            recent_avg = sum(m.get(key, 0) for m in recent) / len(recent)
            earlier_avg = sum(m.get(key, 0) for m in earlier) / len(earlier) if earlier else recent_avg

            delta = recent_avg - earlier_avg
            if abs(delta) > 0.001:
                trend = "↑" if delta > 0 else "↓"
                lines.append(f"- {key}: {trend} ({delta:+.4f})")

        return "\n".join(lines) if len(lines) > 1 else ""
