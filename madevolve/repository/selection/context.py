"""
Inspiration selection for prompt context building.

This module provides utilities for selecting programs to include
as inspiration/examples in LLM prompts.
"""

import logging
import random
from typing import List, Optional

from madevolve.repository.storage.artifact_store import ArtifactStore, ProgramRecord

logger = logging.getLogger(__name__)


class InspirationSelector:
    """
    Selects programs to serve as inspirations for LLM prompts.

    Balances between high-performing programs and diverse examples.
    """

    def __init__(
        self,
        elite_ratio: float = 0.7,
        diversity_weight: float = 0.3,
    ):
        """
        Initialize the inspiration selector.

        Args:
            elite_ratio: Proportion of inspirations from elite archive
            diversity_weight: Weight for diversity in selection
        """
        self.elite_ratio = elite_ratio
        self.diversity_weight = diversity_weight

    def select_inspirations(
        self,
        artifact_store: ArtifactStore,
        count: int,
        exclude_ids: Optional[List[str]] = None,
        parent_embedding: Optional[List[float]] = None,
    ) -> List[ProgramRecord]:
        """
        Select inspiration programs.

        Args:
            artifact_store: Program storage
            count: Number of inspirations to select
            exclude_ids: Program IDs to exclude
            parent_embedding: Parent embedding for diversity calculation

        Returns:
            List of selected programs
        """
        exclude_ids = exclude_ids or []

        # Get candidate pool
        candidates = artifact_store.get_top_programs(count * 3)
        candidates = [p for p in candidates if p.program_id not in exclude_ids]

        if not candidates:
            return []

        # Score candidates
        scored = []
        for program in candidates:
            score = self._compute_inspiration_score(
                program,
                parent_embedding,
                artifact_store,
            )
            scored.append((program, score))

        # Sort by score and select top
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in scored[:count]]

        return selected

    def _compute_inspiration_score(
        self,
        program: ProgramRecord,
        parent_embedding: Optional[List[float]],
        artifact_store: ArtifactStore,
    ) -> float:
        """
        Compute inspiration score for a program.

        Combines performance and diversity considerations.
        """
        # Performance component
        perf_score = program.combined_score

        # Diversity component (if parent embedding available)
        diversity_score = 0.0
        if parent_embedding and program.embedding:
            from madevolve.common.helpers import cosine_similarity
            similarity = cosine_similarity(parent_embedding, program.embedding)
            diversity_score = 1.0 - similarity  # Higher = more diverse

        # Combine scores
        combined = (
            (1.0 - self.diversity_weight) * perf_score +
            self.diversity_weight * diversity_score
        )

        return combined


class PatternTracker:
    """
    Tracks successful modification patterns across evolution.

    Identifies patterns that tend to produce improvements.
    """

    def __init__(self):
        self._patterns: dict = {}
        self._pattern_success: dict = {}

    def record_pattern(
        self,
        pattern_type: str,
        success: bool,
        improvement: float = 0.0,
    ):
        """Record a pattern's success."""
        if pattern_type not in self._patterns:
            self._patterns[pattern_type] = 0
            self._pattern_success[pattern_type] = 0.0

        self._patterns[pattern_type] += 1
        if success:
            self._pattern_success[pattern_type] += improvement

    def get_successful_patterns(self, min_uses: int = 3) -> List[str]:
        """Get patterns that have been successful."""
        successful = []

        for pattern, uses in self._patterns.items():
            if uses >= min_uses:
                avg_improvement = self._pattern_success[pattern] / uses
                if avg_improvement > 0:
                    successful.append(pattern)

        return successful

    def format_patterns_context(self) -> str:
        """Format patterns as context for prompts."""
        successful = self.get_successful_patterns()

        if not successful:
            return ""

        lines = ["Historically successful modification patterns:"]
        for pattern in successful:
            lines.append(f"- {pattern}")

        return "\n".join(lines)
