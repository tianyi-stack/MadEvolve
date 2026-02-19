"""
Parent selection strategies for evolutionary algorithms.

This module provides various strategies for selecting parent programs
to mutate in the evolutionary process.
"""

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from madevolve.repository.storage.artifact_store import ArtifactStore, ProgramRecord

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of parent selection."""
    parent: ProgramRecord
    archive_inspirations: List[ProgramRecord]
    top_k_inspirations: List[ProgramRecord]
    diverse_inspirations: List[ProgramRecord] = None

    def __post_init__(self):
        if self.diverse_inspirations is None:
            self.diverse_inspirations = []


class SelectionStrategy(ABC):
    """Abstract base class for parent selection strategies."""

    @abstractmethod
    def select(
        self,
        candidates: List[ProgramRecord],
        generation: int,
    ) -> ProgramRecord:
        """Select a parent from candidates."""
        pass


class PowerLawSelection(SelectionStrategy):
    """
    Power law (rank-based) selection strategy.

    Higher-ranked programs have proportionally higher selection probability.
    P(i) ∝ rank(i)^(-alpha)
    """

    def __init__(self, alpha: float = 2.0):
        """
        Initialize power law selection.

        Args:
            alpha: Power law exponent (higher = more exploitation)
        """
        self.alpha = alpha

    def select(
        self,
        candidates: List[ProgramRecord],
        generation: int,
    ) -> ProgramRecord:
        if not candidates:
            raise ValueError("No candidates for selection")

        if len(candidates) == 1:
            return candidates[0]

        # Sort by score (descending)
        sorted_candidates = sorted(
            candidates,
            key=lambda p: p.combined_score,
            reverse=True,
        )

        # Compute power law weights
        weights = [(i + 1) ** (-self.alpha) for i in range(len(sorted_candidates))]
        total = sum(weights)
        probabilities = [w / total for w in weights]

        # Sample
        return random.choices(sorted_candidates, weights=probabilities)[0]


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection strategy.

    Randomly samples k candidates and selects the best.
    """

    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.

        Args:
            tournament_size: Number of candidates per tournament
        """
        self.tournament_size = tournament_size

    def select(
        self,
        candidates: List[ProgramRecord],
        generation: int,
    ) -> ProgramRecord:
        if not candidates:
            raise ValueError("No candidates for selection")

        k = min(self.tournament_size, len(candidates))
        tournament = random.sample(candidates, k)

        return max(tournament, key=lambda p: p.combined_score)


class AdaptiveSelection(SelectionStrategy):
    """
    Adaptive selection that combines exploitation and exploration.

    Dynamically adjusts selection pressure based on evolution progress.
    """

    def __init__(
        self,
        exploitation_ratio: float = 0.6,
        pressure_decay: float = 0.995,
    ):
        """
        Initialize adaptive selection.

        Args:
            exploitation_ratio: Base exploitation probability
            pressure_decay: Decay rate for selection pressure
        """
        self.exploitation_ratio = exploitation_ratio
        self.pressure_decay = pressure_decay
        self._generation = 0

    def select(
        self,
        candidates: List[ProgramRecord],
        generation: int,
    ) -> ProgramRecord:
        if not candidates:
            raise ValueError("No candidates for selection")

        self._generation = generation

        # Adjust exploitation ratio based on generation
        current_ratio = self.exploitation_ratio * (self.pressure_decay ** generation)

        if random.random() < current_ratio:
            # Exploit: select from top performers
            sorted_candidates = sorted(
                candidates,
                key=lambda p: p.combined_score,
                reverse=True,
            )
            top_n = max(1, len(candidates) // 4)
            return random.choice(sorted_candidates[:top_n])
        else:
            # Explore: uniform random selection
            return random.choice(candidates)


class ParentSelector:
    """
    Main parent selector that coordinates with population manager.

    Provides the primary interface for selecting parents and
    gathering inspiration programs.
    """

    def __init__(self, container):
        """
        Initialize the parent selector.

        Args:
            container: ServiceContainer with configuration
        """
        self.config = container.config.population
        self._strategy = self._create_strategy()
        self._inspiration_selector = None

    def _create_strategy(self) -> SelectionStrategy:
        """Create the selection strategy based on config."""
        strategy_name = self.config.selection_strategy

        if strategy_name == "power_law":
            return PowerLawSelection(alpha=self.config.selection_pressure)
        elif strategy_name == "tournament":
            return TournamentSelection(
                tournament_size=int(self.config.selection_pressure)
            )
        elif strategy_name == "adaptive":
            return AdaptiveSelection(
                exploitation_ratio=self.config.exploitation_ratio,
            )
        else:
            logger.warning(f"Unknown strategy '{strategy_name}', using adaptive")
            return AdaptiveSelection()

    def sample(
        self,
        generation: int,
        artifact_store: ArtifactStore,
        population,  # HybridPopulationManager
    ) -> SelectionResult:
        """
        Sample a parent and inspiration programs.

        Args:
            generation: Current generation number
            artifact_store: Program storage
            population: Population manager

        Returns:
            SelectionResult with parent and inspirations
        """
        # Get candidate pool from population
        candidates = population.get_selection_pool(generation)

        if not candidates:
            # Fallback to top programs from store
            candidates = artifact_store.get_top_programs(20)

        if not candidates:
            raise RuntimeError("No programs available for selection")

        # Select parent
        parent = self._strategy.select(candidates, generation)

        # Get archive inspirations
        archive_inspirations = self._sample_archive_inspirations(
            artifact_store,
            population,
            parent,
        )

        # Get top-k inspirations
        top_k_inspirations = self._sample_top_k_inspirations(
            artifact_store,
            population,
            parent,
        )

        # Get diverse inspirations from MAP-Elites grid
        diverse_inspirations = self._sample_diverse_inspirations(
            artifact_store,
            population,
            parent,
        )

        return SelectionResult(
            parent=parent,
            archive_inspirations=archive_inspirations,
            top_k_inspirations=top_k_inspirations,
            diverse_inspirations=diverse_inspirations,
        )

    def _sample_archive_inspirations(
        self,
        artifact_store: ArtifactStore,
        population,
        parent: ProgramRecord,
    ) -> List[ProgramRecord]:
        """Sample programs from the elite archive."""
        count = self.config.archive_inspiration_count

        # Get elite programs
        elite_ids = population.get_elite_ids()
        inspirations = []

        for pid in elite_ids:
            if pid != parent.program_id and len(inspirations) < count:
                program = artifact_store.get(pid)
                if program:
                    inspirations.append(program)

        return inspirations

    def _sample_top_k_inspirations(
        self,
        artifact_store: ArtifactStore,
        population,
        parent: ProgramRecord,
    ) -> List[ProgramRecord]:
        """Sample top-performing programs."""
        count = self.config.top_k_inspiration_count
        top_programs = artifact_store.get_top_programs(count * 2)

        inspirations = [
            p for p in top_programs
            if p.program_id != parent.program_id
        ][:count]

        return inspirations

    def _sample_diverse_inspirations(
        self,
        artifact_store: ArtifactStore,
        population,
        parent: ProgramRecord,
    ) -> List[ProgramRecord]:
        """Sample structurally diverse programs from the MAP-Elites grid."""
        count = self.config.diverse_inspiration_count
        if count <= 0:
            return []

        diverse_ids = population.sample_diverse(count)

        # Exclude parent and already-selected elites/top-k
        inspirations = []
        for pid in diverse_ids:
            if pid != parent.program_id and len(inspirations) < count:
                program = artifact_store.get(pid)
                if program:
                    inspirations.append(program)

        return inspirations
