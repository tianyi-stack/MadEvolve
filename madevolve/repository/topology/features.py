"""
Feature extraction for MAP-Elites behavioral characterization.

Computes feature descriptors that place programs into the MAP-Elites grid,
following the approach used by OpenEvolve:
- complexity: code length (character count)
- diversity:  average cosine distance to a reference set of embeddings
- performance: combined fitness score (pass-through)

Custom evaluator metrics are also supported as feature dimensions.
"""

import logging
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from madevolve.common.helpers import cosine_similarity

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts behavioral features for MAP-Elites grid placement.

    Maintains a reference set of program embeddings for diversity
    computation with an LRU cache to avoid redundant calculations.
    """

    def __init__(
        self,
        dimensions: List[str],
        reference_set_size: int = 20,
        cache_size: int = 1000,
    ):
        """
        Initialize the feature extractor.

        Args:
            dimensions: Feature dimension names to compute.
                        Built-in: "complexity", "diversity", "performance".
                        Any other name is looked up in evaluator metrics.
            reference_set_size: Number of embeddings kept for diversity
                                computation (reservoir-sampled).
            cache_size: Maximum entries in the diversity LRU cache.
        """
        self.dimensions = dimensions
        self.reference_set_size = reference_set_size
        self.cache_size = cache_size

        # Reservoir of representative embeddings for diversity calculation
        self._reference_embeddings: List[List[float]] = []
        self._programs_seen: int = 0

        # LRU cache: program_id -> diversity score
        self._diversity_cache: OrderedDict[str, float] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_features(
        self,
        code: str,
        embedding: Optional[List[float]],
        score: float,
        program_id: str,
        evaluator_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute all configured feature dimensions for a program.

        Args:
            code: Full program source code.
            embedding: Code embedding vector (from Vectorizer).
            score: Combined fitness score.
            program_id: Unique program identifier (for caching).
            evaluator_metrics: Optional dict of evaluator-reported metrics
                               that can serve as custom feature dimensions.

        Returns:
            Dict mapping dimension name -> feature value.
        """
        evaluator_metrics = evaluator_metrics or {}
        features: Dict[str, float] = {}

        for dim in self.dimensions:
            if dim == "complexity":
                features[dim] = self._compute_complexity(code)
            elif dim == "diversity":
                features[dim] = self._compute_diversity(
                    embedding, program_id,
                )
            elif dim == "performance":
                features[dim] = score
            elif dim in evaluator_metrics:
                # Custom evaluator metric used as feature dimension
                features[dim] = float(evaluator_metrics[dim])
            else:
                logger.debug(
                    f"Feature dimension '{dim}' not found in built-ins "
                    f"or evaluator metrics; defaulting to 0.0"
                )
                features[dim] = 0.0

        # Update the reference set *after* computing diversity so the
        # program is not compared against itself.
        if embedding is not None:
            self._update_reference_set(embedding)

        return features

    # ------------------------------------------------------------------
    # Built-in feature computations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_complexity(code: str) -> float:
        """
        Compute code complexity as character count.

        This mirrors OpenEvolve's approach: raw ``len(code)`` is used as
        the complexity value.  The MAP-Elites grid normalizes and bins it
        automatically via observed-range tracking.
        """
        return float(len(code))

    def _compute_diversity(
        self,
        embedding: Optional[List[float]],
        program_id: str,
    ) -> float:
        """
        Compute diversity as mean cosine *distance* to the reference set.

        Returns a value in [0, 1] where 1 = maximally diverse (orthogonal
        to every reference program) and 0 = identical.

        Falls back to 1.0 (maximally diverse) when the reference set is
        empty or no embedding is available.
        """
        if embedding is None or not self._reference_embeddings:
            return 1.0

        # Check cache
        if program_id in self._diversity_cache:
            self._diversity_cache.move_to_end(program_id)
            return self._diversity_cache[program_id]

        # Compute mean cosine distance
        total_distance = 0.0
        for ref_emb in self._reference_embeddings:
            similarity = cosine_similarity(embedding, ref_emb)
            total_distance += 1.0 - similarity

        diversity = total_distance / len(self._reference_embeddings)

        # Store in LRU cache
        self._diversity_cache[program_id] = diversity
        if len(self._diversity_cache) > self.cache_size:
            self._diversity_cache.popitem(last=False)

        return diversity

    # ------------------------------------------------------------------
    # Reference set maintenance (reservoir sampling)
    # ------------------------------------------------------------------

    def _update_reference_set(self, embedding: List[float]) -> None:
        """
        Maintain the reference set via reservoir sampling.

        The first ``reference_set_size`` embeddings are added directly;
        thereafter each new embedding replaces a random existing one with
        probability ``reference_set_size / programs_seen``.
        """
        self._programs_seen += 1

        if len(self._reference_embeddings) < self.reference_set_size:
            self._reference_embeddings.append(embedding)
        else:
            # Reservoir sampling (Algorithm R)
            j = random.randrange(self._programs_seen)
            if j < self.reference_set_size:
                self._reference_embeddings[j] = embedding
                # Invalidate cache since reference set changed
                self._diversity_cache.clear()

    # ------------------------------------------------------------------
    # Serialization for checkpointing
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return serializable state for checkpointing."""
        return {
            "reference_embeddings": self._reference_embeddings,
            "programs_seen": self._programs_seen,
        }

    def restore_state(self, state: dict) -> None:
        """Restore from a previously checkpointed state."""
        self._reference_embeddings = state.get("reference_embeddings", [])
        self._programs_seen = state.get("programs_seen", 0)
        self._diversity_cache.clear()
