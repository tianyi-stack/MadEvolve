"""
Model allocation strategies using multi-armed bandit algorithms.

This module provides adaptive model selection for optimal LLM utilization
based on historical performance data.
"""

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelStatistics:
    """Performance statistics for a model."""
    model_name: str
    total_queries: int = 0
    success_count: int = 0
    total_score: float = 0.0
    total_improvement: float = 0.0
    base_weight: float = 1.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_queries == 0:
            return 0.5  # Optimistic prior
        return self.success_count / self.total_queries

    @property
    def avg_score(self) -> float:
        """Calculate average score."""
        if self.total_queries == 0:
            return 0.0
        return self.total_score / self.total_queries

    @property
    def avg_improvement(self) -> float:
        """Calculate average improvement per query."""
        if self.total_queries == 0:
            return 0.0
        return self.total_improvement / self.total_queries


class ModelSelector(ABC):
    """
    Abstract base class for model selection strategies.

    Implements a multi-armed bandit approach to adaptively select
    the best LLM model based on historical performance.
    """

    def __init__(
        self,
        models: List[str],
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize the selector.

        Args:
            models: List of model names
            weights: Optional base weights for each model
        """
        self.models = models
        self.stats: Dict[str, ModelStatistics] = {}

        weights = weights or [1.0] * len(models)

        for model, weight in zip(models, weights):
            self.stats[model] = ModelStatistics(
                model_name=model,
                base_weight=weight,
            )

    @abstractmethod
    def select(self) -> str:
        """Select a model to use."""
        pass

    def record_outcome(
        self,
        model_name: str,
        success: bool,
        score: float = 0.0,
        improvement: float = 0.0,
    ):
        """
        Record the outcome of a query.

        Args:
            model_name: Name of the model used
            success: Whether the query was successful (improved)
            score: The score achieved
            improvement: Score improvement over parent
        """
        if model_name not in self.stats:
            self.stats[model_name] = ModelStatistics(model_name=model_name)

        stats = self.stats[model_name]
        stats.total_queries += 1
        stats.total_score += score
        stats.total_improvement += improvement

        if success:
            stats.success_count += 1

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all models."""
        return {
            name: {
                "queries": stats.total_queries,
                "success_rate": stats.success_rate,
                "avg_score": stats.avg_score,
                "avg_improvement": stats.avg_improvement,
            }
            for name, stats in self.stats.items()
        }

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            "models": self.models,
            "stats": {
                name: {
                    "total_queries": s.total_queries,
                    "success_count": s.success_count,
                    "total_score": s.total_score,
                    "total_improvement": s.total_improvement,
                    "base_weight": s.base_weight,
                }
                for name, s in self.stats.items()
            },
        }

    def restore_state(self, state: Dict[str, Any]):
        """Restore from serialized state."""
        if "stats" in state:
            for name, data in state["stats"].items():
                if name in self.stats:
                    self.stats[name].total_queries = data.get("total_queries", 0)
                    self.stats[name].success_count = data.get("success_count", 0)
                    self.stats[name].total_score = data.get("total_score", 0.0)
                    self.stats[name].total_improvement = data.get("total_improvement", 0.0)


class UCBSelector(ModelSelector):
    """
    Upper Confidence Bound (UCB1) model selector.

    Balances exploration and exploitation using the UCB1 algorithm:
    score = empirical_mean + c * sqrt(ln(total_pulls) / model_pulls)
    """

    def __init__(
        self,
        models: List[str],
        weights: Optional[List[float]] = None,
        exploration_factor: float = 1.0,
        decay_rate: float = 0.99,
        use_improvement: bool = True,
    ):
        """
        Initialize UCB selector.

        Args:
            models: List of model names
            weights: Optional base weights
            exploration_factor: UCB exploration parameter (c)
            decay_rate: Decay for exploration over time
            use_improvement: Use improvement as reward (vs success)
        """
        super().__init__(models, weights)
        self.exploration_factor = exploration_factor
        self.decay_rate = decay_rate
        self.use_improvement = use_improvement
        self._total_queries = 0

    def select(self) -> str:
        """Select model using UCB1 algorithm."""
        self._total_queries += 1

        # Ensure each model is tried at least once
        for model in self.models:
            if self.stats[model].total_queries == 0:
                return model

        # Calculate UCB scores
        ucb_scores = []
        c = self.exploration_factor * (self.decay_rate ** self._total_queries)

        for model in self.models:
            stats = self.stats[model]

            # Empirical mean
            if self.use_improvement:
                mean = stats.avg_improvement
            else:
                mean = stats.success_rate

            # Exploration bonus
            exploration = c * math.sqrt(
                math.log(self._total_queries) / stats.total_queries
            )

            # Apply base weight
            ucb = (mean + exploration) * stats.base_weight
            ucb_scores.append((model, ucb))

        # Select model with highest UCB score
        ucb_scores.sort(key=lambda x: x[1], reverse=True)
        return ucb_scores[0][0]


class ThompsonSelector(ModelSelector):
    """
    Thompson Sampling model selector.

    Uses Bayesian inference with Beta distributions to balance
    exploration and exploitation.
    """

    def __init__(
        self,
        models: List[str],
        weights: Optional[List[float]] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        """
        Initialize Thompson Sampling selector.

        Args:
            models: List of model names
            weights: Optional base weights
            prior_alpha: Beta prior alpha parameter
            prior_beta: Beta prior beta parameter
        """
        super().__init__(models, weights)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def select(self) -> str:
        """Select model using Thompson Sampling."""
        samples = []

        for model in self.models:
            stats = self.stats[model]

            # Posterior parameters
            alpha = self.prior_alpha + stats.success_count
            beta = self.prior_beta + (stats.total_queries - stats.success_count)

            # Sample from Beta distribution
            sample = random.betavariate(alpha, beta) * stats.base_weight
            samples.append((model, sample))

        # Select model with highest sample
        samples.sort(key=lambda x: x[1], reverse=True)
        return samples[0][0]


class EpsilonGreedySelector(ModelSelector):
    """
    Epsilon-Greedy model selector.

    Simple strategy that explores with probability epsilon and
    exploits the best-known model otherwise.
    """

    def __init__(
        self,
        models: List[str],
        weights: Optional[List[float]] = None,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ):
        """
        Initialize Epsilon-Greedy selector.

        Args:
            models: List of model names
            weights: Optional base weights
            epsilon: Initial exploration probability
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum epsilon value
        """
        super().__init__(models, weights)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self._current_epsilon = epsilon

    def select(self) -> str:
        """Select model using epsilon-greedy strategy."""
        # Decay epsilon
        self._current_epsilon = max(
            self.min_epsilon,
            self._current_epsilon * self.epsilon_decay,
        )

        # Explore with probability epsilon
        if random.random() < self._current_epsilon:
            return random.choice(self.models)

        # Exploit: select best-performing model
        best_model = None
        best_score = float("-inf")

        for model in self.models:
            stats = self.stats[model]
            if stats.total_queries > 0:
                score = stats.success_rate * stats.base_weight
                if score > best_score:
                    best_score = score
                    best_model = model

        return best_model or random.choice(self.models)

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["current_epsilon"] = self._current_epsilon
        return state

    def restore_state(self, state: Dict[str, Any]):
        super().restore_state(state)
        self._current_epsilon = state.get("current_epsilon", self.epsilon)


def create_selector(
    algorithm: str,
    models: List[str],
    weights: Optional[List[float]] = None,
    **kwargs,
) -> ModelSelector:
    """
    Factory function to create a model selector.

    Args:
        algorithm: Selection algorithm ("ucb", "thompson", "epsilon_greedy")
        models: List of model names
        weights: Optional base weights
        **kwargs: Additional algorithm-specific parameters

    Returns:
        ModelSelector instance
    """
    selectors = {
        "ucb": UCBSelector,
        "thompson": ThompsonSelector,
        "epsilon_greedy": EpsilonGreedySelector,
    }

    if algorithm not in selectors:
        raise ValueError(f"Unknown selector algorithm: {algorithm}")

    return selectors[algorithm](models, weights, **kwargs)
