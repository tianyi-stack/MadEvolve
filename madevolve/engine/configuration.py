"""
Configuration dataclasses for MadEvolve evolution engine.

This module provides comprehensive configuration options for all aspects
of the evolutionary process, from LLM settings to population management.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class PatchMode(str, Enum):
    """Available patch generation modes."""
    DIFFERENTIAL = "differential"  # Search/Replace diff patches
    HOLISTIC = "holistic"          # Full code rewrite
    SYNTHESIS = "synthesis"        # Crossover/combination of multiple programs


@dataclass
class PatchPolicy:
    """
    Configuration for patch type selection during evolution.

    The weights determine the probability of selecting each patch mode.
    Adaptive mode adjusts weights based on evolution progress.
    """
    modes: List[str] = field(default_factory=lambda: ["differential", "holistic"])
    weights: List[float] = field(default_factory=lambda: [0.7, 0.3])
    adaptive: bool = True
    stagnation_threshold: int = 10  # Generations without improvement before adaptation
    stagnation_boost: float = 0.15  # Boost for holistic mode on stagnation
    max_patch_retries: int = 3      # Max retry attempts with error feedback on extraction failure


@dataclass
class ModelConfig:
    """
    Configuration for LLM models used in evolution.

    Supports multiple models with weighted selection using bandit algorithms
    for adaptive model selection based on performance.
    """
    models: List[str] = field(default_factory=lambda: ["gpt-4o-mini"])
    weights: List[float] = field(default_factory=lambda: [1.0])
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 1200.0

    # Adaptive selection settings
    adaptive_selection: bool = True
    selection_algorithm: str = "ucb"  # Options: "ucb", "thompson", "epsilon_greedy"
    exploration_factor: float = 1.0   # UCB exploration parameter
    decay_rate: float = 0.99          # Exploration decay

    # Provider-specific settings
    provider_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    """Configuration for code embedding generation."""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 32
    cache_embeddings: bool = True


@dataclass
class PartitionConfig:
    """
    Configuration for population partitioning (MAP-Elites style).

    Defines the behavioral dimensions and grid resolution for
    quality-diversity optimization.
    """
    enabled: bool = True
    dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity", "performance"])
    bins_per_dimension: int = 10
    feature_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class IslandConfig:
    """
    Configuration for island-based parallel evolution.

    Islands maintain semi-isolated subpopulations with periodic
    migration to balance exploration and exploitation.
    """
    enabled: bool = True
    num_islands: int = 4
    island_capacity: int = 15
    migration_interval: int = 5     # Generations between migrations
    migration_rate: float = 0.1     # Fraction of population to migrate
    topology: str = "ring"          # Options: "ring", "fully_connected", "random"


@dataclass
class ArchiveConfig:
    """Configuration for elite archive management."""
    enabled: bool = True
    max_size: int = 50
    elite_count: int = 10
    pareto_based: bool = True
    objectives: List[str] = field(default_factory=lambda: ["combined_score"])


@dataclass
class PopulationConfig:
    """
    Unified configuration for hybrid population management.

    Combines MAP-Elites partitioning, island model, and elite archive
    for comprehensive quality-diversity optimization.
    """
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    islands: IslandConfig = field(default_factory=IslandConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)

    # Parent selection strategy
    selection_strategy: str = "adaptive"  # Options: "power_law", "tournament", "adaptive"
    exploitation_ratio: float = 0.6       # Probability of exploiting vs exploring
    selection_pressure: float = 2.0       # Power law exponent or tournament size

    # Inspiration sampling
    archive_inspiration_count: int = 3
    top_k_inspiration_count: int = 5
    diverse_inspiration_count: int = 3  # Structurally diverse neighbors from MAP-Elites grid


@dataclass
class OptimizationConfig:
    """
    Configuration for inner-loop parameter optimization.

    When LLM introduces new hyperparameters, this module automatically
    tunes them within a fixed evaluation budget. Supports two strategies:
    - Derivative-free: grid search, Latin Hypercube, Bayesian optimization
    - Gradient-based: finite-difference gradient estimation with Adam optimizer
    """
    enabled: bool = False
    max_budget: int = 10                # Maximum evaluations per optimization

    # Strategy selection thresholds (for derivative-free)
    grid_threshold: int = 2             # Use grid search for <= N params
    lhs_threshold: int = 5              # Use Latin Hypercube for <= N params
    # Bayesian optimization for > lhs_threshold params

    # Gradient-based optimization settings
    autodiff_fd_step: float = 0.01      # Finite-difference step as fraction of param range
    autodiff_learning_rate: float = 0.1
    autodiff_max_iterations: int = 50
    autodiff_convergence_threshold: float = 1e-5
    autodiff_use_adam: bool = True
    autodiff_adam_b1: float = 0.9       # Adam first moment decay
    autodiff_adam_b2: float = 0.999     # Adam second moment decay
    autodiff_adam_eps: float = 1e-8     # Numerical stability

    # Parameter inheritance
    freeze_after_optimization: bool = True
    inherit_parent_params: bool = True


@dataclass
class ExecutorConfig:
    """
    Configuration for job execution.

    Supports local execution and cluster-based execution (SLURM, etc.)
    """
    mode: str = "local"                 # Options: "local", "slurm", "kubernetes"
    max_parallel_jobs: int = 4
    timeout: float = 7200.0             # Per-job timeout in seconds

    # Local execution settings
    conda_env: Optional[str] = None
    work_dir: Optional[str] = None

    # Cluster settings
    partition: Optional[str] = None
    gpus_per_job: int = 0
    cpus_per_job: int = 1
    memory_gb: float = 4.0


@dataclass
class StorageConfig:
    """Configuration for program storage and persistence."""
    db_path: str = "evolution.db"
    checkpoint_interval: int = 5        # Save state every N generations
    keep_all_programs: bool = True      # Store all evaluated programs
    compress_code: bool = False


@dataclass
class ReportConfig:
    """Configuration for automated report generation."""
    enabled: bool = True
    output_dir: str = "reports"
    include_lineage: bool = True
    include_analysis: bool = True
    generate_plots: bool = True
    export_format: str = "markdown"     # Options: "markdown", "html", "pdf"


@dataclass
class EvolutionConfig:
    """
    Master configuration for the entire evolution process.

    This is the top-level configuration that aggregates all subsystem
    configurations and global evolution parameters.
    """
    # Task description
    task_description: str = ""
    evaluator_script: str = ""
    init_program_path: Optional[str] = None

    # Evolution parameters
    num_generations: int = 100
    seed: Optional[int] = None

    # Subsystem configurations
    models: ModelConfig = field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    patch_policy: PatchPolicy = field(default_factory=PatchPolicy)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # Prompt engineering
    system_prompt_override: Optional[str] = None
    include_text_feedback: bool = True
    include_metrics_history: bool = True

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if not self.evaluator_script:
            issues.append("evaluator_script is required")

        if self.num_generations < 1:
            issues.append("num_generations must be at least 1")

        if len(self.models.models) != len(self.models.weights):
            issues.append("Number of models must match number of weights")

        if self.population.exploitation_ratio < 0 or self.population.exploitation_ratio > 1:
            issues.append("exploitation_ratio must be between 0 and 1")

        return issues
