"""
Base classes and interfaces for the modular report generator.

This module defines the abstract base classes that adapters must implement
to support different evolution scenarios (BAO reconstruction, trading, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class BaseMetrics:
    """
    Base class for evolution metrics.

    Subclasses should define domain-specific metrics.
    All metrics classes must have a combined_score for comparison.
    """
    combined_score: float
    text_feedback: str = ""
    execution_time_mean: float = 0.0
    execution_time_std: float = 0.0

    # Store raw data for custom access
    raw_public: Dict[str, Any] = field(default_factory=dict)
    raw_private: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmInfo:
    """Container for algorithm information."""
    code: str
    generation: int
    metrics: Optional[BaseMetrics]
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Paths to visualization files (e.g., PnL curves)
    chart_paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class EvolutionHistory:
    """Container for evolution history."""
    generations: List[int]
    best_scores: List[float]
    timestamps: List[float]
    total_programs: int
    successful_programs: int
    total_generations: int


@dataclass
class ExperimentConfig:
    """Container for experiment configuration."""
    task_description: str
    llm_models: List[str]
    num_generations: int
    num_islands: int
    migration_interval: int
    raw_config: Dict[str, Any] = field(default_factory=dict)

    # Optional fields that may vary by scenario
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionData:
    """Complete evolution data for report generation."""
    baseline: AlgorithmInfo
    best: AlgorithmInfo
    history: EvolutionHistory
    config: ExperimentConfig
    results_dir: str
    start_time: datetime
    end_time: datetime
    duration_hours: float

    # Scenario identifier
    scenario: str = "unknown"


class MetricsAdapter(ABC):
    """
    Abstract adapter for parsing and displaying domain-specific metrics.

    Implement this class to add support for a new evolution scenario.
    """

    @property
    @abstractmethod
    def scenario_name(self) -> str:
        """Return the scenario name (e.g., 'bao', 'trading')."""
        pass

    @property
    @abstractmethod
    def scenario_description(self) -> str:
        """Return a human-readable description of the scenario."""
        pass

    @abstractmethod
    def parse_metrics(self, metrics_data: Dict[str, Any]) -> BaseMetrics:
        """
        Parse raw metrics.json data into a metrics object.

        Args:
            metrics_data: Raw dictionary from metrics.json

        Returns:
            A BaseMetrics subclass instance
        """
        pass

    @abstractmethod
    def get_metrics_comparison_table(
        self,
        baseline: BaseMetrics,
        best: BaseMetrics
    ) -> str:
        """
        Generate a markdown table comparing baseline and best metrics.

        Args:
            baseline: Baseline algorithm metrics
            best: Best algorithm metrics

        Returns:
            Markdown-formatted comparison table
        """
        pass

    @abstractmethod
    def get_metrics_summary(self, metrics: BaseMetrics) -> str:
        """
        Generate a brief text summary of metrics.

        Args:
            metrics: Metrics to summarize

        Returns:
            Human-readable summary string
        """
        pass

    @abstractmethod
    def get_key_metrics_for_llm(self, metrics: BaseMetrics) -> str:
        """
        Format key metrics for LLM context.

        Args:
            metrics: Metrics to format

        Returns:
            Formatted string for LLM prompts
        """
        pass


class PromptAdapter(ABC):
    """
    Abstract adapter for domain-specific LLM prompts.

    Implement this class to customize how LLMs analyze your specific domain.
    """

    @abstractmethod
    def get_algorithm_analysis_prompt(
        self,
        algorithm: AlgorithmInfo,
        is_baseline: bool,
    ) -> str:
        """
        Generate a prompt for analyzing an algorithm.

        Args:
            algorithm: The algorithm to analyze
            is_baseline: Whether this is the baseline algorithm

        Returns:
            LLM prompt string
        """
        pass

    @abstractmethod
    def get_improvement_analysis_prompt(
        self,
        baseline: AlgorithmInfo,
        best: AlgorithmInfo,
        metrics_comparison: str,
    ) -> str:
        """
        Generate a prompt for analyzing improvements.

        Args:
            baseline: Baseline algorithm
            best: Best evolved algorithm
            metrics_comparison: Formatted metrics comparison

        Returns:
            LLM prompt string
        """
        pass

    @abstractmethod
    def get_executive_summary_prompt(
        self,
        data: EvolutionData,
        metrics_comparison: str,
    ) -> str:
        """
        Generate a prompt for creating an executive summary.

        Args:
            data: Complete evolution data
            metrics_comparison: Formatted metrics comparison

        Returns:
            LLM prompt string
        """
        pass


class ReportTemplateAdapter(ABC):
    """
    Abstract adapter for report templates.

    Implement this to customize the final report structure.
    """

    @abstractmethod
    def get_report_title(self) -> str:
        """Return the main report title."""
        pass

    @abstractmethod
    def get_section_headers(self) -> Dict[str, str]:
        """
        Return section headers for the report.

        Returns:
            Dict mapping section keys to header strings
        """
        pass

    @abstractmethod
    def format_final_report(
        self,
        data: EvolutionData,
        metrics_table: str,
        baseline_analysis: str,
        best_analysis: str,
        improvement_analysis: str,
        executive_summary: str,
    ) -> str:
        """
        Assemble the final report from all components.

        Args:
            data: Evolution data
            metrics_table: Formatted metrics comparison
            baseline_analysis: LLM analysis of baseline
            best_analysis: LLM analysis of best algorithm
            improvement_analysis: LLM improvement analysis
            executive_summary: LLM executive summary

        Returns:
            Complete markdown report
        """
        pass


class ScenarioAdapter(ABC):
    """
    Combined adapter that bundles metrics, prompts, and templates for a scenario.

    This is the main interface that users should implement for new scenarios.
    """

    @property
    @abstractmethod
    def metrics_adapter(self) -> MetricsAdapter:
        """Return the metrics adapter for this scenario."""
        pass

    @property
    @abstractmethod
    def prompt_adapter(self) -> PromptAdapter:
        """Return the prompt adapter for this scenario."""
        pass

    @property
    @abstractmethod
    def template_adapter(self) -> ReportTemplateAdapter:
        """Return the template adapter for this scenario."""
        pass

    def extract_code_block(self, code: str) -> str:
        """
        Extract the relevant code block for analysis.

        Override this if your scenario uses markers like EVOLVE-BLOCK.
        Default implementation returns the full code.

        Args:
            code: Full algorithm code

        Returns:
            Relevant code portion for analysis
        """
        return code
