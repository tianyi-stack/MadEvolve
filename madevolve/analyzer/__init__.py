"""
Modular Report Generator for Evolution Experiments

A flexible system for generating comprehensive reports from evolution results.
Supports multiple scenarios through adapter pattern.

Usage:
    from madevolve.analyzer import (
        DataExtractor,
        ReportGenerator,
        ScenarioAdapter,
        register_adapter,
    )

    # Register your custom adapter
    from your_project.adapters import YourAdapter
    register_adapter('your_scenario', YourAdapter)

    # Create adapter for your scenario
    adapter = get_adapter('your_scenario')

    # Extract data
    extractor = DataExtractor(adapter)
    data = extractor.extract_evolution_data("/path/to/results")

    # Generate report
    generator = ReportGenerator(adapter)
    report = generator.generate_full_report(data, output_path="report.md")

Creating Custom Adapters:
    See base.py for the abstract classes to implement:
    - MetricsAdapter: Parse and display domain-specific metrics
    - PromptAdapter: Generate LLM prompts for analysis
    - ReportTemplateAdapter: Define report structure
    - ScenarioAdapter: Bundle all adapters together
"""

from typing import Dict, Type

from .base import (
    BaseMetrics,
    AlgorithmInfo,
    EvolutionData,
    EvolutionHistory,
    ExperimentConfig,
    MetricsAdapter,
    PromptAdapter,
    ReportTemplateAdapter,
    ScenarioAdapter,
)

from .core import DataExtractor

from .llm_core import (
    LLMClient,
    ReportGenerator,
    LITELLM_AVAILABLE,
    DEFAULT_MODEL,
    export_markdown_to_pdf,
)


# Adapter registry for extensibility
_ADAPTER_REGISTRY: Dict[str, Type[ScenarioAdapter]] = {}


def register_adapter(name: str, adapter_class: Type[ScenarioAdapter]) -> None:
    """
    Register a custom adapter for a scenario.

    Args:
        name: Scenario name (e.g., 'trading', 'bao')
        adapter_class: ScenarioAdapter subclass

    Example:
        from madevolve.analyzer import register_adapter
        from your_project.adapters import YourAdapter

        register_adapter('your_scenario', YourAdapter)
    """
    _ADAPTER_REGISTRY[name] = adapter_class


def get_adapter(scenario: str) -> ScenarioAdapter:
    """
    Get the appropriate adapter for a scenario.

    Args:
        scenario: Scenario name

    Returns:
        ScenarioAdapter instance

    Raises:
        ValueError: If scenario is not registered
    """
    if scenario not in _ADAPTER_REGISTRY:
        available = ', '.join(_ADAPTER_REGISTRY.keys()) if _ADAPTER_REGISTRY else '(none registered)'
        raise ValueError(
            f"Unknown scenario: '{scenario}'. Available: {available}. "
            f"Use register_adapter() to register your custom adapter."
        )

    return _ADAPTER_REGISTRY[scenario]()


def list_adapters() -> Dict[str, Type[ScenarioAdapter]]:
    """
    List all registered adapters.

    Returns:
        Dict mapping scenario names to adapter classes
    """
    return dict(_ADAPTER_REGISTRY)


__all__ = [
    # Base classes
    "BaseMetrics",
    "AlgorithmInfo",
    "EvolutionData",
    "EvolutionHistory",
    "ExperimentConfig",
    "MetricsAdapter",
    "PromptAdapter",
    "ReportTemplateAdapter",
    "ScenarioAdapter",
    # Core
    "DataExtractor",
    # LLM
    "LLMClient",
    "ReportGenerator",
    "LITELLM_AVAILABLE",
    "DEFAULT_MODEL",
    "export_markdown_to_pdf",
    # Registry
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
