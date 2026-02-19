# MadEvolve

**A General-Purpose LLM-Driven Evolution Framework for Code Optimization**

MadEvolve is a flexible framework that combines Large Language Models with evolutionary algorithms to automatically evolve, optimize, and improve code. It implements quality-diversity optimization through MAP-Elites with island models, supports multiple LLM providers, and provides comprehensive result analysis.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [LLM Providers](#llm-providers)
- [Domain Adapters](#domain-adapters)
- [API Reference](#api-reference)
- [License](#license)

## Features

### Evolution Strategies
- **Differential Evolution**: SEARCH/REPLACE patch-based mutations for precise, targeted changes
- **Holistic Rewriting**: Complete function rewrites guided by LLM understanding
- **Synthesis & Crossover**: Combine successful traits from multiple elite programs
- **Hybrid Strategies**: Adaptive switching between mutation strategies based on progress

### Quality-Diversity Optimization
- **MAP-Elites Grid**: Multi-dimensional behavior characterization for diverse solutions
- **Island Model**: Parallel populations with periodic migration for exploration
- **Elite Archives**: Persistent storage of best solutions per behavior niche

### Multi-Provider LLM Support
- Unified interface for OpenAI, Anthropic Claude, Google Gemini, and DeepSeek
- Adaptive model selection using bandit algorithms (UCB, Thompson Sampling)
- Automatic fallback and retry mechanisms
- Cost tracking and budget management

### Execution Backends
- **Native**: Local subprocess execution with timeout handling
- **SLURM**: HPC cluster support for large-scale experiments

### Analysis & Reporting
- LLM-powered insight generation
- Markdown/HTML reports with evolution visualization
- Extensible adapter system for domain-specific analysis (metrics, prompts, templates)

## Installation

### Basic Installation

```bash
pip install -e .
```

### With Optional Dependencies

```bash
# Google Gemini support
pip install -e ".[google]"

# Bayesian optimization for inner-loop tuning
pip install -e ".[optimizer]"

# Syntax tree parsing
pip install -e ".[parsing]"

# Structural similarity metrics
pip install -e ".[similarity]"

# Full installation (all features)
pip install -e ".[full]"

# Development dependencies
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

### Environment Variables

Set API keys for your LLM providers:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

## Quick Start

### Command Line Interface

```bash
# Run evolution with a configuration file
madevolve run -c config.yaml -o ./results

# Resume from a checkpoint
madevolve run -c config.yaml -o ./results --resume checkpoint.pkl

# Run without banner
madevolve run -c config.yaml -o ./results --quiet

# Show version
madevolve version
```

### Python API

```python
from madevolve import (
    EvolutionOrchestrator,
    EvolutionConfig,
    PopulationConfig,
    ModelConfig,
    ExecutorConfig,
)

# Configure the evolution
config = EvolutionConfig(
    population=PopulationConfig(
        size=50,
        elite_size=10,
        island_count=4,
    ),
    model=ModelConfig(
        primary="gpt-4o",
        fallback="claude-sonnet-4-20250514",
    ),
    executor=ExecutorConfig(
        backend="native",
        timeout=300,
    ),
    generations=100,
    mutation_rate=0.8,
    crossover_rate=0.2,
)

# Create and run the orchestrator
orchestrator = EvolutionOrchestrator(
    config=config,
    results_dir="./results",
)
orchestrator.run()
```

## Configuration

Create a YAML configuration file:

```yaml
# config.yaml

# Model configuration
models:
  primary: gpt-4o              # Primary model for evolution
  fallback: claude-sonnet-4-20250514  # Fallback model on failure
  selection_strategy: ucb      # Options: ucb, thompson, random, fixed

# Population settings
population:
  size: 50                     # Total population size
  elite_size: 10               # Number of elite solutions to preserve
  island_count: 4              # Number of parallel islands
  migration_interval: 10       # Generations between migrations
  migration_rate: 0.1          # Fraction of population to migrate

# Evolution parameters
evolution:
  generations: 100             # Maximum generations
  mutation_rate: 0.8           # Probability of mutation
  crossover_rate: 0.2          # Probability of crossover
  strategy: hybrid             # Options: differential, holistic, hybrid

# Execution settings
executor:
  backend: native              # Options: native, slurm
  timeout: 300                 # Evaluation timeout (seconds)
  max_retries: 3               # Retries on failure
  parallel_workers: 4          # Concurrent evaluations

# Storage settings
storage:
  checkpoint_interval: 10      # Save checkpoint every N generations
  cache_enabled: true          # Enable LLM response caching
  database_path: ./cache.db    # SQLite cache location

# Inner-loop optimization (optional)
inner_loop:
  enabled: false
  method: grid_search          # Options: grid_search, bayesian, autodiff
  budget: 100                  # Evaluation budget per candidate
```

### Configuration via JSON

Configuration files can also be in JSON format:

```json
{
  "models": {
    "primary": "gpt-4o",
    "fallback": "claude-sonnet-4-20250514"
  },
  "population": {
    "size": 50,
    "elite_size": 10
  },
  "evolution": {
    "generations": 100
  }
}
```

## Architecture

```
madevolve/
├── engine/              # Evolution orchestration
│   ├── orchestrator.py  # Main evolution loop
│   ├── configuration.py # Config dataclasses
│   ├── container.py     # Dependency injection
│   └── session.py       # Run session management
│
├── provider/            # LLM provider integration
│   ├── gateway.py       # Unified LLM interface
│   ├── vectorizer.py    # Embedding generation
│   └── adapters/        # Provider-specific adapters
│       ├── openai_adapter.py
│       ├── anthropic_adapter.py
│       ├── gemini_adapter.py
│       ├── deepseek_adapter.py
│       ├── tariff.py    # Cost tracking
│       └── response.py  # Response normalization
│
├── repository/          # Population & storage
│   └── topology/
│       └── partitions.py # MAP-Elites + Island model
│
├── transformer/         # Code transformation
│   ├── patcher.py       # SEARCH/REPLACE patches
│   ├── rewriter.py      # Holistic rewrites
│   ├── changeset.py     # Change tracking
│   └── parallel.py      # Parallel transformations
│
├── executor/            # Job execution
│   ├── dispatcher.py    # Job submission & monitoring
│   ├── settings.py      # Executor configuration
│   └── runners/
│       ├── native.py    # Local subprocess
│       └── cluster.py   # SLURM cluster
│
├── synthesizer/         # Prompt composition
│   └── composer.py      # Dynamic prompt building
│
├── templates/           # Prompt templates
│   ├── foundation.py    # Base templates
│   ├── differential.py  # Patch prompts
│   ├── holistic.py      # Rewrite prompts
│   ├── hybrid.py        # Combined strategies
│   ├── bootstrap.py     # Initial population
│   └── insight.py       # Analysis prompts
│
├── analyzer/            # Analysis & reporting
│   ├── base.py          # Adapter base classes & data structures
│   ├── core.py          # Data extraction engine
│   ├── llm_core.py      # LLM-powered report generation
│   └── generate_report.py # CLI report generation
│
├── common/              # Utilities
│   ├── constants.py     # Global constants
│   └── helpers.py       # Helper functions
│
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point
├── branding.py          # Display utilities
└── logo.py              # ASCII logo
```

## LLM Providers

| Provider | Supported Models | Environment Variable |
|----------|------------------|---------------------|
| OpenAI | GPT-4o, GPT-4-Turbo, GPT-4, GPT-3.5-Turbo | `OPENAI_API_KEY` |
| Anthropic | Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku | `ANTHROPIC_API_KEY` |
| Google | Gemini Pro, Gemini Ultra | `GOOGLE_API_KEY` |
| DeepSeek | DeepSeek Coder, DeepSeek Chat | `DEEPSEEK_API_KEY` |

### Model Selection Strategies

- **UCB (Upper Confidence Bound)**: Balances exploration and exploitation based on model performance history
- **Thompson Sampling**: Probabilistic selection based on performance distribution
- **Random**: Uniform random selection across available models
- **Fixed**: Always use the primary model

## Domain Adapters

MadEvolve uses a modular adapter system for domain-specific analysis. Implement your own adapters to customize metrics parsing, LLM prompts, and report templates for your specific use case.

### Adapter Architecture

The analyzer uses three types of adapters bundled into a `ScenarioAdapter`:

| Adapter Type | Purpose |
|--------------|---------|
| `MetricsAdapter` | Parse metrics.json and format domain-specific metrics |
| `PromptAdapter` | Generate LLM prompts for algorithm analysis |
| `ReportTemplateAdapter` | Define report structure and formatting |

### Creating Custom Adapters

Implement the abstract base classes for your domain:

```python
from madevolve.analyzer import (
    ScenarioAdapter,
    MetricsAdapter,
    PromptAdapter,
    ReportTemplateAdapter,
    BaseMetrics,
    register_adapter,
)
from dataclasses import dataclass
from typing import Any, Dict

# 1. Define your metrics structure
@dataclass
class MyMetrics(BaseMetrics):
    accuracy: float = 0.0
    loss: float = 0.0

# 2. Implement the metrics adapter
class MyMetricsAdapter(MetricsAdapter):
    @property
    def scenario_name(self) -> str:
        return "my_scenario"

    @property
    def scenario_description(self) -> str:
        return "My custom evolution scenario"

    def parse_metrics(self, metrics_data: Dict[str, Any]) -> MyMetrics:
        return MyMetrics(
            combined_score=metrics_data.get("score", 0.0),
            accuracy=metrics_data.get("accuracy", 0.0),
            loss=metrics_data.get("loss", 0.0),
        )

    def get_metrics_comparison_table(self, baseline: BaseMetrics, best: BaseMetrics) -> str:
        return f"| Metric | Baseline | Best |\n|--------|----------|------|\n| Score | {baseline.combined_score:.4f} | {best.combined_score:.4f} |"

    def get_metrics_summary(self, metrics: BaseMetrics) -> str:
        return f"Score: {metrics.combined_score:.4f}"

    def get_key_metrics_for_llm(self, metrics: BaseMetrics) -> str:
        return f"Combined Score: {metrics.combined_score}"

# 3. Implement prompt and template adapters (see base.py for full interface)
# 4. Bundle into a ScenarioAdapter
class MyScenarioAdapter(ScenarioAdapter):
    def __init__(self):
        self._metrics = MyMetricsAdapter()
        self._prompts = MyPromptAdapter()  # Your implementation
        self._templates = MyTemplateAdapter()  # Your implementation

    @property
    def metrics_adapter(self) -> MetricsAdapter:
        return self._metrics

    @property
    def prompt_adapter(self) -> PromptAdapter:
        return self._prompts

    @property
    def template_adapter(self) -> ReportTemplateAdapter:
        return self._templates

# 5. Register and use
register_adapter('my_scenario', MyScenarioAdapter)
```

### Using Registered Adapters

```python
from madevolve.analyzer import (
    get_adapter,
    list_adapters,
    DataExtractor,
    ReportGenerator,
)

# List available adapters
print(list_adapters())

# Get adapter and generate report
adapter = get_adapter('my_scenario')
extractor = DataExtractor(adapter)
data = extractor.extract_evolution_data("/path/to/results")

generator = ReportGenerator(adapter)
report = generator.generate_full_report(data, output_path="report.md")
```

## API Reference

### Core Classes

#### `EvolutionOrchestrator`

Main class for running evolution experiments.

```python
orchestrator = EvolutionOrchestrator(
    config: EvolutionConfig,
    results_dir: str = "./results",
    checkpoint_path: Optional[str] = None,
)
orchestrator.run()
```

#### `EvolutionConfig`

Top-level configuration dataclass.

```python
@dataclass
class EvolutionConfig:
    population: PopulationConfig
    model: ModelConfig
    executor: ExecutorConfig
    storage: StorageConfig
    generations: int = 100
    mutation_rate: float = 0.8
    crossover_rate: float = 0.2
```

#### `PopulationConfig`

Population and island model settings.

```python
@dataclass
class PopulationConfig:
    size: int = 50
    elite_size: int = 10
    island_count: int = 4
    migration_interval: int = 10
    migration_rate: float = 0.1
```

#### `ModelConfig`

LLM provider settings.

```python
@dataclass
class ModelConfig:
    primary: str = "gpt-4o"
    fallback: str = "claude-sonnet-4-20250514"
    selection_strategy: str = "ucb"
    max_tokens: int = 4096
    temperature: float = 0.7
```

#### `ExecutorConfig`

Execution backend settings.

```python
@dataclass
class ExecutorConfig:
    backend: str = "native"  # "native" or "slurm"
    timeout: int = 300
    max_retries: int = 3
    parallel_workers: int = 4
```

## Requirements

### Core Dependencies

- Python >= 3.9
- numpy >= 1.21.0
- scipy >= 1.9.0
- openai >= 1.0.0
- anthropic >= 0.18.0
- rich >= 13.0.0
- pyyaml >= 6.0
- tiktoken >= 0.5.0
- scikit-learn >= 1.0.0
- aiosqlite >= 0.19.0

### Optional Dependencies

- google-generativeai >= 0.3.0 (Google Gemini support)
- optuna >= 3.0.0 (Bayesian optimization)
- tree-sitter >= 0.20.0 (Syntax tree parsing)
- zss >= 1.2.0 (Tree edit distance)
