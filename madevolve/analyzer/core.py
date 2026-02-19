"""
Core data extraction module for the report generator.

This module handles reading from:
- evolution_db.sqlite database
- File system (gen_0, best directories)
- experiment_config.yaml

All domain-specific parsing is delegated to adapters.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type

import yaml

from .base import (
    AlgorithmInfo,
    BaseMetrics,
    EvolutionData,
    EvolutionHistory,
    ExperimentConfig,
    ScenarioAdapter,
)

logger = logging.getLogger(__name__)


class DataExtractor:
    """
    Core data extractor that works with any scenario adapter.

    This class handles all file I/O and database operations,
    delegating domain-specific parsing to the provided adapter.
    """

    def __init__(self, adapter: ScenarioAdapter):
        """
        Initialize the extractor with a scenario adapter.

        Args:
            adapter: ScenarioAdapter implementation for the target domain
        """
        self.adapter = adapter
        self.metrics_adapter = adapter.metrics_adapter

    def extract_evolution_data(
        self,
        results_dir: str,
        best_generation: Optional[int] = None,
    ) -> EvolutionData:
        """
        Extract all evolution data from a results directory.

        Args:
            results_dir: Path to the evolution results directory
            best_generation: Optional generation number to use as "best" algorithm.
                           If None, uses the best/ directory (highest score).
                           If specified, loads from gen_{N} directory.

        Returns:
            EvolutionData object containing all extracted data
        """
        results_dir = Path(results_dir)

        # Validate required files exist
        self._validate_results_dir(results_dir)

        # Load baseline (gen_0)
        gen_0_dir = results_dir / 'gen_0'
        baseline = self._load_algorithm_info(str(gen_0_dir), generation=0)

        # Load evolution history
        db_path = results_dir / 'evolution_db.sqlite'
        history = self._load_evolution_history(str(db_path))

        # Load best algorithm - either from specified generation or best/ directory
        if best_generation is not None:
            # Load from specific generation directory
            gen_dir = results_dir / f'gen_{best_generation}'
            if not gen_dir.exists():
                raise FileNotFoundError(
                    f"Generation directory not found: {gen_dir}. "
                    f"Available generations: 0-{history.total_generations}"
                )
            best = self._load_algorithm_info(str(gen_dir), generation=best_generation)

            # Get timestamp from database if available
            if db_path.exists():
                gen_info = self._get_program_info_by_generation(str(db_path), best_generation)
                if gen_info:
                    best.timestamp = gen_info['timestamp']
        else:
            # Load from best/ directory (default behavior)
            best_dir = results_dir / 'best'
            best = self._load_algorithm_info(str(best_dir), generation=-1)

            # Try to find the actual generation of the best algorithm
            if db_path.exists():
                best_info = self._get_best_program_info(str(db_path))
                if best_info:
                    best.generation = best_info['generation']
                    best.timestamp = best_info['timestamp']

        # Load experiment config
        config_path = results_dir / 'experiment_config.yaml'
        config = self._load_experiment_config(str(config_path))

        # Calculate timing information
        if history.timestamps:
            start_ts = min(history.timestamps)
            end_ts = max(history.timestamps)
            start_time = datetime.fromtimestamp(start_ts)
            end_time = datetime.fromtimestamp(end_ts)
            duration_hours = (end_ts - start_ts) / 3600
        else:
            start_time = datetime.now()
            end_time = datetime.now()
            duration_hours = 0.0

        return EvolutionData(
            baseline=baseline,
            best=best,
            history=history,
            config=config,
            results_dir=str(results_dir),
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            scenario=self.metrics_adapter.scenario_name,
        )

    def _validate_results_dir(self, results_dir: Path) -> None:
        """Validate that required files and directories exist."""
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        required = [
            results_dir / 'evolution_db.sqlite',
            results_dir / 'gen_0',
            results_dir / 'best',
        ]

        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required path not found: {path}")

    def _load_algorithm_info(
        self,
        gen_dir: str,
        generation: int = -1
    ) -> AlgorithmInfo:
        """Load algorithm information from a generation directory."""
        gen_path = Path(gen_dir)

        # Try multiple code file patterns
        code_patterns = [
            'main.py.optimized.py',  # MadEvolve optimized
            'main.py',               # Standard
            'algorithm.py',          # Alternative naming
        ]

        code = None
        for pattern in code_patterns:
            code_path = gen_path / pattern
            if code_path.exists():
                with open(code_path, 'r') as f:
                    code = f.read()
                break

        if code is None:
            raise FileNotFoundError(
                f"No code file found in {gen_dir}. "
                f"Tried: {code_patterns}"
            )

        # Load validation metrics
        metrics_path = gen_path / 'results' / 'metrics.json'

        # Load test metrics if available
        test_metrics_path = gen_path / 'test_results' / 'test_metrics.json'
        test_metrics_data = None
        if test_metrics_path.exists():
            try:
                with open(test_metrics_path, 'r') as f:
                    test_metrics_data = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading test metrics from {test_metrics_path}: {e}")

        metrics = self._load_metrics(str(metrics_path), test_metrics_data)

        # Find chart files (PnL curves, etc.)
        chart_paths = self._find_chart_files(gen_path)

        return AlgorithmInfo(
            code=code,
            generation=generation,
            metrics=metrics,
            chart_paths=chart_paths,
        )

    def _load_metrics(self, metrics_path: str, test_metrics_data: Dict[str, Any] = None) -> Optional[BaseMetrics]:
        """Load and parse metrics using the adapter."""
        if not os.path.exists(metrics_path):
            logger.warning(f"Metrics file not found: {metrics_path}")
            return None

        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            # Pass test metrics if adapter supports it
            try:
                return self.metrics_adapter.parse_metrics(data, test_metrics_data)
            except TypeError:
                # Fallback for adapters that don't support test_metrics_data
                return self.metrics_adapter.parse_metrics(data)
        except Exception as e:
            logger.error(f"Error loading metrics from {metrics_path}: {e}")
            return None

    def _find_chart_files(self, gen_path: Path) -> Dict[str, str]:
        """
        Find chart/visualization files in a generation directory.

        Looks for common chart file patterns in the results and test_results subdirectories.

        Args:
            gen_path: Path to the generation directory

        Returns:
            Dict mapping chart type to file path
        """
        chart_paths = {}

        # Check validation results directory
        results_dir = gen_path / 'results'
        if results_dir.exists():
            # Look for PnL curve HTML files
            for html_file in results_dir.glob('pnl_curve*.html'):
                chart_paths['pnl_curve'] = str(html_file)
                break  # Take the first one

            # Look for other common chart patterns
            chart_patterns = {
                'equity_curve': 'equity*.html',
                'drawdown': 'drawdown*.html',
                'returns': 'returns*.html',
                'trades': 'trades*.html',
            }

            for chart_type, pattern in chart_patterns.items():
                for chart_file in results_dir.glob(pattern):
                    chart_paths[chart_type] = str(chart_file)
                    break

            # Also check for PNG/SVG versions
            for img_file in results_dir.glob('*.png'):
                name = img_file.stem.lower()
                if 'pnl' in name and 'pnl_curve_img' not in chart_paths:
                    chart_paths['pnl_curve_img'] = str(img_file)

        # Check test results directory
        test_results_dir = gen_path / 'test_results'
        if test_results_dir.exists():
            # Look for test PnL curve HTML files
            for html_file in test_results_dir.glob('pnl_curve*.html'):
                chart_paths['test_pnl_curve'] = str(html_file)
                break

            # Look for test PnL PNG files
            for img_file in test_results_dir.glob('pnl_curve*.png'):
                chart_paths['test_pnl_curve_img'] = str(img_file)
                break

        return chart_paths

    def _load_evolution_history(self, db_path: str) -> EvolutionHistory:
        """Load evolution history from the SQLite database."""
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            return EvolutionHistory(
                generations=[],
                best_scores=[],
                timestamps=[],
                total_programs=0,
                successful_programs=0,
                total_generations=0,
            )

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            # Get best score per generation for successful programs
            cursor.execute("""
                SELECT generation, MAX(combined_score), MIN(timestamp)
                FROM programs
                WHERE correct = 1 AND combined_score > -1e10
                GROUP BY generation
                ORDER BY generation
            """)
            rows = cursor.fetchall()

            generations = [row[0] for row in rows]
            best_scores = [row[1] for row in rows]
            timestamps = [row[2] for row in rows]

            # Get total and successful program counts
            cursor.execute("SELECT COUNT(*) FROM programs")
            total_programs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM programs WHERE correct = 1")
            successful_programs = cursor.fetchone()[0]

            cursor.execute("SELECT MAX(generation) FROM programs")
            total_generations = cursor.fetchone()[0] or 0

        finally:
            conn.close()

        return EvolutionHistory(
            generations=generations,
            best_scores=best_scores,
            timestamps=timestamps,
            total_programs=total_programs,
            successful_programs=successful_programs,
            total_generations=total_generations,
        )

    def _get_best_program_info(self, db_path: str) -> Optional[Dict[str, Any]]:
        """Get info about the best program from the database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT generation, timestamp FROM programs
                WHERE correct = 1
                ORDER BY combined_score DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                return {'generation': row[0], 'timestamp': row[1]}
            return None
        finally:
            conn.close()

    def _get_program_info_by_generation(
        self,
        db_path: str,
        generation: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get info about the best program from a specific generation.

        Args:
            db_path: Path to the SQLite database
            generation: Generation number to query

        Returns:
            Dict with 'generation', 'timestamp', 'combined_score' or None
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT generation, timestamp, combined_score FROM programs
                WHERE correct = 1 AND generation = ?
                ORDER BY combined_score DESC
                LIMIT 1
            """, (generation,))
            row = cursor.fetchone()
            if row:
                return {
                    'generation': row[0],
                    'timestamp': row[1],
                    'combined_score': row[2],
                }
            return None
        finally:
            conn.close()

    def _load_experiment_config(self, config_path: str) -> ExperimentConfig:
        """Load experiment configuration from YAML file."""
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}")
            return ExperimentConfig(
                task_description="",
                llm_models=[],
                num_generations=0,
                num_islands=1,
                migration_interval=10,
            )

        # Use custom loader that handles unknown tags (like omegaconf objects)
        class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
            pass

        def ignore_unknown(loader, tag_suffix, node):
            if isinstance(node, yaml.MappingNode):
                return loader.construct_mapping(node)
            elif isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node)
            else:
                return loader.construct_scalar(node)

        SafeLoaderIgnoreUnknown.add_multi_constructor('', ignore_unknown)

        try:
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return ExperimentConfig(
                task_description="",
                llm_models=[],
                num_generations=0,
                num_islands=1,
                migration_interval=10,
            )

        if not config:
            config = {}

        evo_config = config.get('evolution_config', {})

        # Extract LLM models (handle various formats)
        llm_models = self._extract_llm_models(evo_config.get('llm_models', []))

        # Extract task description
        task_desc = evo_config.get('task_sys_msg', '')

        return ExperimentConfig(
            task_description=task_desc,
            llm_models=llm_models,
            num_generations=evo_config.get('num_generations', 0),
            num_islands=config.get('database_config', {}).get('num_islands', 1),
            migration_interval=config.get('database_config', {}).get('migration_interval', 10),
            raw_config=config,
        )

    def _extract_llm_models(self, llm_models_data: Any) -> list:
        """Extract LLM model names from various config formats."""
        llm_models = []

        if isinstance(llm_models_data, dict):
            # Handle omegaconf structures
            if '_content' in llm_models_data:
                for item in llm_models_data['_content']:
                    if isinstance(item, dict) and '_val' in item:
                        llm_models.append(item['_val'])
                    elif isinstance(item, str):
                        llm_models.append(item)
        elif isinstance(llm_models_data, list):
            for item in llm_models_data:
                if isinstance(item, str):
                    llm_models.append(item)
                elif isinstance(item, dict) and '_val' in item:
                    llm_models.append(item['_val'])

        return llm_models


def get_metrics_comparison_table(
    baseline: BaseMetrics,
    best: BaseMetrics,
    adapter: ScenarioAdapter
) -> str:
    """
    Generate a metrics comparison table using the adapter.

    Args:
        baseline: Baseline metrics
        best: Best algorithm metrics
        adapter: Scenario adapter

    Returns:
        Markdown-formatted comparison table
    """
    return adapter.metrics_adapter.get_metrics_comparison_table(baseline, best)
