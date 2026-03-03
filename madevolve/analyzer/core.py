"""
Core data extraction module for the report generator.

This module handles reading from:
- evolution.db database
- File system (evaluations/gen_0/{id}/, best/ directories)
- history.json (fallback for timestamps)

All domain-specific parsing is delegated to adapters.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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
                           If specified, loads from evaluations/gen_{N}/ directory.

        Returns:
            EvolutionData object containing all extracted data
        """
        results_dir = Path(results_dir)

        # Validate required files exist
        self._validate_results_dir(results_dir)

        # Load baseline from evaluations/gen_0/{first_program_id}/
        baseline = self._load_baseline(results_dir)

        # Load evolution history from DB, with history.json fallback
        db_path = results_dir / 'evolution.db'
        history = self._load_evolution_history(str(db_path))

        # If DB history is empty, try history.json
        if not history.generations:
            history_json = results_dir / 'history.json'
            if history_json.exists():
                history = self._load_history_from_json(str(history_json))

        # Load best algorithm - either from specified generation or best/ directory
        if best_generation is not None:
            # Load from specific generation directory
            gen_dir = results_dir / 'evaluations' / f'gen_{best_generation}'
            if not gen_dir.exists():
                raise FileNotFoundError(
                    f"Generation directory not found: {gen_dir}. "
                    f"Available generations: 0-{history.total_generations}"
                )
            # Find first program subdirectory
            subdirs = [d for d in gen_dir.iterdir() if d.is_dir() and d.name != '__pycache__']
            if subdirs:
                best = self._load_algorithm_from_dir(subdirs[0], generation=best_generation)
            else:
                best = self._load_algorithm_from_dir(gen_dir, generation=best_generation)

            # Get timestamp from database if available
            if db_path.exists():
                gen_info = self._get_program_info_by_generation(str(db_path), best_generation)
                if gen_info:
                    best.timestamp = gen_info['timestamp']
        else:
            # Load from best/ directory (default behavior)
            best = self._load_best(results_dir)

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
            results_dir / 'evolution.db',
            results_dir / 'evaluations',
        ]

        # Accept either 'best' or 'best_program' (legacy)
        has_best = (results_dir / 'best').exists() or (results_dir / 'best_program').exists()
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required path not found: {path}")
        if not has_best:
            raise FileNotFoundError(
                f"Required path not found: {results_dir / 'best'} (or best_program)"
            )

    def _load_baseline(self, results_dir: Path) -> AlgorithmInfo:
        """Load baseline algorithm from evaluations/gen_0/{first_program_id}/."""
        gen_0_dir = results_dir / 'evaluations' / 'gen_0'
        if not gen_0_dir.exists():
            raise FileNotFoundError(f"Baseline directory not found: {gen_0_dir}")

        # Find the first program subdirectory
        subdirs = [d for d in gen_0_dir.iterdir() if d.is_dir() and d.name != '__pycache__']
        if not subdirs:
            raise FileNotFoundError(f"No program directories found in {gen_0_dir}")
        program_dir = subdirs[0]

        return self._load_algorithm_from_dir(program_dir, generation=0, code_filename='candidate.py')

    def _load_best(self, results_dir: Path) -> AlgorithmInfo:
        """Load best algorithm from best/ (or best_program/) directory."""
        best_dir = results_dir / 'best'
        if not best_dir.exists():
            best_dir = results_dir / 'best_program'
        if not best_dir.exists():
            raise FileNotFoundError(f"Best directory not found in {results_dir}")

        return self._load_algorithm_from_dir(best_dir, generation=-1, code_filename='best.py')

    def _load_algorithm_from_dir(
        self,
        program_dir: Path,
        generation: int = -1,
        code_filename: str = 'candidate.py',
    ) -> AlgorithmInfo:
        """Load algorithm information from a program directory."""
        # Try the specified code filename first, then fallbacks
        code_patterns = [
            code_filename,
            'candidate.py',
            'best.py',
            'main.py.optimized.py',
            'main.py',
            'algorithm.py',
        ]

        code = None
        for pattern in code_patterns:
            code_path = program_dir / pattern
            if code_path.exists():
                with open(code_path, 'r') as f:
                    code = f.read()
                break

        if code is None:
            raise FileNotFoundError(
                f"No code file found in {program_dir}. "
                f"Tried: {code_patterns}"
            )

        # Load metrics from result.json (new format) or results/metrics.json (legacy)
        metrics_path = program_dir / 'result.json'
        if not metrics_path.exists():
            metrics_path = program_dir / 'results' / 'metrics.json'

        metrics = self._load_metrics(str(metrics_path))

        # Find chart/visualization files
        chart_paths = self._find_chart_files(program_dir)

        return AlgorithmInfo(
            code=code,
            generation=generation,
            metrics=metrics,
            chart_paths=chart_paths,
        )

    def _load_metrics(self, metrics_path: str) -> Optional[BaseMetrics]:
        """Load and parse metrics using the adapter."""
        if not os.path.exists(metrics_path):
            logger.warning(f"Metrics file not found: {metrics_path}")
            return None

        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            return self.metrics_adapter.parse_metrics(data)
        except Exception as e:
            logger.error(f"Error loading metrics from {metrics_path}: {e}")
            return None

    def _find_chart_files(self, program_dir: Path) -> Dict[str, str]:
        """
        Find chart/visualization files in a program directory.

        Collects all HTML and PNG files from the program directory and
        its results/ and test_results/ subdirectories. Each file is keyed
        by its stem (filename without extension).

        Args:
            program_dir: Path to the program directory

        Returns:
            Dict mapping chart name (stem) to file path
        """
        chart_paths = {}

        def _collect(directory: Path, prefix: str = '') -> None:
            """Collect HTML and PNG chart files from a directory."""
            if not directory.exists():
                return
            for f in sorted(directory.iterdir()):
                if f.suffix in ('.html', '.png') and f.is_file():
                    key = prefix + f.stem
                    if key not in chart_paths:
                        chart_paths[key] = str(f)

        # Charts directly in the program directory (current format)
        _collect(program_dir)

        # Legacy: results/ subdirectory
        _collect(program_dir / 'results')

        # Legacy: test_results/ subdirectory
        _collect(program_dir / 'test_results', prefix='test_')

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
            # Get column names to handle schema differences
            cursor.execute("PRAGMA table_info(programs)")
            columns = {row[1] for row in cursor.fetchall()}

            has_correct = 'correct' in columns
            # Use created_at (actual schema) or timestamp (legacy)
            ts_col = 'created_at' if 'created_at' in columns else 'timestamp'

            # Build query based on available columns
            where_clause = "WHERE combined_score IS NOT NULL AND combined_score > -1e10"
            if has_correct:
                where_clause += " AND correct = 1"

            cursor.execute(f"""
                SELECT generation, MAX(combined_score), MIN({ts_col})
                FROM programs
                {where_clause}
                GROUP BY generation
                ORDER BY generation
            """)
            rows = cursor.fetchall()

            generations = [row[0] for row in rows]
            best_scores = [row[1] for row in rows]

            # Convert timestamps: created_at is a datetime string, timestamp is epoch
            raw_timestamps = [row[2] for row in rows]
            timestamps = []
            for ts in raw_timestamps:
                if ts is None:
                    timestamps.append(0.0)
                elif isinstance(ts, (int, float)):
                    timestamps.append(float(ts))
                elif isinstance(ts, str):
                    try:
                        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                        timestamps.append(dt.timestamp())
                    except ValueError:
                        timestamps.append(0.0)
                else:
                    timestamps.append(0.0)

            # Get total and successful program counts
            cursor.execute("SELECT COUNT(*) FROM programs")
            total_programs = cursor.fetchone()[0]

            if has_correct:
                cursor.execute("SELECT COUNT(*) FROM programs WHERE correct = 1")
                successful_programs = cursor.fetchone()[0]
            else:
                cursor.execute("SELECT COUNT(*) FROM programs WHERE combined_score IS NOT NULL AND combined_score > -1e10")
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

    def _load_history_from_json(self, history_path: str) -> EvolutionHistory:
        """Load evolution history from history.json as a fallback."""
        try:
            with open(history_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading history.json: {e}")
            return EvolutionHistory(
                generations=[], best_scores=[], timestamps=[],
                total_programs=0, successful_programs=0, total_generations=0,
            )

        gens_data = data.get('generations', [])
        generations = [g['generation'] for g in gens_data]
        best_scores = [g.get('best_score', 0.0) for g in gens_data]
        timestamps = [g.get('timestamp', 0.0) for g in gens_data]

        summary = data.get('summary', {})
        total_programs = summary.get('total_programs', sum(g.get('programs_evaluated', 0) for g in gens_data))
        total_generations = max(generations) if generations else 0

        return EvolutionHistory(
            generations=generations,
            best_scores=best_scores,
            timestamps=timestamps,
            total_programs=total_programs,
            successful_programs=total_programs,  # history.json doesn't distinguish
            total_generations=total_generations,
        )

    def _get_best_program_info(self, db_path: str) -> Optional[Dict[str, Any]]:
        """Get info about the best program from the database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            # Detect available columns
            cursor.execute("PRAGMA table_info(programs)")
            columns = {row[1] for row in cursor.fetchall()}

            ts_col = 'created_at' if 'created_at' in columns else 'timestamp'
            has_correct = 'correct' in columns

            where = "WHERE correct = 1" if has_correct else "WHERE combined_score IS NOT NULL"

            cursor.execute(f"""
                SELECT generation, {ts_col} FROM programs
                {where}
                ORDER BY combined_score DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                ts = row[1]
                # Convert string timestamp to epoch if needed
                if isinstance(ts, str):
                    try:
                        ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()
                    except ValueError:
                        ts = 0.0
                return {'generation': row[0], 'timestamp': ts}
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
            # Detect available columns
            cursor.execute("PRAGMA table_info(programs)")
            columns = {row[1] for row in cursor.fetchall()}

            ts_col = 'created_at' if 'created_at' in columns else 'timestamp'
            has_correct = 'correct' in columns

            where = "WHERE generation = ?"
            if has_correct:
                where += " AND correct = 1"

            cursor.execute(f"""
                SELECT generation, {ts_col}, combined_score FROM programs
                {where}
                ORDER BY combined_score DESC
                LIMIT 1
            """, (generation,))
            row = cursor.fetchone()
            if row:
                ts = row[1]
                if isinstance(ts, str):
                    try:
                        ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()
                    except ValueError:
                        ts = 0.0
                return {
                    'generation': row[0],
                    'timestamp': ts,
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
