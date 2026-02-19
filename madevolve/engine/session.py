"""
Evolution session management for MadEvolve.

This module handles state persistence, checkpointing, and recovery
for long-running evolution processes.
"""

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from madevolve.common.helpers import AtomicFileWriter, generate_uid

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    best_score: float
    avg_score: float
    programs_evaluated: int
    improvements: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvolutionState:
    """
    Complete state of an evolution run.

    This dataclass captures all information needed to resume
    an evolution run from a checkpoint.
    """
    session_id: str
    current_generation: int
    best_program_id: Optional[str]
    best_score: float

    # Progress tracking
    total_programs_evaluated: int
    total_improvements: int
    generations_without_improvement: int

    # Generation history
    generation_stats: List[GenerationStats]

    # Component states (serialized)
    population_state: Dict[str, Any]
    model_selector_state: Dict[str, Any]

    # Timing
    start_time: float
    last_checkpoint_time: float

    # Metadata
    config_hash: str
    version: str = "0.1.0"


@dataclass
class EvolutionSession:
    """
    Manages the lifecycle of an evolution run.

    Handles initialization, checkpointing, resumption, and finalization
    of evolution sessions.
    """

    results_dir: str
    session_id: Optional[str] = None

    _state: Optional[EvolutionState] = None
    _checkpoint_dir: Optional[Path] = None

    def __post_init__(self):
        if self.session_id is None:
            self.session_id = self._generate_session_id()

        self._checkpoint_dir = Path(self.results_dir) / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _generate_session_id(self) -> str:
        """Generate a unique session identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = generate_uid(6)
        return f"session_{timestamp}_{uid}"

    def initialize(self, config_hash: str) -> EvolutionState:
        """
        Initialize a new evolution session.

        Args:
            config_hash: Hash of the configuration for validation

        Returns:
            New EvolutionState instance
        """
        self._state = EvolutionState(
            session_id=self.session_id,
            current_generation=0,
            best_program_id=None,
            best_score=float("-inf"),
            total_programs_evaluated=0,
            total_improvements=0,
            generations_without_improvement=0,
            generation_stats=[],
            population_state={},
            model_selector_state={},
            start_time=time.time(),
            last_checkpoint_time=time.time(),
            config_hash=config_hash,
        )

        logger.info(f"Initialized new session: {self.session_id}")
        return self._state

    def resume(self, checkpoint_path: str) -> EvolutionState:
        """
        Resume from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Restored EvolutionState

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is corrupted
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        try:
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)

            # Convert generation_stats dicts back to GenerationStats objects
            if "generation_stats" in data:
                data["generation_stats"] = [
                    GenerationStats(**s) if isinstance(s, dict) else s
                    for s in data["generation_stats"]
                ]

            self._state = EvolutionState(**data)
            self.session_id = self._state.session_id

            logger.info(
                f"Resumed session {self.session_id} at generation "
                f"{self._state.current_generation}"
            )
            return self._state

        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}")

    def save_checkpoint(self, force: bool = False) -> Optional[str]:
        """
        Save current state to a checkpoint file.

        Args:
            force: Save even if checkpoint interval hasn't elapsed

        Returns:
            Path to saved checkpoint or None if skipped
        """
        if self._state is None:
            logger.warning("No state to checkpoint")
            return None

        checkpoint_name = f"checkpoint_gen{self._state.current_generation:04d}.pkl"
        checkpoint_path = self._checkpoint_dir / checkpoint_name

        try:
            # Convert state to dict for serialization
            state_dict = {
                "session_id": self._state.session_id,
                "current_generation": self._state.current_generation,
                "best_program_id": self._state.best_program_id,
                "best_score": self._state.best_score,
                "total_programs_evaluated": self._state.total_programs_evaluated,
                "total_improvements": self._state.total_improvements,
                "generations_without_improvement": self._state.generations_without_improvement,
                "generation_stats": [
                    {
                        "generation": s.generation,
                        "best_score": s.best_score,
                        "avg_score": s.avg_score,
                        "programs_evaluated": s.programs_evaluated,
                        "improvements": s.improvements,
                        "timestamp": s.timestamp,
                    }
                    for s in self._state.generation_stats
                ],
                "population_state": self._state.population_state,
                "model_selector_state": self._state.model_selector_state,
                "start_time": self._state.start_time,
                "last_checkpoint_time": time.time(),
                "config_hash": self._state.config_hash,
                "version": self._state.version,
            }

            with AtomicFileWriter(str(checkpoint_path), "wb") as f:
                pickle.dump(state_dict, f)

            self._state.last_checkpoint_time = time.time()

            # Create symlink to latest checkpoint
            latest_link = self._checkpoint_dir / "latest.pkl"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(checkpoint_name)

            logger.debug(f"Saved checkpoint: {checkpoint_path}")
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None

    def update_generation(
        self,
        generation: int,
        best_score: float,
        avg_score: float,
        programs_evaluated: int,
        improvements: int,
    ):
        """Update state after a generation completes."""
        if self._state is None:
            return

        self._state.current_generation = generation
        self._state.total_programs_evaluated += programs_evaluated
        self._state.total_improvements += improvements

        if best_score > self._state.best_score:
            self._state.best_score = best_score
            self._state.generations_without_improvement = 0
        else:
            self._state.generations_without_improvement += 1

        stats = GenerationStats(
            generation=generation,
            best_score=best_score,
            avg_score=avg_score,
            programs_evaluated=programs_evaluated,
            improvements=improvements,
        )
        self._state.generation_stats.append(stats)

    def set_best_program(self, program_id: str, score: float):
        """Update the best program."""
        if self._state is not None:
            self._state.best_program_id = program_id
            self._state.best_score = score

    def update_component_state(self, component: str, state: Dict[str, Any]):
        """Update serialized state for a component."""
        if self._state is None:
            return

        if component == "population":
            self._state.population_state = state
        elif component == "model_selector":
            self._state.model_selector_state = state

    def get_component_state(self, component: str) -> Dict[str, Any]:
        """Get serialized state for a component."""
        if self._state is None:
            return {}

        if component == "population":
            return self._state.population_state
        elif component == "model_selector":
            return self._state.model_selector_state

        return {}

    @property
    def state(self) -> Optional[EvolutionState]:
        """Get current evolution state."""
        return self._state

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since session start."""
        if self._state is None:
            return 0.0
        return time.time() - self._state.start_time

    def export_history(self, path: Optional[str] = None) -> str:
        """
        Export generation history to JSON.

        Args:
            path: Output path (default: results_dir/history.json)

        Returns:
            Path to exported file
        """
        if path is None:
            path = os.path.join(self.results_dir, "history.json")

        history = {
            "session_id": self.session_id,
            "generations": [
                {
                    "generation": s.generation,
                    "best_score": s.best_score,
                    "avg_score": s.avg_score,
                    "programs_evaluated": s.programs_evaluated,
                    "improvements": s.improvements,
                    "timestamp": s.timestamp,
                }
                for s in (self._state.generation_stats if self._state else [])
            ],
            "summary": {
                "total_generations": self._state.current_generation if self._state else 0,
                "total_programs": self._state.total_programs_evaluated if self._state else 0,
                "final_best_score": self._state.best_score if self._state else None,
                "elapsed_time": self.elapsed_time,
            },
        }

        with open(path, "w") as f:
            json.dump(history, f, indent=2)

        return path
