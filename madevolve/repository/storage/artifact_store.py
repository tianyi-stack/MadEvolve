"""
Artifact Store - Program storage and lineage tracking.

This module provides persistent storage for evolved programs,
their metrics, embeddings, and lineage information.
"""

import json
import logging
import pickle
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from madevolve.repository.decorators import db_retry
from madevolve.repository.storage.schema import create_schema

logger = logging.getLogger(__name__)


@dataclass
class ProgramRecord:
    """Record of an evolved program."""
    program_id: str
    code: str
    parent_id: Optional[str]
    generation: int
    combined_score: float
    public_metrics: Dict[str, float] = field(default_factory=dict)
    private_metrics: Dict[str, float] = field(default_factory=dict)
    text_feedback: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    @property
    def is_baseline(self) -> bool:
        """Check if this is a baseline program."""
        return self.metadata.get("is_baseline", False)


class ArtifactStore:
    """
    Persistent storage for evolved programs.

    Provides CRUD operations, lineage tracking, and advanced queries
    for program management during evolution.
    """

    def __init__(self, container):
        """
        Initialize the artifact store.

        Args:
            container: ServiceContainer with configuration
        """
        self.config = container.config.storage
        db_path = Path(container.results_dir) / self.config.db_path

        self._conn = sqlite3.connect(str(db_path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row

        create_schema(self._conn)
        logger.info(f"ArtifactStore initialized: {db_path}")

    @db_retry(max_attempts=3)
    def register(
        self,
        program_id: str,
        code: str,
        parent_id: Optional[str],
        generation: int,
        combined_score: float,
        public_metrics: Dict[str, float],
        private_metrics: Dict[str, float],
        text_feedback: str,
        embedding: Optional[List[float]],
        metadata: Dict[str, Any],
    ):
        """
        Register a new program in the store.

        Args:
            program_id: Unique program identifier
            code: Source code
            parent_id: Parent program ID (None for baseline)
            generation: Generation number
            combined_score: Overall fitness score
            public_metrics: Public evaluation metrics
            private_metrics: Private/internal metrics
            text_feedback: Text feedback from evaluator
            embedding: Code embedding vector
            metadata: Additional metadata
        """
        cursor = self._conn.cursor()

        # Serialize data
        public_json = json.dumps(public_metrics)
        private_json = json.dumps(private_metrics)
        metadata_json = json.dumps(metadata)
        embedding_blob = pickle.dumps(embedding) if embedding else None

        cursor.execute(
            """
            INSERT INTO programs (
                program_id, code, parent_id, generation, combined_score,
                public_metrics, private_metrics, text_feedback,
                embedding, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                program_id, code, parent_id, generation, combined_score,
                public_json, private_json, text_feedback,
                embedding_blob, metadata_json,
            ),
        )

        self._conn.commit()
        logger.debug(f"Registered program {program_id} (gen={generation}, score={combined_score:.4f})")

    def get(self, program_id: str) -> Optional[ProgramRecord]:
        """
        Retrieve a program by ID.

        Args:
            program_id: Program identifier

        Returns:
            ProgramRecord or None if not found
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM programs WHERE program_id = ?", (program_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def _row_to_record(self, row: sqlite3.Row) -> ProgramRecord:
        """Convert database row to ProgramRecord."""
        return ProgramRecord(
            program_id=row["program_id"],
            code=row["code"],
            parent_id=row["parent_id"],
            generation=row["generation"],
            combined_score=row["combined_score"] or 0.0,
            public_metrics=json.loads(row["public_metrics"] or "{}"),
            private_metrics=json.loads(row["private_metrics"] or "{}"),
            text_feedback=row["text_feedback"] or "",
            embedding=pickle.loads(row["embedding"]) if row["embedding"] else None,
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=row["created_at"],
        )

    def get_best_program(self) -> Optional[ProgramRecord]:
        """Get the program with highest combined score."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM programs ORDER BY combined_score DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return self._row_to_record(row) if row else None

    def get_by_generation(self, generation: int) -> List[ProgramRecord]:
        """Get all programs from a specific generation."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM programs WHERE generation = ? ORDER BY combined_score DESC",
            (generation,),
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_top_programs(self, n: int = 10, correct_only: bool = False) -> List[ProgramRecord]:
        """
        Get top N programs by score.

        Args:
            n: Number of programs to return
            correct_only: Only include programs with positive scores

        Returns:
            List of ProgramRecords
        """
        cursor = self._conn.cursor()

        if correct_only:
            cursor.execute(
                "SELECT * FROM programs WHERE combined_score > 0 "
                "ORDER BY combined_score DESC LIMIT ?",
                (n,),
            )
        else:
            cursor.execute(
                "SELECT * FROM programs ORDER BY combined_score DESC LIMIT ?",
                (n,),
            )

        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_recent_programs(self, n: int = 10) -> List[ProgramRecord]:
        """Get N most recently registered programs."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM programs ORDER BY created_at DESC LIMIT ?",
            (n,),
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_lineage(self, program_id: str) -> List[ProgramRecord]:
        """
        Get the ancestry chain for a program.

        Args:
            program_id: Program to trace lineage from

        Returns:
            List of ProgramRecords from oldest ancestor to given program
        """
        lineage = []
        current_id = program_id

        while current_id:
            program = self.get(current_id)
            if program is None:
                break
            lineage.append(program)
            current_id = program.parent_id

        lineage.reverse()
        return lineage

    def get_children(self, program_id: str) -> List[ProgramRecord]:
        """Get all direct children of a program."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM programs WHERE parent_id = ? ORDER BY combined_score DESC",
            (program_id,),
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def count_programs(self, generation: Optional[int] = None) -> int:
        """
        Count programs in the store.

        Args:
            generation: Optional generation filter

        Returns:
            Number of programs
        """
        cursor = self._conn.cursor()

        if generation is not None:
            cursor.execute(
                "SELECT COUNT(*) FROM programs WHERE generation = ?",
                (generation,),
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM programs")

        return cursor.fetchone()[0]

    def get_max_generation(self) -> int:
        """Get the highest generation number."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT MAX(generation) FROM programs")
        result = cursor.fetchone()[0]
        return result if result is not None else 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the store."""
        cursor = self._conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total_programs,
                MAX(generation) as max_generation,
                MAX(combined_score) as best_score,
                AVG(combined_score) as avg_score,
                COUNT(DISTINCT generation) as num_generations
            FROM programs
        """)

        row = cursor.fetchone()

        return {
            "total_programs": row[0] or 0,
            "max_generation": row[1] or 0,
            "best_score": row[2] or 0.0,
            "avg_score": row[3] or 0.0,
            "num_generations": row[4] or 0,
        }

    def get_embedding(self, program_id: str) -> Optional[List[float]]:
        """Get embedding for a program."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT embedding FROM programs WHERE program_id = ?",
            (program_id,),
        )
        row = cursor.fetchone()

        if row and row[0]:
            return pickle.loads(row[0])
        return None

    def get_all_embeddings(self) -> Dict[str, List[float]]:
        """Get all embeddings as a dictionary."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT program_id, embedding FROM programs WHERE embedding IS NOT NULL")

        return {
            row[0]: pickle.loads(row[1])
            for row in cursor.fetchall()
        }

    def find_similar(
        self,
        embedding: List[float],
        threshold: float = 0.9,
        limit: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find programs with similar embeddings.

        Args:
            embedding: Query embedding
            threshold: Minimum similarity threshold
            limit: Maximum results to return

        Returns:
            List of (program_id, similarity) tuples
        """
        from madevolve.common.helpers import cosine_similarity

        all_embeddings = self.get_all_embeddings()
        similarities = []

        for program_id, stored_embedding in all_embeddings.items():
            sim = cosine_similarity(embedding, stored_embedding)
            if sim >= threshold:
                similarities.append((program_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    def export_all(self) -> List[Dict[str, Any]]:
        """Export all programs as dictionaries."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM programs ORDER BY generation, combined_score DESC")

        return [
            {
                "program_id": row["program_id"],
                "code": row["code"],
                "parent_id": row["parent_id"],
                "generation": row["generation"],
                "combined_score": row["combined_score"],
                "public_metrics": json.loads(row["public_metrics"] or "{}"),
                "private_metrics": json.loads(row["private_metrics"] or "{}"),
                "text_feedback": row["text_feedback"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in cursor.fetchall()
        ]

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
