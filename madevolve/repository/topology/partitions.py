"""
Population topology management for quality-diversity optimization.

This module implements hybrid population management combining:
- MAP-Elites-style behavioral partitioning
- Island model for parallel evolution
- Elite archive for preserving best solutions
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from madevolve.repository.storage.artifact_store import ProgramRecord

logger = logging.getLogger(__name__)


@dataclass
class CellEntry:
    """Entry in a MAP-Elites cell."""
    program_id: str
    score: float
    features: Tuple[float, ...]


@dataclass
class IslandMember:
    """Member of an island population."""
    program_id: str
    score: float
    generation: int


class PartitionGrid:
    """
    MAP-Elites-style behavioral partitioning.

    Maintains a grid of cells, each containing the best program
    for that behavioral niche.
    """

    def __init__(
        self,
        dimensions: List[str],
        bins_per_dimension: int = 10,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize the partition grid.

        Args:
            dimensions: Names of behavioral dimensions
            bins_per_dimension: Number of bins per dimension
            feature_ranges: Optional fixed ranges for features
        """
        self.dimensions = dimensions
        self.bins = bins_per_dimension
        self.feature_ranges = feature_ranges or {}
        self._grid: Dict[Tuple[int, ...], CellEntry] = {}

        # Auto-computed ranges
        self._observed_ranges: Dict[str, Tuple[float, float]] = {}

    def try_insert(
        self,
        program_id: str,
        score: float,
        features: Dict[str, float],
    ) -> bool:
        """
        Try to insert a program into the grid.

        Args:
            program_id: Program identifier
            score: Program fitness score
            features: Feature values for each dimension

        Returns:
            True if program was inserted (new cell or better than incumbent)
        """
        # Update observed ranges
        for dim in self.dimensions:
            value = features.get(dim, 0.0)
            if dim not in self._observed_ranges:
                self._observed_ranges[dim] = (value, value)
            else:
                low, high = self._observed_ranges[dim]
                self._observed_ranges[dim] = (min(low, value), max(high, value))

        # Compute cell coordinates
        coords = self._discretize(features)

        # Check if cell is empty or new program is better
        current = self._grid.get(coords)

        if current is None or score > current.score:
            feature_tuple = tuple(features.get(d, 0.0) for d in self.dimensions)
            self._grid[coords] = CellEntry(
                program_id=program_id,
                score=score,
                features=feature_tuple,
            )
            return True

        return False

    def _discretize(self, features: Dict[str, float]) -> Tuple[int, ...]:
        """Convert continuous features to grid coordinates."""
        coords = []

        for dim in self.dimensions:
            value = features.get(dim, 0.0)

            # Get range
            if dim in self.feature_ranges:
                low, high = self.feature_ranges[dim]
            elif dim in self._observed_ranges:
                low, high = self._observed_ranges[dim]
            else:
                low, high = 0.0, 1.0

            # Handle zero range
            if high <= low:
                high = low + 1.0

            # Normalize and bin
            normalized = (value - low) / (high - low)
            bin_idx = min(int(normalized * self.bins), self.bins - 1)
            bin_idx = max(0, bin_idx)
            coords.append(bin_idx)

        return tuple(coords)

    def sample_diverse(self, k: int = 5) -> List[str]:
        """Sample k programs from different cells."""
        cells = list(self._grid.values())

        if len(cells) <= k:
            return [c.program_id for c in cells]

        sampled = random.sample(cells, k)
        return [c.program_id for c in sampled]

    def get_coverage(self) -> float:
        """Get fraction of grid cells occupied."""
        total_cells = self.bins ** len(self.dimensions)
        return len(self._grid) / total_cells

    def get_all_program_ids(self) -> List[str]:
        """Get all program IDs in the grid."""
        return [entry.program_id for entry in self._grid.values()]

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            "grid": {
                str(k): {"program_id": v.program_id, "score": v.score, "features": v.features}
                for k, v in self._grid.items()
            },
            "observed_ranges": self._observed_ranges,
        }

    def restore_state(self, state: Dict[str, Any]):
        """Restore from serialized state."""
        self._observed_ranges = state.get("observed_ranges", {})
        self._grid = {}

        for k_str, v in state.get("grid", {}).items():
            # Parse tuple key
            coords = tuple(int(x) for x in k_str.strip("()").split(",") if x.strip())
            self._grid[coords] = CellEntry(
                program_id=v["program_id"],
                score=v["score"],
                features=tuple(v["features"]),
            )


class IslandCluster:
    """
    Island model for parallel evolution.

    Maintains multiple semi-isolated subpopulations with periodic migration.
    """

    def __init__(
        self,
        num_islands: int = 4,
        island_capacity: int = 15,
        migration_interval: int = 5,
        migration_rate: float = 0.1,
        topology: str = "ring",
    ):
        """
        Initialize the island cluster.

        Args:
            num_islands: Number of islands
            island_capacity: Maximum members per island
            migration_interval: Generations between migrations
            migration_rate: Fraction of population to migrate
            topology: Migration topology ("ring" or "fully_connected")
        """
        self.num_islands = num_islands
        self.capacity = island_capacity
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.topology = topology

        self._islands: List[List[IslandMember]] = [[] for _ in range(num_islands)]
        self._assignment: Dict[str, int] = {}  # program_id -> island_idx
        self._next_island = 0

    def assign(
        self,
        program_id: str,
        score: float,
        generation: int,
        parent_id: Optional[str] = None,
    ) -> int:
        """
        Assign a program to an island.

        Args:
            program_id: Program identifier
            score: Program fitness score
            generation: Current generation
            parent_id: Parent program ID (for inheritance)

        Returns:
            Island index assigned to
        """
        # Inherit parent's island if possible
        if parent_id and parent_id in self._assignment:
            island_idx = self._assignment[parent_id]
        else:
            # Round-robin assignment
            island_idx = self._next_island
            self._next_island = (self._next_island + 1) % self.num_islands

        # Add to island
        member = IslandMember(
            program_id=program_id,
            score=score,
            generation=generation,
        )

        self._islands[island_idx].append(member)
        self._assignment[program_id] = island_idx

        # Maintain capacity
        self._enforce_capacity(island_idx)

        return island_idx

    def _enforce_capacity(self, island_idx: int):
        """Remove lowest-scoring members if over capacity."""
        island = self._islands[island_idx]

        if len(island) > self.capacity:
            # Sort by score (ascending) and remove worst
            island.sort(key=lambda m: m.score)
            removed = island[:len(island) - self.capacity]
            self._islands[island_idx] = island[len(island) - self.capacity:]

            # Update assignment
            for member in removed:
                if member.program_id in self._assignment:
                    del self._assignment[member.program_id]

    def maybe_migrate(self, generation: int):
        """
        Perform migration if it's time.

        Args:
            generation: Current generation
        """
        if generation % self.migration_interval != 0:
            return

        logger.debug(f"Performing migration at generation {generation}")

        # Collect migrants from each island
        migrants: List[List[IslandMember]] = []

        for island in self._islands:
            if not island:
                migrants.append([])
                continue

            # Select best members to migrate
            n_migrants = max(1, int(len(island) * self.migration_rate))
            sorted_island = sorted(island, key=lambda m: m.score, reverse=True)
            migrants.append(sorted_island[:n_migrants])

        # Send migrants based on topology
        for src_idx, src_migrants in enumerate(migrants):
            if not src_migrants:
                continue

            if self.topology == "ring":
                dest_idx = (src_idx + 1) % self.num_islands
                self._send_migrants(src_migrants, dest_idx)

            elif self.topology == "fully_connected":
                # Send to all other islands
                for dest_idx in range(self.num_islands):
                    if dest_idx != src_idx:
                        self._send_migrants(src_migrants, dest_idx)

    def _send_migrants(self, migrants: List[IslandMember], dest_idx: int):
        """Send migrants to destination island."""
        for migrant in migrants:
            # Don't remove from source - just copy
            new_member = IslandMember(
                program_id=migrant.program_id,
                score=migrant.score,
                generation=migrant.generation,
            )
            self._islands[dest_idx].append(new_member)

        self._enforce_capacity(dest_idx)

    def get_island_members(self, island_idx: int) -> List[str]:
        """Get program IDs for an island."""
        return [m.program_id for m in self._islands[island_idx]]

    def sample_from_island(self, island_idx: Optional[int] = None) -> Optional[str]:
        """Sample a program from an island (or random island)."""
        if island_idx is None:
            island_idx = random.randrange(self.num_islands)

        island = self._islands[island_idx]
        if not island:
            return None

        # Weighted by score
        scores = [m.score for m in island]
        min_score = min(scores)
        weights = [s - min_score + 0.1 for s in scores]

        selected = random.choices(island, weights=weights)[0]
        return selected.program_id

    def get_statistics(self) -> Dict[str, Any]:
        """Get island statistics."""
        return {
            "num_islands": self.num_islands,
            "island_sizes": [len(island) for island in self._islands],
            "total_members": sum(len(island) for island in self._islands),
        }

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            "islands": [
                [{"program_id": m.program_id, "score": m.score, "generation": m.generation}
                 for m in island]
                for island in self._islands
            ],
            "assignment": self._assignment,
            "next_island": self._next_island,
        }

    def restore_state(self, state: Dict[str, Any]):
        """Restore from serialized state."""
        self._assignment = state.get("assignment", {})
        self._next_island = state.get("next_island", 0)
        self._islands = []

        for island_data in state.get("islands", []):
            island = [
                IslandMember(
                    program_id=m["program_id"],
                    score=m["score"],
                    generation=m["generation"],
                )
                for m in island_data
            ]
            self._islands.append(island)


class EliteVault:
    """
    Elite archive for preserving best solutions.

    Maintains a fixed-size archive of the best programs found.
    """

    def __init__(self, max_size: int = 50):
        """
        Initialize the elite vault.

        Args:
            max_size: Maximum archive size
        """
        self.max_size = max_size
        self._entries: Dict[str, float] = {}  # program_id -> score

    def add(self, program_id: str, score: float) -> bool:
        """
        Try to add a program to the vault.

        Args:
            program_id: Program identifier
            score: Program fitness score

        Returns:
            True if added to vault
        """
        # Check if already in vault
        if program_id in self._entries:
            self._entries[program_id] = max(self._entries[program_id], score)
            return True

        # Check if vault is full
        if len(self._entries) >= self.max_size:
            # Only add if better than worst
            min_id = min(self._entries, key=self._entries.get)
            if score > self._entries[min_id]:
                del self._entries[min_id]
                self._entries[program_id] = score
                return True
            return False

        self._entries[program_id] = score
        return True

    def get_top(self, k: int = 10) -> List[str]:
        """Get top k programs by score."""
        sorted_ids = sorted(
            self._entries.keys(),
            key=lambda x: self._entries[x],
            reverse=True,
        )
        return sorted_ids[:k]

    def sample(self, k: int = 1) -> List[str]:
        """Sample k programs weighted by score."""
        if not self._entries:
            return []

        ids = list(self._entries.keys())
        scores = [self._entries[pid] for pid in ids]

        # Normalize weights
        min_score = min(scores)
        weights = [s - min_score + 0.1 for s in scores]

        k = min(k, len(ids))
        sampled = random.choices(ids, weights=weights, k=k)

        return sampled

    def contains(self, program_id: str) -> bool:
        """Check if program is in vault."""
        return program_id in self._entries

    def get_all_ids(self) -> List[str]:
        """Get all program IDs in vault."""
        return list(self._entries.keys())

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {"entries": self._entries.copy()}

    def restore_state(self, state: Dict[str, Any]):
        """Restore from serialized state."""
        self._entries = state.get("entries", {})


class HybridPopulationManager:
    """
    Unified population manager combining all topology mechanisms.

    Coordinates MAP-Elites partitioning, island model, and elite vault
    for comprehensive quality-diversity optimization.
    """

    def __init__(self, container):
        """
        Initialize the hybrid population manager.

        Args:
            container: ServiceContainer with configuration
        """
        self.config = container.config.population

        # Initialize components
        self._grid = None
        self._islands = None
        self._vault = None

        if self.config.partition.enabled:
            self._grid = PartitionGrid(
                dimensions=self.config.partition.dimensions,
                bins_per_dimension=self.config.partition.bins_per_dimension,
                feature_ranges=self.config.partition.feature_ranges,
            )

        if self.config.islands.enabled:
            self._islands = IslandCluster(
                num_islands=self.config.islands.num_islands,
                island_capacity=self.config.islands.island_capacity,
                migration_interval=self.config.islands.migration_interval,
                migration_rate=self.config.islands.migration_rate,
                topology=self.config.islands.topology,
            )

        if self.config.archive.enabled:
            self._vault = EliteVault(max_size=self.config.archive.max_size)

        self._all_program_ids: Set[str] = set()

    def register(
        self,
        program_id: str,
        score: float,
        embedding: Optional[List[float]],
        features: Optional[Dict[str, float]] = None,
        parent_id: Optional[str] = None,
        generation: int = 0,
    ):
        """
        Register a program with all population mechanisms.

        Args:
            program_id: Program identifier
            score: Program fitness score
            embedding: Code embedding
            features: Behavioral features for partitioning
            parent_id: Parent program ID
            generation: Current generation
        """
        self._all_program_ids.add(program_id)

        # Add to partition grid
        if self._grid and features:
            self._grid.try_insert(program_id, score, features)

        # Assign to island
        if self._islands:
            self._islands.assign(program_id, score, generation, parent_id)

        # Add to elite vault
        if self._vault:
            self._vault.add(program_id, score)

    def get_selection_pool(self, generation: int) -> List[ProgramRecord]:
        """
        Get candidate pool for parent selection.

        This is called by ParentSelector to get the initial candidate set.
        Returns program IDs that need to be resolved to ProgramRecords.
        """
        # This will be populated by ParentSelector using artifact_store
        # Here we just return the IDs
        return []  # ParentSelector handles this with artifact_store

    def get_elite_ids(self) -> List[str]:
        """Get IDs of elite programs."""
        if self._vault:
            return self._vault.get_all_ids()
        return []

    def sample_diverse(self, k: int = 3) -> List[str]:
        """Sample structurally diverse programs from different MAP-Elites cells."""
        if self._grid:
            return self._grid.sample_diverse(k)
        return []

    def maybe_migrate(self, generation: int):
        """Trigger migration if using island model."""
        if self._islands:
            self._islands.maybe_migrate(generation)

    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics."""
        stats = {
            "total_programs": len(self._all_program_ids),
        }

        if self._grid:
            stats["grid_coverage"] = self._grid.get_coverage()
            stats["grid_cells"] = len(self._grid._grid)

        if self._islands:
            island_stats = self._islands.get_statistics()
            stats.update(island_stats)

        if self._vault:
            stats["elite_count"] = len(self._vault._entries)

        return stats

    def get_state(self) -> Dict[str, Any]:
        """Get complete serializable state."""
        state = {
            "all_program_ids": list(self._all_program_ids),
        }

        if self._grid:
            state["grid"] = self._grid.get_state()

        if self._islands:
            state["islands"] = self._islands.get_state()

        if self._vault:
            state["vault"] = self._vault.get_state()

        return state

    def restore_state(self, state: Dict[str, Any]):
        """Restore from serialized state."""
        self._all_program_ids = set(state.get("all_program_ids", []))

        if self._grid and "grid" in state:
            self._grid.restore_state(state["grid"])

        if self._islands and "islands" in state:
            self._islands.restore_state(state["islands"])

        if self._vault and "vault" in state:
            self._vault.restore_state(state["vault"])
