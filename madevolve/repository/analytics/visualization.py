"""
Report generation and visualization for evolution analysis.

This module provides automated report generation including
lineage analysis, performance tracking, and evolution insights.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from madevolve.repository.storage.artifact_store import ArtifactStore, ProgramRecord

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive evolution reports.

    Produces markdown reports with lineage analysis, performance
    metrics, and LLM-generated insights.
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        results_dir: str,
        config,
    ):
        """
        Initialize the report generator.

        Args:
            artifact_store: Program storage
            results_dir: Directory for outputs
            config: Evolution configuration
        """
        self.artifact_store = artifact_store
        self.results_dir = Path(results_dir)
        self.config = config

    def generate(
        self,
        best_program_id: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive evolution report.

        Args:
            best_program_id: ID of the best evolved program
            output_dir: Optional output directory override

        Returns:
            Path to the generated report
        """
        output_path = Path(output_dir) if output_dir else self.results_dir / "report"
        output_path.mkdir(parents=True, exist_ok=True)

        # Gather data
        best_program = self.artifact_store.get(best_program_id)
        baseline = self._get_baseline()
        lineage = self.artifact_store.get_lineage(best_program_id)
        stats = self.artifact_store.get_statistics()

        # Generate report sections
        sections = [
            self._generate_header(),
            self._generate_summary(best_program, baseline, stats),
            self._generate_lineage_section(lineage),
            self._generate_metrics_section(stats),
            self._generate_code_comparison(baseline, best_program),
        ]

        # Write report
        report_content = "\n\n".join(sections)
        report_path = output_path / "evolution_report.md"

        with open(report_path, "w") as f:
            f.write(report_content)

        # Export data
        self._export_data(output_path, best_program, lineage, stats)

        logger.info(f"Report generated: {report_path}")
        return str(report_path)

    def _get_baseline(self) -> Optional[ProgramRecord]:
        """Get the baseline (generation 0) program."""
        gen0 = self.artifact_store.get_by_generation(0)
        return gen0[0] if gen0 else None

    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""# MadEvolve Evolution Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---"""

    def _generate_summary(
        self,
        best: Optional[ProgramRecord],
        baseline: Optional[ProgramRecord],
        stats: Dict[str, Any],
    ) -> str:
        """Generate summary section."""
        lines = ["## Summary", ""]

        if best:
            lines.append(f"**Best Score:** {best.combined_score:.4f}")

        if baseline:
            lines.append(f"**Baseline Score:** {baseline.combined_score:.4f}")

        if best and baseline:
            improvement = best.combined_score - baseline.combined_score
            pct = (improvement / max(abs(baseline.combined_score), 0.001)) * 100
            lines.append(f"**Improvement:** {improvement:+.4f} ({pct:+.1f}%)")

        lines.extend([
            "",
            f"**Total Programs Evaluated:** {stats.get('total_programs', 0)}",
            f"**Total Generations:** {stats.get('max_generation', 0)}",
            f"**Average Score:** {stats.get('avg_score', 0):.4f}",
        ])

        return "\n".join(lines)

    def _generate_lineage_section(self, lineage: List[ProgramRecord]) -> str:
        """Generate lineage analysis section."""
        if not lineage:
            return "## Lineage\n\nNo lineage data available."

        lines = ["## Lineage", "", "Evolution path from baseline to best:"]

        for i, program in enumerate(lineage):
            prefix = "  " * i + "├── " if i > 0 else ""
            score_str = f"{program.combined_score:.4f}"
            lines.append(f"{prefix}**Gen {program.generation}** (Score: {score_str})")

            # Show improvement from parent
            if i > 0:
                parent = lineage[i - 1]
                delta = program.combined_score - parent.combined_score
                if delta != 0:
                    lines.append(f"{' ' * (i * 2)}    Δ = {delta:+.4f}")

        return "\n".join(lines)

    def _generate_metrics_section(self, stats: Dict[str, Any]) -> str:
        """Generate metrics analysis section."""
        lines = [
            "## Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]

        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |")
            else:
                lines.append(f"| {key} | {value} |")

        return "\n".join(lines)

    def _generate_code_comparison(
        self,
        baseline: Optional[ProgramRecord],
        best: Optional[ProgramRecord],
    ) -> str:
        """Generate code comparison section."""
        if not baseline or not best:
            return "## Code Comparison\n\nInsufficient data for comparison."

        lines = [
            "## Code Comparison",
            "",
            "### Baseline Code",
            "",
            "```python",
            baseline.code[:2000] + ("..." if len(baseline.code) > 2000 else ""),
            "```",
            "",
            "### Best Evolved Code",
            "",
            "```python",
            best.code[:2000] + ("..." if len(best.code) > 2000 else ""),
            "```",
        ]

        return "\n".join(lines)

    def _export_data(
        self,
        output_path: Path,
        best: Optional[ProgramRecord],
        lineage: List[ProgramRecord],
        stats: Dict[str, Any],
    ):
        """Export raw data as JSON."""
        data = {
            "stats": stats,
            "best_program": {
                "program_id": best.program_id if best else None,
                "score": best.combined_score if best else None,
                "generation": best.generation if best else None,
            },
            "lineage": [
                {
                    "program_id": p.program_id,
                    "generation": p.generation,
                    "score": p.combined_score,
                }
                for p in lineage
            ],
        }

        with open(output_path / "evolution_data.json", "w") as f:
            json.dump(data, f, indent=2)


class ProgressTracker:
    """
    Tracks evolution progress for real-time monitoring.
    """

    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def record(
        self,
        generation: int,
        best_score: float,
        avg_score: float,
        num_programs: int,
        improvements: int,
    ):
        """Record progress for a generation."""
        self._history.append({
            "generation": generation,
            "best_score": best_score,
            "avg_score": avg_score,
            "num_programs": num_programs,
            "improvements": improvements,
            "timestamp": datetime.now().isoformat(),
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """Get complete history."""
        return self._history.copy()

    def export_csv(self, path: str):
        """Export history to CSV."""
        if not self._history:
            return

        import csv

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._history[0].keys())
            writer.writeheader()
            writer.writerows(self._history)
