"""
Cluster job runner implementation.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class ClusterRunner:
    """
    Runs evaluation jobs on a cluster (SLURM, etc).

    Provides job submission, status monitoring, and
    result collection for cluster-based execution.
    """

    def __init__(
        self,
        scheduler: str = "slurm",
        partition: Optional[str] = None,
        gpus: int = 0,
        cpus: int = 1,
        memory_gb: float = 4.0,
        timeout: float = 300.0,
        conda_env: Optional[str] = None,
    ):
        """
        Initialize the cluster runner.

        Args:
            scheduler: Cluster scheduler type (slurm)
            partition: Cluster partition/queue name
            gpus: GPUs per job
            cpus: CPUs per job
            memory_gb: Memory in GB
            timeout: Job timeout in seconds
            conda_env: Conda environment to activate
        """
        self.scheduler = scheduler
        self.partition = partition
        self.gpus = gpus
        self.cpus = cpus
        self.memory_gb = memory_gb
        self.timeout = timeout
        self.conda_env = conda_env

    def submit(
        self,
        evaluator_script: str,
        code_path: str,
        work_dir: str,
        job_name: str = "madevolve",
    ) -> Optional[str]:
        """
        Submit a job to the cluster.

        Args:
            evaluator_script: Path to evaluator script
            code_path: Path to code file
            work_dir: Working directory
            job_name: Name for the job

        Returns:
            Job ID or None if submission failed
        """
        if self.scheduler == "slurm":
            return self._submit_slurm(evaluator_script, code_path, work_dir, job_name)
        else:
            logger.error(f"Unknown scheduler: {self.scheduler}")
            return None

    def _submit_slurm(
        self,
        evaluator_script: str,
        code_path: str,
        work_dir: str,
        job_name: str,
    ) -> Optional[str]:
        """Submit job to SLURM."""
        script_path = Path(work_dir) / "job.slurm"
        self._write_slurm_script(
            script_path,
            evaluator_script,
            code_path,
            work_dir,
            job_name,
        )

        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            cwd=work_dir,
        )

        if result.returncode == 0:
            # Parse job ID from output
            job_id = result.stdout.strip().split()[-1]
            return job_id

        logger.error(f"SLURM submission failed: {result.stderr}")
        return None

    def _write_slurm_script(
        self,
        script_path: Path,
        evaluator_script: str,
        code_path: str,
        work_dir: str,
        job_name: str,
    ):
        """Generate SLURM batch script."""
        timeout_mins = int(self.timeout / 60) + 5

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={work_dir}/stdout.log",
            f"#SBATCH --error={work_dir}/stderr.log",
            f"#SBATCH --cpus-per-task={self.cpus}",
            f"#SBATCH --mem={int(self.memory_gb)}G",
            f"#SBATCH --time={timeout_mins}",
        ]

        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")

        if self.gpus > 0:
            lines.append(f"#SBATCH --gres=gpu:{self.gpus}")

        lines.append("")

        if self.conda_env:
            lines.extend([
                "source $(conda info --base)/etc/profile.d/conda.sh",
                f"conda activate {self.conda_env}",
                "",
            ])

        lines.extend([
            f"cd {work_dir}",
            f"python {evaluator_script} {code_path}",
        ])

        script_path.write_text("\n".join(lines))

    def check_status(self, job_id: str) -> str:
        """
        Check job status.

        Args:
            job_id: Cluster job ID

        Returns:
            Status string (pending, running, completed, failed)
        """
        if self.scheduler == "slurm":
            return self._check_slurm_status(job_id)
        return "unknown"

    def _check_slurm_status(self, job_id: str) -> str:
        """Check SLURM job status."""
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%t"],
            capture_output=True,
            text=True,
        )

        status = result.stdout.strip()

        if not status:
            # Job not in queue - check if completed
            return "completed"

        status_map = {
            "PD": "pending",
            "R": "running",
            "CG": "completing",
            "CD": "completed",
            "F": "failed",
            "CA": "cancelled",
        }

        return status_map.get(status, "unknown")

    def cancel(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Cluster job ID

        Returns:
            True if cancellation was successful
        """
        if self.scheduler == "slurm":
            result = subprocess.run(["scancel", job_id])
            return result.returncode == 0

        return False
