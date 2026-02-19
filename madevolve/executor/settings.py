"""
Executor settings and configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutorSettings:
    """Settings for job execution."""
    mode: str = "local"  # local, slurm, kubernetes
    max_parallel_jobs: int = 4
    timeout: float = 300.0

    # Local execution
    conda_env: Optional[str] = None
    work_dir: Optional[str] = None

    # Cluster execution
    partition: Optional[str] = None
    gpus_per_job: int = 0
    cpus_per_job: int = 1
    memory_gb: float = 4.0
    container_image: Optional[str] = None
