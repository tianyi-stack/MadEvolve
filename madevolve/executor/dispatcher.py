"""
Job Dispatcher - Coordinates job execution across different backends.

This module handles job submission, monitoring, and result collection
for local and cluster-based execution.
"""

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class JobHandle:
    """Handle for tracking a submitted job."""
    job_id: str
    program_id: str
    work_dir: str
    submit_time: float
    status: str = "pending"  # pending, running, completed, failed
    process: Optional[subprocess.Popen] = None
    result: Optional[Dict[str, Any]] = None


class JobDispatcher:
    """
    Dispatches and manages evaluation jobs.

    Supports local subprocess execution and cluster backends.
    """

    def __init__(self, container):
        """
        Initialize the job dispatcher.

        Args:
            container: ServiceContainer with configuration
        """
        self.config = container.config.executor
        self._jobs: Dict[str, JobHandle] = {}
        self._job_counter = 0
        self._lock = threading.Lock()

        # Create runner based on mode
        if self.config.mode == "local":
            self._runner = LocalRunner(self.config)
        elif self.config.mode == "slurm":
            self._runner = SlurmRunner(self.config)
        else:
            logger.warning(f"Unknown executor mode '{self.config.mode}', using local")
            self._runner = LocalRunner(self.config)

    def submit(
        self,
        program_id: str,
        code: str,
        evaluator_script: str,
        work_dir: str,
    ) -> str:
        """
        Submit a job for evaluation.

        Args:
            program_id: Program identifier
            code: Source code to evaluate
            evaluator_script: Path to evaluator script
            work_dir: Working directory for the job

        Returns:
            Job ID
        """
        with self._lock:
            self._job_counter += 1
            job_id = f"job_{self._job_counter:06d}"

        # Create work directory
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        # Write code to file
        code_path = work_path / "candidate.py"
        with open(code_path, "w") as f:
            f.write(code)

        # Submit to runner
        handle = self._runner.submit(
            job_id=job_id,
            program_id=program_id,
            code_path=str(code_path),
            evaluator_script=evaluator_script,
            work_dir=str(work_path),
        )

        with self._lock:
            self._jobs[job_id] = handle

        logger.debug(f"Submitted job {job_id} for program {program_id}")
        return job_id

    def is_complete(self, job_id: str) -> bool:
        """
        Check if a job is complete.

        Args:
            job_id: Job identifier

        Returns:
            True if job has completed
        """
        with self._lock:
            handle = self._jobs.get(job_id)
            if handle is None:
                return True

        return self._runner.is_complete(handle)

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a completed job.

        Args:
            job_id: Job identifier

        Returns:
            Result dictionary or None if not available
        """
        with self._lock:
            handle = self._jobs.get(job_id)
            if handle is None:
                return None

        return self._runner.collect_result(handle)

    def wait_for_completion(
        self,
        job_id: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Wait for a job to complete.

        Args:
            job_id: Job identifier
            timeout: Optional timeout in seconds

        Returns:
            True if job completed, False if timed out
        """
        start_time = time.time()
        effective_timeout = timeout or self.config.timeout

        while True:
            if self.is_complete(job_id):
                return True

            if time.time() - start_time > effective_timeout:
                return False

            time.sleep(0.5)

    def get_active_count(self) -> int:
        """Get number of active jobs."""
        with self._lock:
            return sum(
                1 for h in self._jobs.values()
                if h.status in ("pending", "running")
            )

    def cancel(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled
        """
        with self._lock:
            handle = self._jobs.get(job_id)
            if handle is None:
                return False

        return self._runner.cancel(handle)

    def cleanup(self):
        """Cancel all active jobs and cleanup."""
        with self._lock:
            for job_id in list(self._jobs.keys()):
                self.cancel(job_id)
            self._jobs.clear()


class LocalRunner:
    """
    Executes jobs as local subprocesses.
    """

    def __init__(self, config):
        """Initialize the local runner."""
        self.config = config
        self._processes: Dict[str, subprocess.Popen] = {}

    def submit(
        self,
        job_id: str,
        program_id: str,
        code_path: str,
        evaluator_script: str,
        work_dir: str,
    ) -> JobHandle:
        """Submit a job as subprocess."""
        # Resolve paths to absolute (they may be relative to the launch
        # directory, not the work_dir where the subprocess runs)
        evaluator_script = str(Path(evaluator_script).resolve())
        code_path = str(Path(code_path).resolve())

        # Build command
        cmd = ["python", evaluator_script, code_path]

        if self.config.conda_env:
            # Activate conda environment
            cmd = [
                "conda", "run", "-n", self.config.conda_env,
                "python", evaluator_script, code_path,
            ]

        # Set up output files
        stdout_path = Path(work_dir) / "stdout.log"
        stderr_path = Path(work_dir) / "stderr.log"

        stdout_file = open(stdout_path, "w")
        stderr_file = open(stderr_path, "w")

        # Start process
        process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=stdout_file,
            stderr=stderr_file,
            env=os.environ.copy(),
        )

        handle = JobHandle(
            job_id=job_id,
            program_id=program_id,
            work_dir=work_dir,
            submit_time=time.time(),
            status="running",
            process=process,
        )

        self._processes[job_id] = process

        return handle

    def is_complete(self, handle: JobHandle) -> bool:
        """Check if job process has completed."""
        if handle.process is None:
            return True

        poll = handle.process.poll()
        if poll is not None:
            handle.status = "completed" if poll == 0 else "failed"
            return True

        # Check timeout
        elapsed = time.time() - handle.submit_time
        if elapsed > self.config.timeout:
            self.cancel(handle)
            handle.status = "failed"
            return True

        return False

    def collect_result(self, handle: JobHandle) -> Optional[Dict[str, Any]]:
        """Collect results from completed job."""
        result_path = Path(handle.work_dir) / "result.json"

        if result_path.exists():
            try:
                with open(result_path, "r") as f:
                    result = json.load(f)

                # Add standard fields
                result.setdefault("success", result.get("combined_score", 0) > 0)

                # Read logs
                stdout_path = Path(handle.work_dir) / "stdout.log"
                stderr_path = Path(handle.work_dir) / "stderr.log"

                if stdout_path.exists():
                    result["stdout"] = stdout_path.read_text()[:10000]

                if stderr_path.exists():
                    result["stderr"] = stderr_path.read_text()[:10000]

                return result

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse result for {handle.job_id}: {e}")
                return {"success": False, "error": f"JSON parse error: {e}"}

        return {"success": False, "error": "No result file found"}

    def cancel(self, handle: JobHandle) -> bool:
        """Cancel a running job."""
        if handle.process and handle.process.poll() is None:
            try:
                handle.process.terminate()
                handle.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handle.process.kill()
            return True
        return False


class SlurmRunner:
    """
    Executes jobs via SLURM cluster scheduler.
    """

    def __init__(self, config):
        """Initialize the SLURM runner."""
        self.config = config
        self._slurm_ids: Dict[str, str] = {}

    def submit(
        self,
        job_id: str,
        program_id: str,
        code_path: str,
        evaluator_script: str,
        work_dir: str,
    ) -> JobHandle:
        """Submit a job to SLURM."""
        # Generate SLURM script
        script_path = Path(work_dir) / "job.slurm"
        self._generate_slurm_script(
            script_path,
            evaluator_script,
            code_path,
            work_dir,
            job_id,
        )

        # Submit to SLURM
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            cwd=work_dir,
        )

        if result.returncode != 0:
            logger.error(f"SLURM submission failed: {result.stderr}")
            return JobHandle(
                job_id=job_id,
                program_id=program_id,
                work_dir=work_dir,
                submit_time=time.time(),
                status="failed",
            )

        # Parse SLURM job ID
        slurm_id = result.stdout.strip().split()[-1]
        self._slurm_ids[job_id] = slurm_id

        return JobHandle(
            job_id=job_id,
            program_id=program_id,
            work_dir=work_dir,
            submit_time=time.time(),
            status="pending",
        )

    def _generate_slurm_script(
        self,
        script_path: Path,
        evaluator_script: str,
        code_path: str,
        work_dir: str,
        job_name: str,
    ):
        """Generate SLURM batch script."""
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={work_dir}/stdout.log",
            f"#SBATCH --error={work_dir}/stderr.log",
            f"#SBATCH --cpus-per-task={self.config.cpus_per_job}",
            f"#SBATCH --mem={int(self.config.memory_gb)}G",
        ]

        if self.config.partition:
            lines.append(f"#SBATCH --partition={self.config.partition}")

        if self.config.gpus_per_job > 0:
            lines.append(f"#SBATCH --gres=gpu:{self.config.gpus_per_job}")

        # Time limit
        timeout_minutes = int(self.config.timeout / 60) + 5
        lines.append(f"#SBATCH --time={timeout_minutes}")

        lines.append("")

        # Conda activation if specified
        if self.config.conda_env:
            lines.extend([
                "source $(conda info --base)/etc/profile.d/conda.sh",
                f"conda activate {self.config.conda_env}",
            ])

        # Run command
        lines.extend([
            "",
            f"cd {work_dir}",
            f"python {evaluator_script} {code_path}",
        ])

        script_path.write_text("\n".join(lines))

    def is_complete(self, handle: JobHandle) -> bool:
        """Check if SLURM job has completed."""
        slurm_id = self._slurm_ids.get(handle.job_id)
        if not slurm_id:
            return True

        result = subprocess.run(
            ["squeue", "-j", slurm_id, "-h"],
            capture_output=True,
            text=True,
        )

        # If job not in queue, it's complete
        if not result.stdout.strip():
            return True

        return False

    def collect_result(self, handle: JobHandle) -> Optional[Dict[str, Any]]:
        """Collect results from completed SLURM job."""
        # Same as local runner
        result_path = Path(handle.work_dir) / "result.json"

        if result_path.exists():
            try:
                with open(result_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"success": False, "error": "JSON parse error"}

        return {"success": False, "error": "No result file found"}

    def cancel(self, handle: JobHandle) -> bool:
        """Cancel a SLURM job."""
        slurm_id = self._slurm_ids.get(handle.job_id)
        if slurm_id:
            subprocess.run(["scancel", slurm_id])
            return True
        return False
