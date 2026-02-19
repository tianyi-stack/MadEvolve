"""
Native (local) job runner implementation.
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class NativeRunner:
    """
    Runs evaluation jobs as native local processes.

    Provides process management, output capture, and
    timeout handling for local job execution.
    """

    def __init__(self, timeout: float = 300.0, conda_env: Optional[str] = None):
        """
        Initialize the native runner.

        Args:
            timeout: Job timeout in seconds
            conda_env: Optional conda environment to activate
        """
        self.timeout = timeout
        self.conda_env = conda_env
        self._active_processes: Dict[str, subprocess.Popen] = {}

    def run_sync(
        self,
        evaluator_script: str,
        code_path: str,
        work_dir: str,
    ) -> Dict[str, Any]:
        """
        Run evaluation synchronously.

        Args:
            evaluator_script: Path to evaluator script
            code_path: Path to code file
            work_dir: Working directory

        Returns:
            Result dictionary
        """
        cmd = self._build_command(evaluator_script, code_path)

        stdout_path = Path(work_dir) / "stdout.log"
        stderr_path = Path(work_dir) / "stderr.log"

        try:
            with open(stdout_path, "w") as stdout_f, \
                 open(stderr_path, "w") as stderr_f:

                result = subprocess.run(
                    cmd,
                    cwd=work_dir,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    timeout=self.timeout,
                    env=os.environ.copy(),
                )

            return self._collect_result(work_dir, result.returncode)

        except subprocess.TimeoutExpired:
            logger.warning(f"Job timed out after {self.timeout}s")
            return {"success": False, "error": "Timeout"}

        except Exception as e:
            logger.error(f"Job execution failed: {e}")
            return {"success": False, "error": str(e)}

    def _build_command(self, evaluator_script: str, code_path: str) -> list:
        """Build the execution command."""
        if self.conda_env:
            return [
                "conda", "run", "-n", self.conda_env,
                "python", evaluator_script, code_path,
            ]
        return ["python", evaluator_script, code_path]

    def _collect_result(
        self,
        work_dir: str,
        return_code: int,
    ) -> Dict[str, Any]:
        """Collect result from work directory."""
        import json

        result_path = Path(work_dir) / "result.json"

        if result_path.exists():
            try:
                with open(result_path) as f:
                    result = json.load(f)
                result["success"] = result.get("success", return_code == 0)
                return result
            except json.JSONDecodeError:
                pass

        return {
            "success": return_code == 0,
            "combined_score": 0.0,
            "public_metrics": {},
            "private_metrics": {},
            "error": f"Return code: {return_code}" if return_code != 0 else None,
        }
