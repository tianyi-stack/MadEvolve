"""
Evolution Orchestrator - The main engine driving the evolutionary process.

This module coordinates all aspects of LLM-driven code evolution,
including population management, LLM interactions, job execution,
and strategic analysis.
"""

import logging
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from madevolve.branding import print_generation_header, print_result, print_summary, print_step, print_substep, print_error
from madevolve.common.helpers import compute_hash, format_duration, generate_uid
from madevolve.engine.configuration import EvolutionConfig, PatchMode
from madevolve.engine.container import ServiceContainer
from madevolve.engine.session import EvolutionSession
from madevolve.repository.topology.features import FeatureExtractor
from madevolve.transformer.blocks import extract_mutable_content, has_evolve_blocks, replace_mutable_content

logger = logging.getLogger(__name__)


@dataclass
class PendingJob:
    """Represents a job awaiting completion."""
    job_id: str
    program_id: str
    generation: int
    parent_id: Optional[str]
    code: str
    patch_mode: str
    model_used: str
    submit_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PreparedCandidate:
    """A candidate fully prepared by the LLM, ready for evaluation submission."""
    program_id: str
    code: str
    generation: int
    parent_id: str
    patch_mode: str
    model_used: str
    metadata: Dict[str, Any]


@dataclass
class EvaluationResult:
    """Result from program evaluation."""
    program_id: str
    success: bool
    combined_score: float
    public_metrics: Dict[str, float]
    private_metrics: Dict[str, float]
    text_feedback: str
    execution_time: float
    error_message: Optional[str] = None


class EvolutionOrchestrator:
    """
    Main orchestration engine for LLM-driven code evolution.

    This class coordinates the entire evolutionary process:
    1. Population initialization and management
    2. Parent selection and inspiration sampling
    3. LLM-based code mutation and generation
    4. Job submission and result processing
    5. Strategic analysis and adaptation
    6. Checkpointing and reporting
    """

    def __init__(
        self,
        config: EvolutionConfig,
        results_dir: str,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize the evolution orchestrator.

        Args:
            config: Complete evolution configuration
            results_dir: Directory for outputs and checkpoints
            checkpoint_path: Optional path to resume from checkpoint
        """
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Validate configuration
        issues = config.validate()
        if issues:
            raise ValueError(f"Configuration errors: {issues}")

        # Initialize random seed only for fresh runs (not resuming from checkpoint)
        # When resuming, we skip seeding to avoid generating duplicate UIDs
        if config.seed is not None and checkpoint_path is None:
            random.seed(config.seed)

        # Create service container
        self.container = ServiceContainer(
            config=config,
            results_dir=str(self.results_dir),
        )

        # Initialize session
        self.session = EvolutionSession(
            results_dir=str(self.results_dir),
        )

        if checkpoint_path:
            self.session.resume(checkpoint_path)
        else:
            config_hash = compute_hash(str(config))
            self.session.initialize(config_hash)

        # Runtime state
        self._pending_jobs: Dict[str, PendingJob] = {}
        self._state_lock = threading.RLock()

        # Restore best-score tracking from checkpoint, or start fresh
        if checkpoint_path and self.session.state:
            self._best_program_id: Optional[str] = self.session.state.best_program_id
            self._best_score: float = self.session.state.best_score
            self._stagnation_counter: int = self.session.state.generations_without_improvement
            logger.info(
                f"Restored best_score={self._best_score:.4f} "
                f"(program={self._best_program_id}) from checkpoint"
            )
        else:
            self._best_program_id: Optional[str] = None
            self._best_score: float = float("-inf")
            self._stagnation_counter: int = 0

        # Initialize components lazily
        self._gateway = None
        self._vectorizer = None
        self._artifact_store = None
        self._population = None
        self._selector = None
        self._composer = None
        self._dispatcher = None
        self._optimizer = None
        self._feature_extractor = None

        logger.info(f"Orchestrator initialized. Results dir: {self.results_dir}")

    def _init_services(self):
        """Lazily initialize all required services."""
        if self._gateway is not None:
            return

        print_step("Initializing services...")
        logger.info("Initializing services...")

        self._gateway = self.container.get_gateway()
        self._vectorizer = self.container.get_vectorizer()
        self._artifact_store = self.container.get_artifact_store()
        self._population = self.container.get_population_manager()
        self._selector = self.container.get_parent_selector()
        self._composer = self.container.get_composer()
        self._dispatcher = self.container.get_dispatcher()

        if self.config.optimization.enabled:
            self._optimizer = self.container.get_parameter_optimizer()

        # Initialize MAP-Elites feature extractor
        if self.config.population.partition.enabled:
            self._feature_extractor = FeatureExtractor(
                dimensions=self.config.population.partition.dimensions,
            )

        # Restore component states from checkpoint
        if self.session.state and self.session.state.current_generation > 0:
            self._restore_component_states()

        logger.info("Services initialized")

    def _restore_component_states(self):
        """Restore component states from session."""
        pop_state = self.session.get_component_state("population")
        if pop_state:
            self._population.restore_state(pop_state)

        fe_state = self.session.get_component_state("feature_extractor")
        if fe_state and self._feature_extractor:
            self._feature_extractor.restore_state(fe_state)

        selector_state = self.session.get_component_state("model_selector")
        if selector_state:
            self._gateway.restore_selector_state(selector_state)

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete evolution process.

        Returns:
            Dictionary containing evolution results and statistics
        """
        self._init_services()

        start_gen = self.session.state.current_generation if self.session.state else 0
        total_gens = self.config.num_generations

        # total_budget: number of candidates to submit (not idle loop iterations)
        total_budget = total_gens
        max_parallel = self.config.executor.max_parallel_jobs

        logger.info(
            f"Starting evolution from generation {start_gen}, "
            f"budget={total_budget} candidates, max_parallel={max_parallel}"
        )

        try:
            # Bootstrap generation 0 if needed
            if start_gen == 0:
                existing = self._artifact_store.get_by_generation(0)
                if existing:
                    # Reuse existing baseline from a previous run
                    best = max(existing, key=lambda p: p.combined_score)
                    self._best_program_id = best.program_id
                    self._best_score = best.combined_score

                    # Re-populate the population manager with all existing gen_0 programs
                    for prog in existing:
                        features = None
                        if self._feature_extractor and prog.embedding:
                            features = self._feature_extractor.compute_features(
                                code=prog.code,
                                embedding=prog.embedding,
                                score=prog.combined_score,
                                program_id=prog.program_id,
                                evaluator_metrics=prog.public_metrics,
                            )
                        self._population.register(
                            prog.program_id, prog.combined_score, prog.embedding,
                            features=features, parent_id=prog.parent_id, generation=0,
                        )

                    self._save_best_program()
                    logger.info(f"Reused {len(existing)} existing gen_0 programs (best={self._best_score:.4f})")
                    print_substep(f"Found {len(existing)} existing baseline(s), best score: {self._best_score:.4f}")
                else:
                    self._bootstrap_initial_generation()
                start_gen = 1

            # ── Event-driven main loop with LLM prefetch ──
            submission_count = 0
            last_stats_gen = start_gen - 1

            # Determine number of LLM producer threads
            num_producers = self.config.executor.num_llm_threads
            if num_producers <= 0:
                num_producers = min(max_parallel, 4)

            prefetch_queue: queue.Queue[PreparedCandidate] = queue.Queue(maxsize=max_parallel * 2)
            producer_stop = threading.Event()

            # Thread-safe slot counter for producer workers
            _slot_lock = threading.Lock()
            _next_slot = [0]  # mutable container for closure access

            def _claim_slot() -> Optional[int]:
                """Atomically claim the next slot number. Returns None if budget exhausted."""
                with _slot_lock:
                    if _next_slot[0] >= total_budget:
                        return None
                    slot = _next_slot[0]
                    _next_slot[0] += 1
                    return slot

            def _producer_worker():
                """Worker thread: claim slots and prepare candidates via LLM queries."""
                max_attempts_per_slot = 3
                while not producer_stop.is_set():
                    slot = _claim_slot()
                    if slot is None:
                        return
                    gen = slot // max_parallel + start_gen
                    candidate = None
                    for attempt in range(max_attempts_per_slot):
                        if producer_stop.is_set():
                            return
                        try:
                            candidate = self._prepare_candidate(gen)
                            if candidate is not None:
                                break
                        except Exception as e:
                            logger.warning(
                                f"Candidate preparation failed (slot {slot}, attempt {attempt + 1}/"
                                f"{max_attempts_per_slot}): {e}"
                            )
                    if candidate is not None:
                        while not producer_stop.is_set():
                            try:
                                prefetch_queue.put(candidate, timeout=2.0)
                                break
                            except queue.Full:
                                continue
                    else:
                        logger.warning(f"Candidate slot {slot} failed all preparation attempts")

            producer_pool = ThreadPoolExecutor(max_workers=num_producers, thread_name_prefix="llm-producer")
            producer_futures: List[Future] = [
                producer_pool.submit(_producer_worker) for _ in range(num_producers)
            ]

            def _producers_alive() -> bool:
                return any(not f.done() for f in producer_futures)

            try:
                while submission_count < total_budget or self._pending_jobs:
                    # 1. Fill open eval slots from prefetch queue
                    while len(self._pending_jobs) < max_parallel and submission_count < total_budget:
                        try:
                            candidate = prefetch_queue.get(timeout=1.0)
                        except queue.Empty:
                            # Producers might be slow or dead
                            if not _producers_alive() and prefetch_queue.empty():
                                break
                            continue
                        self._submit_candidate(candidate)
                        submission_count += 1

                        generation = (submission_count - 1) // max_parallel + start_gen
                        if generation > last_stats_gen:
                            total_display_gens = (total_budget - 1) // max_parallel + start_gen
                            print_generation_header(generation, total_display_gens)

                    # 2. If nothing is running and producers are dead, break
                    if not self._pending_jobs:
                        if not _producers_alive() and prefetch_queue.empty():
                            break
                        continue

                    # 3. Block until at least one eval completes
                    self._wait_for_any_completion()

                    # 4. Process completed evaluations (under state_lock)
                    with self._state_lock:
                        self._process_pending_jobs()

                    # 5. Periodic maintenance (based on submission count, not inflated counter)
                    generation = (submission_count - 1) // max_parallel + start_gen if submission_count > 0 else start_gen
                    if generation > last_stats_gen:
                        self._population.maybe_migrate(generation)
                        self._update_session_stats(generation)
                        if generation % self.config.storage.checkpoint_interval == 0:
                            self._save_checkpoint()
                        last_stats_gen = generation

            finally:
                producer_stop.set()
                producer_pool.shutdown(wait=True, cancel_futures=True)

            # Finalization
            return self._finalize()

        except KeyboardInterrupt:
            logger.warning("Evolution interrupted by user")
            self._save_checkpoint()
            raise

        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            self._save_checkpoint()
            raise

    def _bootstrap_initial_generation(self):
        """Initialize the population with a baseline program."""
        print_step("Bootstrapping generation 0...")
        logger.info("Bootstrapping generation 0...")

        if self.config.init_program_path:
            # Load initial program from file
            print_substep(f"Loading initial program from {self.config.init_program_path}")
            with open(self.config.init_program_path, "r") as f:
                initial_code = f.read()
            logger.info(f"Loaded initial program from {self.config.init_program_path}")
        else:
            # Generate initial program using LLM
            print_substep("Generating initial program via LLM...")
            initial_code = self._generate_initial_program()

        # Evaluate the initial program
        print_substep("Evaluating baseline program...")
        program_id = generate_uid()
        result = self._evaluate_program(program_id, initial_code, generation=0)

        if result.success:
            # Register in artifact store
            embedding = self._vectorizer.embed(extract_mutable_content(initial_code))
            self._artifact_store.register(
                program_id=program_id,
                code=initial_code,
                parent_id=None,
                generation=0,
                combined_score=result.combined_score,
                public_metrics=result.public_metrics,
                private_metrics=result.private_metrics,
                text_feedback=result.text_feedback,
                embedding=embedding,
                metadata={"is_baseline": True},
            )

            # Compute MAP-Elites features
            features = None
            if self._feature_extractor:
                features = self._feature_extractor.compute_features(
                    code=initial_code,
                    embedding=embedding,
                    score=result.combined_score,
                    program_id=program_id,
                    evaluator_metrics=result.public_metrics,
                )

            # Add to population
            self._population.register(
                program_id, result.combined_score, embedding,
                features=features, parent_id=None, generation=0,
            )

            self._best_program_id = program_id
            self._best_score = result.combined_score
            self._save_best_program()

            print_substep(f"Baseline score: {result.combined_score:.4f}")
            logger.info(f"Baseline program registered: score={result.combined_score:.4f}")
        else:
            print_error(f"Baseline evaluation failed: {result.error_message}")
            raise RuntimeError(f"Failed to evaluate initial program: {result.error_message}")

    def _generate_initial_program(self) -> str:
        """Generate an initial program using LLM."""
        from madevolve.templates.bootstrap import build_initial_prompt

        prompt = build_initial_prompt(self.config.task_description)
        response = self._gateway.query(
            system_message="You are an expert programmer. Generate clean, functional code.",
            user_message=prompt,
        )

        # Extract code from response
        from madevolve.transformer.changeset import extract_code_block
        code = extract_code_block(response.content)

        if not code:
            raise ValueError("Failed to extract code from LLM response")

        return code

    def _prepare_candidate(self, generation: int) -> Optional[PreparedCandidate]:
        """Prepare a candidate: select parent, query LLM, apply patch. Thread-safe.

        Phase 1 (under _state_lock): read shared state — parent selection,
        patch mode, prompt composition (~100ms).
        Phase 2 (no lock): LLM query + patch application (10-30s).
        """
        # Phase 1: read shared state under lock
        with self._state_lock:
            selection = self._selector.sample(
                generation=generation,
                artifact_store=self._artifact_store,
                population=self._population,
            )
            if hasattr(selection, "parent"):
                parent = selection.parent
                archive_inspirations = selection.archive_inspirations
                top_k_inspirations = selection.top_k_inspirations
                diverse_inspirations = getattr(selection, "diverse_inspirations", [])
            else:
                parent, archive_inspirations, top_k_inspirations = selection
                diverse_inspirations = []

            patch_mode = self._select_patch_mode()

            prompt = self._composer.compose(
                parent=parent,
                archive_inspirations=archive_inspirations,
                top_k_inspirations=top_k_inspirations,
                patch_mode=patch_mode,
                task_description=self.config.task_description,
                diverse_inspirations=diverse_inspirations,
            )
            system_message = self._composer.get_system_message(patch_mode)

        # Phase 2: LLM query + patch application (no lock, slow)
        print_substep(f"Parent {parent.program_id[:8]}... (score={parent.combined_score:.4f}) | mode={patch_mode}")

        max_retries = self.config.patch_policy.max_patch_retries
        new_code = None
        total_cost = 0.0
        last_raw_output = ""
        model_used = None
        last_attempt = 0

        conversation_history = [
            {"role": "user", "content": prompt},
        ]

        for attempt in range(1, max_retries + 1):
            last_attempt = attempt
            if attempt == 1:
                response = self._gateway.query(
                    system_message=system_message,
                    user_message=prompt,
                )
            else:
                response = self._gateway.query_multiturn(
                    messages=conversation_history,
                    system_message=system_message,
                    model=model_used,
                )

            model_used = response.model_name
            total_cost += response.cost
            last_raw_output = response.content

            print_substep(f"LLM response from {response.model_name} (attempt {attempt}/{max_retries})")

            # Check for truncation (finish_reason="length")
            if response.finish_reason == "length":
                error_msg = (
                    "Response was truncated (hit token limit). "
                    f"Got {response.completion_tokens} tokens."
                )
                logger.warning(f"LLM response truncated (attempt {attempt}/{max_retries}): {error_msg}")

                debug_dir = self.results_dir / "debug" / f"gen_{generation}"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_file = debug_dir / f"raw_llm_attempt_{attempt}.txt"
                debug_file.write_text(response.content)

                if attempt < max_retries:
                    retry_instruction = self._build_retry_message(patch_mode, error_msg)
                    conversation_history.append({"role": "assistant", "content": response.content})
                    conversation_history.append({"role": "user", "content": retry_instruction})
                    print_substep(f"Retrying due to truncation: {error_msg}")
                continue

            # Try to apply patch
            new_code, error_msg = self._apply_patch(parent.code, response.content, patch_mode)

            if new_code is not None:
                break

            logger.warning(f"Patch extraction failed (attempt {attempt}/{max_retries}): {error_msg}")

            debug_dir = self.results_dir / "debug" / f"gen_{generation}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / f"raw_llm_attempt_{attempt}.txt"
            debug_file.write_text(response.content)

            if attempt < max_retries:
                retry_instruction = self._build_retry_message(patch_mode, error_msg)
                conversation_history.append({"role": "assistant", "content": response.content})
                conversation_history.append({"role": "user", "content": retry_instruction})
                print_substep(f"Retrying with error feedback: {error_msg}")

        if new_code is None:
            print_error(f"Failed to apply {patch_mode} patch after {max_retries} attempts")
            logger.warning(f"Failed to apply {patch_mode} patch after {max_retries} attempts")
            return None

        # Run inner-loop optimization if enabled
        if self._optimizer and self.config.optimization.enabled:
            new_code = self._optimizer.optimize(
                new_code,
                parent.code,
                self._evaluate_program_quick,
            )

        program_id = generate_uid()
        return PreparedCandidate(
            program_id=program_id,
            code=new_code,
            generation=generation,
            parent_id=parent.program_id,
            patch_mode=patch_mode,
            model_used=model_used,
            metadata={
                "patch_retries": last_attempt,
                "llm_total_cost": total_cost,
                "llm_raw_output": last_raw_output[:2000],
            },
        )

    def _submit_candidate(self, candidate: PreparedCandidate) -> str:
        """Submit a prepared candidate for evaluation. Fast (milliseconds)."""
        job_id = self._dispatcher.submit(
            program_id=candidate.program_id,
            code=candidate.code,
            evaluator_script=self.config.evaluator_script,
            work_dir=str(self.results_dir / "evaluations" / f"gen_{candidate.generation}" / candidate.program_id),
        )

        job = PendingJob(
            job_id=job_id,
            program_id=candidate.program_id,
            generation=candidate.generation,
            parent_id=candidate.parent_id,
            code=candidate.code,
            patch_mode=candidate.patch_mode,
            model_used=candidate.model_used,
            submit_time=time.time(),
            metadata=candidate.metadata,
        )
        self._pending_jobs[job_id] = job
        return job_id

    def _wait_for_any_completion(self, timeout: float = 300.0):
        """Block until at least one pending job completes."""
        start = time.time()
        while time.time() - start < timeout:
            for job_id in list(self._pending_jobs.keys()):
                if self._dispatcher.is_complete(job_id):
                    return
            time.sleep(0.5)

    def _diagnose_extraction_error(self, llm_content: str, patch_mode: str) -> str:
        """Diagnose why code extraction failed from LLM output."""
        import re

        if not llm_content or not llm_content.strip():
            return "LLM returned empty content"

        if patch_mode == "differential":
            # Check SEARCH/REPLACE markers
            from madevolve.transformer.patcher import validate_patch_syntax
            is_valid, issues = validate_patch_syntax(llm_content)
            if issues:
                return "; ".join(issues)
            return "Patch blocks present but search text not found in code"

        # For holistic / synthesis: check fenced code blocks
        has_fences = bool(re.search(r"```", llm_content))
        if not has_fences:
            return "No code block found (missing ``` fences)"

        code_match = re.search(r"```\w*\s*(.*?)\s*```", llm_content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            try:
                compile(code, "<string>", "exec")
            except SyntaxError as e:
                return f"Syntax error in extracted code: {e}"
            return "Code extracted but failed validation (missing def/class or too short)"

        return "Code block markers found but content extraction failed"

    @staticmethod
    def _build_retry_message(patch_mode: str, error_msg: str) -> str:
        """Build a mode-aware retry instruction with the actual error."""
        is_truncation = "truncated" in error_msg.lower() or "token limit" in error_msg.lower()

        if is_truncation:
            header = (
                "Your previous response was cut off (hit the token limit). "
                "Please try again but keep your response shorter. "
            )
        else:
            header = (
                "The previous edit was not successful. "
                f"This was the error message:\n\n{error_msg}\n\n"
                "Try again. "
            )

        if patch_mode == "differential":
            if is_truncation:
                return header + (
                    "Use fewer SEARCH/REPLACE blocks. Only change the lines "
                    "that absolutely need to change. Keep SEARCH sections minimal "
                    "(just enough context to match uniquely).\n"
                    "Format:\n"
                    "<DIFF>\n<<<<<<< SEARCH\n(exact text)\n=======\n"
                    "(replacement)\n>>>>>>> REPLACE\n</DIFF>"
                )
            return header + (
                "Make sure every SEARCH/REPLACE block is complete and properly "
                "terminated with all three markers, wrapped in <DIFF> tags:\n"
                "<DIFF>\n<<<<<<< SEARCH\n(exact text to find)\n=======\n"
                "(replacement text)\n>>>>>>> REPLACE\n</DIFF>\n\n"
                "The SEARCH text must match the original code exactly "
                "(including indentation). Keep each block short — edit only "
                "the lines that need to change."
            )
        else:
            # holistic / synthesis
            if is_truncation:
                return header + (
                    "Focus on the essential changes only. Provide the complete "
                    "code inside a single ```python ... ``` fenced block. "
                    "Remove all comments and docstrings that are not essential. "
                    "Do not include any explanation outside the code block."
                )
            return header + (
                "Please provide the complete code inside a single "
                "```python ... ``` fenced block. "
                "Do not include any explanation outside the code block."
            )

    def _select_patch_mode(self) -> str:
        """Select patch mode based on policy and stagnation."""
        modes = self.config.patch_policy.modes
        weights = list(self.config.patch_policy.weights)

        # Adaptive adjustment based on stagnation
        if self.config.patch_policy.adaptive:
            if self._stagnation_counter > self.config.patch_policy.stagnation_threshold:
                # Boost holistic mode when stagnating
                if "holistic" in modes:
                    idx = modes.index("holistic")
                    boost = min(
                        self.config.patch_policy.stagnation_boost,
                        self._stagnation_counter * 0.01,
                    )
                    weights[idx] += boost
                    # Normalize
                    total = sum(weights)
                    weights = [w / total for w in weights]

        return random.choices(modes, weights=weights)[0]

    def _apply_patch(
        self, parent_code: str, llm_output: str, patch_mode: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Apply LLM-generated patch to parent code.

        Returns:
            (new_code, error_message).  error_message is None on success.
        """
        from madevolve.transformer.patcher import _apply_patches
        from madevolve.transformer.rewriter import apply_holistic_rewrite
        from madevolve.transformer.changeset import apply_synthesis_patch

        try:
            parent_has_blocks = has_evolve_blocks(parent_code)

            if patch_mode == "differential":
                result = _apply_patches(parent_code, llm_output)
                if result.success:
                    new_code = result.code
                else:
                    return None, result.error_message
            elif patch_mode == "holistic":
                new_code = apply_holistic_rewrite(llm_output, has_evolve_block=parent_has_blocks)
                if new_code is None:
                    return None, self._diagnose_extraction_error(llm_output, patch_mode)
            elif patch_mode == "synthesis":
                new_code = apply_synthesis_patch(parent_code, llm_output)
                if new_code is None:
                    return None, self._diagnose_extraction_error(llm_output, patch_mode)
            else:
                return None, f"Unknown patch mode: {patch_mode}"

            # Reassemble if parent had evolve blocks and mode is holistic/synthesis
            if new_code and parent_has_blocks and patch_mode in ("holistic", "synthesis"):
                new_code = replace_mutable_content(parent_code, new_code)

            return new_code, None
        except Exception as e:
            logger.warning(f"Patch application failed: {e}")
            return None, str(e)

    def _process_pending_jobs(self):
        """Poll and process completed jobs."""
        completed = []

        for job_id, pending in self._pending_jobs.items():
            if self._dispatcher.is_complete(job_id):
                completed.append(job_id)
                self._process_completed_job(pending)

        for job_id in completed:
            del self._pending_jobs[job_id]

        # Wait if all jobs are pending
        if self._pending_jobs and not completed:
            print_substep(f"Waiting for {len(self._pending_jobs)} pending jobs...")
            time.sleep(1.0)

    def _process_completed_job(self, pending: PendingJob):
        """Process a completed evaluation job."""
        result = self._dispatcher.get_result(pending.job_id)

        if result is None:
            logger.warning(f"Job {pending.job_id} returned no result")
            return

        evaluation = EvaluationResult(
            program_id=pending.program_id,
            success=result.get("success", False),
            combined_score=result.get("combined_score", 0.0),
            public_metrics=result.get("public_metrics", {}),
            private_metrics=result.get("private_metrics", {}),
            text_feedback=result.get("text_feedback", ""),
            execution_time=time.time() - pending.submit_time,
            error_message=result.get("error"),
        )

        improved = False
        if evaluation.success:
            # Generate embedding from mutable content only
            embedding = self._vectorizer.embed(extract_mutable_content(pending.code))

            # Register in artifact store (merge job metadata with patch info)
            artifact_metadata = {
                "patch_mode": pending.patch_mode,
                "model_used": pending.model_used,
            }
            artifact_metadata.update(pending.metadata)
            self._artifact_store.register(
                program_id=pending.program_id,
                code=pending.code,
                parent_id=pending.parent_id,
                generation=pending.generation,
                combined_score=evaluation.combined_score,
                public_metrics=evaluation.public_metrics,
                private_metrics=evaluation.private_metrics,
                text_feedback=evaluation.text_feedback,
                embedding=embedding,
                metadata=artifact_metadata,
            )

            # Compute MAP-Elites features
            features = None
            if self._feature_extractor:
                features = self._feature_extractor.compute_features(
                    code=pending.code,
                    embedding=embedding,
                    score=evaluation.combined_score,
                    program_id=pending.program_id,
                    evaluator_metrics=evaluation.public_metrics,
                )

            # Update population
            self._population.register(
                pending.program_id,
                evaluation.combined_score,
                embedding,
                features=features,
                parent_id=pending.parent_id,
                generation=pending.generation,
            )

            # Track best program
            if evaluation.combined_score > self._best_score:
                self._best_score = evaluation.combined_score
                self._best_program_id = pending.program_id
                self._stagnation_counter = 0
                improved = True

                self.session.set_best_program(pending.program_id, evaluation.combined_score)
                self._save_best_program()
            else:
                self._stagnation_counter += 1

            # Update model selector
            self._gateway.record_outcome(
                model_name=pending.model_used,
                success=improved,
                score=evaluation.combined_score,
            )

        print_result(pending.program_id, evaluation.combined_score, improved)

    def _evaluate_program(
        self,
        program_id: str,
        code: str,
        generation: int,
    ) -> EvaluationResult:
        """Synchronously evaluate a single program."""
        print_substep("Submitting evaluation...")
        work_dir = self.results_dir / "evaluations" / f"gen_{generation}" / program_id
        work_dir.mkdir(parents=True, exist_ok=True)

        job_id = self._dispatcher.submit(
            program_id=program_id,
            code=code,
            evaluator_script=self.config.evaluator_script,
            work_dir=str(work_dir),
        )

        # Wait for completion
        while not self._dispatcher.is_complete(job_id):
            time.sleep(0.5)

        result = self._dispatcher.get_result(job_id)
        print_substep("Evaluation complete")

        return EvaluationResult(
            program_id=program_id,
            success=result.get("success", False) if result else False,
            combined_score=result.get("combined_score", 0.0) if result else 0.0,
            public_metrics=result.get("public_metrics", {}) if result else {},
            private_metrics=result.get("private_metrics", {}) if result else {},
            text_feedback=result.get("text_feedback", "") if result else "",
            execution_time=0.0,
            error_message=result.get("error") if result else "No result",
        )

    def _evaluate_program_quick(self, code: str) -> float:
        """Quick evaluation for inner-loop optimization."""
        program_id = generate_uid()
        result = self._evaluate_program(program_id, code, generation=-1)
        return result.combined_score if result.success else float("-inf")

    def _update_session_stats(self, generation: int):
        """Update session statistics after a generation."""
        stats = self._population.get_statistics()
        self.session.update_generation(
            generation=generation,
            best_score=self._best_score,
            avg_score=stats.get("avg_score", 0.0),
            programs_evaluated=stats.get("total_programs", 0),
            improvements=stats.get("improvements", 0),
        )

    def _save_best_program(self):
        """Save the current best program to the best/ directory with full results and artifacts."""
        if not self._best_program_id:
            return

        best_program = self._artifact_store.get(self._best_program_id)
        if not best_program:
            return

        import json
        import shutil

        best_dir = self.results_dir / "best"
        best_dir.mkdir(exist_ok=True)

        # Save code
        with open(best_dir / "best.py", "w") as f:
            f.write(best_program.code)

        # Save full evaluation results
        best_info = {
            "program_id": best_program.program_id,
            "generation": best_program.generation,
            "combined_score": best_program.combined_score,
            "public_metrics": best_program.public_metrics,
            "private_metrics": best_program.private_metrics,
            "text_feedback": best_program.text_feedback,
            "parent_id": best_program.parent_id,
            "created_at": best_program.created_at,
            "metadata": best_program.metadata,
        }
        with open(best_dir / "result.json", "w") as f:
            json.dump(best_info, f, indent=2)

        # Copy evaluation artifacts (HTML plots, logs, etc.)
        eval_dir = (
            self.results_dir
            / "evaluations"
            / f"gen_{best_program.generation}"
            / best_program.program_id
        )
        if eval_dir.exists():
            # Clear previous artifacts (they belong to the old best)
            for old in best_dir.iterdir():
                if old.name not in ("best.py", "result.json"):
                    old.unlink()
            for artifact in eval_dir.iterdir():
                if artifact.name not in ("candidate.py", "__pycache__") and artifact.is_file():
                    shutil.copy2(artifact, best_dir / artifact.name)

        logger.info(f"Saved best program {best_program.program_id} (score={best_program.combined_score:.4f}) to {best_dir}")

    def _save_checkpoint(self):
        """Save current state to checkpoint."""
        # Update component states
        self.session.update_component_state(
            "population",
            self._population.get_state(),
        )

        self.session.update_component_state(
            "model_selector",
            self._gateway.get_selector_state(),
        )

        if self._feature_extractor:
            self.session.update_component_state(
                "feature_extractor",
                self._feature_extractor.get_state(),
            )

        path = self.session.save_checkpoint()
        if path:
            print_substep("Checkpoint saved")
            logger.info(f"Checkpoint saved: {path}")

    def _finalize(self) -> Dict[str, Any]:
        """Finalize evolution and generate reports."""
        print_step("Finalizing evolution...")
        logger.info("Finalizing evolution...")

        # Save final checkpoint
        self._save_checkpoint()

        # Export history
        history_path = self.session.export_history()

        # Final save of best program (ensures best/ is up to date)
        print_substep("Saving best program...")
        self._save_best_program()

        # Generate report if enabled
        report_path = None
        if self.config.report.enabled:
            print_substep("Generating report...")
            report_path = self._generate_report()

        # Print summary
        stats = {
            "Total Generations": self.session.state.current_generation,
            "Total Programs Evaluated": self.session.state.total_programs_evaluated,
            "Best Score": f"{self._best_score:.4f}",
            "Total Improvements": self.session.state.total_improvements,
            "Elapsed Time": format_duration(self.session.elapsed_time),
        }
        print_summary(stats)

        # Cleanup
        self.container.shutdown()

        return {
            "best_program_id": self._best_program_id,
            "best_score": self._best_score,
            "total_generations": self.session.state.current_generation,
            "total_programs": self.session.state.total_programs_evaluated,
            "history_path": history_path,
            "report_path": report_path,
        }

    def _generate_report(self) -> Optional[str]:
        """Generate evolution report."""
        try:
            from madevolve.repository.analytics.visualization import ReportGenerator

            generator = ReportGenerator(
                artifact_store=self._artifact_store,
                results_dir=str(self.results_dir),
                config=self.config,
            )

            report_path = generator.generate(
                best_program_id=self._best_program_id,
                output_dir=str(self.results_dir / self.config.report.output_dir),
            )

            logger.info(f"Report generated: {report_path}")
            return report_path

        except Exception as e:
            logger.warning(f"Failed to generate report: {e}")
            return None
