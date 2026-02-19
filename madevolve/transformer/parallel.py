"""
Parameter Optimizer - Inner-loop hyperparameter optimization.

This module implements budget-constrained parameter optimization
for automatically tuning hyperparameters in evolved algorithms.

Supports two optimization strategies:
- Derivative-free: grid search, Latin Hypercube, Bayesian optimization
- Gradient-based: finite-difference gradient estimation with Adam optimizer
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TunableParameter:
    """Definition of a tunable parameter."""
    name: str
    default_value: float
    bounds: Tuple[float, float]
    method: str = "grid"  # grid, lhs, bayesian, autodiff


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_params: Dict[str, float]
    best_score: float
    evaluations: int
    history: List[Dict[str, Any]] = field(default_factory=list)
    converged: bool = False
    final_grad_norm: Optional[float] = None


class ParameterExtractor:
    """
    Extracts tunable parameters from code.

    Supports the format:
    # TUNABLE: param_name = value, bounds=(min, max), method=autodiff
    """

    # Pattern to match TUNABLE declarations with optional method
    TUNABLE_PATTERN = re.compile(
        r'#\s*TUNABLE:\s*(\w+)\s*=\s*([^,]+)'
        r'(?:,\s*bounds\s*=\s*\(([^)]+)\))?'
        r'(?:,\s*method\s*=\s*(\w+))?',
        re.IGNORECASE
    )

    def extract(self, code: str) -> List[TunableParameter]:
        """
        Extract tunable parameters from code.

        Args:
            code: Source code to analyze

        Returns:
            List of TunableParameter definitions
        """
        params = []

        for match in self.TUNABLE_PATTERN.finditer(code):
            name = match.group(1)
            default_str = match.group(2).strip()
            bounds_str = match.group(3)
            method = match.group(4) or "grid"

            try:
                default = float(default_str)

                if bounds_str:
                    bounds_parts = bounds_str.split(",")
                    bounds = (float(bounds_parts[0].strip()), float(bounds_parts[1].strip()))
                else:
                    # Default bounds: ±50% of default
                    if default > 0:
                        bounds = (default * 0.5, default * 1.5)
                    elif default < 0:
                        bounds = (default * 1.5, default * 0.5)
                    else:
                        bounds = (-1.0, 1.0)

                params.append(TunableParameter(
                    name=name,
                    default_value=default,
                    bounds=bounds,
                    method=method.lower(),
                ))
                logger.debug(f"Found TUNABLE: {name}={default}, bounds={bounds}, method={method}")
            except ValueError as e:
                logger.warning(f"Failed to parse TUNABLE for {name}: {e}")

        return params

    def detect_new_parameters(
        self,
        candidate_code: str,
        parent_code: str,
    ) -> Tuple[List[TunableParameter], List[TunableParameter]]:
        """
        Detect new parameters that weren't in parent.

        Args:
            candidate_code: New code with potential new parameters
            parent_code: Parent code

        Returns:
            Tuple of (new_params, inherited_params)
        """
        candidate_params = self.extract(candidate_code)
        parent_params = {p.name for p in self.extract(parent_code)}

        new_params = [p for p in candidate_params if p.name not in parent_params]
        inherited_params = [p for p in candidate_params if p.name in parent_params]

        return new_params, inherited_params


class AutodiffOptimizer:
    """
    Gradient-based parameter optimizer using finite-difference gradients and Adam.

    Estimates gradients via forward finite differences and updates parameters
    using the Adam algorithm. Works with any black-box evaluator, including
    subprocess-based evaluation.
    """

    def __init__(self, config):
        """
        Initialize the gradient-based optimizer.

        Args:
            config: OptimizationConfig with gradient optimization settings
        """
        self.config = config

    def optimize(
        self,
        params: List[TunableParameter],
        loss_fn: Callable[[Dict[str, float]], float],
        fixed_params: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """
        Run gradient-based optimization using finite-difference gradients and Adam.

        Gradients are estimated via forward finite differences, making this
        compatible with any black-box evaluator (including subprocess-based).

        Args:
            params: Parameters to optimize (must have method='autodiff')
            loss_fn: Loss function that takes param dict and returns scalar loss
                     (lower is better, will be minimized)
            fixed_params: Fixed parameters to include in loss_fn calls

        Returns:
            OptimizationResult with optimized parameters
        """
        fixed_params = fixed_params or {}
        n = len(params)
        param_names = [p.name for p in params]
        bounds_low = np.array([p.bounds[0] for p in params])
        bounds_high = np.array([p.bounds[1] for p in params])

        # Current parameter values
        x = np.array([p.default_value for p in params], dtype=np.float64)

        # Finite-difference step sizes: fraction of parameter range
        fd_rel = getattr(self.config, "autodiff_fd_step", 0.01)
        fd_eps = np.maximum((bounds_high - bounds_low) * fd_rel, 1e-8)

        # Adam state
        m = np.zeros(n)
        v = np.zeros(n)
        t = 0

        history = []
        best_params = dict(zip(param_names, x.tolist()))
        best_loss = float("inf")
        converged = False
        final_grad_norm = None
        total_evals = 0

        def _eval(param_values: np.ndarray) -> float:
            """Evaluate loss at given parameter values."""
            nonlocal total_evals
            pd = dict(zip(param_names, param_values.tolist()))
            pd.update(fixed_params)
            total_evals += 1
            return loss_fn(pd)

        logger.info(f"Starting gradient optimization with {n} parameters")
        logger.info(f"  Learning rate: {self.config.autodiff_learning_rate}")
        logger.info(f"  Max iterations: {self.config.autodiff_max_iterations}")
        logger.info(f"  Using Adam: {self.config.autodiff_use_adam}")
        logger.info(f"  FD step fraction: {fd_rel}")

        for iteration in range(self.config.autodiff_max_iterations):
            # Evaluate at current point
            try:
                current_loss = _eval(x)
            except Exception as e:
                logger.warning(f"Evaluation failed at iteration {iteration + 1}: {e}")
                break

            if not np.isfinite(current_loss):
                logger.warning(f"Non-finite loss at iteration {iteration + 1}, stopping")
                break

            # Track best
            current_params = dict(zip(param_names, x.tolist()))
            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss
                best_params = current_params.copy()

            # Estimate gradients via forward finite differences
            grads = np.zeros(n)
            for i in range(n):
                x_fwd = x.copy()
                x_fwd[i] = min(x[i] + fd_eps[i], bounds_high[i])
                delta = x_fwd[i] - x[i]
                # If at upper bound, use backward difference
                if abs(delta) < 1e-12:
                    x_fwd[i] = max(x[i] - fd_eps[i], bounds_low[i])
                    delta = x_fwd[i] - x[i]
                if abs(delta) < 1e-12:
                    continue
                try:
                    loss_fwd = _eval(x_fwd)
                except Exception:
                    continue
                if np.isfinite(loss_fwd):
                    grads[i] = (loss_fwd - current_loss) / delta

            grad_norm = float(np.linalg.norm(grads))
            final_grad_norm = grad_norm

            # Record history
            history.append({
                "iteration": iteration + 1,
                "params": current_params.copy(),
                "loss": current_loss,
                "score": -current_loss,
                "grad_norm": grad_norm,
                "is_best": is_best,
            })

            # Log progress
            if (iteration + 1) % 5 == 0 or iteration == 0:
                param_str = ", ".join(f"{k}={v:.4f}" for k, v in current_params.items())
                logger.info(
                    f"  Iter {iteration + 1}: loss={current_loss:.6f}, "
                    f"grad_norm={grad_norm:.2e}, {param_str}"
                )

            # Check convergence
            if grad_norm < self.config.autodiff_convergence_threshold:
                logger.info(f"Converged at iteration {iteration + 1} (grad_norm={grad_norm:.2e})")
                converged = True
                break

            # Parameter update
            if self.config.autodiff_use_adam:
                t += 1
                m = self.config.autodiff_adam_b1 * m + (1 - self.config.autodiff_adam_b1) * grads
                v = self.config.autodiff_adam_b2 * v + (1 - self.config.autodiff_adam_b2) * grads ** 2
                m_hat = m / (1 - self.config.autodiff_adam_b1 ** t)
                v_hat = v / (1 - self.config.autodiff_adam_b2 ** t)
                x = x - self.config.autodiff_learning_rate * m_hat / (
                    np.sqrt(v_hat) + self.config.autodiff_adam_eps
                )
            else:
                x = x - self.config.autodiff_learning_rate * grads

            # Clip to bounds
            x = np.clip(x, bounds_low, bounds_high)

        logger.info(f"Gradient optimization complete: best_loss={best_loss:.6f}, evals={total_evals}")
        logger.info(f"  Best params: {best_params}")

        return OptimizationResult(
            best_params=best_params,
            best_score=-best_loss,
            evaluations=total_evals,
            history=history,
            converged=converged,
            final_grad_norm=final_grad_norm,
        )


class ParameterOptimizer:
    """
    Inner-loop parameter optimizer.

    Automatically tunes hyperparameters in evolved algorithms
    within a fixed evaluation budget. Supports both derivative-free
    methods and autodiff-based optimization.
    """

    def __init__(self, container):
        """
        Initialize the parameter optimizer.

        Args:
            container: ServiceContainer with configuration
        """
        self.config = container.config.optimization
        self._extractor = ParameterExtractor()
        self._autodiff_optimizer = AutodiffOptimizer(self.config)
        self._frozen_params: Dict[str, float] = {}

    def optimize(
        self,
        candidate_code: str,
        parent_code: str,
        evaluator: Callable[[str], float],
    ) -> str:
        """
        Optimize parameters in candidate code.

        Args:
            candidate_code: Code with tunable parameters
            parent_code: Parent code (for detecting new params)
            evaluator: Function that evaluates code and returns score

        Returns:
            Code with optimized parameter values
        """
        if not self.config.enabled:
            return candidate_code

        # Extract parameters
        new_params, inherited_params = self._extractor.detect_new_parameters(
            candidate_code, parent_code
        )

        if not new_params:
            # No new parameters to optimize
            return self._inject_frozen_params(candidate_code, inherited_params)

        logger.info(f"Optimizing {len(new_params)} new parameters")

        # Separate autodiff and derivative-free parameters
        autodiff_params = [p for p in new_params if p.method == "autodiff"]
        df_params = [p for p in new_params if p.method != "autodiff"]

        all_optimized = {}

        # Optimize autodiff parameters first
        if autodiff_params:
            logger.info(f"Running autodiff optimization for {len(autodiff_params)} parameters")
            autodiff_result = self._optimize_autodiff(
                candidate_code, autodiff_params, evaluator
            )
            all_optimized.update(autodiff_result.best_params)

        # Then optimize derivative-free parameters
        if df_params:
            logger.info(f"Running derivative-free optimization for {len(df_params)} parameters")
            # Include already-optimized autodiff params as fixed
            df_result = self._optimize_derivative_free(
                candidate_code, df_params, evaluator, fixed_params=all_optimized
            )
            all_optimized.update(df_result.best_params)

        # Freeze optimized parameters
        if self.config.freeze_after_optimization:
            for name, value in all_optimized.items():
                self._frozen_params[name] = value

        # Inject optimized values
        optimized_code = self._inject_params(candidate_code, all_optimized)

        return optimized_code

    def _optimize_autodiff(
        self,
        code: str,
        params: List[TunableParameter],
        evaluator: Callable[[str], float],
    ) -> OptimizationResult:
        """
        Optimize parameters using autodiff.

        Args:
            code: Code template
            params: Parameters to optimize (method='autodiff')
            evaluator: Evaluation function (returns score, higher is better)

        Returns:
            OptimizationResult
        """
        # Create loss function that wraps the evaluator
        # evaluator returns score (higher is better), we need loss (lower is better)
        def loss_fn(param_dict: Dict[str, float]) -> float:
            test_code = self._inject_params(code, param_dict)
            try:
                score = evaluator(test_code)
                return -score  # Negate to convert score to loss
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                return float("inf")

        return self._autodiff_optimizer.optimize(params, loss_fn)

    def _optimize_derivative_free(
        self,
        code: str,
        params: List[TunableParameter],
        evaluator: Callable[[str], float],
        fixed_params: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """
        Optimize parameters using derivative-free methods.

        Args:
            code: Code template
            params: Parameters to optimize
            evaluator: Evaluation function
            fixed_params: Already-optimized parameters to include

        Returns:
            OptimizationResult
        """
        fixed_params = fixed_params or {}

        # Inject fixed params into code first
        code_with_fixed = self._inject_params(code, fixed_params)

        # Select strategy
        strategy = self._select_strategy(len(params))

        if strategy == "grid":
            return self._grid_search(code_with_fixed, params, evaluator)
        elif strategy == "lhs":
            return self._latin_hypercube(code_with_fixed, params, evaluator)
        else:
            return self._bayesian_optimization(code_with_fixed, params, evaluator)

    def _select_strategy(self, num_params: int) -> str:
        """Select optimization strategy based on parameter count."""
        if num_params <= self.config.grid_threshold:
            return "grid"
        elif num_params <= self.config.lhs_threshold:
            return "lhs"
        else:
            return "bayesian"

    def _grid_search(
        self,
        code: str,
        params: List[TunableParameter],
        evaluator: Callable[[str], float],
    ) -> OptimizationResult:
        """Run grid search optimization."""
        history = []
        best_params = {p.name: p.default_value for p in params}
        best_score = float("-inf")

        # Determine grid points per parameter
        if len(params) == 1:
            points_per_param = min(self.config.max_budget, 10)
        else:
            points_per_param = int(self.config.max_budget ** (1 / len(params)))
            points_per_param = max(2, min(points_per_param, 5))

        # Generate grid
        grids = []
        for p in params:
            low, high = p.bounds
            grids.append(np.linspace(low, high, points_per_param))

        # Evaluate grid points
        from itertools import product
        for values in product(*grids):
            if len(history) >= self.config.max_budget:
                break

            param_dict = {p.name: v for p, v in zip(params, values)}
            test_code = self._inject_params(code, param_dict)

            try:
                score = evaluator(test_code)
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                score = float("-inf")

            history.append({"params": param_dict, "score": score})

            if score > best_score:
                best_score = score
                best_params = param_dict.copy()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            evaluations=len(history),
            history=history,
        )

    def _latin_hypercube(
        self,
        code: str,
        params: List[TunableParameter],
        evaluator: Callable[[str], float],
    ) -> OptimizationResult:
        """Run Latin Hypercube Sampling optimization."""
        try:
            from scipy.stats import qmc
        except ImportError:
            logger.warning("scipy not available, falling back to grid search")
            return self._grid_search(code, params, evaluator)

        history = []
        best_params = {p.name: p.default_value for p in params}
        best_score = float("-inf")

        n_samples = self.config.max_budget

        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=len(params))
        samples = sampler.random(n=n_samples)

        # Scale samples to parameter bounds
        for i, sample in enumerate(samples):
            param_dict = {}
            for j, p in enumerate(params):
                low, high = p.bounds
                param_dict[p.name] = low + sample[j] * (high - low)

            test_code = self._inject_params(code, param_dict)

            try:
                score = evaluator(test_code)
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                score = float("-inf")

            history.append({"params": param_dict, "score": score})

            if score > best_score:
                best_score = score
                best_params = param_dict.copy()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            evaluations=len(history),
            history=history,
        )

    def _bayesian_optimization(
        self,
        code: str,
        params: List[TunableParameter],
        evaluator: Callable[[str], float],
    ) -> OptimizationResult:
        """Run Bayesian optimization with GP surrogate."""
        history = []
        best_params = {p.name: p.default_value for p in params}
        best_score = float("-inf")

        # Initial random samples
        n_init = min(3, self.config.max_budget // 3)
        n_iter = self.config.max_budget - n_init

        X = []
        y = []

        # Random initialization
        for _ in range(n_init):
            param_dict = {}
            x = []
            for p in params:
                low, high = p.bounds
                value = np.random.uniform(low, high)
                param_dict[p.name] = value
                x.append(value)

            test_code = self._inject_params(code, param_dict)

            try:
                score = evaluator(test_code)
            except Exception:
                score = float("-inf")

            X.append(x)
            y.append(score)
            history.append({"params": param_dict, "score": score})

            if score > best_score:
                best_score = score
                best_params = param_dict.copy()

        # Bayesian optimization iterations
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel

            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

            for _ in range(n_iter):
                # Fit GP
                gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
                gp.fit(np.array(X), np.array(y))

                # Acquisition function (Expected Improvement)
                best_next_x = self._acquire_next(gp, params, best_score)

                # Evaluate
                param_dict = {p.name: v for p, v in zip(params, best_next_x)}
                test_code = self._inject_params(code, param_dict)

                try:
                    score = evaluator(test_code)
                except Exception:
                    score = float("-inf")

                X.append(list(best_next_x))
                y.append(score)
                history.append({"params": param_dict, "score": score})

                if score > best_score:
                    best_score = score
                    best_params = param_dict.copy()

        except ImportError:
            # Fall back to random search
            for _ in range(n_iter):
                param_dict = {}
                for p in params:
                    low, high = p.bounds
                    param_dict[p.name] = np.random.uniform(low, high)

                test_code = self._inject_params(code, param_dict)

                try:
                    score = evaluator(test_code)
                except Exception:
                    score = float("-inf")

                history.append({"params": param_dict, "score": score})

                if score > best_score:
                    best_score = score
                    best_params = param_dict.copy()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            evaluations=len(history),
            history=history,
        )

    def _acquire_next(
        self,
        gp,
        params: List[TunableParameter],
        best_score: float,
        n_candidates: int = 100,
    ) -> np.ndarray:
        """Find next point using Expected Improvement."""
        best_ei = -float("inf")
        best_x = None

        for _ in range(n_candidates):
            x = np.array([
                np.random.uniform(p.bounds[0], p.bounds[1])
                for p in params
            ])

            mu, sigma = gp.predict([x], return_std=True)

            # Expected Improvement
            from scipy.stats import norm
            z = (mu[0] - best_score) / (sigma[0] + 1e-8)
            ei = (mu[0] - best_score) * norm.cdf(z) + sigma[0] * norm.pdf(z)

            if ei > best_ei:
                best_ei = ei
                best_x = x

        return best_x

    def _inject_params(self, code: str, params: Dict[str, float]) -> str:
        """Inject parameter values into code."""
        result = code

        for name, value in params.items():
            # Format value appropriately
            if abs(value) < 0.001 or abs(value) > 10000:
                value_str = f"{value:.6g}"
            else:
                value_str = f"{value:.6f}".rstrip('0').rstrip('.')

            # Replace TUNABLE comment assignments
            pattern = rf"(#\s*TUNABLE:\s*{name}\s*=\s*)[-]?[\d.e+-]+"
            result = re.sub(pattern, rf"\g<1>{value_str}", result)

            # Replace function parameter defaults
            pattern = rf"(\b{name}\s*(?::\s*\w+)?\s*=\s*)[-]?[\d.e+-]+"
            result = re.sub(pattern, rf"\g<1>{value_str}", result)

        return result

    def _inject_frozen_params(
        self,
        code: str,
        inherited: List[TunableParameter],
    ) -> str:
        """Inject frozen parameter values from previous optimization."""
        params_to_inject = {}

        for p in inherited:
            if p.name in self._frozen_params:
                params_to_inject[p.name] = self._frozen_params[p.name]

        if params_to_inject:
            return self._inject_params(code, params_to_inject)

        return code

    def get_frozen_params(self) -> Dict[str, float]:
        """Get currently frozen parameters."""
        return self._frozen_params.copy()

    def restore_frozen_params(self, params: Dict[str, float]):
        """Restore frozen parameters from saved state."""
        self._frozen_params = params.copy()
