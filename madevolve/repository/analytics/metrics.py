"""
Code complexity analysis and metrics computation.

This module provides multi-dimensional code quality analysis
including complexity, maintainability, and structural metrics.
"""

import ast
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ComplexityMetrics:
    """Comprehensive code complexity metrics."""
    lines_of_code: int = 0
    logical_lines: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    cyclomatic_complexity: int = 1
    max_nesting_depth: int = 0
    num_functions: int = 0
    num_classes: int = 0
    avg_function_length: float = 0.0
    halstead_volume: float = 0.0
    maintainability_index: float = 100.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "loc": self.lines_of_code,
            "lloc": self.logical_lines,
            "cyclomatic": self.cyclomatic_complexity,
            "nesting": self.max_nesting_depth,
            "functions": self.num_functions,
            "maintainability": self.maintainability_index,
        }


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor for computing complexity metrics."""

    def __init__(self):
        self.complexity = 1  # Base complexity
        self.max_depth = 0
        self.current_depth = 0
        self.num_functions = 0
        self.num_classes = 0
        self.function_lengths = []
        self.operators = set()
        self.operands = set()

    def visit_If(self, node):
        self.complexity += 1
        self._update_depth()
        self.generic_visit(node)
        self._decrease_depth()

    def visit_For(self, node):
        self.complexity += 1
        self._update_depth()
        self.generic_visit(node)
        self._decrease_depth()

    def visit_While(self, node):
        self.complexity += 1
        self._update_depth()
        self.generic_visit(node)
        self._decrease_depth()

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node):
        self._update_depth()
        self.generic_visit(node)
        self._decrease_depth()

    def visit_BoolOp(self, node):
        # Each 'and' or 'or' adds to complexity
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.num_functions += 1
        self.function_lengths.append(len(node.body))
        self._update_depth()
        self.generic_visit(node)
        self._decrease_depth()

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.num_classes += 1
        self._update_depth()
        self.generic_visit(node)
        self._decrease_depth()

    def visit_ListComp(self, node):
        self.complexity += len(node.generators)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.complexity += len(node.generators)
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.complexity += len(node.generators)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.complexity += len(node.generators)
        self.generic_visit(node)

    def visit_Lambda(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def _update_depth(self):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)

    def _decrease_depth(self):
        self.current_depth = max(0, self.current_depth - 1)


class CodeAnalyzer:
    """
    Analyzes code to compute various quality metrics.

    Provides complexity analysis, structural metrics, and
    maintainability scoring for Python code.
    """

    def __init__(self):
        self._cache: Dict[str, ComplexityMetrics] = {}

    def analyze(self, code: str, use_cache: bool = True) -> ComplexityMetrics:
        """
        Analyze code and compute metrics.

        Args:
            code: Python source code
            use_cache: Whether to use cached results

        Returns:
            ComplexityMetrics instance
        """
        # Check cache
        if use_cache:
            import hashlib
            cache_key = hashlib.md5(code.encode()).hexdigest()[:16]
            if cache_key in self._cache:
                return self._cache[cache_key]

        metrics = ComplexityMetrics()

        # Line counts
        lines = code.split("\n")
        metrics.lines_of_code = len(lines)

        for line in lines:
            stripped = line.strip()
            if not stripped:
                metrics.blank_lines += 1
            elif stripped.startswith("#"):
                metrics.comment_lines += 1
            else:
                metrics.logical_lines += 1

        # AST-based metrics
        try:
            tree = ast.parse(code)
            visitor = ComplexityVisitor()
            visitor.visit(tree)

            metrics.cyclomatic_complexity = visitor.complexity
            metrics.max_nesting_depth = visitor.max_depth
            metrics.num_functions = visitor.num_functions
            metrics.num_classes = visitor.num_classes

            if visitor.function_lengths:
                metrics.avg_function_length = sum(visitor.function_lengths) / len(visitor.function_lengths)

        except SyntaxError as e:
            logger.warning(f"Syntax error in code analysis: {e}")

        # Compute maintainability index
        metrics.maintainability_index = self._compute_maintainability(metrics)

        # Estimate Halstead volume
        metrics.halstead_volume = self._estimate_halstead(code)

        # Cache result
        if use_cache:
            self._cache[cache_key] = metrics

        return metrics

    def _compute_maintainability(self, metrics: ComplexityMetrics) -> float:
        """
        Compute maintainability index (0-100 scale).

        Based on simplified MI formula.
        """
        import math

        v = max(1, metrics.halstead_volume or metrics.lines_of_code * 8)
        g = metrics.cyclomatic_complexity
        lloc = max(1, metrics.logical_lines)

        # MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)
        mi = 171 - 5.2 * math.log(v) - 0.23 * g - 16.2 * math.log(lloc)

        # Normalize to 0-100
        mi = max(0, min(100, mi * 100 / 171))

        return round(mi, 2)

    def _estimate_halstead(self, code: str) -> float:
        """
        Estimate Halstead volume from code.

        Simplified estimation based on token counting.
        """
        import math

        # Operators (simplified)
        operators = re.findall(
            r'[\+\-\*/%=<>!&|^~]+|if|else|elif|for|while|try|except|with|'
            r'return|import|from|class|def|lambda|and|or|not|in|is',
            code
        )

        # Operands (identifiers and literals)
        operands = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+\.?\d*\b', code)

        n1 = len(set(operators))  # Unique operators
        n2 = len(set(operands))   # Unique operands
        N1 = len(operators)       # Total operators
        N2 = len(operands)        # Total operands

        n = n1 + n2
        N = N1 + N2

        if n == 0:
            return 0.0

        V = N * math.log2(max(1, n))  # Halstead Volume
        return round(V, 2)

    def compute_diversity_score(
        self,
        code: str,
        reference_codes: List[str],
    ) -> float:
        """
        Compute structural diversity score compared to references.

        Args:
            code: Code to analyze
            reference_codes: Reference codes to compare against

        Returns:
            Diversity score (0 to 1, higher = more diverse)
        """
        if not reference_codes:
            return 1.0

        metrics = self.analyze(code)
        ref_metrics = [self.analyze(ref) for ref in reference_codes]

        # Compare key metrics
        distances = []
        for ref in ref_metrics:
            dist = 0.0

            # Complexity difference
            dist += abs(metrics.cyclomatic_complexity - ref.cyclomatic_complexity) / max(metrics.cyclomatic_complexity, ref.cyclomatic_complexity, 1)

            # Structure difference
            dist += abs(metrics.num_functions - ref.num_functions) / max(metrics.num_functions, ref.num_functions, 1)

            # Nesting difference
            dist += abs(metrics.max_nesting_depth - ref.max_nesting_depth) / max(metrics.max_nesting_depth, ref.max_nesting_depth, 1)

            distances.append(dist / 3)

        # Average distance (converted to diversity)
        avg_distance = sum(distances) / len(distances)
        return min(1.0, avg_distance)

    def extract_behavioral_features(self, code: str) -> Dict[str, float]:
        """
        Extract features for MAP-Elites partitioning.

        Returns:
            Dictionary of feature names to values
        """
        metrics = self.analyze(code)

        return {
            "complexity": metrics.cyclomatic_complexity,
            "diversity": 0.5,  # Will be updated with actual embedding-based diversity
            "performance": 0.0,  # Will be updated with actual score
        }
