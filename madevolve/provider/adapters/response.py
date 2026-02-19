"""
Response dataclasses for LLM and embedding services.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Response from an LLM query."""
    content: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        """Alias for cost field."""
        return self.cost

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for conversation history."""
        return {
            "content": self.content,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
        }


@dataclass
class EmbeddingResponse:
    """Response from an embedding service."""
    embeddings: List[List[float]]
    model_name: str
    total_tokens: int = 0
    cost: float = 0.0
    dimensions: int = 0

    @property
    def single_embedding(self) -> List[float]:
        """Get single embedding (for single input)."""
        return self.embeddings[0] if self.embeddings else []


@dataclass
class BatchLLMResponse:
    """Response from a batch LLM query."""
    responses: List[LLMResponse]
    total_cost: float = 0.0
    total_latency_ms: float = 0.0

    def __post_init__(self):
        if self.total_cost == 0.0:
            self.total_cost = sum(r.cost for r in self.responses)
        if self.total_latency_ms == 0.0:
            self.total_latency_ms = max((r.latency_ms for r in self.responses), default=0)


@dataclass
class ProviderUsage:
    """Tracks usage statistics for a provider."""
    total_queries: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    success_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0

    def record(self, response: LLMResponse, success: bool = True):
        """Record a query result."""
        self.total_queries += 1
        self.total_tokens += response.total_tokens
        self.total_cost += response.cost

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        # Update running average latency
        n = self.total_queries
        self.avg_latency_ms = (
            (self.avg_latency_ms * (n - 1) + response.latency_ms) / n
        )
