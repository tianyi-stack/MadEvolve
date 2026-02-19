"""
Model Gateway - Unified interface for LLM providers.

This module provides a single entry point for all LLM operations,
handling provider routing, model selection, and cost tracking.
"""

import logging
from typing import Any, Dict, List, Optional

from madevolve.engine.configuration import ModelConfig
from madevolve.provider.adapters.response import LLMResponse, ProviderUsage
from madevolve.provider.strategy.allocation import ModelSelector, create_selector

logger = logging.getLogger(__name__)


class ModelGateway:
    """
    Unified gateway for LLM interactions.

    Routes requests to appropriate providers, manages model selection,
    and tracks usage/costs across all providers.
    """

    def __init__(self, container):
        """
        Initialize the model gateway.

        Args:
            container: ServiceContainer with configuration
        """
        self.config: ModelConfig = container.config.models
        self._adapters = {}
        self._selector: Optional[ModelSelector] = None
        self._usage: Dict[str, ProviderUsage] = {}

        self._init_adapters()
        self._init_selector()

    def _init_adapters(self):
        """Initialize provider adapters lazily."""
        # Adapters are initialized on first use for each provider
        pass

    def _init_selector(self):
        """Initialize model selector."""
        if self.config.adaptive_selection:
            self._selector = create_selector(
                algorithm=self.config.selection_algorithm,
                models=self.config.models,
                weights=self.config.weights,
                exploration_factor=self.config.exploration_factor,
                decay_rate=self.config.decay_rate,
            )
        else:
            self._selector = None

    def _get_adapter(self, model_name: str):
        """Get or create adapter for the given model."""
        # Determine provider from model name
        provider = self._infer_provider(model_name)

        if provider not in self._adapters:
            self._adapters[provider] = self._create_adapter(provider)

        return self._adapters[provider]

    def _infer_provider(self, model_name: str) -> str:
        """Infer provider from model name."""
        model_lower = model_name.lower()

        if "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "deepseek" in model_lower:
            return "deepseek"
        else:
            # Default to OpenAI
            return "openai"

    def _create_adapter(self, provider: str):
        """Create adapter for the given provider."""
        provider_kwargs = self.config.provider_kwargs.get(provider, {})

        if provider == "openai":
            from madevolve.provider.adapters.openai_adapter import OpenAIAdapter
            return OpenAIAdapter(timeout=self.config.timeout, **provider_kwargs)

        elif provider == "anthropic":
            from madevolve.provider.adapters.anthropic_adapter import AnthropicAdapter
            return AnthropicAdapter(timeout=self.config.timeout, **provider_kwargs)

        elif provider == "google":
            from madevolve.provider.adapters.gemini_adapter import GeminiAdapter
            return GeminiAdapter(timeout=self.config.timeout, **provider_kwargs)

        elif provider == "deepseek":
            from madevolve.provider.adapters.deepseek_adapter import DeepSeekAdapter
            return DeepSeekAdapter(timeout=self.config.timeout, **provider_kwargs)

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def query(
        self,
        system_message: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a query to an LLM.

        Args:
            system_message: System prompt
            user_message: User message
            model: Optional model override (uses selector if not provided)
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse from the selected model
        """
        # Select model
        if model is None:
            if self._selector:
                model = self._selector.select()
            else:
                model = self.config.models[0]

        # Get adapter
        adapter = self._get_adapter(model)

        # Set defaults
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        # Make query
        response = adapter.query(
            model=model,
            system_message=system_message,
            user_message=user_message,
            temperature=temp,
            max_tokens=tokens,
            **kwargs,
        )

        # Track usage
        self._record_usage(model, response)

        return response

    def query_multiturn(
        self,
        messages: List[Dict[str, str]],
        system_message: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a multi-turn conversation to an LLM.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            system_message: System prompt
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse from the selected model
        """
        if model is None:
            if self._selector:
                model = self._selector.select()
            else:
                model = self.config.models[0]

        adapter = self._get_adapter(model)

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        response = adapter.query(
            model=model,
            system_message=system_message,
            user_message="",  # ignored when messages is provided
            temperature=temp,
            max_tokens=tokens,
            messages=messages,
            **kwargs,
        )

        self._record_usage(model, response)
        return response

    def _record_usage(self, model: str, response: LLMResponse):
        """Record usage statistics."""
        if model not in self._usage:
            self._usage[model] = ProviderUsage()

        self._usage[model].record(response)

    def record_outcome(
        self,
        model_name: str,
        success: bool,
        score: float = 0.0,
        improvement: float = 0.0,
    ):
        """
        Record outcome for model selection.

        Args:
            model_name: Name of the model
            success: Whether the query produced improvement
            score: The score achieved
            improvement: Score improvement over parent
        """
        if self._selector:
            self._selector.record_outcome(
                model_name=model_name,
                success=success,
                score=score,
                improvement=improvement,
            )

    def get_usage_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all models."""
        stats = {}

        for model, usage in self._usage.items():
            stats[model] = {
                "total_queries": usage.total_queries,
                "total_tokens": usage.total_tokens,
                "total_cost": usage.total_cost,
                "success_rate": usage.success_count / max(usage.total_queries, 1),
                "avg_latency_ms": usage.avg_latency_ms,
            }

        return stats

    def get_selector_state(self) -> Dict[str, Any]:
        """Get selector state for checkpointing."""
        if self._selector:
            return self._selector.get_state()
        return {}

    def restore_selector_state(self, state: Dict[str, Any]):
        """Restore selector state from checkpoint."""
        if self._selector and state:
            self._selector.restore_state(state)

    def get_total_cost(self) -> float:
        """Get total cost across all models."""
        return sum(usage.total_cost for usage in self._usage.values())

    def close(self):
        """Close all adapters."""
        self._adapters.clear()
