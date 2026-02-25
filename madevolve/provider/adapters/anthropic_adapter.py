"""
Anthropic API adapter for MadEvolve.
"""

import logging
import os
import time
from typing import Dict, List, Optional

from madevolve.common.helpers import retry_with_backoff
from madevolve.provider.adapters.response import LLMResponse
from madevolve.provider.adapters.tariff import calculate_cost

logger = logging.getLogger(__name__)

# Thinking budget as a multiple of the user-requested max_tokens.
# Added ON TOP of max_tokens so visible output keeps the full budget.
# E.g., with max_tokens=4096 and fraction=3.0:
#   thinking_budget = 4096 * 3.0 = 12288
#   effective_max_tokens = 4096 + 12288 = 16384
_THINKING_BUDGET_FRACTION = 3.0


def _get_thinking_mode(model: str) -> Optional[str]:
    """Determine thinking mode for a Claude model.

    Returns:
        "adaptive" for models supporting adaptive thinking (Opus 4.6, Sonnet 4.6),
        "manual" for models supporting manual extended thinking (older Claude 4+),
        None for models without thinking support.
    """
    m = model.lower()

    # Adaptive thinking models (newest, recommended)
    if "opus-4-6" in m or "sonnet-4-6" in m:
        return "adaptive"

    # Manual thinking models:
    #   Claude 4.x Opus/Sonnet, Claude 3.7 Sonnet, Haiku 4.5
    if any(x in m for x in [
        "opus-4", "sonnet-4", "3-7-sonnet", "sonnet-3-7", "haiku-4",
    ]):
        return "manual"

    return None


def _supports_max_effort(model: str) -> bool:
    """Check if model supports effort='max' (Opus 4.6 only)."""
    return "opus-4-6" in model.lower()


class AnthropicAdapter:
    """
    Adapter for Anthropic Claude API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize the Anthropic adapter.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            timeout: Request timeout in seconds
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=timeout,
        )

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    def query(
        self,
        model: str,
        system_message: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a query to Anthropic Claude.

        Automatically enables extended thinking for supported models:
        - Opus 4.6 / Sonnet 4.6: adaptive thinking with max/high effort
        - Older Claude 4+: manual thinking with budget_tokens
        Temperature is dropped when thinking is active (API constraint).

        Args:
            model: Model name (e.g., "claude-opus-4-6")
            system_message: System prompt
            user_message: User message
            temperature: Sampling temperature (ignored when thinking is active)
            max_tokens: Maximum tokens for *visible* output
            messages: Optional pre-built message list for multi-turn conversations.
            **kwargs: Additional API parameters

        Returns:
            LLMResponse with the generated content
        """
        start_time = time.time()

        api_messages = messages if messages is not None else [{"role": "user", "content": user_message}]

        thinking_mode = _get_thinking_mode(model)

        create_kwargs: Dict = {
            "model": model,
            "system": system_message,
            "messages": api_messages,
        }

        if thinking_mode == "adaptive":
            # Adaptive thinking: model decides when/how much to think.
            # Inflate max_tokens so visible output still gets the full budget.
            thinking_budget = int(max_tokens * _THINKING_BUDGET_FRACTION)
            effective_max_tokens = max_tokens + thinking_budget

            create_kwargs["max_tokens"] = effective_max_tokens
            create_kwargs["thinking"] = {"type": "adaptive"}

            effort = "max" if _supports_max_effort(model) else "high"
            create_kwargs["output_config"] = {"effort": effort}

            # Temperature is NOT compatible with thinking
            logger.debug(
                f"Adaptive thinking enabled: effort={effort}, "
                f"max_tokens={effective_max_tokens} "
                f"(requested visible={max_tokens})"
            )

        elif thinking_mode == "manual":
            # Manual thinking: explicit budget_tokens.
            thinking_budget = int(max_tokens * _THINKING_BUDGET_FRACTION)
            effective_max_tokens = max_tokens + thinking_budget

            create_kwargs["max_tokens"] = effective_max_tokens
            create_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

            # Temperature is NOT compatible with thinking
            logger.debug(
                f"Manual thinking enabled: budget_tokens={thinking_budget}, "
                f"max_tokens={effective_max_tokens} "
                f"(requested visible={max_tokens})"
            )

        else:
            # No thinking support (e.g. Claude 3.5 Sonnet)
            create_kwargs["max_tokens"] = max_tokens
            create_kwargs["temperature"] = temperature

        # Allow caller overrides via **kwargs
        create_kwargs.update(kwargs)

        # SDK requires streaming when max_tokens > 21333 to avoid HTTP timeouts.
        if create_kwargs["max_tokens"] > 21333:
            with self.client.messages.stream(**create_kwargs) as stream:
                response = stream.get_final_message()
        else:
            response = self.client.messages.create(**create_kwargs)

        latency_ms = (time.time() - start_time) * 1000

        # Extract visible content (skip thinking / redacted_thinking blocks)
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        usage = response.usage
        cost = calculate_cost(model, usage.input_tokens, usage.output_tokens)

        return LLMResponse(
            content=content,
            model_name=model,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            finish_reason=response.stop_reason or "stop",
        )

    def supports_model(self, model_name: str) -> bool:
        """Check if this adapter supports the given model."""
        return "claude" in model_name.lower()
