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

        Args:
            model: Model name (e.g., "claude-3-5-sonnet-20241022")
            system_message: System prompt
            user_message: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            messages: Optional pre-built message list for multi-turn conversations.
            **kwargs: Additional API parameters

        Returns:
            LLMResponse with the generated content
        """
        start_time = time.time()

        api_messages = messages if messages is not None else [{"role": "user", "content": user_message}]

        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=api_messages,
            **kwargs,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract content
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
