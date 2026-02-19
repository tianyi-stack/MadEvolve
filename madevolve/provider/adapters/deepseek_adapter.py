"""
DeepSeek API adapter for MadEvolve.
"""

import logging
import os
import time
from typing import Dict, List, Optional

from madevolve.common.helpers import retry_with_backoff
from madevolve.provider.adapters.response import LLMResponse
from madevolve.provider.adapters.tariff import calculate_cost

logger = logging.getLogger(__name__)


class DeepSeekAdapter:
    """
    Adapter for DeepSeek API (compatible with OpenAI API format).
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize the DeepSeek adapter.

        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            timeout: Request timeout in seconds
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.DEEPSEEK_BASE_URL,
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
        Send a query to DeepSeek.

        Args:
            model: Model name (e.g., "deepseek-chat")
            system_message: System prompt
            user_message: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            messages: Optional pre-built message list for multi-turn conversations.
            **kwargs: Additional API parameters

        Returns:
            LLMResponse with the generated content
        """
        if messages is not None:
            api_messages = [{"role": "system", "content": system_message}] + messages
        else:
            api_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        latency_ms = (time.time() - start_time) * 1000

        usage = response.usage
        cost = calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model_name=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost=cost,
            latency_ms=latency_ms,
            finish_reason=response.choices[0].finish_reason or "stop",
        )

    def supports_model(self, model_name: str) -> bool:
        """Check if this adapter supports the given model."""
        return "deepseek" in model_name.lower()
