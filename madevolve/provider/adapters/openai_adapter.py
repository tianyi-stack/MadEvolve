"""
OpenAI API adapter for MadEvolve.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from madevolve.common.helpers import retry_with_backoff
from madevolve.provider.adapters.response import LLMResponse, EmbeddingResponse
from madevolve.provider.adapters.tariff import calculate_cost, calculate_embedding_cost

logger = logging.getLogger(__name__)

# Models that require max_completion_tokens instead of max_tokens,
# and don't support temperature (forced to 1.0).
# Reference: ShinkaEvolve/shinka/llm/models/pricing.py REASONING_OAI_MODELS
REASONING_OAI_MODELS = {
    "o1", "o1-2024-12-17", "o1-preview",
    "o3", "o3-mini", "o3-2025-04-16", "o3-mini-2025-01-31",
    "o4-mini", "o4-mini-2025-04-16",
    "gpt-5", "gpt-5.2", "gpt-5-mini", "gpt-5-nano",
}


def _is_reasoning_model(model: str) -> bool:
    """Check if a model requires reasoning-model API parameters."""
    if model in REASONING_OAI_MODELS:
        return True
    # Prefix fallback for future model variants (e.g. gpt-5.2-preview)
    return model.startswith(("o1", "o3", "o4", "gpt-5"))


class OpenAIAdapter:
    """
    Adapter for OpenAI API (including Azure OpenAI).

    Supports both chat completions and embeddings.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize the OpenAI adapter.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: Optional organization ID
            base_url: Optional custom base URL (for Azure or proxies)
            timeout: Request timeout in seconds
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        client_kwargs = {
            "api_key": self.api_key,
            "timeout": timeout,
        }

        if organization:
            client_kwargs["organization"] = organization

        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = openai.OpenAI(**client_kwargs)
        self._async_client = None

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
        Send a query to OpenAI.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            system_message: System prompt
            user_message: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            messages: Optional pre-built message list for multi-turn conversations.
                When provided, ``user_message`` is ignored and these messages are
                used directly (with system_message prepended).
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

        # Handle reasoning models (o1, o3, o4, gpt-5, gpt-5.2, etc.)
        # These models use max_completion_tokens instead of max_tokens
        # and don't support system messages or temperature
        if _is_reasoning_model(model):
            if messages is not None:
                # Merge system into first user message for multi-turn
                merged = list(api_messages)
                merged.pop(0)  # remove system message
                if merged and merged[0]["role"] == "user":
                    merged[0] = {"role": "user", "content": f"{system_message}\n\n{merged[0]['content']}"}
                else:
                    merged.insert(0, {"role": "user", "content": system_message})
                api_messages = merged
            else:
                api_messages = [{"role": "user", "content": f"{system_message}\n\n{user_message}"}]
            response = self.client.chat.completions.create(
                model=model,
                messages=api_messages,
                max_completion_tokens=max_tokens,
                **kwargs,
            )
        else:
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

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    def embed(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Embedding model name
            dimensions: Optional dimension override

        Returns:
            EmbeddingResponse with embeddings
        """
        kwargs = {"model": model, "input": texts}

        if dimensions and "3-" in model:
            kwargs["dimensions"] = dimensions

        response = self.client.embeddings.create(**kwargs)

        embeddings = [item.embedding for item in response.data]
        total_tokens = response.usage.total_tokens
        cost = calculate_embedding_cost(model, total_tokens)

        return EmbeddingResponse(
            embeddings=embeddings,
            model_name=model,
            total_tokens=total_tokens,
            cost=cost,
            dimensions=len(embeddings[0]) if embeddings else 0,
        )

    def supports_model(self, model_name: str) -> bool:
        """Check if this adapter supports the given model."""
        openai_prefixes = ["gpt-", "o1", "o3", "o4", "text-embedding"]
        return any(model_name.lower().startswith(p) for p in openai_prefixes)
