"""
Google Gemini API adapter for MadEvolve.
"""

import logging
import os
import re
import time
from typing import Dict, List, Optional

from madevolve.common.helpers import retry_with_backoff
from madevolve.provider.adapters.response import LLMResponse
from madevolve.provider.adapters.tariff import calculate_cost

logger = logging.getLogger(__name__)

# Thinking budget as a fraction of the user-requested max_tokens.
# Added ON TOP of max_tokens so visible output keeps the full budget.
_THINKING_BUDGET_FRACTION = 0.4


def _is_thinking_model(model: str) -> bool:
    """Check if a model is a thinking-capable Gemini model (2.5+, 3.x)."""
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", model.lower())
    if not m:
        return False
    major = int(m.group(1))
    minor = int(m.group(2)) if m.group(2) else 0
    return major >= 3 or (major == 2 and minor >= 5)


class GeminiAdapter:
    """
    Adapter for Google Gemini API.

    Uses the new google-genai package (replacing deprecated google-generativeai).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize the Gemini adapter.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            timeout: Request timeout in seconds
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package required. "
                "Install with: pip install google-genai"
            )

        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided")

        # google-genai expects timeout in milliseconds
        self._timeout_ms = int(timeout * 1000)
        self._client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=self._timeout_ms),
        )
        self._types = types

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
        Send a query to Google Gemini.

        Args:
            model: Model name (e.g., "gemini-2.0-flash")
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

        thinking_config = None
        effective_max_output = max_tokens
        if _is_thinking_model(model):
            thinking_budget = int(max_tokens * _THINKING_BUDGET_FRACTION)
            thinking_config = self._types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
            # Gemini's max_output_tokens is the TOTAL budget (thinking + visible).
            # Add thinking_budget on top so visible output still gets the full max_tokens.
            effective_max_output = max_tokens + thinking_budget
            logger.debug(
                f"Thinking model detected: thinking_budget={thinking_budget}, "
                f"max_output_tokens={effective_max_output} "
                f"(requested visible={max_tokens})"
            )

        config = self._types.GenerateContentConfig(
            system_instruction=system_message,
            temperature=temperature,
            max_output_tokens=effective_max_output,
            thinking_config=thinking_config,
            http_options=self._types.HttpOptions(timeout=self._timeout_ms),
        )

        if messages is not None:
            # Convert messages to Gemini Content format
            role_map = {"user": "user", "assistant": "model"}
            contents = [
                self._types.Content(
                    role=role_map.get(msg["role"], msg["role"]),
                    parts=[self._types.Part(text=msg["content"])],
                )
                for msg in messages
            ]
        else:
            contents = user_message

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract usage info
        prompt_tokens = 0
        completion_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        cost = calculate_cost(model, prompt_tokens, completion_tokens)

        # Extract actual finish_reason from response candidates
        finish_reason = "stop"
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            raw_reason = getattr(candidate, "finish_reason", None)
            if raw_reason is not None:
                # google-genai returns enum values like "STOP", "MAX_TOKENS", "SAFETY", etc.
                reason_str = str(raw_reason).upper()
                if "MAX_TOKENS" in reason_str:
                    finish_reason = "length"
                elif "SAFETY" in reason_str:
                    finish_reason = "safety"
                elif "STOP" in reason_str:
                    finish_reason = "stop"
                else:
                    finish_reason = reason_str.lower()
                if finish_reason != "stop":
                    logger.warning(f"Gemini response truncated: finish_reason={finish_reason}")

        # Extract content safely (response.text can raise if blocked)
        try:
            content = response.text
        except Exception:
            content = ""
            if hasattr(response, "candidates") and response.candidates:
                parts = getattr(response.candidates[0], "content", None)
                if parts and hasattr(parts, "parts"):
                    content = "".join(
                        getattr(p, "text", "") for p in parts.parts
                    )

        return LLMResponse(
            content=content,
            model_name=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )

    def supports_model(self, model_name: str) -> bool:
        """Check if this adapter supports the given model."""
        return "gemini" in model_name.lower()
