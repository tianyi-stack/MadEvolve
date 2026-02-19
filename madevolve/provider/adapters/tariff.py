"""
Pricing information for various LLM providers.
"""

from typing import Dict, Tuple

# Pricing per 1M tokens: (input_price, output_price)
# Reference: ShinkaEvolve/shinka/llm/models/pricing.py
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI Models - Legacy
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # OpenAI Models - GPT-4.1 family
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 1.40),
    # OpenAI Models - GPT-4.5
    "gpt-4.5-preview": (75.00, 150.00),
    # OpenAI Models - GPT-5 family
    "gpt-5": (1.25, 10.00),
    "gpt-5.2": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    # OpenAI Models - Reasoning (o-series)
    "o1-preview": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o1": (15.00, 60.00),
    "o3": (10.00, 40.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),

    # Anthropic Models
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-sonnet-20240229": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-7-sonnet-20250219": (3.00, 15.00),
    "claude-4-sonnet-20250514": (3.00, 15.00),

    # Google Models
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash-exp": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3-pro-preview": (2.50, 15.00),
    "gemini-3-flash-preview": (0.60, 4.00),

    # DeepSeek Models
    "deepseek-chat": (0.27, 1.10),
    "deepseek-coder": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
}

# Embedding model pricing per 1M tokens
EMBEDDING_PRICING: Dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


def calculate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """
    Calculate the cost of an LLM query.

    Args:
        model_name: Name of the model
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    # Find matching pricing: exact match first, then substring fallback
    model_lower = model_name.lower()
    pricing = MODEL_PRICING.get(model_lower) or MODEL_PRICING.get(model_name)

    if pricing is None:
        # Substring match for model variants (e.g. "gpt-4o-2024-08-06" → "gpt-4o")
        for key, price in MODEL_PRICING.items():
            if key in model_lower or model_lower in key:
                pricing = price
                break

    if pricing is None:
        pricing = MODEL_PRICING["gpt-4o-mini"]

    input_price, output_price = pricing
    cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
    return cost


def calculate_embedding_cost(
    model_name: str,
    total_tokens: int,
) -> float:
    """
    Calculate the cost of an embedding request.

    Args:
        model_name: Name of the embedding model
        total_tokens: Number of tokens

    Returns:
        Cost in USD
    """
    price = EMBEDDING_PRICING.get(model_name, 0.02)  # Default to small
    return (total_tokens * price) / 1_000_000
