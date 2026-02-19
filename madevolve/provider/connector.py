"""
Provider connector utilities.
"""

import os
from typing import Dict, Optional


def get_api_key(provider: str, env_var: Optional[str] = None) -> Optional[str]:
    """
    Get API key for a provider.

    Args:
        provider: Provider name
        env_var: Optional environment variable override

    Returns:
        API key or None if not found
    """
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }

    var_name = env_var or env_vars.get(provider.lower())
    if var_name:
        return os.environ.get(var_name)
    return None


def check_provider_availability() -> Dict[str, bool]:
    """
    Check which providers have API keys configured.

    Returns:
        Dictionary mapping provider names to availability
    """
    providers = ["openai", "anthropic", "google", "deepseek"]
    return {
        provider: get_api_key(provider) is not None
        for provider in providers
    }


def get_default_model(provider: str) -> str:
    """
    Get default model for a provider.

    Args:
        provider: Provider name

    Returns:
        Default model name
    """
    defaults = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
        "google": "gemini-1.5-flash",
        "deepseek": "deepseek-chat",
    }

    return defaults.get(provider.lower(), "gpt-4o-mini")
