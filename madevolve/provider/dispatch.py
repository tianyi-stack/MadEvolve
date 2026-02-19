"""
Request dispatch utilities for provider module.
"""

import asyncio
import logging
from typing import List, Optional

from madevolve.provider.adapters.response import LLMResponse, BatchLLMResponse
from madevolve.provider.gateway import ModelGateway

logger = logging.getLogger(__name__)


class BatchDispatcher:
    """
    Dispatcher for batch LLM requests.

    Provides utilities for parallel query execution and
    rate limiting.
    """

    def __init__(
        self,
        gateway: ModelGateway,
        max_concurrent: int = 5,
        rate_limit_delay: float = 0.1,
    ):
        """
        Initialize the batch dispatcher.

        Args:
            gateway: ModelGateway instance
            max_concurrent: Maximum concurrent requests
            rate_limit_delay: Delay between requests in seconds
        """
        self.gateway = gateway
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay

    def dispatch_batch(
        self,
        requests: List[dict],
        model: Optional[str] = None,
    ) -> BatchLLMResponse:
        """
        Dispatch a batch of requests sequentially.

        Args:
            requests: List of request dictionaries with:
                - system_message: str
                - user_message: str
                - kwargs: Optional additional parameters
            model: Optional model override

        Returns:
            BatchLLMResponse with all responses
        """
        responses = []

        for req in requests:
            try:
                response = self.gateway.query(
                    system_message=req["system_message"],
                    user_message=req["user_message"],
                    model=model,
                    **req.get("kwargs", {}),
                )
                responses.append(response)
            except Exception as e:
                logger.warning(f"Batch request failed: {e}")
                # Create error response
                responses.append(LLMResponse(
                    content="",
                    model_name=model or "unknown",
                    metadata={"error": str(e)},
                ))

        return BatchLLMResponse(responses=responses)
