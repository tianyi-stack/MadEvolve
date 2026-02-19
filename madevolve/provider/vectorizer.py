"""
Vectorizer - Code embedding service.

This module provides embedding generation for code similarity
computation and diversity analysis.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

from madevolve.engine.configuration import EmbeddingConfig

logger = logging.getLogger(__name__)


class Vectorizer:
    """
    Service for generating code embeddings.

    Supports caching to avoid redundant API calls for identical code.
    """

    def __init__(self, container):
        """
        Initialize the vectorizer.

        Args:
            container: ServiceContainer with configuration
        """
        self.config: EmbeddingConfig = container.config.embedding
        self._adapter = None
        self._cache: Dict[str, List[float]] = {}
        self._total_cost: float = 0.0

    def _get_adapter(self):
        """Lazily initialize the embedding adapter."""
        if self._adapter is None:
            from madevolve.provider.adapters.openai_adapter import OpenAIAdapter
            self._adapter = OpenAIAdapter()
        return self._adapter

    def embed(self, code: str) -> List[float]:
        """
        Generate embedding for code.

        Args:
            code: Source code to embed

        Returns:
            Embedding vector
        """
        if self.config.cache_embeddings:
            cache_key = self._compute_cache_key(code)
            if cache_key in self._cache:
                return self._cache[cache_key]

        response = self._get_adapter().embed(
            texts=[code],
            model=self.config.model,
            dimensions=self.config.dimensions,
        )

        embedding = response.single_embedding
        self._total_cost += response.cost

        if self.config.cache_embeddings:
            self._cache[cache_key] = embedding

        return embedding

    def embed_batch(self, codes: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple codes.

        Args:
            codes: List of source codes to embed

        Returns:
            List of embedding vectors
        """
        results = []
        to_embed = []
        to_embed_indices = []

        # Check cache first
        for i, code in enumerate(codes):
            if self.config.cache_embeddings:
                cache_key = self._compute_cache_key(code)
                if cache_key in self._cache:
                    results.append(self._cache[cache_key])
                    continue

            results.append(None)  # Placeholder
            to_embed.append(code)
            to_embed_indices.append(i)

        # Embed uncached codes in batches
        if to_embed:
            for batch_start in range(0, len(to_embed), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(to_embed))
                batch = to_embed[batch_start:batch_end]

                response = self._get_adapter().embed(
                    texts=batch,
                    model=self.config.model,
                    dimensions=self.config.dimensions,
                )

                self._total_cost += response.cost

                for j, embedding in enumerate(response.embeddings):
                    idx = to_embed_indices[batch_start + j]
                    results[idx] = embedding

                    if self.config.cache_embeddings:
                        cache_key = self._compute_cache_key(to_embed[batch_start + j])
                        self._cache[cache_key] = embedding

        return results

    def _compute_cache_key(self, code: str) -> str:
        """Compute cache key for code."""
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity (0 to 1)
        """
        from madevolve.common.helpers import cosine_similarity
        return cosine_similarity(embedding1, embedding2)

    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Find most similar embeddings.

        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of results to return

        Returns:
            List of (index, similarity) tuples
        """
        similarities = [
            (i, self.compute_similarity(query_embedding, emb))
            for i, emb in enumerate(candidate_embeddings)
        ]

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_total_cost(self) -> float:
        """Get total embedding cost."""
        return self._total_cost

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()

    def close(self):
        """Close the vectorizer."""
        self._adapter = None
        self._cache.clear()
