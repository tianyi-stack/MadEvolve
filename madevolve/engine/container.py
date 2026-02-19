"""
Dependency injection container for MadEvolve services.

This module provides a centralized way to manage and inject dependencies
across the evolution framework, enabling loose coupling and testability.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar

from madevolve.engine.configuration import EvolutionConfig

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    """
    Central container for all MadEvolve services.

    Provides lazy initialization and dependency injection for
    all major components of the evolution system.
    """

    config: EvolutionConfig
    results_dir: str

    # Service instances (lazily initialized)
    _services: Dict[str, Any] = None

    def __post_init__(self):
        self._services = {}
        self._factories = {}
        self._register_default_factories()

    def _register_default_factories(self):
        """Register default service factories."""
        # These will be populated when respective modules are imported
        pass

    def register_factory(self, name: str, factory: callable):
        """Register a factory function for a service."""
        self._factories[name] = factory

    def get(self, service_type: Type[T], name: Optional[str] = None) -> T:
        """
        Get or create a service instance.

        Args:
            service_type: The type of service to retrieve
            name: Optional name override for the service

        Returns:
            The service instance
        """
        key = name or service_type.__name__

        if key not in self._services:
            self._services[key] = self._create_service(service_type, key)

        return self._services[key]

    def _create_service(self, service_type: Type[T], key: str) -> T:
        """Create a new service instance."""
        # Check for registered factory
        if key in self._factories:
            return self._factories[key](self)

        # Try to create from service type
        try:
            return service_type(self)
        except TypeError:
            # Service doesn't accept container, try default constructor
            return service_type()

    def get_gateway(self):
        """Get the LLM gateway service."""
        from madevolve.provider.gateway import ModelGateway
        return self.get(ModelGateway)

    def get_vectorizer(self):
        """Get the embedding vectorizer service."""
        from madevolve.provider.vectorizer import Vectorizer
        return self.get(Vectorizer)

    def get_artifact_store(self):
        """Get the artifact storage service."""
        from madevolve.repository.storage.artifact_store import ArtifactStore
        return self.get(ArtifactStore)

    def get_population_manager(self):
        """Get the hybrid population manager."""
        from madevolve.repository.topology.partitions import HybridPopulationManager
        return self.get(HybridPopulationManager)

    def get_parent_selector(self):
        """Get the parent selection service."""
        from madevolve.repository.selection.ancestry import ParentSelector
        return self.get(ParentSelector)

    def get_composer(self):
        """Get the prompt composer service."""
        from madevolve.synthesizer.composer import PromptComposer
        return self.get(PromptComposer)

    def get_dispatcher(self):
        """Get the job dispatcher service."""
        from madevolve.executor.dispatcher import JobDispatcher
        return self.get(JobDispatcher)

    def get_parameter_optimizer(self):
        """Get the inner-loop parameter optimizer."""
        from madevolve.transformer.parallel import ParameterOptimizer
        return self.get(ParameterOptimizer)

    def shutdown(self):
        """Cleanup all services."""
        for name, service in self._services.items():
            if hasattr(service, "close"):
                try:
                    service.close()
                except Exception as e:
                    logger.warning(f"Error closing service {name}: {e}")

        self._services.clear()


class ServiceRegistry:
    """
    Global service registry for managing singletons.

    This is used for services that should have exactly one instance
    across the entire application lifetime.
    """

    _instance: Optional["ServiceRegistry"] = None
    _services: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._services = {}
        return cls._instance

    @classmethod
    def register(cls, name: str, service: Any):
        """Register a global service."""
        cls._services[name] = service

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """Get a registered global service."""
        return cls._services.get(name)

    @classmethod
    def clear(cls):
        """Clear all registered services."""
        cls._services.clear()
