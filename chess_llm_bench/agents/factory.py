"""
Agent Factory

This module provides factory functions for creating appropriate chess agents
from bot specifications. It handles the creation of different agent types
(LLM, random, etc.) while maintaining compatibility with the existing
chess LLM benchmark system.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type, Union

from .base import ChessAgent, ChessAgentConfig
from .llm_agent import LLMChessAgent
from .random_agent import RandomPlayerAgent
from .auto_reply_agent import AutoReplyAgent
from ..core.models import BotSpec
from ..llm.client import LLMProviderError

logger = logging.getLogger(__name__)


class AgentCreationError(Exception):
    """Raised when agent creation fails."""
    pass


class AgentFactory:
    """
    Factory class for creating chess agents from bot specifications.

    This factory maintains a registry of agent types and their corresponding
    creation functions, allowing for easy extension with new agent types.
    """

    def __init__(self):
        """Initialize the agent factory."""
        self._agent_registry: Dict[str, Type[ChessAgent]] = {}
        self._creation_stats = {
            "agents_created": 0,
            "creation_failures": 0,
            "provider_usage": {},
        }

        # Register default agent types
        self._register_default_agents()

    def _register_default_agents(self) -> None:
        """Register the default agent types."""
        # LLM providers
        self.register_agent_type("openai", LLMChessAgent)
        self.register_agent_type("anthropic", LLMChessAgent)
        self.register_agent_type("gemini", LLMChessAgent)
        self.register_agent_type("google", LLMChessAgent)  # Alias for gemini

        # Non-LLM agents
        self.register_agent_type("random", RandomPlayerAgent)

        logger.info("Registered default agent types")

    def register_agent_type(self, provider: str, agent_class: Type[ChessAgent]) -> None:
        """
        Register a new agent type for a provider.

        Args:
            provider: Provider name (e.g., 'openai', 'random')
            agent_class: Agent class to instantiate for this provider
        """
        self._agent_registry[provider.lower()] = agent_class
        logger.debug(f"Registered agent type: {provider} -> {agent_class.__name__}")

    def create_agent(
        self,
        bot_spec: BotSpec,
        temperature: float = 0.7,
        timeout_seconds: float = 30.0,
        **kwargs
    ) -> ChessAgent:
        """
        Create a chess agent from a bot specification.

        Args:
            bot_spec: Bot specification with provider and model info
            temperature: LLM temperature for move generation
            timeout_seconds: Timeout for move generation
            **kwargs: Additional agent-specific parameters

        Returns:
            Configured chess agent

        Raises:
            AgentCreationError: If agent creation fails
        """
        try:
            provider = bot_spec.provider.lower()

            # Update usage statistics
            self._creation_stats["provider_usage"][provider] = (
                self._creation_stats["provider_usage"].get(provider, 0) + 1
            )

            # Get the appropriate agent class
            if provider not in self._agent_registry:
                raise AgentCreationError(
                    f"Unknown provider: {provider}. "
                    f"Available providers: {list(self._agent_registry.keys())}"
                )

            agent_class = self._agent_registry[provider]

            # Create agent based on type
            if agent_class == LLMChessAgent:
                agent = self._create_llm_agent(
                    bot_spec, temperature, timeout_seconds, **kwargs
                )
            elif agent_class == RandomPlayerAgent:
                agent = self._create_random_agent(bot_spec, **kwargs)
            else:
                # Generic agent creation
                agent = agent_class(
                    bot_spec=bot_spec,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    **kwargs
                )

            self._creation_stats["agents_created"] += 1
            logger.info(f"Created agent: {agent.name} ({provider}:{bot_spec.model})")

            return agent

        except Exception as e:
            self._creation_stats["creation_failures"] += 1
            logger.error(f"Failed to create agent for {bot_spec}: {e}")
            raise AgentCreationError(f"Agent creation failed: {e}") from e

    def _create_llm_agent(
        self,
        bot_spec: BotSpec,
        temperature: float,
        timeout_seconds: float,
        **kwargs
    ) -> LLMChessAgent:
        """Create an LLM-based chess agent."""
        return LLMChessAgent(
            bot_spec=bot_spec,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def _create_random_agent(self, bot_spec: BotSpec, **kwargs) -> RandomPlayerAgent:
        """Create a random chess agent."""
        name = bot_spec.name or "RandomBot"
        return RandomPlayerAgent(name=name, bot_spec=bot_spec, **kwargs)

    def create_agents_from_specs(
        self,
        bot_specs: List[BotSpec],
        temperature: float = 0.7,
        timeout_seconds: float = 30.0,
        **kwargs
    ) -> List[ChessAgent]:
        """
        Create multiple agents from a list of bot specifications.

        Args:
            bot_specs: List of bot specifications
            temperature: Default LLM temperature
            timeout_seconds: Default timeout
            **kwargs: Additional agent parameters

        Returns:
            List of created chess agents

        Raises:
            AgentCreationError: If any agent creation fails
        """
        agents = []
        failures = []

        for bot_spec in bot_specs:
            try:
                agent = self.create_agent(
                    bot_spec=bot_spec,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    **kwargs
                )
                agents.append(agent)
            except AgentCreationError as e:
                failures.append((bot_spec, str(e)))
                logger.error(f"Failed to create agent for {bot_spec}: {e}")

        if failures and not agents:
            # All agents failed to create
            failure_details = "; ".join([f"{spec}: {error}" for spec, error in failures])
            raise AgentCreationError(f"All agent creations failed: {failure_details}")
        elif failures:
            # Some agents failed
            logger.warning(f"Failed to create {len(failures)} out of {len(bot_specs)} agents")

        logger.info(f"Successfully created {len(agents)} agents")
        return agents

    def get_supported_providers(self) -> List[str]:
        """Get list of supported providers."""
        return list(self._agent_registry.keys())

    def is_provider_supported(self, provider: str) -> bool:
        """Check if a provider is supported."""
        return provider.lower() in self._agent_registry

    def get_creation_stats(self) -> Dict[str, any]:
        """Get agent creation statistics."""
        return self._creation_stats.copy()

    def reset_stats(self) -> None:
        """Reset creation statistics."""
        self._creation_stats = {
            "agents_created": 0,
            "creation_failures": 0,
            "provider_usage": {},
        }

    def validate_bot_spec(self, bot_spec: BotSpec) -> tuple[bool, Optional[str]]:
        """
        Validate a bot specification.

        Args:
            bot_spec: Bot specification to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not bot_spec.provider:
            return False, "Provider is required"

        if not self.is_provider_supported(bot_spec.provider):
            return False, f"Unsupported provider: {bot_spec.provider}"

        # Provider-specific validation
        provider = bot_spec.provider.lower()

        if provider in ["openai", "anthropic", "gemini", "google"]:
            # LLM providers should have valid model names
            if not bot_spec.model or not bot_spec.model.strip():
                return False, f"Model name is required for {provider}"
        elif provider == "random":
            # Random provider doesn't require a model name
            pass
        else:
            # For other providers, model is generally required
            if not bot_spec.model:
                return False, f"Model is required for provider {provider}"

        return True, None


# Global factory instance
_default_factory = AgentFactory()


def create_agent(
    bot_spec: BotSpec,
    temperature: float = 0.7,
    timeout_seconds: float = 30.0,
    factory: Optional[AgentFactory] = None,
    **kwargs
) -> ChessAgent:
    """
    Create a chess agent from a bot specification using the default factory.

    Args:
        bot_spec: Bot specification
        temperature: LLM temperature
        timeout_seconds: Move generation timeout
        factory: Custom factory instance (uses default if None)
        **kwargs: Additional agent parameters

    Returns:
        Configured chess agent
    """
    factory = factory or _default_factory
    return factory.create_agent(
        bot_spec=bot_spec,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        **kwargs
    )


def create_agents(
    bot_specs: List[BotSpec],
    temperature: float = 0.7,
    timeout_seconds: float = 30.0,
    factory: Optional[AgentFactory] = None,
    **kwargs
) -> List[ChessAgent]:
    """
    Create multiple chess agents from bot specifications.

    Args:
        bot_specs: List of bot specifications
        temperature: Default LLM temperature
        timeout_seconds: Default timeout
        factory: Custom factory instance (uses default if None)
        **kwargs: Additional agent parameters

    Returns:
        List of created chess agents
    """
    factory = factory or _default_factory
    return factory.create_agents_from_specs(
        bot_specs=bot_specs,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        **kwargs
    )


def create_random_agent(name: str = "RandomBot") -> RandomPlayerAgent:
    """
    Create a random chess agent.

    Args:
        name: Agent name

    Returns:
        RandomPlayerAgent instance
    """
    bot_spec = BotSpec(provider="random", model="baseline", name=name)
    return _default_factory.create_agent(bot_spec)


def create_llm_agent(
    provider: str,
    model: str,
    name: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> LLMChessAgent:
    """
    Create an LLM-based chess agent.

    Args:
        provider: LLM provider (openai, anthropic, gemini)
        model: Model name
        name: Agent name (auto-generated if None)
        temperature: LLM temperature
        **kwargs: Additional agent parameters

    Returns:
        LLMChessAgent instance
    """
    bot_spec = BotSpec(
        provider=provider,
        model=model,
        name=name or f"{provider}_{model}"
    )
    return _default_factory.create_agent(
        bot_spec=bot_spec,
        temperature=temperature,
        **kwargs
    )


def get_supported_providers() -> List[str]:
    """Get list of supported providers from the default factory."""
    return _default_factory.get_supported_providers()


def is_provider_supported(provider: str) -> bool:
    """Check if a provider is supported by the default factory."""
    return _default_factory.is_provider_supported(provider)


def validate_bot_spec(bot_spec: BotSpec) -> tuple[bool, Optional[str]]:
    """Validate a bot specification using the default factory."""
    return _default_factory.validate_bot_spec(bot_spec)


def register_custom_agent(
    provider: str,
    agent_class: Type[ChessAgent],
    factory: Optional[AgentFactory] = None
) -> None:
    """
    Register a custom agent type with the factory.

    Args:
        provider: Provider name
        agent_class: Agent class to register
        factory: Factory instance (uses default if None)
    """
    factory = factory or _default_factory
    factory.register_agent_type(provider, agent_class)


# Convenience functions for common agent types
def create_openai_agent(
    model: str = "gpt-4o-mini",
    name: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> LLMChessAgent:
    """Create an OpenAI chess agent."""
    return create_llm_agent("openai", model, name, temperature, **kwargs)


def create_anthropic_agent(
    model: str = "claude-3-5-sonnet-20241022",
    name: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> LLMChessAgent:
    """Create an Anthropic chess agent."""
    return create_llm_agent("anthropic", model, name, temperature, **kwargs)


def create_gemini_agent(
    model: str = "gemini-1.5-flash",
    name: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> LLMChessAgent:
    """Create a Gemini chess agent."""
    return create_llm_agent("gemini", model, name, temperature, **kwargs)


def get_factory_stats() -> Dict[str, any]:
    """Get statistics from the default factory."""
    return _default_factory.get_creation_stats()
