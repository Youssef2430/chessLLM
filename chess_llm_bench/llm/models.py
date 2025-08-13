"""
Model configurations for all supported LLM providers.

This module defines the available models for each provider along with their
characteristics and recommended configurations for chess gameplay.
"""

from typing import Dict, List, NamedTuple
from ..core.models import BotSpec


class ModelInfo(NamedTuple):
    """Information about a specific model."""
    name: str
    display_name: str
    provider: str
    model_id: str
    description: str
    recommended: bool = False
    context_length: int = 0
    notes: str = ""


# OpenAI Models
OPENAI_MODELS = [
    ModelInfo(
        name="gpt-4o",
        display_name="GPT-4o",
        provider="openai",
        model_id="gpt-4o",
        description="Latest GPT-4 Omni model with enhanced reasoning",
        recommended=True,
        context_length=128000,
        notes="Best overall performance, multimodal capabilities"
    ),
    ModelInfo(
        name="gpt-4o-mini",
        display_name="GPT-4o Mini",
        provider="openai",
        model_id="gpt-4o-mini",
        description="Faster, cost-effective GPT-4o variant",
        recommended=True,
        context_length=128000,
        notes="Great balance of speed and intelligence"
    ),
    ModelInfo(
        name="gpt-4-turbo",
        display_name="GPT-4 Turbo",
        provider="openai",
        model_id="gpt-4-turbo",
        description="High-performance GPT-4 with large context",
        recommended=False,
        context_length=128000,
        notes="Powerful but slower than 4o variants"
    ),
    ModelInfo(
        name="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        provider="openai",
        model_id="gpt-3.5-turbo",
        description="Fast and efficient legacy model",
        recommended=False,
        context_length=16385,
        notes="Budget option, decent chess performance"
    ),
]

# Anthropic Models
ANTHROPIC_MODELS = [
    ModelInfo(
        name="claude-3-5-sonnet",
        display_name="Claude 3.5 Sonnet",
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        description="Most intelligent Claude model with excellent reasoning",
        recommended=True,
        context_length=200000,
        notes="Top-tier performance, excellent at strategic thinking"
    ),
    ModelInfo(
        name="claude-3-5-haiku",
        display_name="Claude 3.5 Haiku",
        provider="anthropic",
        model_id="claude-3-5-haiku-20241022",
        description="Fast and efficient Claude model",
        recommended=True,
        context_length=200000,
        notes="Good balance of speed and capability"
    ),
    ModelInfo(
        name="claude-3-opus",
        display_name="Claude 3 Opus",
        provider="anthropic",
        model_id="claude-3-opus-20240229",
        description="Most capable legacy Claude model",
        recommended=False,
        context_length=200000,
        notes="Powerful but expensive, superseded by 3.5 Sonnet"
    ),
    ModelInfo(
        name="claude-3-haiku",
        display_name="Claude 3 Haiku",
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
        description="Fast and cost-effective legacy model",
        recommended=False,
        context_length=200000,
        notes="Budget option, decent performance"
    ),
]

# Google Gemini Models
GEMINI_MODELS = [
    ModelInfo(
        name="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        provider="gemini",
        model_id="gemini-2.5-pro",
        description="Most capable Gemini model with large context",
        recommended=True,
        context_length=1000000,
        notes="Excellent reasoning, massive context window"
    ),
    ModelInfo(
        name="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        provider="gemini",
        model_id="gemini-2.5-flash",
        description="Fast and efficient Gemini model",
        recommended=True,
        context_length=1000000,
        notes="Great speed-to-performance ratio"
    ),
    ModelInfo(
        name="gemini-1.0-pro",
        display_name="Gemini 1.0 Pro",
        provider="gemini",
        model_id="gemini-1.0-pro",
        description="Legacy Gemini Pro model",
        recommended=False,
        context_length=30720,
        notes="Older model, use 2.5 variants instead"
    ),
]

# All models combined
ALL_MODELS = OPENAI_MODELS + ANTHROPIC_MODELS + GEMINI_MODELS

# Models grouped by provider
MODELS_BY_PROVIDER: Dict[str, List[ModelInfo]] = {
    "openai": OPENAI_MODELS,
    "anthropic": ANTHROPIC_MODELS,
    "gemini": GEMINI_MODELS,
}

# Recommended models for quick access
RECOMMENDED_MODELS = [model for model in ALL_MODELS if model.recommended]


def get_model_info(provider: str, model_id: str) -> ModelInfo:
    """
    Get model information by provider and model ID.

    Args:
        provider: Provider name (openai, anthropic, gemini)
        model_id: Model identifier

    Returns:
        ModelInfo object

    Raises:
        ValueError: If model not found
    """
    provider = provider.lower()

    if provider not in MODELS_BY_PROVIDER:
        raise ValueError(f"Unknown provider: {provider}")

    for model in MODELS_BY_PROVIDER[provider]:
        if model.model_id == model_id or model.name == model_id:
            return model

    raise ValueError(f"Unknown model {model_id} for provider {provider}")


def get_provider_models(provider: str) -> List[ModelInfo]:
    """
    Get all models for a specific provider.

    Args:
        provider: Provider name

    Returns:
        List of ModelInfo objects
    """
    provider = provider.lower()
    return MODELS_BY_PROVIDER.get(provider, [])


def get_recommended_models(provider: str = None) -> List[ModelInfo]:
    """
    Get recommended models, optionally filtered by provider.

    Args:
        provider: Optional provider filter

    Returns:
        List of recommended ModelInfo objects
    """
    if provider:
        provider = provider.lower()
        return [model for model in RECOMMENDED_MODELS if model.provider == provider]
    return RECOMMENDED_MODELS


def create_bot_specs(models: List[ModelInfo]) -> List[BotSpec]:
    """
    Create BotSpec objects from ModelInfo objects.

    Args:
        models: List of ModelInfo objects

    Returns:
        List of BotSpec objects
    """
    return [
        BotSpec(
            provider=model.provider,
            model=model.model_id,
            name=model.display_name
        )
        for model in models
    ]


def get_premium_bot_lineup() -> List[BotSpec]:
    """
    Get a premium lineup of the best models from each provider.

    Returns:
        List of BotSpec objects for top models
    """
    premium_models = [
        get_model_info("openai", "gpt-4o"),
        get_model_info("openai", "gpt-4o-mini"),
        get_model_info("anthropic", "claude-3-5-sonnet"),
        get_model_info("anthropic", "claude-3-5-haiku"),
        get_model_info("gemini", "gemini-2.5-pro"),
        get_model_info("gemini", "gemini-2.5-flash"),
    ]
    return create_bot_specs(premium_models)


def get_budget_bot_lineup() -> List[BotSpec]:
    """
    Get a budget-friendly lineup focusing on speed and cost.

    Returns:
        List of BotSpec objects for budget models
    """
    budget_models = [
        get_model_info("openai", "gpt-4o-mini"),
        get_model_info("openai", "gpt-3.5-turbo"),
        get_model_info("anthropic", "claude-3-5-haiku"),
        get_model_info("anthropic", "claude-3-haiku"),
        get_model_info("gemini", "gemini-2.5-flash"),
    ]
    return create_bot_specs(budget_models)


def get_all_recommended_bots() -> List[BotSpec]:
    """
    Get all recommended models as BotSpec objects.

    Returns:
        List of BotSpec objects for all recommended models
    """
    return create_bot_specs(RECOMMENDED_MODELS)


# Preset configurations for common use cases
PRESET_CONFIGS = {
    "premium": {
        "bots": get_premium_bot_lineup(),
        "description": "Top-tier models from each provider"
    },
    "budget": {
        "bots": get_budget_bot_lineup(),
        "description": "Cost-effective models with good performance"
    },
    "recommended": {
        "bots": get_all_recommended_bots(),
        "description": "All recommended models across providers"
    },
    "openai": {
        "bots": create_bot_specs(get_recommended_models("openai")),
        "description": "OpenAI's best models"
    },
    "anthropic": {
        "bots": create_bot_specs(get_recommended_models("anthropic")),
        "description": "Anthropic's best models"
    },
    "gemini": {
        "bots": create_bot_specs(get_recommended_models("gemini")),
        "description": "Google's best Gemini models"
    },
}


def format_bot_spec_string(bot_specs: List[BotSpec]) -> str:
    """
    Format a list of BotSpec objects as a comma-separated string.

    Args:
        bot_specs: List of BotSpec objects

    Returns:
        Formatted string for CLI usage
    """
    return ",".join(f"{bot.provider}:{bot.model}:{bot.name}" for bot in bot_specs)


def print_available_models():
    """Print a nicely formatted list of all available models."""
    print("\n🤖 Available LLM Models for Chess Benchmarking\n")

    for provider, models in MODELS_BY_PROVIDER.items():
        print(f"📡 {provider.upper()}")
        print("-" * (len(provider) + 3))

        for model in models:
            status = "⭐ RECOMMENDED" if model.recommended else "  Available"
            print(f"{status} {model.display_name}")
            print(f"    Model ID: {model.model_id}")
            print(f"    Context: {model.context_length:,} tokens")
            print(f"    Description: {model.description}")
            if model.notes:
                print(f"    Notes: {model.notes}")
            print()

        print()
