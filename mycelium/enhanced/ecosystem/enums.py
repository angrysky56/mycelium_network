"""
Enumerations for the enhanced ecosystem module.

This module defines various enums used across the ecosystem components.
"""

from enum import Enum, auto


class NutrientNeed(Enum):
    """Nutrient requirements for different organism types."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class ReproductionStrategy(Enum):
    """Different reproduction strategies for organisms."""
    ASEXUAL = auto()    # Single-parent reproduction (e.g., budding, division)
    SEXUAL = auto()      # Two-parent reproduction
    SPORES = auto()      # Fungal spore reproduction
    SEEDS = auto()       # Plant seed-based reproduction
