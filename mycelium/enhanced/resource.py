"""
Resource types and environmental factors for the enhanced mycelium environment.

This module defines enumerations and data structures for resources and 
environmental conditions.
"""

import enum
from dataclasses import dataclass


class ResourceType(enum.Enum):
    """Types of resources available in the environment."""
    CARBON = "carbon"
    NITROGEN = "nitrogen"
    PHOSPHORUS = "phosphorus"
    WATER = "water"
    SUGAR = "sugar"
    PROTEIN = "protein"
    MINERAL = "mineral"
    LIGHT = "light"


@dataclass
class Environmental_Factors:
    """Environmental conditions that affect growth and behavior."""
    temperature: float = 0.5  # 0-1 scale (0=cold, 1=hot)
    moisture: float = 0.5     # 0-1 scale (0=dry, 1=wet)
    ph: float = 7.0           # pH scale (0-14)
    light_level: float = 0.5  # 0-1 scale (0=dark, 1=bright)
    toxicity: float = 0.0     # 0-1 scale (0=clean, 1=toxic)
    oxygen: float = 0.5       # 0-1 scale (0=anoxic, 1=oxygen-rich)
    wind: float = 0.0         # 0-1 scale (0=still, 1=strong wind)
    gravity: float = 1.0      # relative to Earth gravity
    season: int = 0           # 0=spring, 1=summer, 2=fall, 3=winter
