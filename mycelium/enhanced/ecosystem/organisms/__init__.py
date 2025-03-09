"""
Organism implementations for the enhanced ecosystem.

This module contains the various organism types that can exist
in the enhanced mycelium network environment.
"""

from mycelium.enhanced.ecosystem.organisms.base import Organism
from mycelium.enhanced.ecosystem.organisms.plant import Plant
from mycelium.enhanced.ecosystem.organisms.herbivore import Herbivore
from mycelium.enhanced.ecosystem.organisms.decomposer import Decomposer

__all__ = [
    'Organism',
    'Plant',
    'Herbivore',
    'Decomposer'
]
