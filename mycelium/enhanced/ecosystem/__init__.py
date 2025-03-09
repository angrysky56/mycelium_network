"""
Enhanced ecosystem module for mycelium network.

This module extends the core environment with a more complex ecosystem, 
including diverse organism types, predator-prey relationships, and complex
resource cycles.
"""

from mycelium.enhanced.ecosystem.organisms.base import Organism
from mycelium.enhanced.ecosystem.organisms.plant import Plant
from mycelium.enhanced.ecosystem.organisms.herbivore import Herbivore
from mycelium.enhanced.ecosystem.organisms.decomposer import Decomposer
from mycelium.enhanced.ecosystem.ecosystem import Ecosystem
from mycelium.enhanced.ecosystem.interaction import InteractionRegistry, PredatorPreyInteraction

__all__ = [
    'Organism',
    'Plant',
    'Herbivore',
    'Decomposer',
    'Ecosystem',
    'InteractionRegistry',
    'PredatorPreyInteraction'
]
