"""
Enhanced Mycelium components with advanced environment and network capabilities.

This package provides extensions to the base Mycelium Network implementation
with more sophisticated environmental modeling and adaptive network behavior.
"""

from mycelium.enhanced.resource import ResourceType, Environmental_Factors
from mycelium.enhanced.environment import TerrainLayer
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork

__all__ = [
    'ResourceType', 
    'Environmental_Factors',
    'TerrainLayer',
    'RichEnvironment',
    'AdaptiveMyceliumNetwork',
]
