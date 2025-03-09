"""
Enhanced environment implementation for the mycelium network.

This module provides a more sophisticated environmental model with:
- Multi-layered terrain with different properties
- Dynamic environmental factors (temperature, moisture, pH, light)
- Resource cycles and interaction between different resource types
- Advanced stimuli and environmental effects
"""

import math
import random
import time
import enum
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from dataclasses import dataclass
import numpy as np

from mycelium.environment import Environment


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


class TerrainLayer:
    """
    Represents a layer of terrain with specific properties.
    
    Each layer has:
    - Height range
    - Conductivity (how easily nodes can grow through it)
    - Resource concentrations
    - Other properties
    """
    
    def __init__(
        self,
        name: str,
        height_range: Tuple[float, float],
        conductivity: float = 1.0,
        density: float = 1.0,
        resource_affinity: Dict[ResourceType, float] = None,
        description: str = ""
    ):
        """
        Initialize a terrain layer.
        
        Args:
            name: Name of the layer (e.g., "topsoil")
            height_range: (min_height, max_height) within the environment
            conductivity: How easily mycelium can grow through (0-1)
            density: Physical density of the layer (affects growth)
            resource_affinity: How well different resources accumulate in this layer
            description: Text description of the layer
        """
        self.name = name
        self.height_range = height_range
        self.conductivity = conductivity
        self.density = density
        self.resource_affinity = resource_affinity or {}
        self.description = description
        
        # Resources present in this layer
        self.resources = {}  # {position: {resource_type: amount}}
        
        # Additional properties
        self.temperature_modifier = 0.0  # How this layer affects temperature
        self.moisture_retention = 1.0    # How well it retains moisture
        self.ph_value = 7.0              # pH of this layer
    
    def contains_point(self, position: Tuple[float, ...]) -> bool:
        """Check if a position is within this layer."""
        if len(position) < 3:
            return True  # In 2D, all points are in all layers
        
        height = position[2]
        return self.height_range[0] <= height <= self.height_range[1]
    
    def get_conductivity_at(self, position: Tuple[float, ...]) -> float:
        """Get the conductivity at a specific position."""
        if not self.contains_point(position):
            return 0.0
        
        # Base conductivity plus random variation
        return max(0.1, min(1.0, self.conductivity * (0.9 + 0.2 * random.random())))
    
    def add_resource(self, position: Tuple[float, ...], resource_type: ResourceType, amount: float) -> None:
        """Add a resource to this layer at the specified position."""
        if not self.contains_point(position):
            return
        
        # Apply resource affinity modifier
        affinity = self.resource_affinity.get(resource_type, 1.0)
        effective_amount = amount * affinity
        
        if position not in self.resources:
            self.resources[position] = {}
        
        current_amount = self.resources[position].get(resource_type, 0.0)
        self.resources[position][resource_type] = current_amount + effective_amount
    
    def get_resources_at(self, position: Tuple[float, ...]) -> Dict[ResourceType, float]:
        """Get all resources at a specific position."""
        if not self.contains_point(position):
            return {}
        
        return self.resources.get(position, {}).copy()
    
    def get_resources_in_range(self, position: Tuple[float, ...], radius: float) -> Dict[Tuple[float, ...], Dict[ResourceType, float]]:
        """Get all resources within a certain radius of a position."""
        result = {}
        for res_pos, resources in self.resources.items():
            # Calculate distance in the x-y plane for 3D environments
            if len(position) == len(res_pos) == 3:
                distance = math.sqrt((position[0] - res_pos[0]) ** 2 + (position[1] - res_pos[1]) ** 2)
            else:
                # Use Euclidean distance for all dimensions
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(position, res_pos)))
            
            if distance <= radius:
                result[res_pos] = resources.copy()
        
        return result
    
    def update(self, delta_time: float = 0.1) -> None:
        """Update the layer state over time."""
        # Decay resources slightly
        for pos in list(self.resources.keys()):
            for resource_type in list(self.resources[pos].keys()):
                # Different decay rates for different resource types
                decay_rate = 0.01  # Default decay rate
                
                if resource_type == ResourceType.WATER:
                    decay_rate = 0.05 / self.moisture_retention  # Water evaporates faster
                elif resource_type == ResourceType.SUGAR:
                    decay_rate = 0.03  # Sugar breaks down
                
                # Apply decay
                self.resources[pos][resource_type] *= (1 - decay_rate * delta_time)
                
                # Remove if too small
                if self.resources[pos][resource_type] < 0.01:
                    del self.resources[pos][resource_type]
            
            # Remove position if no resources left
            if not self.resources[pos]:
                del self.resources[pos]
