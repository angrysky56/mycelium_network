"""
Utility functions for the enhanced mycelium environment.

This module contains helper functions for the RichEnvironment class that
were too large to fit in the main file.
"""

import random
from typing import Dict, Tuple, List, Optional, Any

from mycelium.enhanced.resource import ResourceType
from mycelium.enhanced.layers import TerrainLayer


def count_resources_in_layer(layer: TerrainLayer, resource_type: ResourceType) -> float:
    """
    Count the total amount of a specific resource type in a layer.
    
    Args:
        layer: The terrain layer
        resource_type: Type of resource to count
        
    Returns:
        Total amount of that resource
    """
    total = 0.0
    for pos, resources in layer.resources.items():
        total += resources.get(resource_type, 0.0)
    return total


def apply_seasonal_spring_effect(env, phase: float, delta_time: float, intensity: float):
    """
    Apply spring season effects (more water, growth).
    
    Args:
        env: The RichEnvironment instance
        phase: Current phase within the season (0-1)
        delta_time: Time step size
        intensity: Effect strength
    """
    # Randomly add some water resources in top layers
    if random.random() < 0.3 * intensity * delta_time:  # Spring showers
        # Generate rain
        num_drops = int(5 * intensity)
        for _ in range(num_drops):
            x = random.random() * env.size
            y = random.random() * env.size
            if env.dimensions >= 3:
                # In 3D environments, rain falls from the top
                position = (x, y, 0.7 + random.random() * 0.3)
            else:
                position = (x, y)
            
            env.add_resource(position, 0.2 * intensity, ResourceType.WATER)
    
    # Add some nitrogen (growth nutrients)
    if random.random() < 0.1 * intensity * delta_time:
        x = random.random() * env.size
        y = random.random() * env.size
        if env.dimensions >= 3:
            # Add to topsoil layer
            position = (x, y, 0.6 + random.random() * 0.1)
        else:
            position = (x, y)
        
        env.add_resource(position, 0.1 * intensity, ResourceType.NITROGEN)


def apply_seasonal_summer_effect(env, phase: float, delta_time: float, intensity: float):
    """
    Apply summer season effects (more light, evaporation).
    
    Args:
        env: The RichEnvironment instance
        phase: Current phase within the season (0-1)
        delta_time: Time step size
        intensity: Effect strength
    """
    # Increase light level
    light_boost = 0.2 * intensity * (1 - phase)  # Decreases toward end of summer
    env.factors.light_level = min(1.0, env.factors.light_level + light_boost)
    
    # Evaporate some water
    evaporation_rate = 0.05 * intensity * delta_time
    for pos in list(env.resources.keys()):
        if isinstance(env.resources[pos], dict) and ResourceType.WATER in env.resources[pos]:
            env.resources[pos][ResourceType.WATER] *= (1 - evaporation_rate)
            
            # Remove if too small
            if env.resources[pos][ResourceType.WATER] < 0.01:
                del env.resources[pos][ResourceType.WATER]
            
            # Clean up empty entries
            if not env.resources[pos]:
                del env.resources[pos]


def apply_seasonal_fall_effect(env, phase: float, delta_time: float, intensity: float):
    """
    Apply fall season effects (falling leaves, decreasing light).
    
    Args:
        env: The RichEnvironment instance
        phase: Current phase within the season (0-1)
        delta_time: Time step size
        intensity: Effect strength
    """
    # Decrease light level
    light_reduction = 0.1 * intensity * phase  # Increases toward end of fall
    env.factors.light_level = max(0.3, env.factors.light_level - light_reduction)
    
    # Add some carbon (falling leaves)
    if random.random() < 0.2 * intensity * delta_time:
        x = random.random() * env.size
        y = random.random() * env.size
        if env.dimensions >= 3:
            # Add to ground level
            position = (x, y, 0.6)
        else:
            position = (x, y)
        
        env.add_resource(position, 0.1 * intensity, ResourceType.CARBON)


def apply_seasonal_winter_effect(env, phase: float, delta_time: float, intensity: float):
    """
    Apply winter season effects (cold, dormancy).
    
    Args:
        env: The RichEnvironment instance
        phase: Current phase within the season (0-1)
        delta_time: Time step size
        intensity: Effect strength
    """
    # Decrease temperature
    temp_reduction = 0.2 * intensity * (1 - phase)  # Increases in early winter
    env.factors.temperature = max(0.1, env.factors.temperature - temp_reduction)
    
    # Slow down all resource interactions
    env.factors.moisture = max(0.2, env.factors.moisture - 0.05 * intensity * delta_time)
    
    # Occasionally add snow (special water resource)
    if random.random() < 0.1 * intensity * delta_time:
        num_snowflakes = int(3 * intensity)
        for _ in range(num_snowflakes):
            x = random.random() * env.size
            y = random.random() * env.size
            if env.dimensions >= 3:
                # Snow accumulates on top layers
                position = (x, y, 0.7)
            else:
                position = (x, y)
            
            # Add frozen water (special property)
            env.add_resource(position, 0.05 * intensity, ResourceType.WATER)
            
            # Mark as frozen (if environment supports properties)
            if position in env.resources and isinstance(env.resources[position], dict):
                if ResourceType.WATER in env.resources[position]:
                    env.resources[position]['properties'] = env.resources[position].get('properties', {})
                    env.resources[position]['properties']['frozen'] = True
