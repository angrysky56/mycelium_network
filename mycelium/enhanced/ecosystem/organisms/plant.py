"""
Plant organism for the enhanced ecosystem.

This module defines plant organisms that produce energy through photosynthesis.
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Any

from mycelium.enhanced.ecosystem.enums import NutrientNeed, ReproductionStrategy
from mycelium.enhanced.ecosystem.organisms.base import Organism


class Plant(Organism):
    """
    Plant organism that produces energy through photosynthesis.
    
    Plants convert light and resources into energy, serving as primary
    producers in the ecosystem food web.
    """
    
    def __init__(
        self,
        organism_id: str,
        position: Tuple[float, ...],
        energy: float = 1.0,
        size: float = 1.0,
        properties: Dict[str, Any] = None
    ):
        """
        Initialize a plant.
        
        Args:
            organism_id: Unique identifier for this plant
            position: Spatial position of the plant
            energy: Current energy level (0-1)
            size: Size of the plant
            properties: Additional properties for this plant
        """
        properties = properties or {}
        
        # Plant-specific properties
        plant_properties = {
            "photosynthesis_efficiency": properties.get("photosynthesis_efficiency", 0.1),
            "water_absorption_rate": properties.get("water_absorption_rate", 0.1),
            "root_depth": properties.get("root_depth", 0.2),
            "leaf_area": properties.get("leaf_area", size * 0.5),
            "growth_rate": properties.get("growth_rate", 0.01)
        }
        properties.update(plant_properties)
        
        super().__init__(
            organism_id=organism_id,
            position=position,
            energy=energy,
            size=size,
            lifespan=100.0 + random.uniform(-20, 20),  # Variable lifespan
            reproduction_rate=0.005,  # Slower reproduction
            reproduction_strategy=ReproductionStrategy.SEEDS,
            properties=properties
        )
        
        # Plants need water, carbon, and minerals
        self.nutrient_needs = {
            "WATER": NutrientNeed.HIGH,
            "CARBON": NutrientNeed.MEDIUM,
            "NITROGEN": NutrientNeed.MEDIUM,
            "MINERAL": NutrientNeed.LOW
        }
    
    def update(self, environment, delta_time: float) -> Dict[str, Any]:
        """
        Update plant state for a time step.
        
        Args:
            environment: The environment the plant exists in
            delta_time: Time step size
            
        Returns:
            State changes from this update
        """
        # Run base organism update
        result = super().update(environment, delta_time)
        
        if not self.alive:
            return result
        
        # Get environmental factors at plant's position
        env_factors = environment.get_environmental_factors_at(self.position)
        light_level = env_factors.light_level
        
        # Calculate season information
        try:
            season_idx = int(((environment.time % environment.year_length) / environment.year_length) * 4) % 4
            season_names = ["Spring", "Summer", "Fall", "Winter"]
            current_season = season_names[season_idx]
            
            # Season-specific growth rates
            season_growth_factors = {
                "Spring": 1.2,  # Rapid growth
                "Summer": 1.0,  # Normal growth
                "Fall": 0.6,    # Slowing growth
                "Winter": 0.2   # Minimal growth
            }
            season_factor = season_growth_factors[current_season]
        except AttributeError:
            # If environment doesn't have year_length, default to no seasonal effect
            season_factor = 1.0
        
        # Modify photosynthesis based on environmental factors
        light_factor = min(1.5, max(0.2, light_level * 2))  # Light is crucial
        moisture_factor = min(1.2, max(0.3, env_factors.moisture * 1.5))  # Moisture important
        
        # Photosynthesis affected by light, moisture, and season
        photosynthesis_rate = (
            self.properties["photosynthesis_efficiency"] *
            light_factor * 
            moisture_factor *
            season_factor
        )
        
        # Energy gain from photosynthesis
        energy_gain = photosynthesis_rate * delta_time
        self.energy = min(1.0, self.energy + energy_gain)
        
        # Consume resources from environment
        resource_types = ["WATER", "CARBON", "NITROGEN", "MINERAL"]
        for resource_type in resource_types:
            if hasattr(environment, "get_resources_at"):
                resources = environment.get_resources_at(self.position)
                if resource_type in resources and resources[resource_type] > 0:
                    # Consume based on need level
                    consumption = 0.01 * delta_time * self.size * self.nutrient_needs.get(resource_type, NutrientNeed.LOW).value
                    # (Note: In a real implementation, we would update environment resources here)
        
        # Growth affected by all environmental factors and season
        growth_rate = self.properties["growth_rate"] * season_factor
        
        # Growth conditions improved to be more sensitive to environment
        growth_conditions = env_factors.moisture > 0.3 and light_level > 0.3 and self.energy > 0.6
        
        # Calculate growth based on conditions
        growth = 0
        if growth_conditions:
            combined_factor = light_factor * moisture_factor * season_factor
            growth = growth_rate * combined_factor * delta_time
            self.size += growth
            self.energy -= growth * 0.5  # Growth costs energy
            
            # Output seasonal info for debugging
            if hasattr(environment, "year_length"):
                result["season"] = current_season
                result["season_factor"] = season_factor
        
        result.update({
            "photosynthesis": energy_gain,
            "growth": growth,
            "light_factor": light_factor,
            "moisture_factor": moisture_factor
        })
        
        return result
    
    def interact(self, other_organism, environment, interaction_type: str) -> Dict[str, Any]:
        """
        Interact with another organism.
        
        Args:
            other_organism: The organism to interact with
            environment: The environment context
            interaction_type: Type of interaction to perform
            
        Returns:
            Interaction results
        """
        result = {"interaction": interaction_type, "effect": "none"}
        
        if interaction_type == "competition":
            # Plants can compete for light and resources
            if isinstance(other_organism, Plant):
                # Larger plants shade smaller ones
                if self.size > other_organism.size:
                    light_reduction = 0.2 * (self.size / other_organism.size)
                    result["effect"] = "shading"
                    result["intensity"] = light_reduction
        
        return result
    
    def reproduce(self, environment, partner=None) -> Optional['Plant']:
        """
        Create a new plant through seed reproduction.
        
        Args:
            environment: The environment context
            partner: Not needed for plants (pollination is implicit)
            
        Returns:
            New plant or None if reproduction failed
        """
        if not self.can_reproduce(environment):
            return None
        
        # Plants need good conditions to reproduce
        env_factors = environment.get_environmental_factors_at(self.position)
        if env_factors.moisture < 0.4 or self.energy < 0.7:
            return None
        
        # Calculate seed spread direction
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0.1, 0.3) * self.size
        
        # New position for offspring
        new_position = list(self.position)
        new_position[0] += math.cos(angle) * distance
        new_position[1] += math.sin(angle) * distance
        
        # Ensure within bounds
        for i in range(len(new_position)):
            new_position[i] = max(0, min(environment.size, new_position[i]))
        
        # Create new plant with some genetic variation
        seed_id = f"{self.id}_offspring_{random.randint(1000, 9999)}"
        
        # Inherit properties with some variation
        child_properties = self.properties.copy()
        for key in child_properties:
            # Add up to ±10% variation to numeric properties
            if isinstance(child_properties[key], (int, float)):
                variation = random.uniform(-0.1, 0.1)
                child_properties[key] *= (1 + variation)
        
        # Create new plant
        new_plant = Plant(
            organism_id=seed_id,
            position=tuple(new_position),
            energy=0.5,  # Start with less energy
            size=self.size * 0.3,  # Start smaller
            properties=child_properties
        )
        
        # Reproduction costs energy
        self.energy -= 0.3
        
        return new_plant
