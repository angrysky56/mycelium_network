"""
Decomposer organism for the enhanced ecosystem.

This module defines decomposers like fungi that break down organic matter.
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Any

from mycelium.enhanced.ecosystem.enums import NutrientNeed, ReproductionStrategy
from mycelium.enhanced.ecosystem.organisms.base import Organism


class Decomposer(Organism):
    """
    Decomposer organism that breaks down dead organic matter.
    
    Decomposers play a crucial role in nutrient cycling by breaking down
    dead organisms and releasing nutrients back into the environment.
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
        Initialize a decomposer.
        
        Args:
            organism_id: Unique identifier for this decomposer
            position: Spatial position of the decomposer
            energy: Current energy level (0-1)
            size: Size of the decomposer
            properties: Additional properties for this decomposer
        """
        properties = properties or {}
        
        # Decomposer-specific properties
        decomposer_properties = {
            "decomposition_rate": properties.get("decomposition_rate", 0.1),
            "growth_rate": properties.get("growth_rate", 0.03),
            "spread_rate": properties.get("spread_rate", 0.02),
            "enzyme_efficiency": properties.get("enzyme_efficiency", 0.4),
            "moisture_preference": properties.get("moisture_preference", 0.7)  # Prefer damp environments
        }
        properties.update(decomposer_properties)
        
        super().__init__(
            organism_id=organism_id,
            position=position,
            energy=energy,
            size=size,
            lifespan=75.0 + random.uniform(-15, 15),  # Variable lifespan
            reproduction_rate=0.03,  # Faster reproduction
            reproduction_strategy=ReproductionStrategy.SPORES,
            properties=properties
        )
        
        # Lower metabolism rate for decomposers
        self.metabolism_rate = 0.005
        
        # Nutrient needs
        self.nutrient_needs = {
            "WATER": NutrientNeed.MEDIUM,
            "CARBON": NutrientNeed.HIGH,
            "NITROGEN": NutrientNeed.MEDIUM
        }
        
        # Decomposer state
        self.attached_to = None  # ID of dead organism being decomposed
        self.hypha_network = []  # Network of connected points
        self.nutrients_released = {
            "CARBON": 0.0,
            "NITROGEN": 0.0,
            "PHOSPHORUS": 0.0,
            "MINERAL": 0.0
        }
    
    def update(self, environment, delta_time: float) -> Dict[str, Any]:
        """
        Update decomposer state for a time step.
        
        Args:
            environment: The environment the decomposer exists in
            delta_time: Time step size
            
        Returns:
            State changes from this update
        """
        # Run base organism update
        result = super().update(environment, delta_time)
        
        if not self.alive:
            return result
        
        # Get environmental factors
        env_factors = environment.get_environmental_factors_at(self.position)
        
        # Decomposers thrive in moist environments
        moisture_factor = self._calculate_moisture_factor(env_factors.moisture)
        
        # Find dead organisms to decompose
        if not self.attached_to and hasattr(environment, "get_organisms_in_range"):
            organisms = environment.get_organisms_in_range(
                self.position, 
                0.1,  # Short range, must be close
            )
            # Find dead organisms
            dead_organisms = {id: org for id, org in organisms.items() 
                             if not org["alive"] and not org.get("type") == "Decomposer"}
            
            if dead_organisms:
                # Attach to a random dead organism
                dead_id = random.choice(list(dead_organisms.keys()))
                self.attached_to = dead_id
                result["attached_to"] = dead_id
        
        # If attached to a dead organism, decompose it
        nutrients_released = {}
        if self.attached_to:
            # Decompose at a rate based on environmental conditions
            decomposition_rate = (
                self.properties["decomposition_rate"] *
                moisture_factor *
                self.properties["enzyme_efficiency"] * 
                delta_time
            )
            
            # Add randomness to decomposition for more natural behavior
            activity_factor = random.uniform(0.8, 1.2)
            actual_decomp_rate = decomposition_rate * activity_factor
            
            # Convert dead matter to nutrients (simulation)
            for nutrient_type in self.nutrients_released:
                release_amount = actual_decomp_rate * random.uniform(0.5, 1.0)
                self.nutrients_released[nutrient_type] += release_amount
                nutrients_released[nutrient_type] = release_amount
                
                # Add nutrients back to environment (in a real implementation)
                # environment.add_resource(self.position, release_amount, nutrient_type)
            
            # Get energy from decomposition
            energy_gain = actual_decomp_rate * 0.5
            self.energy = min(1.0, self.energy + energy_gain)
            
            # Important: Make sure this value is non-zero for tracking
            if energy_gain > 0:
                print(f"Decomposer {self.id} processed dead organism {self.attached_to} and gained {energy_gain:.3f} energy")
            
            result["decomposing"] = {
                "target": self.attached_to,
                "rate": actual_decomp_rate,
                "energy_gain": energy_gain,
                "nutrients_released": nutrients_released
            }
        
        # Grow and spread if conditions are favorable
        if env_factors.moisture > 0.5 and self.energy > 0.6:
            # Calculate growth
            growth = self.properties["growth_rate"] * moisture_factor * delta_time
            self.size += growth
            
            # Extend hypha network
            if random.random() < self.properties["spread_rate"] * delta_time:
                # Add a new hypha connection point
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0.05, 0.1) * self.size
                
                # New hyphal position
                new_x = self.position[0] + math.cos(angle) * distance
                new_y = self.position[1] + math.sin(angle) * distance
                
                # Ensure within bounds
                new_x = max(0, min(environment.size, new_x))
                new_y = max(0, min(environment.size, new_y))
                
                # Add to network
                if len(self.position) == 2:
                    self.hypha_network.append((new_x, new_y))
                else:
                    # For 3D environments
                    new_z = self.position[2]  # Keep same layer
                    self.hypha_network.append((new_x, new_y, new_z))
            
            result["growth"] = growth
        
        return result
    
    def _calculate_moisture_factor(self, moisture: float) -> float:
        """Calculate efficiency factor based on environment moisture."""
        # Optimal moisture is stored in properties
        optimal = self.properties["moisture_preference"]
        
        # Distance from optimal (0 = optimal, 1 = furthest)
        distance = abs(moisture - optimal)
        
        # Convert to factor (1 at optimal, lower as you move away)
        return max(0.1, 1 - distance * 2)
    
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
        result = {"interaction": interaction_type, "success": False}
        
        if interaction_type == "decomposition":
            # Decomposers break down dead organisms
            if isinstance(other_organism, dict) and not other_organism.get("alive", True):
                # Calculate decomposition amount
                decomp_amount = (
                    self.properties["decomposition_rate"] * 
                    self.properties["enzyme_efficiency"] * 
                    random.uniform(0.8, 1.2)
                )
                
                result.update({
                    "success": True,
                    "amount": decomp_amount,
                    "nutrients": {
                        "CARBON": decomp_amount * 0.5,
                        "NITROGEN": decomp_amount * 0.2,
                        "PHOSPHORUS": decomp_amount * 0.15,
                        "MINERAL": decomp_amount * 0.1
                    }
                })
        
        elif interaction_type == "symbiosis":
            # Decomposers can form symbiotic relationships with plants
            if isinstance(other_organism, dict) and other_organism.get("type") == "Plant":
                # Mycorrhizal relationship - exchange nutrients for sugars
                plant_size = other_organism.get("size", 1.0)
                
                # Simulate nutrient exchange
                nutrients_provided = self.size * 0.1
                sugar_received = plant_size * 0.05
                
                result.update({
                    "success": True,
                    "type": "mycorrhizal",
                    "nutrients_provided": nutrients_provided,
                    "sugar_received": sugar_received
                })
        
        return result
    
    def reproduce(self, environment, partner=None) -> Optional['Decomposer']:
        """
        Create a new decomposer through spore reproduction.
        
        Args:
            environment: The environment context
            partner: Not needed for spore reproduction
            
        Returns:
            New decomposer or None if reproduction failed
        """
        if not self.can_reproduce(environment):
            return None
        
        # Spore dispersal is affected by environment
        env_factors = environment.get_environmental_factors_at(self.position)
        if env_factors.moisture < 0.3:  # Need some moisture for spores
            return None
        
        # Calculate spore dispersal
        angle = random.uniform(0, 2 * math.pi)
        
        # Distance affected by wind
        base_distance = random.uniform(0.1, 0.3) * self.size
        wind_effect = env_factors.wind * 0.2
        distance = base_distance * (1 + wind_effect)
        
        # New position for offspring
        new_position = list(self.position)
        new_position[0] += math.cos(angle) * distance
        new_position[1] += math.sin(angle) * distance
        
        # Ensure within bounds
        for i in range(len(new_position)):
            new_position[i] = max(0, min(environment.size, new_position[i]))
        
        # Create new decomposer with genetic variation
        spore_id = f"{self.id}_spore_{random.randint(1000, 9999)}"
        
        # Inherit properties with variation
        child_properties = self.properties.copy()
        for key in child_properties:
            if isinstance(child_properties[key], (int, float)):
                variation = random.uniform(-0.15, 0.15)  # Higher variation for fungi
                child_properties[key] *= (1 + variation)
        
        # Create new decomposer
        new_decomposer = Decomposer(
            organism_id=spore_id,
            position=tuple(new_position),
            energy=0.4,  # Start with less energy
            size=self.size * 0.2,  # Start much smaller
            properties=child_properties
        )
        
        # Reproduction costs energy
        self.energy -= 0.2  # Less energy cost for spore reproduction
        
        return new_decomposer
