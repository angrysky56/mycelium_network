"""
Rich Environment implementation for the enhanced mycelium network.

This module provides a more sophisticated environmental model extending the
base Environment class with:
- Multi-layered terrain with different properties
- Dynamic environmental factors (temperature, moisture, pH, light)
- Resource cycles and interaction between different resource types
- Advanced stimuli and environmental effects
- Seasonal cycles and time-based effects
"""

import math
import random
import time
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import numpy as np

from mycelium.environment import Environment
from mycelium.enhanced.resource import ResourceType, Environmental_Factors
from mycelium.enhanced.environment import TerrainLayer


class RichEnvironment(Environment):
    """
    Enhanced environment with richer simulation capabilities.
    
    Features:
    - 3D terrain with different layers (soil, air, etc.)
    - Dynamic environmental factors
    - Multiple resource types with interactions
    - Seasonal cycles and time-based effects
    - Organisms and ecosystem simulation
    """
    
    def __init__(
        self,
        dimensions: int = 2,
        size: float = 1.0,
        name: str = "Rich Environment"
    ):
        """
        Initialize a rich environment.
        
        Args:
            dimensions: Number of spatial dimensions (2 or 3)
            size: Size of the environment (side length of square/cube)
            name: Name of the environment
        """
        super().__init__(dimensions=dimensions, size=size)
        
        self.name = name
        
        # Environmental factors
        self.factors = Environmental_Factors()
        
        # Time tracking
        self.time = 0.0
        self.day_length = 24.0
        self.year_length = 365.0
        self.seasonal_intensity = 0.5
        
        # Terrain layers (for 3D environments)
        self.layers = []
        if dimensions >= 3:
            self._create_default_layers()
        
        # Organisms in the environment
        self.organisms = {}  # {id: organism_data}
        
        # Convert legacy resources format if needed
        self._convert_legacy_resources()

    def _create_default_layers(self):
        """Create default terrain layers for 3D environments."""
        # Deep soil layer
        deep_soil = TerrainLayer(
            name="Deep Soil",
            height_range=(0.0, 0.3),
            conductivity=0.5,
            density=1.5,
            resource_affinity={
                ResourceType.MINERAL: 1.2,
                ResourceType.WATER: 0.7,
                ResourceType.CARBON: 0.4
            },
            description="Dense, mineral-rich deep soil with less organic matter"
        )
        deep_soil.temperature_modifier = -0.1
        deep_soil.moisture_retention = 1.1
        deep_soil.ph_value = 6.8
        
        # Topsoil layer
        topsoil = TerrainLayer(
            name="Topsoil",
            height_range=(0.3, 0.6),
            conductivity=0.8,
            density=1.0,
            resource_affinity={
                ResourceType.CARBON: 1.3,
                ResourceType.NITROGEN: 1.2,
                ResourceType.WATER: 1.0,
                ResourceType.MINERAL: 0.8
            },
            description="Fertile topsoil with organic matter and nutrients"
        )
        topsoil.temperature_modifier = 0.0
        topsoil.moisture_retention = 1.0
        topsoil.ph_value = 6.5
        
        # Surface layer
        surface = TerrainLayer(
            name="Surface",
            height_range=(0.6, 0.7),
            conductivity=1.0,
            density=0.8,
            resource_affinity={
                ResourceType.LIGHT: 1.5,
                ResourceType.CARBON: 1.0,
                ResourceType.SUGAR: 1.1,
                ResourceType.WATER: 0.9
            },
            description="Surface layer with leaf litter and exposed elements"
        )
        surface.temperature_modifier = 0.1
        surface.moisture_retention = 0.8
        surface.ph_value = 6.3
        
        # Air layer
        air = TerrainLayer(
            name="Air",
            height_range=(0.7, 1.0),
            conductivity=0.3,
            density=0.1,
            resource_affinity={
                ResourceType.LIGHT: 1.8,
                ResourceType.WATER: 0.3,
                ResourceType.CARBON: 0.5
            },
            description="Atmospheric layer above the ground"
        )
        air.temperature_modifier = 0.2
        air.moisture_retention = 0.6
        air.ph_value = 7.0
        
        # Add layers in order from bottom to top
        self.layers = [deep_soil, topsoil, surface, air]

    def _convert_legacy_resources(self):
        """Convert legacy resources format to new typed format."""
        new_resources = {}
        
        for pos, res in self.resources.items():
            if not isinstance(res, dict):
                # Convert scalar to dictionary
                new_resources[pos] = {ResourceType.CARBON: res}
            else:
                # Already in new format
                new_resources[pos] = res
        
        self.resources = new_resources

    def get_layer_at_position(self, position: Tuple[float, ...]) -> Optional[TerrainLayer]:
        """
        Get the terrain layer at a specific position.
        
        Args:
            position: Position to check
            
        Returns:
            TerrainLayer or None if no layer contains this position
        """
        if not self.layers or len(position) < 3:
            return None
        
        for layer in self.layers:
            if layer.contains_point(position):
                return layer
        
        return None

    def add_nutrient_cluster(
        self, 
        center: Tuple[float, ...],
        radius: float,
        resource_type: ResourceType,
        total_amount: float,
        distribution: str = "gaussian"
    ):
        """
        Add a cluster of resources around a center point.
        
        Args:
            center: Center of the cluster
            radius: Radius of the cluster
            resource_type: Type of resource to add
            total_amount: Total amount of resource to distribute
            distribution: Distribution pattern ('gaussian', 'uniform')
        """
        # Determine number of points based on total amount
        num_points = max(5, int(total_amount * 20))
        
        # Distribute points
        for _ in range(num_points):
            # Generate offset based on distribution
            if distribution == "gaussian":
                # Gaussian distribution centered at origin
                offset = tuple(random.gauss(0, radius/2) for _ in range(len(center)))
            else:
                # Uniform distribution within radius
                offset = tuple(random.uniform(-radius, radius) for _ in range(len(center)))
            
            # Calculate position with offset
            pos = tuple(max(0, min(self.size, c + o)) for c, o in zip(center, offset))
            
            # Calculate amount based on distance from center
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(center, pos)))
            amount_factor = max(0, 1 - (dist / radius)) if radius > 0 else 1
            
            if distribution == "gaussian":
                # More concentrated at center for gaussian
                amount_factor = amount_factor ** 2
            
            # Calculate amount for this point
            point_amount = (total_amount / num_points) * amount_factor * 2
            
            # Add resource at position
            self.add_resource(pos, point_amount, resource_type)

    def add_resource(self, position: Tuple[float, ...], amount: float, resource_type: ResourceType = ResourceType.CARBON) -> None:
        """
        Add a resource at a specific position.
        
        Args:
            position: Position to add the resource
            amount: Amount of resource to add
            resource_type: Type of resource to add
        """
        # For 3D environments with layers
        if len(position) >= 3 and self.layers:
            layer = self.get_layer_at_position(position)
            if layer:
                # Apply resource affinity
                affinity = layer.resource_affinity.get(resource_type, 1.0)
                effective_amount = amount * affinity
                
                # Add resource to layer
                layer.add_resource(position, resource_type, effective_amount)
                return
        
        # For 2D environments or if no layer found
        if position not in self.resources:
            self.resources[position] = {}
        
        if not isinstance(self.resources[position], dict):
            # Convert legacy format
            self.resources[position] = {ResourceType.CARBON: self.resources[position]}
        
        # Add resource amount
        current = self.resources[position].get(resource_type, 0.0)
        self.resources[position][resource_type] = current + amount

    def get_resources_at(self, position: Tuple[float, ...]) -> Dict[ResourceType, float]:
        """
        Get all resources at a specific position.
        
        Args:
            position: Position to check
            
        Returns:
            Dictionary of resource types and amounts
        """
        # For 3D environments with layers
        if len(position) >= 3 and self.layers:
            layer = self.get_layer_at_position(position)
            if layer:
                return layer.get_resources_at(position)
        
        # For 2D environments or if no layer found
        if position in self.resources:
            if isinstance(self.resources[position], dict):
                return self.resources[position].copy()
            else:
                # Legacy format
                return {ResourceType.CARBON: self.resources[position]}
        
        return {}

    def get_resources_in_range(self, position: Tuple[float, ...], radius: float) -> Dict[Tuple[float, ...], Dict[ResourceType, float]]:
        """
        Get all resources within a certain radius of a position.
        
        Args:
            position: Position to check
            radius: Radius to search within
            
        Returns:
            Dictionary mapping positions to resource dictionaries
        """
        result = {}
        
        # For 3D environments with layers
        if len(position) >= 3 and self.layers:
            # Get layer at position
            layer = self.get_layer_at_position(position)
            if layer:
                # Get resources from layer
                layer_resources = layer.get_resources_in_range(position, radius)
                result.update(layer_resources)
                
                # Also check neighboring layers
                layer_idx = self.layers.index(layer)
                
                # Check layer above if not at top
                if layer_idx < len(self.layers) - 1:
                    above_layer = self.layers[layer_idx + 1]
                    # Only check near the boundary
                    if position[2] + radius >= above_layer.height_range[0]:
                        above_resources = above_layer.get_resources_in_range(position, radius)
                        result.update(above_resources)
                
                # Check layer below if not at bottom
                if layer_idx > 0:
                    below_layer = self.layers[layer_idx - 1]
                    # Only check near the boundary
                    if position[2] - radius <= below_layer.height_range[1]:
                        below_resources = below_layer.get_resources_in_range(position, radius)
                        result.update(below_resources)
                
                return result
        
        # For 2D environments or if no layer found
        for pos, resources in self.resources.items():
            # Calculate Euclidean distance
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(position, pos[:len(position)])))
            
            if distance <= radius:
                if isinstance(resources, dict):
                    result[pos] = resources.copy()
                else:
                    # Legacy format
                    result[pos] = {ResourceType.CARBON: resources}
        
        return result

    def get_total_resources(self, resource_type: Optional[ResourceType] = None) -> float:
        """
        Get the total amount of a specific resource type or all resources.
        
        Args:
            resource_type: Type of resource to sum, or None for all types
            
        Returns:
            Total amount of resources
        """
        total = 0.0
        
        # For layered 3D environments
        if self.layers:
            for layer in self.layers:
                for pos_resources in layer.resources.values():
                    if resource_type:
                        total += pos_resources.get(resource_type, 0.0)
                    else:
                        total += sum(pos_resources.values())
            
            return total
        
        # For 2D environments or no layers
        for resources in self.resources.values():
            if isinstance(resources, dict):
                if resource_type:
                    total += resources.get(resource_type, 0.0)
                else:
                    total += sum(resources.values())
            else:
                # Legacy format
                if resource_type is None or resource_type == ResourceType.CARBON:
                    total += resources
        
        return total

    def add_organism(self, organism_id: str, position: Tuple[float, ...], organism_type: str, properties: Dict[str, Any] = None) -> None:
        """
        Add an organism to the environment.
        
        Args:
            organism_id: Unique identifier for the organism
            position: Position of the organism
            organism_type: Type of organism (e.g., 'plant', 'herbivore')
            properties: Additional properties of the organism
        """
        properties = properties or {}
        self.organisms[organism_id] = {
            "id": organism_id,
            "position": position,
            "type": organism_type,
            "alive": True,
            "age": 0.0,
            "energy": properties.get("energy", 1.0),
            "properties": properties
        }

    def update_organism(self, organism_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an organism's properties.
        
        Args:
            organism_id: ID of the organism to update
            updates: Dictionary of properties to update
            
        Returns:
            True if organism was found and updated
        """
        if organism_id not in self.organisms:
            return False
        
        organism = self.organisms[organism_id]
        
        # Apply updates
        for key, value in updates.items():
            if key == "properties":
                # Update nested properties
                organism["properties"].update(value)
            else:
                # Update top-level properties
                organism[key] = value
        
        return True

    def remove_organism(self, organism_id: str) -> bool:
        """
        Remove an organism from the environment.
        
        Args:
            organism_id: ID of the organism to remove
            
        Returns:
            True if organism was found and removed
        """
        if organism_id in self.organisms:
            del self.organisms[organism_id]
            return True
        return False

    def get_organisms_in_range(self, position: Tuple[float, ...], radius: float, organism_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get all organisms within a certain radius.
        
        Args:
            position: Center position to search
            radius: Radius to search within
            organism_type: Optional filter for organism type
            
        Returns:
            Dictionary of organism IDs to organism data
        """
        result = {}
        
        for org_id, organism in self.organisms.items():
            if not organism["alive"]:
                continue
            
            if organism_type and organism["type"] != organism_type:
                continue
            
            # Calculate distance
            org_pos = organism["position"]
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(position, org_pos[:len(position)])))
            
            if distance <= radius:
                result[org_id] = organism.copy()
        
        return result

    def update(self, delta_time: float = 0.1) -> None:
        """
        Update the environment state over time.
        
        Args:
            delta_time: Time step size
        """
        # Update time
        self.time += delta_time
        
        # Apply seasonal effects if enabled
        if hasattr(self, "year_length") and self.year_length > 0:
            self.apply_seasonal_effects(delta_time)
        
        # Update terrain layers
        if self.layers:
            for layer in self.layers:
                layer.update(delta_time)
        
        # Update organisms
        for org_id, organism in list(self.organisms.items()):
            if not organism["alive"]:
                continue
            
            # Age organism
            organism["age"] += delta_time
            
            # Deplete energy
            organism["energy"] -= 0.01 * delta_time
            
            # Mark as dead if out of energy
            if organism["energy"] <= 0:
                organism["alive"] = False
        
        # Basic environmental factors cycling
        self._update_environmental_factors(delta_time)

    def _update_environmental_factors(self, delta_time: float) -> None:
        """
        Update basic environmental factors over time.
        
        Args:
            delta_time: Time step size
        """
        # Day-night cycle
        day_phase = (self.time % self.day_length) / self.day_length
        
        # Light follows day-night cycle
        if day_phase < 0.25 or day_phase > 0.75:
            # Night time
            self.factors.light_level = max(0.1, self.factors.light_level - 0.1 * delta_time)
        else:
            # Day time
            self.factors.light_level = min(1.0, self.factors.light_level + 0.1 * delta_time)
    
    def get_environmental_factors_at(self, position: Tuple[float, ...]) -> Environmental_Factors:
        """
        Get environmental factors at a specific position.
        
        Args:
            position: Position to check
            
        Returns:
            Environmental factors at that position
        """
        # Start with global factors
        factors = Environmental_Factors(
            temperature=self.factors.temperature,
            moisture=self.factors.moisture,
            ph=self.factors.ph,
            light_level=self.factors.light_level,
            toxicity=self.factors.toxicity,
            oxygen=self.factors.oxygen,
            wind=self.factors.wind,
            gravity=self.factors.gravity,
            season=self.factors.season
        )
        
        # Modify based on layer properties for 3D environments
        if len(position) >= 3 and self.layers:
            layer = self.get_layer_at_position(position)
            if layer:
                factors.temperature += layer.temperature_modifier
                factors.moisture *= layer.moisture_retention
                factors.ph = (factors.ph + layer.ph_value) / 2
                
                # Light decreases with depth
                factors.light_level *= max(0.1, min(1.0, position[2]))
        
        # Clamp values to valid ranges
        factors.temperature = max(0.0, min(1.0, factors.temperature))
        factors.moisture = max(0.0, min(1.0, factors.moisture))
        factors.ph = max(0.0, min(14.0, factors.ph))
        factors.light_level = max(0.0, min(1.0, factors.light_level))
        factors.toxicity = max(0.0, min(1.0, factors.toxicity))
        factors.oxygen = max(0.0, min(1.0, factors.oxygen))
        factors.wind = max(0.0, min(1.0, factors.wind))
        
        return factors

    def create_seasonal_cycle(self, year_length: float = 365.0, intensity: float = 0.5):
        """
        Configure the environment to simulate seasonal cycles.
        
        Args:
            year_length: Length of a year in simulation time units
            intensity: How strongly seasons affect the environment (0-1)
        """
        self.year_length = year_length
        self.seasonal_intensity = intensity

    def apply_seasonal_effects(self, delta_time: float):
        """
        Apply effects based on the current season.
        
        Args:
            delta_time: Time step size
        """
        # Determine season and phase within season
        year_phase = (self.time % self.year_length) / self.year_length
        season_idx = int(year_phase * 4) % 4
        season_phase = (year_phase * 4) % 1.0  # Phase within the current season
        
        # Update season factor
        self.factors.season = season_idx
        
        # Apply appropriate seasonal effect
        intensity = self.seasonal_intensity
        
        if season_idx == 0:  # Spring
            self._apply_spring_effects(season_phase, delta_time, intensity)
        elif season_idx == 1:  # Summer
            self._apply_summer_effects(season_phase, delta_time, intensity)
        elif season_idx == 2:  # Fall
            self._apply_fall_effects(season_phase, delta_time, intensity)
        elif season_idx == 3:  # Winter
            self._apply_winter_effects(season_phase, delta_time, intensity)

    def _apply_spring_effects(self, phase: float, delta_time: float, intensity: float):
        """
        Apply spring season effects (more water, growth).
        
        Args:
            phase: Current phase within the season (0-1)
            delta_time: Time step size
            intensity: Effect strength
        """
        # Randomly add some water resources in top layers
        if random.random() < 0.3 * intensity * delta_time:  # Spring showers
            # Generate rain
            num_drops = int(5 * intensity)
            for _ in range(num_drops):
                x = random.random() * self.size
                y = random.random() * self.size
                if self.dimensions >= 3:
                    # In 3D environments, rain falls from the top
                    position = (x, y, 0.7 + random.random() * 0.3)
                else:
                    position = (x, y)
                
                self.add_resource(position, 0.2 * intensity, ResourceType.WATER)
        
        # Add some nitrogen (growth nutrients)
        if random.random() < 0.1 * intensity * delta_time:
            x = random.random() * self.size
            y = random.random() * self.size
            if self.dimensions >= 3:
                # Add to topsoil layer
                position = (x, y, 0.6 + random.random() * 0.1)
            else:
                position = (x, y)
            
            self.add_resource(position, 0.1 * intensity, ResourceType.NITROGEN)

    def _apply_summer_effects(self, phase: float, delta_time: float, intensity: float):
        """
        Apply summer season effects (more light, evaporation).
        
        Args:
            phase: Current phase within the season (0-1)
            delta_time: Time step size
            intensity: Effect strength
        """
        # Increase light level
        light_boost = 0.2 * intensity * (1 - phase)  # Decreases toward end of summer
        self.factors.light_level = min(1.0, self.factors.light_level + light_boost)
        
        # Evaporate some water
        evaporation_rate = 0.05 * intensity * delta_time
        for pos in list(self.resources.keys()):
            if isinstance(self.resources[pos], dict) and ResourceType.WATER in self.resources[pos]:
                self.resources[pos][ResourceType.WATER] *= (1 - evaporation_rate)
                
                # Remove if too small
                if self.resources[pos][ResourceType.WATER] < 0.01:
                    del self.resources[pos][ResourceType.WATER]
                
                # Clean up empty entries
                if not self.resources[pos]:
                    del self.resources[pos]

    def _apply_fall_effects(self, phase: float, delta_time: float, intensity: float):
        """
        Apply fall season effects (falling leaves, decreasing light).
        
        Args:
            phase: Current phase within the season (0-1)
            delta_time: Time step size
            intensity: Effect strength
        """
        # Decrease light level
        light_reduction = 0.1 * intensity * phase  # Increases toward end of fall
        self.factors.light_level = max(0.3, self.factors.light_level - light_reduction)
        
        # Add some carbon (falling leaves)
        if random.random() < 0.2 * intensity * delta_time:
            x = random.random() * self.size
            y = random.random() * self.size
            if self.dimensions >= 3:
                # Add to ground level
                position = (x, y, 0.6)
            else:
                position = (x, y)
            
            self.add_resource(position, 0.1 * intensity, ResourceType.CARBON)

    def _apply_winter_effects(self, phase: float, delta_time: float, intensity: float):
        """
        Apply winter season effects (cold, dormancy).
        
        Args:
            phase: Current phase within the season (0-1)
            delta_time: Time step size
            intensity: Effect strength
        """
        # Decrease temperature
        temp_reduction = 0.2 * intensity * (1 - phase)  # Increases in early winter
        self.factors.temperature = max(0.1, self.factors.temperature - temp_reduction)
        
        # Slow down all resource interactions
        self.factors.moisture = max(0.2, self.factors.moisture - 0.05 * intensity * delta_time)
        
        # Occasionally add snow (special water resource)
        if random.random() < 0.1 * intensity * delta_time:
            num_snowflakes = int(3 * intensity)
            for _ in range(num_snowflakes):
                x = random.random() * self.size
                y = random.random() * self.size
                if self.dimensions >= 3:
                    # Snow accumulates on top layers
                    position = (x, y, 0.7)
                else:
                    position = (x, y)
                
                # Add frozen water (special property)
                self.add_resource(position, 0.05 * intensity, ResourceType.WATER)

    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current environment state.
        
        Returns:
            Dictionary containing environment state
        """
        snapshot = {
            "time": self.time,
            "day_phase": (self.time % self.day_length) / self.day_length,
            "year_phase": (self.time % self.year_length) / self.year_length,
            "global_factors": {
                "temperature": self.factors.temperature,
                "moisture": self.factors.moisture,
                "light_level": self.factors.light_level,
                "season": self.factors.season,
                "wind": self.factors.wind,
                "ph": self.factors.ph,
                "oxygen": self.factors.oxygen,
            },
            "resources": {
                "total": self.get_total_resources(),
                "carbon": self.get_total_resources(ResourceType.CARBON),
                "water": self.get_total_resources(ResourceType.WATER),
                "nitrogen": self.get_total_resources(ResourceType.NITROGEN),
                "sugar": self.get_total_resources(ResourceType.SUGAR),
            },
            "organisms": {
                "count": len([o for o in self.organisms.values() if o["alive"]]),
                "by_type": {}
            }
        }
        
        # Count organisms by type
        for organism in self.organisms.values():
            if organism["alive"]:
                org_type = organism["type"]
                if org_type not in snapshot["organisms"]["by_type"]:
                    snapshot["organisms"]["by_type"][org_type] = 0
                snapshot["organisms"]["by_type"][org_type] += 1
        
        # Layer information for 3D environments
        if self.layers:
            snapshot["layers"] = []
            for layer in self.layers:
                # Count resources in this layer
                layer_resources = {}
                for resource_type in ResourceType:
                    total = 0.0
                    for pos, resources in layer.resources.items():
                        total += resources.get(resource_type, 0.0)
                    if total > 0:
                        layer_resources[resource_type.name] = total
                
                # Add layer data
                layer_data = {
                    "name": layer.name,
                    "height_range": layer.height_range,
                    "resources": layer_resources
                }
                snapshot["layers"].append(layer_data)
        
        return snapshot
