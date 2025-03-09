"""
Final implementation of RichEnvironment class with all methods.
This file completes the implementation where the previous attempt was cut off.
"""

def _update_position_resources(self, position: Tuple[float, ...], resources: Dict[ResourceType, float]) -> None:
    """
    Update resources at a specific position.
    
    Args:
        position: Position to update
        resources: Updated resource dictionary
    """
    # For 3D environments with layers
    if len(position) >= 3 and self.layers:
        layer = self.get_layer_at_position(position)
        if layer:
            # Update layer resources
            for res_type, amount in resources.items():
                if amount < 0.01:
                    # Remove if too small
                    if position in layer.resources and res_type in layer.resources[position]:
                        del layer.resources[position][res_type]
                else:
                    # Update amount
                    if position not in layer.resources:
                        layer.resources[position] = {}
                    layer.resources[position][res_type] = amount
            
            # Clean up empty entries
            if position in layer.resources and not layer.resources[position]:
                del layer.resources[position]
            
            return
    
    # For 2D environments or if no layer found
    if position in self.resources:
        # Update resources
        for res_type, amount in resources.items():
            if amount < 0.01:
                # Remove if too small
                if isinstance(self.resources[position], dict) and res_type in self.resources[position]:
                    del self.resources[position][res_type]
            else:
                # Update amount
                if not isinstance(self.resources[position], dict):
                    # Convert legacy format
                    self.resources[position] = {ResourceType.CARBON: self.resources[position]}
                
                self.resources[position][res_type] = amount
        
        # Clean up empty entries
        if isinstance(self.resources[position], dict) and not self.resources[position]:
            del self.resources[position]
    else:
        # Create new entry if resources exist
        new_resources = {res_type: amount for res_type, amount in resources.items() if amount >= 0.01}
        if new_resources:
            self.resources[position] = new_resources

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
