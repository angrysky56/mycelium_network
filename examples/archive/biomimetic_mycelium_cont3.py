    def calculate_distance(self, other_node) -> float:
        """
        Calculate Euclidean distance to another node.
        
        Args:
            other_node: Other node object
            
        Returns:
            Distance value
        """
        if len(self.position) != len(other_node.position):
            raise ValueError("Position dimensions don't match")
            
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, other_node.position)))
        
    def process_signal(self, input_value: float) -> float:
        """
        Enhanced signal processing with circadian modulation.
        
        Args:
            input_value: Input signal value
            
        Returns:
            Output signal value
        """
        # Update circadian phase
        self.cycle_phase = (self.cycle_phase + 2 * math.pi / self.cycle_duration) % (2 * math.pi)
        
        # Circadian modulation of sensitivity (±20%)
        circadian_factor = 1.0 + 0.2 * math.sin(self.cycle_phase)
        
        # Stress reduces processing effectiveness
        stress_factor = 1.0 - (self.stress_level * 0.3)
        
        # Combined factors
        effective_sensitivity = self.sensitivity * circadian_factor * stress_factor
        
        # Apply sigmoid activation with effective sensitivity
        try:
            self.activation = 1 / (1 + math.exp(-input_value * effective_sensitivity))
        except OverflowError:
            self.activation = 0.0 if input_value < 0 else 1.0
        
        # Resource consumption with circadian and stress influences
        metabolic_demand = self.metabolic_rate * (1.0 + 0.2 * self.activation)
        
        # Circadian rhythm affects metabolism (reduced at "night")
        if math.sin(self.cycle_phase) < 0:
            metabolic_demand *= 0.7
            
        self.energy = max(0.1, self.energy - metabolic_demand)
        
        # Age the node
        self.age += 1
        
        return self.activation
        
    def allocate_resources(self, amount: float, resource_type: str = 'general') -> None:
        """
        Enhanced resource allocation with specific nutrient types.
        
        Args:
            amount: Amount of resources to receive
            resource_type: Type of resource ('carbon', 'nitrogen', 'phosphorus', 'general')
        """
        # Add to specific nutrient storage
        self.nutrient_storage[resource_type] += amount
        
        # Different nutrients have different effects
        if resource_type == 'carbon':
            # Carbon increases energy
            self.energy = min(1.5, self.energy + amount * 0.2)
            
        elif resource_type == 'nitrogen':
            # Nitrogen enhances enzyme production
            for enzyme in self.enzymes.values():
                enzyme.strength = min(1.0, enzyme.strength + amount * 0.1)
                
        elif resource_type == 'phosphorus':
            # Phosphorus enhances connection formation
            self.growth_potential += amount * 0.05
            
        else:  # General nutrients
            # General resources affect overall resource level
            self.resource_level = min(2.0, self.resource_level + amount)
            self.energy = min(1.0, self.energy + amount * 0.1)
        
        # Substantial resources extend longevity
        if amount > 0.3:
            self.longevity += 2