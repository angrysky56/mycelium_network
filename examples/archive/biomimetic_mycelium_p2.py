        
        # Sense obstacles
        obstacles = environment.get_obstacles_in_radius(self.position, radius)
        sensed_info['obstacles'] = obstacles
        
        # Update spatial memory
        for resource in resources:
            pos_hash = hash(tuple(resource['position']))
            self.spatial_memory[pos_hash] = (resource['level'], time.time())
            
        # Record stress from obstacles
        for obstacle in obstacles:
            if obstacle['distance'] < 0.1:  # Close obstacle causes stress
                self.add_stress('obstacle', 0.2 * (1 - obstacle['distance'] / 0.1))
                
        return sensed_info
        
    def add_stress(self, stress_type: str, level: float) -> None:
        """
        Add a stress response.
        
        Args:
            stress_type: Type of stress ('drought', 'obstacle', 'toxin', etc.)
            level: Intensity of stress (0.0 to 1.0)
        """
        self.stress_level = min(1.0, self.stress_level + level)
        self.stress_memory.append((stress_type, level, time.time()))
        
        # Stress triggers adaptation
        if stress_type == 'drought':
            # Drought stress increases energy efficiency
            self.metabolic_rate = max(0.01, self.metabolic_rate * 0.9)
            
        elif stress_type == 'obstacle':
            # Obstacle stress activates enzymes
            for enzyme in self.enzymes.values():
                # Higher chance to activate relevant enzymes
                if random.random() < level * 0.5:
                    enzyme.activate(self.energy)
                    
        elif stress_type == 'toxin':
            # Toxin stress increases adaptability
            self.adaptability = min(2.0, self.adaptability * 1.1)
            # Emit warning signal
            warning = self.emit_signal('danger', level, {'type': 'toxin'})
            return warning
            
        # All stress gradually reduces as the node adapts
        self._adapt_to_stress()
        
    def _adapt_to_stress(self) -> None:
        """Gradually adapt to stress conditions."""
        # Decay old stress
        recent_stress = [s for s in self.stress_memory if time.time() - s[2] < 50]
        if recent_stress:
            # Calculate average recent stress
            avg_stress = sum(s[1] for s in recent_stress) / len(recent_stress)
            # Adaptive response proportional to stress level and adaptability
            adaptation_factor = 0.01 * self.adaptability * avg_stress
            
            # Adaptations
            self.sensitivity = min(2.0, self.sensitivity * (1 + adaptation_factor))
            self.longevity += int(10 * adaptation_factor)
        
        # Reduce current stress level
        self.stress_level = max(0.0, self.stress_level - 0.01)
        
    def attempt_anastomosis(self, other_node, success_probability: float = 0.3) -> bool:
        """
        Attempt to fuse with another node (hyphal anastomosis).
        
        Args:
            other_node: Target node for fusion
            success_probability: Base probability of successful fusion
            
        Returns:
            True if fusion successful, False otherwise
        """
        # Already fused
        if other_node.id in self.anastomosis_targets:
            return True
            
        # Factors affecting anastomosis success
        genetic_similarity = 0.8  # Same network = high similarity
        distance_factor = 1.0 - min(1.0, self.calculate_distance(other_node) / 0.2)
        energy_factor = min(1.0, (self.energy + other_node.energy) / 1.5)
        
        # Calculate success chance
        fusion_chance = success_probability * genetic_similarity * distance_factor * energy_factor
        
        # Attempt fusion
        if random.random() < fusion_chance:
            self.anastomosis_targets.add(other_node.id)
            other_node.anastomosis_targets.add(self.id)
            
            # Strengthen connection dramatically
            if other_node.id in self.connections:
                self.connections[other_node.id] *= 2.0
            else:
                self.connect_to(other_node, strength=0.5)
                
            # Energy cost for both nodes
            self.energy = max(0.1, self.energy - 0.15)
            other_node.energy = max(0.1, other_node.energy - 0.15)
            
            # Signal successful fusion
            fusion_signal = self.emit_signal(
                'anastomosis', 
                0.8, 
                {'target_id': other_node.id}
            )
            
            return True
        else:
            return False
            
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