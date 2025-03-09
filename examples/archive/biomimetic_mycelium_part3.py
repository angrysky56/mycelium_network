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
            
    def grow_hyphal_branch(self, environment: Environment, direction: Optional[Tuple[float, ...]] = None) -> Optional[int]:
        """
        Grow a new hyphal branch in a specific or random direction.
        
        Args:
            environment: Environment object
            direction: Optional growth direction vector
            
        Returns:
            ID of the new node if created, None otherwise
        """
        # Check if growth is possible
        if self.energy < 0.3 or self.resource_level < 0.5:
            return None
            
        # Determine growth direction
        if direction is None:
            # Random direction if none specified
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0.05, 0.1) * self.growth_potential
            
            # In 2D space
            direction = (math.cos(angle), math.sin(angle))
        
        # Calculate new position
        new_position = tuple(p + d * 0.1 for p, d in zip(self.position, direction))
        
        # Ensure new position is within bounds
        new_position = tuple(max(0, min(1, p)) for p in new_position)
        
        # Check if position is valid in environment
        if not environment.is_position_valid(new_position):
            # Attempt to degrade obstacles
            obstacles = environment.get_obstacles_in_radius(new_position, 0.05)
            
            if obstacles:
                obstacle = obstacles[0]
                enzyme_type = self._select_enzyme_for_obstacle(obstacle['type'])
                
                if enzyme_type and self.enzymes[enzyme_type].active:
                    # Attempt to degrade obstacle
                    effectiveness = self.enzymes[enzyme_type].apply_to_obstacle(obstacle['type'])
                    
                    if random.random() < effectiveness:
                        # Successfully cleared obstacle
                        environment.remove_obstacle(obstacle['id'])
                    else:
                        # Failed to clear obstacle
                        return None
                else:
                    # No appropriate enzyme active
                    return None
        
        # Create new node with a new ID
        new_id = max(environment.get_highest_node_id() + 1, 1000 * self.id + len(self.hyphal_branches))
        
        # Create the node
        new_node = BiomimeticNode(new_id, new_position, 'regular')
        
        # Inherit some properties from parent
        new_node.sensitivity = self.sensitivity * random.uniform(0.9, 1.1)
        new_node.adaptability = self.adaptability * random.uniform(0.9, 1.1)
        
        # Initialize with some resources from parent
        resource_transfer = min(0.3, self.resource_level * 0.3)
        self.resource_level -= resource_transfer
        new_node.resource_level = resource_transfer
        
        energy_transfer = min(0.2, self.energy * 0.3)
        self.energy -= energy_transfer
        new_node.energy = energy_transfer
        
        # Connect to the parent node
        self.connect_to(new_node, strength=0.7)  # Strong initial connection
        
        # Add to hyphal branches
        self.hyphal_branches.append(new_id)
        
        return new_id
        
    def _select_enzyme_for_obstacle(self, obstacle_type: str) -> Optional[str]:
        """
        Select the most appropriate enzyme for an obstacle.
        
        Args:
            obstacle_type: Type of obstacle
            
        Returns:
            Enzyme type or None if no suitable enzyme
        """
        enzyme_mapping = {
            'cellulose': 'cellulase',
            'lignin': 'lignin_peroxidase',
            'protein': 'protease'
        }
        
        return enzyme_mapping.get(obstacle_type)
        
    def update_circadian_rhythm(self, external_cue: float = None) -> None:
        """
        Update the node's circadian rhythm, optionally synchronized to external cues.
        
        Args:
            external_cue: Optional external synchronization signal (0.0 to 1.0)
        """
        # Natural phase progression
        self.cycle_phase = (self.cycle_phase + 2 * math.pi / self.cycle_duration) % (2 * math.pi)
        
        # External synchronization if provided
        if external_cue is not None:
            # Convert external cue to target phase
            target_phase = external_cue * 2 * math.pi
            # Gradually adjust phase (entrainment)
            phase_difference = target_phase - self.cycle_phase
            # Normalize to -π to π
            if phase_difference > math.pi:
                phase_difference -= 2 * math.pi
            elif phase_difference < -math.pi:
                phase_difference += 2 * math.pi
                
            # Gradual adjustment
            adjustment = phase_difference * 0.1
            self.cycle_phase = (self.cycle_phase + adjustment) % (2 * math.pi)