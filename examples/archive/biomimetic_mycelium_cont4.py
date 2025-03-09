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