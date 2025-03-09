    def update(self) -> None:
        """Update the environment (resource regeneration, depletion, etc.)."""
        # Update environmental conditions
        self._update_environment_conditions()
        
        # Update resources
        self._update_resources()
        
    def _update_environment_conditions(self) -> None:
        """Update environmental conditions with small fluctuations."""
        # Small random fluctuations
        self.moisture_level = max(0.1, min(1.0, self.moisture_level + random.uniform(-0.03, 0.03)))
        self.temperature = max(0.1, min(1.0, self.temperature + random.uniform(-0.02, 0.02)))
        self.ph_level = max(0.1, min(1.0, self.ph_level + random.uniform(-0.01, 0.01)))
        
        # Light follows a day/night cycle
        day_cycle = (time.time() % 100) / 100  # 0.0 to 1.0
        self.light_level = 0.5 + 0.4 * math.sin(day_cycle * 2 * math.pi)
        
    def _update_resources(self) -> None:
        """Update resources (depletion, regeneration)."""
        # Deplete resources
        for resource in self.resources:
            resource['level'] = max(0.0, resource['level'] - self.resource_depletion_rate)
            
        # Remove depleted resources
        self.resources = [r for r in self.resources if r['level'] > 0.1]
        
        # Regenerate resources with small probability
        if random.random() < 0.05:
            self._create_resource_clusters(1, random.randint(2, 5))
            
    def get_resource_at_position(self, position: Tuple[float, ...]) -> Optional[Dict]:
        """
        Get resource at a specific position.
        
        Args:
            position: Position to check
            
        Returns:
            Resource dictionary or None
        """
        for resource in self.resources:
            distance = self.calculate_distance(position, resource['position'])
            if distance < 0.05:  # Within resource radius
                return resource
        return None
        
    def get_resources_in_radius(self, center: Tuple[float, ...], radius: float) -> List[Dict]:
        """
        Get all resources within a radius.
        
        Args:
            center: Center position
            radius: Search radius
            
        Returns:
            List of resources with distance information
        """
        result = []
        for resource in self.resources:
            distance = self.calculate_distance(center, resource['position'])
            if distance <= radius:
                resource_copy = resource.copy()
                resource_copy['distance'] = distance
                result.append(resource_copy)
                
        # Sort by distance
        result.sort(key=lambda r: r['distance'])
        return result
        
    def get_obstacles_in_radius(self, center: Tuple[float, ...], radius: float) -> List[Dict]:
        """
        Get all obstacles within a radius.
        
        Args:
            center: Center position
            radius: Search radius
            
        Returns:
            List of obstacles with distance information
        """
        result = []
        for obstacle in self.obstacles:
            distance = self.calculate_distance(center, obstacle['position'])
            if distance <= radius + obstacle['size']:
                obstacle_copy = obstacle.copy()
                obstacle_copy['distance'] = distance
                result.append(obstacle_copy)
                
        # Sort by distance
        result.sort(key=lambda o: o['distance'])
        return result
        
    def remove_obstacle(self, obstacle_id: int) -> bool:
        """
        Remove an obstacle from the environment.
        
        Args:
            obstacle_id: ID of the obstacle to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, obstacle in enumerate(self.obstacles):
            if obstacle['id'] == obstacle_id:
                self.obstacles.pop(i)
                return True
        return False