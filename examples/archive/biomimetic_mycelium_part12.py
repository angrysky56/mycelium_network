    def fit(
        self,
        X: List[List[float]],
        y: List[int],
        epochs: int = 10,
        learning_rate: float = 0.1,
        validation_split: float = 0.2,
        verbose: bool = True,
        environmental_complexity: float = 0.5  # 0.0 to 1.0
    ) -> Dict[str, List[float]]:
        """
        Train the classifier with environmental adaptations.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors
        y : List[int]
            Target labels (class indices)
        epochs : int
            Number of training epochs (default: 10)
        learning_rate : float
            Learning rate for training (default: 0.1)
        validation_split : float
            Proportion of data to use for validation (default: 0.2)
        verbose : bool
            Whether to print progress (default: True)
        environmental_complexity : float
            Complexity of environmental changes during training (0.0 to 1.0)
            
        Returns:
        --------
        Dict[str, List[float]]
            Training history with metrics
        """
        # Enhanced training with environmental adaptation
        environment = self.network.environment
        
        # Adjust environment periodically during training
        for epoch in range(epochs):
            # Adjust environment every few epochs
            if epoch % 3 == 0 and environmental_complexity > 0:
                # Adjust moisture and temperature
                environment.moisture_level = max(0.1, min(1.0, 
                    environment.moisture_level + environmental_complexity * random.uniform(-0.2, 0.2)))
                environment.temperature = max(0.1, min(1.0, 
                    environment.temperature + environmental_complexity * random.uniform(-0.2, 0.2)))
                
                # Add or remove resources
                if random.random() < environmental_complexity * 0.3:
                    environment._create_resource_clusters(1, random.randint(2, 5))
                    
                # Add obstacles
                if random.random() < environmental_complexity * 0.2:
                    environment._create_obstacles(random.randint(1, 3))
                
                if verbose:
                    print(f"Environment adjusted: Moisture={environment.moisture_level:.2f}, "
                          f"Temperature={environment.temperature:.2f}, "
                          f"Resources={len(environment.resources)}, "
                          f"Obstacles={len(environment.obstacles)}")
        
        # Call parent fit method
        history = super().fit(X, y, epochs, learning_rate, validation_split, verbose)
        
        # Record adaptations
        self.train_history['adaptations'] = self.network.adaptation_history
        
        return history