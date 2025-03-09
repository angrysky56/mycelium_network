class BiomimeticClassifier(MyceliumClassifier):
    """
    Classifier implementation using the biomimetic network architecture.
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 2,
        hidden_nodes: int = 20,
        environment: Optional[EnhancedEnvironment] = None
    ):
        """
        Initialize the biomimetic classifier.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        num_classes : int
            Number of output classes (default: 2 for binary classification)
        hidden_nodes : int
            Number of initial hidden nodes (default: 20)
        environment : EnhancedEnvironment, optional
            Custom environment for the network
        """
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Create environment if not provided
        if environment is None:
            environment = EnhancedEnvironment()
        
        # Initialize the network
        output_size = num_classes if num_classes > 2 else 1
        self.network = BiomimeticNetwork(
            input_size=input_size,
            output_size=output_size,
            initial_nodes=hidden_nodes,
            environment=environment
        )
        
        # Training metrics
        self.train_history = {
            'accuracy': [],
            'loss': [],
            'adaptations': []
        }
    
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
    
    def get_network_insights(self) -> Dict:
        """
        Get biological insights about the network's adaptations.
        
        Returns:
        --------
        Dict
            Dictionary with network insights
        """
        stats = self.network.get_network_statistics()
        
        # Calculate meaningful biological metrics
        insights = {
            'adaptive_capacity': stats.get('adaptation_progress', 1.0),
            'network_complexity': stats['connection_count'] / max(1, stats['node_count']),
            'resource_efficiency': stats['total_resources'] / max(1, stats['node_count']),
            'stress_resilience': 1.0 - stats.get('average_stress', 0.0),
            'growth_pattern': len(stats.get('enzymatic_activities', {})),
            'connectivity': stats.get('anastomosis_count', 0) / max(1, stats['node_count'])
        }
        
        # Overall resilience score
        insights['resilience_score'] = (
            0.3 * insights['adaptive_capacity'] +
            0.2 * insights['network_complexity'] +
            0.2 * insights['resource_efficiency'] +
            0.2 * insights['stress_resilience'] +
            0.1 * insights['connectivity']
        )
        
        return insights


def generate_complex_data(n_samples: int = 200, noise: float = 0.1) -> Tuple[List[List[float]], List[int]]:
    """
    Generate a more complex classification dataset with non-linear decision boundary.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Amount of noise to add
        
    Returns:
    --------
    Tuple[List[List[float]], List[int]]
        Feature vectors and binary labels
    """
    features = []
    labels = []
    
    for _ in range(n_samples):
        # Generate points in a 2D space
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        # Add noise
        x += random.uniform(-noise, noise)
        y += random.uniform(-noise, noise)
        
        # Keep within bounds
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        
        features.append([x, y])
        
        # Complex decision boundary: circular pattern
        distance_from_center = math.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        
        # Alternating concentric circles
        if (0.15 < distance_from_center < 0.25) or (0.35 < distance_from_center < 0.45):
            labels.append(1)
        else:
            labels.append(0)
    
    # Shuffle data
    combined = list(zip(features, labels))
    random.shuffle(combined)
    features, labels = zip(*combined)
    
    return list(features), list(labels)


def main():
    """Demonstrate the biomimetic mycelium network on a classification task."""
    print("Biomimetic Mycelium Network Example")
    print("===================================")
    
    # Create environment
    print("\nInitializing enhanced environment...")
    env = EnhancedEnvironment(dimensions=2)
    
    # Generate complex dataset
    print("\nGenerating complex dataset...")
    features, labels = generate_complex_data(200, noise=0.05)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Create classifier
    print("\nInitializing biomimetic classifier...")
    classifier = BiomimeticClassifier(
        input_size=2,
        num_classes=2,
        hidden_nodes=30,
        environment=env
    )
    
    # Set fixed random seed for reproducibility
    random.seed(42)
    
    # Train the classifier
    print("\nTraining classifier with environmental adaptations...")
    history = classifier.fit(
        X_train, 
        y_train, 
        epochs=15,
        learning_rate=0.15,
        validation_split=0.2,
        verbose=True,
        environmental_complexity=0.6
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = classifier.evaluate(X_test, y_test)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Network insights
    print("\nBiomimetic Network Insights:")
    insights = classifier.get_network_insights()
    for key, value in insights.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    
    # Network statistics
    print("\nNetwork Statistics:")
    stats = classifier.network.get_network_statistics()
    print(f"Nodes: {stats['node_count']} (input: {stats['input_nodes']}, output: {stats['output_nodes']}, regular: {stats['regular_nodes']})")
    print(f"Connections: {stats['connection_count']} (avg. {stats['avg_connections_per_node']:.2f} per node)")
    print(f"Anastomosis connections: {stats.get('anastomosis_count', 0)}")
    print(f"Average stress level: {stats.get('average_stress', 0):.4f}")
    
    # Environment status
    print("\nEnvironment Status:")
    env_stats = {
        'Moisture level': env.moisture_level,
        'Temperature': env.temperature,
        'pH level': env.ph_level,
        'Light level': env.light_level,
        'Resources': len(env.resources),
        'Obstacles': len(env.obstacles)
    }
    for key, value in env_stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Visualize network (save to file)
    print("\nSaving network visualization data...")
    classifier.network.visualize_network("biomimetic_network_visualization.json")
    print("Visualization data saved to 'biomimetic_network_visualization.json'")
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()
