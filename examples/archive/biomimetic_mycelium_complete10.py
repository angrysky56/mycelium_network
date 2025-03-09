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