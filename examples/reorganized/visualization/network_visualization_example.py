#!/usr/bin/env python3
"""
Network Visualization Example

This script demonstrates how to create visualizations of mycelium networks,
including network structure, activation patterns, and growth dynamics.
"""

import os
import sys
import random
import json
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from mycelium import MyceliumClassifier, Environment
from mycelium.network import AdvancedMyceliumNetwork

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "utils"))
    
import data_utils
import visualization_utils


def create_sample_network() -> AdvancedMyceliumNetwork:
    """Create a sample network for visualization."""
    # Create environment
    env = Environment(dimensions=2)
    env.create_grid_resources(grid_size=5, resource_value=1.2)
    env.create_random_obstacles(num_obstacles=3, max_radius=0.1)
    
    # Create network
    network = AdvancedMyceliumNetwork(
        environment=env,
        input_size=2,
        output_size=1,
        initial_nodes=25
    )
    
    # Run a few iterations to develop the network
    for _ in range(10):
        # Generate random input
        random_input = [random.random() for _ in range(2)]
        network.forward(random_input)
    
    return network


def create_network_growth_sequence() -> List[Dict]:
    """
    Create a sequence of network states showing growth over time.
    
    Returns:
    --------
    List[Dict]
        List of network visualization data at different timesteps
    """
    # Create environment
    env = Environment(dimensions=2)
    env.create_grid_resources(grid_size=5, resource_value=1.2)
    
    # Create network with fewer initial nodes
    network = AdvancedMyceliumNetwork(
        environment=env,
        input_size=2,
        output_size=1,
        initial_nodes=10
    )
    
    # Set higher growth rate
    network.growth_rate = 0.2
    
    # Track network states
    states = []
    
    # Initial state
    states.append(network.visualize_network())
    
    # Run for multiple iterations and record states
    for i in range(15):
        # Generate random input
        random_input = [random.random() for _ in range(2)]
        network.forward(random_input)
        
        # Record state every few iterations
        if (i + 1) % 3 == 0:
            states.append(network.visualize_network())
    
    return states


def main():
    """Run the network visualization example."""
    print("Mycelium Network Visualization Example")
    print("=====================================")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Part 1: Basic network visualization
    print("\n1. Creating basic network visualization...")
    network = create_sample_network()
    
    # Get network statistics
    stats = network.get_network_statistics()
    print(f"Network created with {stats['node_count']} nodes and {stats['connection_count']} connections")
    
    # Save network visualization data
    network_json_path = os.path.join(output_dir, "network_structure.json") 
    visualization_utils.save_network_visualization_data(
        network,
        network_json_path,
        include_node_details=True,
        include_connection_details=True
    )
    
    # Generate HTML visualization
    html_output_path = os.path.join(output_dir, "network_visualization.html")
    visualization_utils.generate_html_visualization(
        network_json_path,
        html_output_path,
        title="Mycelium Network Structure Visualization"
    )
    
    print(f"Network visualization saved to {html_output_path}")
    
    # Part 2: Network growth visualization
    print("\n2. Creating network growth visualization...")
    growth_states = create_network_growth_sequence()
    
    # Save growth animation
    growth_html_path = os.path.join(output_dir, "network_growth_animation.html")
    visualization_utils.visualize_network_growth(
        growth_states,
        growth_html_path,
        interval_ms=800
    )
    
    print(f"Network growth animation saved to {growth_html_path}")
    
    try:
        # Part 3: Performance visualization with different configurations
        print("\n3. Creating performance comparison visualization...")
        
        # Create and train multiple networks with different configurations
        configurations = [
            {"hidden_nodes": 15, "name": "Small Network (15 nodes)"},
            {"hidden_nodes": 30, "name": "Medium Network (30 nodes)"},
            {"hidden_nodes": 50, "name": "Large Network (50 nodes)"}
        ]
        
        # Generate synthetic data
        features, labels = data_utils.generate_synthetic_classification_data(
            n_samples=200,
            n_features=4,
            n_classes=2,
            noise=0.2,
            random_seed=42
        )
        
        # Split data
        X_train, y_train, X_test, y_test = data_utils.train_test_split(
            features, labels, test_size=0.2, stratify=True
        )
        
        # Train different configurations
        performance_data = []
        
        for config in configurations:
            print(f"Training {config['name']}...")
            
            # Create classifier
            classifier = MyceliumClassifier(
                input_size=4,
                num_classes=2,
                hidden_nodes=config['hidden_nodes'],
                environment=Environment(dimensions=4)
            )
            
            # Train
            history = classifier.fit(
                X_train,
                y_train,
                epochs=10,
                verbose=False
            )
            
            # Record performance
            performance_data.append({
                "name": config['name'],
                "values": history['accuracy']
            })
        
        # Plot comparison
        performance_plot_path = os.path.join(output_dir, "network_performance_comparison.png")
        visualization_utils.plot_network_performance_comparison(
            performance_data,
            metric_name="accuracy",
            title="Performance Comparison of Different Network Configurations",
            filename=performance_plot_path
        )
        
        print(f"Performance comparison plot saved to {performance_plot_path}")
        
    except ImportError:
        print("Matplotlib not installed. Skipping performance visualization.")
    
    print("\nVisualization example complete! Check the output directory for results.")


if __name__ == "__main__":
    main()
