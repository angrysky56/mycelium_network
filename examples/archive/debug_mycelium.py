#!/usr/bin/env python3
"""
Debug script for the Mycelium Network.

This script tests the basic functionality of the network
and prints debug information to diagnose learning issues.
"""

import os
import sys
import random
import csv
import time
import numpy as np
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium.environment import Environment
from mycelium.node import MyceliumNode, Signal
from mycelium.network import AdvancedMyceliumNetwork


def load_iris_data() -> Tuple[List[List[float]], List[int]]:
    """
    Load the Iris dataset from the datasets directory.
    
    Returns:
    --------
    Tuple[List[List[float]], List[int]]
        Feature vectors and binary labels (1 for setosa, 0 for others)
    """
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets/iris.csv")
    
    features = []
    labels = []
    
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 5 or i == 0:  # Skip header or empty rows
                continue
                
            # Extract features (first 4 columns)
            feature_values = [float(x) for x in row[:4]]
            features.append(feature_values)
            
            # Convert to binary classification (setosa vs. not-setosa)
            label = 1 if "setosa" in row[4] else 0
            labels.append(label)
    
    print(f"Loaded Iris dataset: {len(features)} samples, {len(features[0])} features")
    print(f"Class distribution: {labels.count(1)} setosa, {labels.count(0)} others")
    
    return features, labels


def test_network_forward_pass():
    """Test the forward pass of the network with simple inputs."""
    print("Testing Network Forward Pass")
    print("===========================")
    
    # Create a simple environment
    env = Environment(dimensions=2, size=1.0)
    
    # Create a network with 2 inputs, 1 output, and 5 hidden nodes
    network = AdvancedMyceliumNetwork(
        environment=env,
        input_size=2,
        output_size=1,
        initial_nodes=5
    )
    
    # Print initial network state
    print(f"Initial network has {len(network.nodes)} nodes")
    print(f"Input nodes: {network.input_nodes}")
    print(f"Output nodes: {network.output_nodes}")
    print(f"Regular nodes: {network.regular_nodes}")
    
    # Test with simple inputs
    inputs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    
    for input_vector in inputs:
        output = network.forward(input_vector)
        print(f"Input: {input_vector}, Output: {output}")
        
        # Print node activations
        print("Node activations:")
        for node_id, node in network.nodes.items():
            node_type = "input" if node_id in network.input_nodes else "output" if node_id in network.output_nodes else "regular"
            print(f"  Node {node_id} ({node_type}): activation={node.activation:.4f}, resource={node.resource_level:.4f}, energy={node.energy:.4f}")
    
    # Print connection strengths
    print("\nConnection strengths:")
    for node_id, node in network.nodes.items():
        if node.connections:
            print(f"  Node {node_id} connects to:")
            for target_id, strength in node.connections.items():
                print(f"    Node {target_id} with strength {strength:.4f}")


def test_network_training():
    """Test the training process with the Iris dataset."""
    print("Testing Network Training Process")
    print("==============================")
    
    # Load Iris dataset
    features, labels = load_iris_data()
    
    # Create a suitable environment
    env = Environment(dimensions=len(features[0]), size=1.0)
    
    # Create a network
    network = AdvancedMyceliumNetwork(
        environment=env,
        input_size=len(features[0]),
        output_size=1,
        initial_nodes=10
    )
    
    # Print initial network state
    print(f"Initial network has {len(network.nodes)} nodes")
    print(f"Input nodes: {network.input_nodes}")
    print(f"Output nodes: {network.output_nodes}")
    print(f"Regular nodes: {network.regular_nodes}")
    
    # Normalize features
    min_values = [min(feature[i] for feature in features) for i in range(len(features[0]))]
    max_values = [max(feature[i] for feature in features) for i in range(len(features[0]))]
    
    normalized_features = []
    for feature in features:
        normalized_feature = []
        for i, value in enumerate(feature):
            if max_values[i] == min_values[i]:
                normalized_value = 0.5
            else:
                normalized_value = (value - min_values[i]) / (max_values[i] - min_values[i])
            normalized_feature.append(normalized_value)
        normalized_features.append(normalized_feature)
    
    # Select a few samples for testing
    test_indices = random.sample(range(len(normalized_features)), 5)
    
    # Test initial predictions
    print("\nInitial predictions (before training):")
    for idx in test_indices:
        output = network.forward(normalized_features[idx])
        prediction = 1 if output[0] >= 0.5 else 0
        print(f"Sample {idx}: True={labels[idx]}, Predicted={prediction}, Output={output[0]:.4f}")
    
    # Print initial connection strengths for some nodes
    print("\nInitial connection strengths:")
    for node_id in network.input_nodes:
        node = network.nodes[node_id]
        if node.connections:
            print(f"  Input Node {node_id} connects to:")
            for target_id, strength in node.connections.items():
                print(f"    Node {target_id} with strength {strength:.4f}")
    
    # Train the network for a few epochs
    print("\nTraining network for 5 epochs...")
    epochs = 5
    learning_rate = 0.2
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Shuffle the data
        indices = list(range(len(normalized_features)))
        random.shuffle(indices)
        
        # Training loop
        for idx in indices:
            # Forward pass
            outputs = network.forward(normalized_features[idx])
            
            # Calculate error
            target = labels[idx]
            error = target - outputs[0]
            
            # Debug print for some samples
            if idx in test_indices:
                print(f"  Sample {idx}: Target={target}, Output={outputs[0]:.4f}, Error={error:.4f}")
                
                # Print active signals
                print(f"    Active signals: {len(network.active_signals)}")
                
                # Print output node state
                output_node_id = network.output_nodes[0]
                output_node = network.nodes[output_node_id]
                print(f"    Output node {output_node_id}: activation={output_node.activation:.4f}, resource={output_node.resource_level:.4f}")
                
                # Print incoming connections to output node
                incoming = {}
                for src_id, node in network.nodes.items():
                    if output_node_id in node.connections:
                        incoming[src_id] = node.connections[output_node_id]
                print(f"    Incoming connections to output: {incoming}")
        
        epoch_end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_end_time - epoch_start_time:.2f} seconds")
    
    # Test final predictions
    print("\nFinal predictions (after training):")
    for idx in test_indices:
        output = network.forward(normalized_features[idx])
        prediction = 1 if output[0] >= 0.5 else 0
        print(f"Sample {idx}: True={labels[idx]}, Predicted={prediction}, Output={output[0]:.4f}")
    
    # Print final connection strengths
    print("\nFinal connection strengths:")
    for node_id in network.input_nodes:
        node = network.nodes[node_id]
        if node.connections:
            print(f"  Input Node {node_id} connects to:")
            for target_id, strength in node.connections.items():
                print(f"    Node {target_id} with strength {strength:.4f}")


def main():
    """Run all tests."""
    print("Mycelium Network Debug Script")
    print("============================\n")
    
    # Test forward pass
    # test_network_forward_pass()
    
    # Test training process
    test_network_training()


if __name__ == "__main__":
    main()
