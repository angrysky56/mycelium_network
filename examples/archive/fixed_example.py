#!/usr/bin/env python3
"""
Fixed example of using the MyceliumClassifier for binary classification.

This script demonstrates how to use the mycelium-based neural network
with a fixed initialization that ensures there are proper connections.
"""

import os
import sys
import random
import time
import csv
import numpy as np
from typing import List, Tuple, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium import Environment
from mycelium.node import MyceliumNode, Signal
from mycelium.network import AdvancedMyceliumNetwork


class FixedMyceliumNetwork(AdvancedMyceliumNetwork):
    """
    A fixed version of the mycelium network that ensures proper connectivity.
    """
    
    def _initialize_connections(self) -> None:
        """Initialize the network with connections ensuring paths to outputs."""
        # Connect each input node to some regular nodes
        for input_id in self.input_nodes:
            input_node = self.nodes[input_id]
            
            # Find closest regular nodes
            regular_distances = [
                (reg_id, self.environment.calculate_distance(
                    input_node.position, self.nodes[reg_id].position
                ))
                for reg_id in self.regular_nodes
            ]
            
            # Sort by distance and connect to the closest ones
            regular_distances.sort(key=lambda x: x[1])
            for reg_id, distance in regular_distances[:3]:  # Connect to 3 closest
                if input_node.can_connect_to(self.nodes[reg_id], self.environment):
                    input_node.connect_to(self.nodes[reg_id], strength=0.5)  # Stronger initial connection
        
        # Connect regular nodes to each other based on proximity
        for i, node_id in enumerate(self.regular_nodes):
            node = self.nodes[node_id]
            
            # Find potential connections
            for other_id in self.regular_nodes[i+1:]:
                other_node = self.nodes[other_id]
                
                # Connect if possible and with some probability
                if (node.can_connect_to(other_node, self.environment) and 
                    random.random() < 0.4):  # Higher probability
                    node.connect_to(other_node, strength=0.3)
        
        # CRITICAL FIX: Ensure all regular nodes connect to output nodes
        for reg_id in self.regular_nodes:
            reg_node = self.nodes[reg_id]
            
            for output_id in self.output_nodes:
                # Always connect regular nodes to outputs with strong connection
                reg_node.connect_to(self.nodes[output_id], strength=0.5)  # Stronger initial connection


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


def normalize_features(features: List[List[float]]) -> List[List[float]]:
    """Normalize features to the [0, 1] range."""
    min_values = [min(f[i] for f in features) for i in range(len(features[0]))]
    max_values = [max(f[i] for f in features) for i in range(len(features[0]))]
    
    normalized = []
    for feature in features:
        normalized_feature = []
        for i, value in enumerate(feature):
            if max_values[i] == min_values[i]:
                normalized_value = 0.5
            else:
                normalized_value = (value - min_values[i]) / (max_values[i] - min_values[i])
            normalized_feature.append(normalized_value)
        normalized.append(normalized_feature)
    
    return normalized


def train_test_split(
    features: List[List[float]], 
    labels: List[int], 
    test_size: float = 0.2
) -> Tuple[List[List[float]], List[int], List[List[float]], List[int]]:
    """Split data into training and testing sets."""
    indices = list(range(len(features)))
    random.shuffle(indices)
    
    split_idx = int(len(features) * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = [features[i] for i in train_indices]
    y_train = [labels[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    y_test = [labels[i] for i in test_indices]
    
    return X_train, y_train, X_test, y_test


def main():
    """Run a fixed classification example."""
    print("Fixed Mycelium Network Example")
    print("============================")
    
    # Load and prepare data
    features, labels = load_iris_data()
    normalized_features = normalize_features(features)
    X_train, y_train, X_test, y_test = train_test_split(normalized_features, labels, test_size=0.2)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Create environment
    env = Environment(dimensions=len(features[0]), size=1.0)
    
    # Create fixed network
    network = FixedMyceliumNetwork(
        environment=env,
        input_size=len(features[0]),
        output_size=1,
        initial_nodes=15
    )
    
    # Print initial network state
    print("\nInitial network:")
    print(f"Nodes: {len(network.nodes)} ({len(network.input_nodes)} input, {len(network.output_nodes)} output, {len(network.regular_nodes)} regular)")
    
    # Check connections to output
    output_id = network.output_nodes[0]
    incoming_connections = 0
    for node_id, node in network.nodes.items():
        if output_id in node.connections:
            incoming_connections += 1
    
    print(f"Output node has {incoming_connections} incoming connections")
    
    # Test initial predictions (before training)
    correct_before = 0
    test_indices = random.sample(range(len(X_test)), min(5, len(X_test)))
    
    print("\nInitial predictions (before training):")
    for i in test_indices:
        output = network.forward(X_test[i])
        prediction = 1 if output[0] >= 0.5 else 0
        if prediction == y_test[i]:
            correct_before += 1
        print(f"Sample {i}: True={y_test[i]}, Predicted={prediction}, Output={output[0]:.4f}")
    
    # Manual training
    print("\nTraining network...")
    epochs = 5
    
    for epoch in range(epochs):
        epoch_correct = 0
        
        # Shuffle training data
        train_data = list(zip(X_train, y_train))
        random.shuffle(train_data)
        X_train_shuffled, y_train_shuffled = zip(*train_data)
        
        for i, (features, target) in enumerate(zip(X_train_shuffled, y_train_shuffled)):
            # Forward pass
            output = network.forward(features)
            
            # Calculate error
            error = target - output[0]
            
            # Update weights using a simple rule
            # (This is a manual implementation, not using the built-in train method)
            for node_id in network.regular_nodes:
                node = network.nodes[node_id]
                if output_id in node.connections:
                    # Get the node's activation
                    node_activation = node.activation
                    
                    # Update the connection
                    delta = 0.1 * error * node_activation
                    node.connections[output_id] += delta
            
            # Prediction
            prediction = 1 if output[0] >= 0.5 else 0
            if prediction == target:
                epoch_correct += 1
        
        # Epoch accuracy
        epoch_accuracy = epoch_correct / len(X_train)
        print(f"Epoch {epoch+1}/{epochs} - accuracy: {epoch_accuracy:.4f}")
    
    # Test final predictions (after training)
    correct_after = 0
    
    print("\nFinal predictions (after training):")
    for i in test_indices:
        output = network.forward(X_test[i])
        prediction = 1 if output[0] >= 0.5 else 0
        if prediction == y_test[i]:
            correct_after += 1
        print(f"Sample {i}: True={y_test[i]}, Predicted={prediction}, Output={output[0]:.4f}")
    
    # Evaluate on the entire test set
    correct = 0
    for i in range(len(X_test)):
        output = network.forward(X_test[i])
        prediction = 1 if output[0] >= 0.5 else 0
        if prediction == y_test[i]:
            correct += 1
    
    accuracy = correct / len(X_test)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Debug: check some final connection strengths
    print("\nFinal connection strengths to output node:")
    connections = []
    for node_id in network.regular_nodes:
        node = network.nodes[node_id]
        if output_id in node.connections:
            connections.append((node_id, node.connections[output_id]))
    
    # Sort by strength
    connections.sort(key=lambda x: abs(x[1]), reverse=True)
    for node_id, strength in connections[:5]:  # Show top 5
        print(f"Node {node_id} -> Output: {strength:.4f}")


if __name__ == "__main__":
    main()
