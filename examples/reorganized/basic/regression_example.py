#!/usr/bin/env python3
"""
Basic Regression Example using Mycelium Neural Network

This script demonstrates how to use the mycelium-based neural network
for regression tasks using a synthetic dataset.
"""

import os
import sys
import random
import numpy as np
import math
from typing import List, Tuple, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from mycelium import MyceliumRegressor, Environment


def generate_regression_data(
    n_samples: int = 100, 
    noise: float = 0.2
) -> Tuple[List[List[float]], List[float]]:
    """
    Generate synthetic regression data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Noise level to add to the targets
        
    Returns:
    --------
    Tuple[List[List[float]], List[float]]
        Feature vectors and target values
    """
    features = []
    targets = []
    
    for i in range(n_samples):
        # Generate a 2D feature vector
        x1 = random.uniform(-3, 3)
        x2 = random.uniform(-3, 3)
        
        # Target function: y = sin(x1) + cos(x2) + x1*x2/5
        target = math.sin(x1) + math.cos(x2) + (x1 * x2 / 5)
        
        # Add noise
        target += random.uniform(-noise, noise)
        
        features.append([x1, x2])
        targets.append(target)
    
    # Shuffle data
    combined = list(zip(features, targets))
    random.shuffle(combined)
    features, targets = zip(*combined)
    
    print(f"Generated {n_samples} samples with 2 features")
    print(f"Target range: {min(targets):.2f} to {max(targets):.2f}")
    
    return list(features), list(targets)


def train_test_split(
    features: List[List[float]], 
    targets: List[float], 
    test_size: float = 0.2
) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    features : List[List[float]]
        Feature vectors
    targets : List[float]
        Target values
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    Tuple[List[List[float]], List[float], List[List[float]], List[float]]
        Train features, train targets, test features, test targets
    """
    # Create indices and shuffle
    indices = list(range(len(features)))
    random.shuffle(indices)
    
    # Split point
    split_idx = int(len(features) * (1 - test_size))
    
    # Split data
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Extract data
    X_train = [features[i] for i in train_indices]
    y_train = [targets[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    y_test = [targets[i] for i in test_indices]
    
    return X_train, y_train, X_test, y_test


def main():
    """Run the regression example."""
    print("Mycelium Network Regressor Example")
    print("=================================")
    
    # Generate data
    features, targets = generate_regression_data(200, noise=0.3)
    
    # Split data
    X_train, y_train, X_test, y_test = train_test_split(features, targets, test_size=0.2)
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Create an environment with some resources
    env = Environment(dimensions=len(features[0]))
    env.create_grid_resources(grid_size=5, resource_value=1.5)
    
    # Create and train regressor
    print("\nTraining regressor...")
    regressor = MyceliumRegressor(
        input_size=len(features[0]),
        output_size=1,
        hidden_nodes=30,
        environment=env
    )
    
    # Train the regressor
    history = regressor.fit(
        X_train, 
        y_train, 
        epochs=20, 
        learning_rate=0.1,
        verbose=True
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = regressor.evaluate(X_test, y_test)
    
    print(f"\nTest Results:")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    print(f"R-squared: {metrics['r2']:.4f}")
    
    # Network statistics
    print("\nMycelium Network Statistics:")
    stats = regressor.network.get_network_statistics()
    print(f"Nodes: {stats['node_count']} ({stats['input_nodes']} input, {stats['output_nodes']} output, {stats['regular_nodes']} regular)")
    print(f"Connections: {stats['connection_count']} (avg {stats['avg_connections_per_node']:.2f} per node)")
    
    # Make some example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(X_test))):
        pred = regressor.predict([X_test[i]])[0]
        print(f"Sample {i+1}: True={y_test[i]:.4f}, Predicted={pred:.4f}, Error={abs(pred - y_test[i]):.4f}")


if __name__ == "__main__":
    main()
