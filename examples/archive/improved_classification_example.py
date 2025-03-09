#!/usr/bin/env python3
"""
Improved example of using the MyceliumClassifier for binary classification.

This script demonstrates how to use the mycelium-based neural network
for a classification task using the Iris dataset with improved parameters.
"""

import os
import sys
import random
import time
import numpy as np
from typing import List, Tuple, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium import MyceliumClassifier, Environment


def load_iris_data() -> Tuple[List[List[float]], List[int]]:
    """
    Load the Iris dataset and convert it to a binary classification problem.
    
    Returns:
    --------
    Tuple[List[List[float]], List[int]]
        Feature vectors and binary labels (1 for setosa, 0 for others)
    """
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        
        features = iris.data.tolist()
        # Convert to binary: setosa (0) vs others (1, 2)
        labels = [1 if label == 0 else 0 for label in iris.target]
        
        print(f"Loaded Iris dataset: {len(features)} samples, {len(features[0])} features")
        print(f"Class distribution: {labels.count(1)} setosa, {labels.count(0)} others")
        
        return features, labels
    except ImportError:
        print("scikit-learn not found. Using synthetic data instead.")
        # Generate synthetic data if sklearn is not available
        return generate_synthetic_data(150, 4)


def train_test_split(
    features: List[List[float]], 
    labels: List[int], 
    test_size: float = 0.3
) -> Tuple[List[List[float]], List[int], List[List[float]], List[int]]:
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    features : List[List[float]]
        Feature vectors
    labels : List[int]
        Target labels
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    Tuple[List[List[float]], List[int], List[List[float]], List[int]]
        Train features, train labels, test features, test labels
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
    y_train = [labels[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    y_test = [labels[i] for i in test_indices]
    
    return X_train, y_train, X_test, y_test


def main():
    """Run the classification example with improved parameters."""
    print("Improved Mycelium Network Classifier Example")
    print("===========================================")
    
    # Load data
    features, labels = load_iris_data()
    
    # Split data
    X_train, y_train, X_test, y_test = train_test_split(features, labels, test_size=0.3)
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Create a more complex environment with resources
    env = Environment(dimensions=len(features[0]), size=2.0)
    env.create_grid_resources(grid_size=5, resource_value=1.5)
    env.create_random_obstacles(num_obstacles=3, max_radius=0.1)
    
    print("\nEnvironment:")
    print(f"Dimensions: {env.dimensions}")
    print(f"Size: {env.size}")
    print(f"Resources: {len(env.resources)}")
    print(f"Obstacles: {len(env.obstacles)}")
    
    # Create and train classifier with more nodes and longer training
    print("\nTraining classifier...")
    start_time = time.time()
    
    classifier = MyceliumClassifier(
        input_size=len(features[0]),
        num_classes=2,
        hidden_nodes=40,  # More hidden nodes
        environment=env
    )
    
    # Train the classifier with more epochs
    history = classifier.fit(
        X_train, 
        y_train, 
        epochs=30,  # More epochs
        learning_rate=0.2,  # Higher learning rate
        validation_split=0.2,
        verbose=True
    )
    
    end_time = time.time()
    train_time = end_time - start_time
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = classifier.evaluate(X_test, y_test)
    
    print(f"\nTraining Time: {train_time:.2f} seconds")
    print(f"\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"True Positives: {cm['tp']}")
    print(f"False Positives: {cm['fp']}")
    print(f"True Negatives: {cm['tn']}")
    print(f"False Negatives: {cm['fn']}")
    
    # Network visualization data
    print("\nMycelium Network Statistics:")
    stats = classifier.network.get_network_statistics()
    print(f"Nodes: {stats['node_count']} ({stats['input_nodes']} input, {stats['output_nodes']} output, {stats['regular_nodes']} regular)")
    print(f"Connections: {stats['connection_count']} (avg {stats['avg_connections_per_node']:.2f} per node)")
    print(f"Average resources: {stats.get('avg_resource_level', 0):.2f}")
    print(f"Average energy: {stats.get('avg_energy', 0):.2f}")
    
    # Skip plotting to avoid permission issues
    print("\nSkipping plot generation due to permission limitations.")
    
    # Make some example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(X_test))):
        probs = classifier.predict_proba([X_test[i]])[0]
        pred = classifier.predict([X_test[i]])[0]
        print(f"Sample {i+1}: True={y_test[i]}, Predicted={pred}, Probability={probs[pred]:.4f}")
    
    # Skip network visualization to avoid permission issues
    print("\nSkipping network visualization data due to permission limitations.")


if __name__ == "__main__":
    main()
