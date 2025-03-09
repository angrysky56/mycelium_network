#!/usr/bin/env python3
"""
Simple example of using the MyceliumClassifier for binary classification.

This script demonstrates how to use the mycelium-based neural network
for a very simple binary classification problem.
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


def generate_simple_data(n_samples: int = 100) -> Tuple[List[List[float]], List[int]]:
    """
    Generate a very simple binary classification dataset with two features.
    The data has a clear linear separation.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    Tuple[List[List[float]], List[int]]
        Feature vectors and binary labels
    """
    features = []
    labels = []
    
    # Generate class 0: points in bottom-left quadrant
    for _ in range(n_samples // 2):
        x = random.uniform(0, 0.4)
        y = random.uniform(0, 0.4)
        features.append([x, y])
        labels.append(0)
    
    # Generate class 1: points in top-right quadrant
    for _ in range(n_samples // 2):
        x = random.uniform(0.6, 1.0)
        y = random.uniform(0.6, 1.0)
        features.append([x, y])
        labels.append(1)
    
    # Shuffle data
    combined = list(zip(features, labels))
    random.shuffle(combined)
    features, labels = zip(*combined)
    
    print(f"Generated simple dataset: {len(features)} samples, 2 features")
    print(f"Class distribution: {labels.count(1)} positive, {labels.count(0)} negative")
    
    return list(features), list(labels)


def train_test_split(
    features: List[List[float]], 
    labels: List[int], 
    test_size: float = 0.2
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


def print_data_visualization(features, labels):
    """Print a basic ASCII visualization of the data."""
    print("\nData Visualization (ASCII):")
    print("--------------------------")
    print("   y")
    print("   ^")
    print("1.0|" + "".join(["+" if labels[i] == 1 else "o" for i in range(len(features)) if features[i][1] > 0.9]))
    print("0.9|" + "".join(["+" if labels[i] == 1 else "o" for i in range(len(features)) if 0.8 < features[i][1] <= 0.9]))
    print("0.8|" + "".join(["+" if labels[i] == 1 else "o" for i in range(len(features)) if 0.7 < features[i][1] <= 0.8]))
    print("0.7|" + "".join(["+" if labels[i] == 1 else "o" for i in range(len(features)) if 0.6 < features[i][1] <= 0.7]))
    print("0.6|" + "".join(["+" if labels[i] == 1 else "o" for i in range(len(features)) if 0.5 < features[i][1] <= 0.6]))
    print("0.5|")
    print("0.4|" + "".join(["+" if labels[i] == 1 else "o" for i in range(len(features)) if 0.3 < features[i][1] <= 0.4]))
    print("0.3|" + "".join(["+" if labels[i] == 1 else "o" for i in range(len(features)) if 0.2 < features[i][1] <= 0.3]))
    print("0.2|" + "".join(["+" if labels[i] == 1 else "o" for i in range(len(features)) if 0.1 < features[i][1] <= 0.2]))
    print("0.1|" + "".join(["+" if labels[i] == 1 else "o" for i in range(len(features)) if features[i][1] <= 0.1]))
    print("   +---------------------> x")
    print("     0.1 0.3 0.5 0.7 0.9  ")
    print("\nLegend: o = Class 0, + = Class 1")


def main():
    """Run a simple classification example."""
    print("Simple Mycelium Network Classifier Example")
    print("=========================================")
    
    # Generate simple dataset
    features, labels = generate_simple_data(100)
    
    # Print a basic visualization
    print_data_visualization(features, labels)
    
    # Split data
    X_train, y_train, X_test, y_test = train_test_split(features, labels, test_size=0.2)
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Create a simple environment
    env = Environment(dimensions=2, size=1.0)
    
    # Create and train classifier
    print("\nTraining classifier...")
    start_time = time.time()
    
    classifier = MyceliumClassifier(
        input_size=2,
        num_classes=2,
        hidden_nodes=10,
        environment=env
    )
    
    # Train the classifier
    history = classifier.fit(
        X_train, 
        y_train, 
        epochs=20,
        learning_rate=0.2,
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
    
    # Make some example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(X_test))):
        probs = classifier.predict_proba([X_test[i]])[0]
        pred = classifier.predict([X_test[i]])[0]
        print(f"Sample {i+1}: Features={[f'{x:.2f}' for x in X_test[i]]}, True={y_test[i]}, Predicted={pred}, Probability={probs[pred]:.4f}")


if __name__ == "__main__":
    main()
