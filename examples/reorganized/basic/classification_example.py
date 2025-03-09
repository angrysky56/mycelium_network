#!/usr/bin/env python3
"""
Basic Classification Example using Mycelium Neural Network

This script demonstrates how to use the mycelium-based neural network
for a classification task using the Iris dataset.
"""

import os
import sys
import random
import numpy as np
from typing import List, Tuple, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

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


def generate_synthetic_data(n_samples: int = 100, n_features: int = 4) -> Tuple[List[List[float]], List[int]]:
    """
    Generate synthetic binary classification data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features per sample
        
    Returns:
    --------
    Tuple[List[List[float]], List[int]]
        Feature vectors and binary labels
    """
    features = []
    labels = []
    
    for i in range(n_samples):
        # Class 0: Features tend to be smaller
        if i < n_samples // 2:
            feature_vector = [random.uniform(0, 0.5) for _ in range(n_features)]
            label = 0
        # Class 1: Features tend to be larger
        else:
            feature_vector = [random.uniform(0.5, 1.0) for _ in range(n_features)]
            label = 1
        
        # Add some noise
        feature_vector = [f + random.uniform(-0.1, 0.1) for f in feature_vector]
        features.append(feature_vector)
        labels.append(label)
    
    # Shuffle data
    combined = list(zip(features, labels))
    random.shuffle(combined)
    features, labels = zip(*combined)
    
    print(f"Generated synthetic data: {len(features)} samples, {n_features} features")
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


def main():
    """Run the classification example."""
    print("Mycelium Network Classifier Example")
    print("==================================")
    
    # Load data
    features, labels = load_iris_data()
    
    # Split data
    X_train, y_train, X_test, y_test = train_test_split(features, labels, test_size=0.2)
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Create an environment with some obstacles
    env = Environment(dimensions=len(features[0]))
    env.create_random_obstacles(num_obstacles=2, max_radius=0.1)
    
    # Create and train classifier
    print("\nTraining classifier...")
    classifier = MyceliumClassifier(
        input_size=len(features[0]),
        num_classes=2,
        hidden_nodes=25,
        environment=env
    )
    
    # Train the classifier
    history = classifier.fit(
        X_train, 
        y_train, 
        epochs=15, 
        learning_rate=0.1,
        verbose=True
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = classifier.evaluate(X_test, y_test)
    
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
    
    # Make some example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(X_test))):
        probs = classifier.predict_proba([X_test[i]])[0]
        pred = classifier.predict([X_test[i]])[0]
        print(f"Sample {i+1}: True={y_test[i]}, Predicted={pred}, Probability={probs[pred]:.4f}")


if __name__ == "__main__":
    main()
