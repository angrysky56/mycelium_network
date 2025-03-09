#!/usr/bin/env python3
"""
Basic Anomaly Detection Example using Mycelium Neural Network

This script demonstrates how to use the mycelium-based neural network
for anomaly detection using a synthetic dataset.
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

from mycelium import MyceliumAnomalyDetector, Environment


def generate_anomaly_data(
    n_samples: int = 200,
    n_features: int = 5,
    contamination: float = 0.1
) -> Tuple[List[List[float]], List[int]]:
    """
    Generate synthetic data for anomaly detection.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features per sample
    contamination : float
        Proportion of anomalies in the dataset
        
    Returns:
    --------
    Tuple[List[List[float]], List[int]]
        Feature vectors and binary labels (1 for anomalies, 0 for normal samples)
    """
    features = []
    labels = []
    
    # Generate normal samples
    n_normal = int(n_samples * (1 - contamination))
    for _ in range(n_normal):
        # Normal samples are clustered
        feature_vector = [random.normalvariate(0, 1) for _ in range(n_features)]
        features.append(feature_vector)
        labels.append(0)  # Normal
    
    # Generate anomalies
    n_anomalies = n_samples - n_normal
    for _ in range(n_anomalies):
        # Anomalies are scattered
        feature_vector = [random.normalvariate(0, 1) * 3 + (random.random() * 10 - 5) 
                         for _ in range(n_features)]
        features.append(feature_vector)
        labels.append(1)  # Anomaly
    
    # Shuffle data
    combined = list(zip(features, labels))
    random.shuffle(combined)
    features, labels = zip(*combined)
    
    print(f"Generated synthetic data: {len(features)} samples, {n_features} features")
    print(f"Class distribution: {labels.count(0)} normal, {labels.count(1)} anomalies")
    
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
        Binary labels
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    Tuple[List[List[float]], List[int], List[List[float]], List[int]]
        Train features, train labels, test features, test labels
    """
    # Split normal and anomaly samples
    normal_indices = [i for i, label in enumerate(labels) if label == 0]
    anomaly_indices = [i for i, label in enumerate(labels) if label == 1]
    
    # Use a higher proportion of normal samples for training
    train_normal = int(len(normal_indices) * (1 - test_size))
    train_anomalies = int(len(anomaly_indices) * 0.5)  # Use fewer anomalies in training
    
    # Shuffle
    random.shuffle(normal_indices)
    random.shuffle(anomaly_indices)
    
    # Create train/test sets
    train_indices = normal_indices[:train_normal] + anomaly_indices[:train_anomalies]
    test_indices = normal_indices[train_normal:] + anomaly_indices[train_anomalies:]
    
    # Extract data
    X_train = [features[i] for i in train_indices]
    y_train = [labels[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    y_test = [labels[i] for i in test_indices]
    
    # Shuffle train/test data
    train_combined = list(zip(X_train, y_train))
    test_combined = list(zip(X_test, y_test))
    random.shuffle(train_combined)
    random.shuffle(test_combined)
    X_train, y_train = zip(*train_combined)
    X_test, y_test = zip(*test_combined)
    
    return list(X_train), list(y_train), list(X_test), list(y_test)


def main():
    """Run the anomaly detection example."""
    print("Mycelium Network Anomaly Detector Example")
    print("========================================")
    
    # Generate data
    contamination = 0.1
    features, labels = generate_anomaly_data(200, 5, contamination)
    
    # Split data
    X_train, y_train, X_test, y_test = train_test_split(features, labels, test_size=0.3)
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Count anomalies in training set
    train_anomalies = sum(y_train)
    print(f"Training set contamination: {train_anomalies / len(y_train):.2%}")
    
    # Create an environment with obstacles to create irregular patterns
    env = Environment(dimensions=5)
    env.create_random_obstacles(num_obstacles=3, max_radius=0.15)
    
    # Create and train anomaly detector
    print("\nTraining anomaly detector...")
    detector = MyceliumAnomalyDetector(
        input_size=5,
        hidden_nodes=25,
        contamination=contamination,
        environment=env
    )
    
    # Train the detector (unsupervised, ignoring labels)
    history = detector.fit(
        X_train, 
        epochs=15, 
        verbose=True
    )
    
    # Make predictions on test set
    print("\nDetecting anomalies in test set...")
    y_pred = detector.predict(X_test)
    
    # Calculate metrics
    print("\nEvaluating results...")
    tp = sum(1 for true, pred in zip(y_test, y_pred) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_test, y_pred) if true == 0 and pred == 1)
    tn = sum(1 for true, pred in zip(y_test, y_pred) if true == 0 and pred == 0)
    fn = sum(1 for true, pred in zip(y_test, y_pred) if true == 1 and pred == 0)
    
    accuracy = (tp + tn) / len(y_test)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    
    # Network statistics
    print("\nMycelium Network Statistics:")
    stats = detector.network.get_network_statistics()
    print(f"Nodes: {stats['node_count']} ({stats['input_nodes']} input, {stats['output_nodes']} output, {stats['regular_nodes']} regular)")
    print(f"Connections: {stats['connection_count']} (avg {stats['avg_connections_per_node']:.2f} per node)")
    
    # Get decision scores
    print("\nExample anomalies with scores:")
    scores = detector.decision_function(X_test)
    
    # Sort by descending anomaly score
    sample_scores = list(zip(range(len(X_test)), y_test, y_pred, scores))
    sample_scores.sort(key=lambda x: x[3], reverse=True)
    
    # Show top 5 anomalies
    for i, true, pred, score in sample_scores[:5]:
        print(f"Sample {i}: True={true}, Predicted={pred}, Anomaly Score={score:.4f}")


if __name__ == "__main__":
    main()
