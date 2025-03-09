#!/usr/bin/env python3
"""
Comparison of the mycelium network with conventional neural networks.

This script compares the mycelium network with scikit-learn's MLPClassifier
on the same dataset to provide a performance benchmark.
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

# Import the hybrid mycelium network
from hybrid_example_fixed import HybridMyceliumNetwork, load_breast_cancer_data, normalize_features

# Try to import scikit-learn
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    sklearn_available = True
except ImportError:
    print("scikit-learn not available, comparison will only use mycelium network")
    sklearn_available = False


def train_and_evaluate_mycelium(X_train, y_train, X_test, y_test):
    """Train and evaluate the mycelium network."""
    print("\n=== Mycelium Network ===")
    
    start_time = time.time()
    
    # Create environment with dimensions matching feature count
    from mycelium import Environment
    env = Environment(dimensions=len(X_train[0]), size=1.0)
    
    # Create hybrid network
    network = HybridMyceliumNetwork(
        environment=env,
        input_size=len(X_train[0]),
        output_size=1,
        initial_nodes=20
    )
    
    # Create validation split
    train_size = int(0.8 * len(X_train))
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    X_train_subset = X_train[:train_size]
    y_train_subset = y_train[:train_size]
    
    # Train the network
    network.train(
        X_train_subset,
        y_train_subset,
        epochs=15,
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=True
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Evaluate on test set
    predictions = network.predict(X_test)
    
    # Calculate metrics
    accuracy = sum(1 for p, t in zip(predictions, y_test) if p == t) / len(y_test)
    
    # Confusion matrix
    tp = sum(1 for p, t in zip(predictions, y_test) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(predictions, y_test) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(predictions, y_test) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(predictions, y_test) if p == 0 and t == 1)
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    
    # Network statistics
    stats = network.get_network_statistics()
    print(f"\nFinal network stats:")
    print(f"Nodes: {stats['node_count']} ({stats['input_nodes']} input, {stats['output_nodes']} output, {stats['regular_nodes']} regular)")
    print(f"Connections: {stats['connection_count']} (avg {stats['avg_connections_per_node']:.2f} per node)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
        'network_stats': stats
    }


def train_and_evaluate_sklearn_mlp(X_train, y_train, X_test, y_test):
    """Train and evaluate a scikit-learn MLP classifier."""
    if not sklearn_available:
        print("scikit-learn not available, skipping MLP comparison")
        return None
    
    print("\n=== Scikit-learn MLP Classifier ===")
    
    # Convert to numpy arrays
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)
    y_test_np = np.array(y_test)
    
    # Create and train the model
    start_time = time.time()
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(20,),  # Comparable to our mycelium network
        activation='logistic',     # Sigmoid activation (like our mycelium nodes)
        solver='adam',             # Adam optimizer
        alpha=0.0001,              # L2 regularization
        batch_size=16,             # Same as mycelium
        learning_rate_init=0.001,  # Initial learning rate
        max_iter=100,              # Maximum iterations
        early_stopping=True,       # Use early stopping
        validation_fraction=0.2,   # Validation split
        verbose=True,              # Show training progress
        random_state=42            # For reproducibility
    )
    
    mlp.fit(X_train_np, y_train_np)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Make predictions
    y_pred = mlp.predict(X_test_np)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_np, y_pred)
    precision = precision_score(y_test_np, y_pred, zero_division=0)
    recall = recall_score(y_test_np, y_pred, zero_division=0)
    f1 = f1_score(y_test_np, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test_np, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Print results
    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
        'iterations': mlp.n_iter_
    }


def custom_train_test_split(features, labels, test_size=0.2, random_state=42):
    """Custom train-test split function."""
    # Set random seed
    random.seed(random_state)
    
    # Create indices and shuffle
    indices = list(range(len(features)))
    random.shuffle(indices)
    
    # Split point
    split_idx = int(len(features) * (1 - test_size))
    
    # Split data
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = [features[i] for i in train_indices]
    y_train = [labels[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    y_test = [labels[i] for i in test_indices]
    
    return X_train, y_train, X_test, y_test


def print_comparison(mycelium_results, sklearn_results):
    """Print a comparison of the results."""
    print("\n=== Performance Comparison ===")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'training_time']
    
    # Calculate the maximum width needed for the metric names
    max_width = max(len(metric) for metric in metrics)
    
    # Print header
    print(f"{'Metric':{max_width}} | {'Mycelium':10} | {'MLP':10}")
    print(f"{'-' * max_width}-+-{'-' * 10}-+-{'-' * 10}")
    
    # Print metrics
    for metric in metrics:
        mycelium_value = mycelium_results[metric]
        sklearn_value = sklearn_results[metric] if sklearn_results else "N/A"
        
        if metric == 'training_time':
            print(f"{metric:{max_width}} | {mycelium_value:10.2f} | {sklearn_value if isinstance(sklearn_value, str) else sklearn_value:10.2f}")
        else:
            print(f"{metric:{max_width}} | {mycelium_value:10.4f} | {sklearn_value if isinstance(sklearn_value, str) else sklearn_value:10.4f}")
    
    # Print unique mycelium features
    print("\nUnique Mycelium Network Features:")
    print("1. Spatial awareness - nodes positioned in multi-dimensional environment")
    print("2. Chemical signaling - activation signals propagate through network")
    print("3. Resource allocation - active nodes receive more resources")
    print("4. Dynamic growth - network structure adapts during training")
    print("5. Hybrid learning - combines gradient-based learning with bio-inspired mechanisms")


def main():
    """Run the comparison."""
    print("Neural Network Comparison: Mycelium vs. Traditional")
    print("=================================================")
    
    # Load and prepare data
    features, labels = load_breast_cancer_data()
    normalized_features = normalize_features(features)
    
    # Split the data
    X_train, y_train, X_test, y_test = custom_train_test_split(normalized_features, labels, test_size=0.2)
    
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Train and evaluate the mycelium network
    mycelium_results = train_and_evaluate_mycelium(X_train, y_train, X_test, y_test)
    
    # Train and evaluate scikit-learn MLP (if available)
    if sklearn_available:
        sklearn_results = train_and_evaluate_sklearn_mlp(X_train, y_train, X_test, y_test)
    else:
        sklearn_results = None
    
    # Print comparison
    if sklearn_results:
        print_comparison(mycelium_results, sklearn_results)
    
    # Print conclusion
    print("\nConclusion:")
    if sklearn_results:
        if mycelium_results['accuracy'] > sklearn_results['accuracy']:
            print("The mycelium network outperformed the traditional MLP classifier!")
        elif mycelium_results['accuracy'] < sklearn_results['accuracy']:
            print("The traditional MLP classifier outperformed the mycelium network.")
        else:
            print("Both approaches achieved similar accuracy.")
        
        print("\nNote: While the mycelium network may not always achieve better numerical results,")
        print("it offers unique biomimetic features like adaptive growth and spatial awareness")
        print("that could be valuable in specific domains requiring more dynamic neural architectures.")
    else:
        print("The mycelium network demonstrates the potential of biomimetic approaches,")
        print("but further development and optimization are needed to compete with")
        print("traditional machine learning algorithms.")


if __name__ == "__main__":
    main()
