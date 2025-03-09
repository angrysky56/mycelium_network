#!/usr/bin/env python3
"""
Revised fixed example with direct weight updates.

This script fixes the mycelium neural network to properly learn from data
by implementing a more direct supervised learning approach.
"""

import os
import sys
import random
import time
import csv
import math
import numpy as np
from typing import List, Tuple, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium import Environment
from mycelium.node import MyceliumNode, Signal
from mycelium.network import AdvancedMyceliumNetwork


class RevisedMyceliumNetwork(AdvancedMyceliumNetwork):
    """
    A revised mycelium network with improved learning capabilities.
    """
    
    def _initialize_connections(self) -> None:
        """Initialize the network with strong connections."""
        # Connect each input node to a subset of hidden nodes
        for input_id in self.input_nodes:
            input_node = self.nodes[input_id]
            
            # Connect to a random subset of hidden nodes
            num_connections = max(3, len(self.regular_nodes) // 2)
            target_nodes = random.sample(self.regular_nodes, min(num_connections, len(self.regular_nodes)))
            
            for target_id in target_nodes:
                # Random initial weight
                strength = random.uniform(-0.5, 0.5)
                input_node.connect_to(self.nodes[target_id], strength=strength)
        
        # Connect each hidden node to all output nodes
        for hidden_id in self.regular_nodes:
            hidden_node = self.nodes[hidden_id]
            
            for output_id in self.output_nodes:
                # Random initial weight
                strength = random.uniform(-0.5, 0.5)
                hidden_node.connect_to(self.nodes[output_id], strength=strength)
    
    def direct_train(self, 
                     X_train: List[List[float]], 
                     y_train: List[float],
                     epochs: int = 10,
                     learning_rate: float = 0.1) -> Dict[str, List[float]]:
        """
        Train the network using direct weight updates.
        
        Parameters:
        -----------
        X_train : List[List[float]]
            Training features
        y_train : List[float]
            Training targets
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for weight updates
            
        Returns:
        --------
        Dict[str, List[float]]
            Training history
        """
        history = {
            'loss': [],
            'accuracy': []
        }
        
        for epoch in range(epochs):
            # Shuffle data
            indices = list(range(len(X_train)))
            random.shuffle(indices)
            
            epoch_loss = 0.0
            epoch_correct = 0
            
            # Process each sample
            for idx in indices:
                features = X_train[idx]
                target = y_train[idx]
                
                # Forward pass
                outputs = self.forward(features)
                prediction = 1 if outputs[0] >= 0.5 else 0
                
                # Calculate error
                error = target - outputs[0]
                loss = 0.5 * error**2
                epoch_loss += loss
                
                if prediction == target:
                    epoch_correct += 1
                
                # Backward pass - update weights
                self._update_weights(error, learning_rate)
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(X_train)
            accuracy = epoch_correct / len(X_train)
            
            # Store metrics
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - accuracy: {accuracy:.4f}")
            
            # Adjust learning rate based on progress
            if epoch > 0 and avg_loss > history['loss'][-2]:
                learning_rate *= 0.8  # Reduce learning rate if loss increases
        
        return history
    
    def _update_weights(self, error: float, learning_rate: float) -> None:
        """
        Update weights based on error.
        
        Parameters:
        -----------
        error : float
            Error value (target - output)
        learning_rate : float
            Learning rate for weight updates
        """
        # Get output node ID
        output_id = self.output_nodes[0]
        output_node = self.nodes[output_id]
        
        # Use a sigmoid derivative approximation
        output_value = output_node.activation
        output_derivative = output_value * (1 - output_value)
        output_delta = error * output_derivative
        
        # Update hidden -> output weights
        for hidden_id in self.regular_nodes:
            hidden_node = self.nodes[hidden_id]
            
            if output_id in hidden_node.connections:
                # Calculate weight update
                hidden_activation = hidden_node.activation
                weight_update = learning_rate * output_delta * hidden_activation
                
                # Update the weight
                hidden_node.connections[output_id] += weight_update
                
                # Use the mycelium network's signal mechanism
                if abs(weight_update) > 0.01:
                    # Create a reinforcement signal
                    strength = min(1.0, abs(weight_update) * 5)
                    signal = hidden_node.emit_signal(
                        'reinforcement',
                        strength,
                        {'connection_id': output_id, 'error': error}
                    )
                    self.active_signals.append((hidden_id, signal))
        
        # Process signals to propagate learning information
        self._process_signals()
        
        # Approximation of backpropagation to input layer
        for input_id in self.input_nodes:
            input_node = self.nodes[input_id]
            input_value = input_node.activation
            
            # Update input -> hidden weights based on the output error
            for hidden_id in self.regular_nodes:
                hidden_node = self.nodes[hidden_id]
                
                if hidden_id in input_node.connections and output_id in hidden_node.connections:
                    # Approximation of hidden layer delta
                    hidden_activation = hidden_node.activation
                    hidden_derivative = hidden_activation * (1 - hidden_activation)
                    hidden_delta = output_delta * hidden_node.connections[output_id] * hidden_derivative
                    
                    # Calculate weight update
                    weight_update = learning_rate * hidden_delta * input_value
                    
                    # Update the weight
                    input_node.connections[hidden_id] += weight_update


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


def evaluate_model(network, X_test, y_test) -> Dict[str, float]:
    """
    Evaluate the model on test data.
    
    Parameters:
    -----------
    network : RevisedMyceliumNetwork
        The trained network
    X_test : List[List[float]]
        Test features
    y_test : List[int]
        Test labels
        
    Returns:
    --------
    Dict[str, float]
        Evaluation metrics
    """
    predictions = []
    for features in X_test:
        output = network.forward(features)
        prediction = 1 if output[0] >= 0.5 else 0
        predictions.append(prediction)
    
    # Calculate metrics
    correct = sum(1 for pred, true in zip(predictions, y_test) if pred == true)
    accuracy = correct / len(y_test)
    
    # Confusion matrix
    tp = sum(1 for pred, true in zip(predictions, y_test) if pred == 1 and true == 1)
    fp = sum(1 for pred, true in zip(predictions, y_test) if pred == 1 and true == 0)
    tn = sum(1 for pred, true in zip(predictions, y_test) if pred == 0 and true == 0)
    fn = sum(1 for pred, true in zip(predictions, y_test) if pred == 0 and true == 1)
    
    # Additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
    }


def main():
    """Run the revised fixed example."""
    print("Revised Mycelium Network Example")
    print("===============================")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load and prepare data
    features, labels = load_iris_data()
    normalized_features = normalize_features(features)
    X_train, y_train, X_test, y_test = train_test_split(normalized_features, labels, test_size=0.2)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Create environment
    env = Environment(dimensions=len(features[0]), size=1.0)
    
    # Create revised network
    print("\nInitializing network...")
    network = RevisedMyceliumNetwork(
        environment=env,
        input_size=len(features[0]),
        output_size=1,
        initial_nodes=15
    )
    
    # Print initial network state
    print(f"Nodes: {len(network.nodes)} ({len(network.input_nodes)} input, {len(network.output_nodes)} output, {len(network.regular_nodes)} regular)")
    
    # Test initial predictions (before training)
    test_indices = random.sample(range(len(X_test)), min(5, len(X_test)))
    
    print("\nInitial predictions (before training):")
    for i in test_indices:
        output = network.forward(X_test[i])
        prediction = 1 if output[0] >= 0.5 else 0
        print(f"Sample {i}: True={y_test[i]}, Predicted={prediction}, Output={output[0]:.4f}")
    
    # Train the network
    print("\nTraining network...")
    history = network.direct_train(
        X_train,
        y_train,
        epochs=20,
        learning_rate=0.1
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    metrics = evaluate_model(network, X_test, y_test)
    
    print("\nTest Results:")
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
    
    # Test final predictions (after training)
    print("\nFinal predictions (after training):")
    for i in test_indices:
        output = network.forward(X_test[i])
        prediction = 1 if output[0] >= 0.5 else 0
        print(f"Sample {i}: True={y_test[i]}, Predicted={prediction}, Output={output[0]:.4f}")
    
    # Print network statistics
    print("\nNetwork statistics:")
    stats = network.get_network_statistics()
    print(f"Nodes: {stats['node_count']} ({stats['input_nodes']} input, {stats['output_nodes']} output, {stats['regular_nodes']} regular)")
    print(f"Connections: {stats['connection_count']} (avg {stats['avg_connections_per_node']:.2f} per node)")
    
    # Highlight unique mycelium network features
    print("\nUnique Mycelium Network Features:")
    print("1. Spatial awareness of nodes in environment")
    print("2. Chemical signaling between nodes")
    print("3. Resource allocation based on activity")
    print("4. Dynamic growth and pruning")


if __name__ == "__main__":
    main()
