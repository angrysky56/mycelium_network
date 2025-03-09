#!/usr/bin/env python3
"""
Hybrid mycelium-neural network example with numerical stability fixes.

This script combines the unique features of mycelium networks 
with traditional neural network learning techniques and adds
numerical stability improvements.
"""

import os
import sys
import random
import time
import csv
import math
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict, deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium import Environment
from mycelium.node import MyceliumNode, Signal
from mycelium.network import AdvancedMyceliumNetwork


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


class HybridMyceliumNetwork(AdvancedMyceliumNetwork):
    """
    A hybrid approach that combines mycelium network features with 
    traditional neural network learning.
    """
    
    def __init__(
        self, 
        environment=None,
        input_size=3, 
        output_size=1, 
        initial_nodes=20
    ):
        """Initialize the hybrid network."""
        super().__init__(environment, input_size, output_size, initial_nodes)
        
        # Additional hyperparameters
        self.learning_rate = 0.01  # Reduced learning rate for stability
        self.momentum = 0.9
        self.dropout_rate = 0.2
        self.weight_decay = 0.0001
        
        # Weight constraints
        self.max_weight = 5.0
        
        # Store previous weight updates for momentum
        self.previous_updates = defaultdict(lambda: defaultdict(float))
        
        # Initialize weights with Xavier/Glorot initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        # Input to hidden connections
        for input_id in self.input_nodes:
            input_node = self.nodes[input_id]
            
            # Calculate fan-in and fan-out
            fan_in = self.input_size
            fan_out = len(self.regular_nodes)
            limit = math.sqrt(6 / (fan_in + fan_out))
            
            # Connect to all hidden nodes
            for hidden_id in self.regular_nodes:
                # Xavier/Glorot uniform initialization
                weight = random.uniform(-limit, limit)
                input_node.connect_to(self.nodes[hidden_id], strength=weight)
        
        # Hidden to output connections
        for hidden_id in self.regular_nodes:
            hidden_node = self.nodes[hidden_id]
            
            # Calculate fan-in and fan-out
            fan_in = len(self.regular_nodes)
            fan_out = self.output_size
            limit = math.sqrt(6 / (fan_in + fan_out))
            
            # Connect to all output nodes
            for output_id in self.output_nodes:
                # Xavier/Glorot uniform initialization
                weight = random.uniform(-limit, limit)
                hidden_node.connect_to(self.nodes[output_id], strength=weight)
    
    def train(self, 
              X_train: List[List[float]], 
              y_train: List[float],
              epochs: int = 20,
              batch_size: int = 16,
              validation_data: Tuple[List[List[float]], List[float]] = None,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the network using a hybrid approach.
        
        Parameters:
        -----------
        X_train : List[List[float]]
            Training features
        y_train : List[float]
            Training targets
        epochs : int
            Number of training epochs
        batch_size : int
            Mini-batch size
        validation_data : Tuple[List[List[float]], List[float]]
            Validation data (features, targets)
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        Dict[str, List[float]]
            Training history
        """
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Learning rate schedule
        initial_learning_rate = self.learning_rate
        
        for epoch in range(epochs):
            # Reset epoch metrics
            epoch_loss = 0.0
            epoch_correct = 0
            
            # Create mini-batches
            indices = list(range(len(X_train)))
            random.shuffle(indices)
            
            # Process mini-batches
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:min(start_idx + batch_size, len(indices))]
                batch_X = [X_train[i] for i in batch_indices]
                batch_y = [y_train[i] for i in batch_indices]
                
                # Train on batch
                batch_loss, batch_correct = self._train_on_batch(batch_X, batch_y)
                
                epoch_loss += batch_loss
                epoch_correct += batch_correct
                
                # Apply mycelium-specific features
                self._apply_mycelium_features()
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(X_train)
            accuracy = epoch_correct / len(X_train)
            
            # Store metrics
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            # Validate if validation data is provided
            if validation_data:
                val_X, val_y = validation_data
                val_loss, val_accuracy = self._evaluate(val_X, val_y)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - accuracy: {accuracy:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - accuracy: {accuracy:.4f}")
            
            # Learning rate decay
            self.learning_rate = initial_learning_rate * (1 / (1 + 0.1 * epoch))
            
            # Dynamic network growth/pruning
            if epoch % 5 == 0 and epoch > 0:
                self._dynamic_network_adaptation()
        
        return history
    
    def _train_on_batch(self, 
                        batch_X: List[List[float]], 
                        batch_y: List[float]) -> Tuple[float, int]:
        """
        Train the network on a mini-batch.
        
        Parameters:
        -----------
        batch_X : List[List[float]]
            Batch features
        batch_y : List[float]
            Batch targets
            
        Returns:
        --------
        Tuple[float, int]
            Batch loss and number of correct predictions
        """
        batch_loss = 0.0
        batch_correct = 0
        
        # Accumulate gradients
        weight_gradients = defaultdict(lambda: defaultdict(float))
        
        # Process each sample in the batch
        for features, target in zip(batch_X, batch_y):
            # Forward pass with dropout
            outputs = self._forward_with_dropout(features)
            prediction = 1 if outputs[0] >= 0.5 else 0
            
            # Check if prediction is correct
            if prediction == target:
                batch_correct += 1
            
            # Calculate error and loss
            error = target - outputs[0]
            loss = 0.5 * (error ** 2)
            batch_loss += loss
            
            # Backward pass
            self._backward_pass(features, error, weight_gradients)
        
        # Update weights using accumulated gradients
        self._update_weights_with_momentum(weight_gradients, len(batch_X))
        
        return batch_loss, batch_correct
    
    def _forward_with_dropout(self, features: List[float]) -> List[float]:
        """
        Forward pass with dropout regularization.
        
        Parameters:
        -----------
        features : List[float]
            Input features
            
        Returns:
        --------
        List[float]
            Output values
        """
        # Set input node activations
        for i, value in enumerate(features):
            input_id = self.input_nodes[i]
            self.nodes[input_id].activation = value
        
        # Apply dropout to hidden nodes
        dropout_mask = {}
        for node_id in self.regular_nodes:
            # Bernoulli distribution with p = 1 - dropout_rate
            dropout_mask[node_id] = 1 if random.random() > self.dropout_rate else 0
        
        # Process hidden nodes
        for hidden_id in self.regular_nodes:
            hidden_node = self.nodes[hidden_id]
            
            # Calculate weighted sum of inputs
            weighted_sum = 0.0
            for input_id in self.input_nodes:
                input_node = self.nodes[input_id]
                if hidden_id in input_node.connections:
                    weighted_sum += input_node.activation * input_node.connections[hidden_id]
            
            # Apply activation function (with numerical stability)
            hidden_node.activation = sigmoid(weighted_sum)
            
            # Apply dropout
            if dropout_mask[hidden_id] == 0:
                hidden_node.activation = 0.0
            else:
                # Scale activation to maintain expected value
                hidden_node.activation /= (1 - self.dropout_rate)
        
        # Process output nodes
        outputs = []
        for output_id in self.output_nodes:
            output_node = self.nodes[output_id]
            
            # Calculate weighted sum of hidden nodes
            weighted_sum = 0.0
            for hidden_id in self.regular_nodes:
                hidden_node = self.nodes[hidden_id]
                if output_id in hidden_node.connections:
                    weighted_sum += hidden_node.activation * hidden_node.connections[output_id]
            
            # Apply activation function (with numerical stability)
            output_node.activation = sigmoid(weighted_sum)
            outputs.append(output_node.activation)
        
        return outputs
    
    def _backward_pass(self, 
                       features: List[float], 
                       error: float, 
                       weight_gradients: Dict[int, Dict[int, float]]) -> None:
        """
        Backward pass to calculate gradients.
        
        Parameters:
        -----------
        features : List[float]
            Input features
        error : float
            Output error
        weight_gradients : Dict[int, Dict[int, float]]
            Dictionary to accumulate weight gradients
        """
        # Output node delta
        output_id = self.output_nodes[0]
        output_node = self.nodes[output_id]
        output_activation = output_node.activation
        output_delta = error * output_activation * (1 - output_activation)
        
        # Update hidden -> output weights
        for hidden_id in self.regular_nodes:
            hidden_node = self.nodes[hidden_id]
            
            if output_id in hidden_node.connections:
                # Calculate gradient
                gradient = output_delta * hidden_node.activation
                
                # Add weight decay term (L2 regularization)
                gradient -= self.weight_decay * hidden_node.connections[output_id]
                
                # Accumulate gradient
                weight_gradients[hidden_id][output_id] += gradient
        
        # Calculate hidden node deltas
        hidden_deltas = {}
        for hidden_id in self.regular_nodes:
            hidden_node = self.nodes[hidden_id]
            hidden_activation = hidden_node.activation
            
            # Error term from output
            error_term = 0.0
            if output_id in hidden_node.connections:
                error_term = output_delta * hidden_node.connections[output_id]
            
            # Calculate delta
            hidden_deltas[hidden_id] = error_term * hidden_activation * (1 - hidden_activation)
        
        # Update input -> hidden weights
        for input_id in self.input_nodes:
            input_node = self.nodes[input_id]
            input_value = input_node.activation
            
            for hidden_id in self.regular_nodes:
                if hidden_id in input_node.connections:
                    # Calculate gradient
                    gradient = hidden_deltas[hidden_id] * input_value
                    
                    # Add weight decay term
                    gradient -= self.weight_decay * input_node.connections[hidden_id]
                    
                    # Accumulate gradient
                    weight_gradients[input_id][hidden_id] += gradient
    
    def _update_weights_with_momentum(self, 
                                      weight_gradients: Dict[int, Dict[int, float]],
                                      batch_size: int) -> None:
        """
        Update weights using accumulated gradients with momentum.
        
        Parameters:
        -----------
        weight_gradients : Dict[int, Dict[int, float]]
            Accumulated weight gradients
        batch_size : int
            Batch size for averaging gradients
        """
        # Process each source node
        for source_id, targets in weight_gradients.items():
            source_node = self.nodes[source_id]
            
            # Process each target connection
            for target_id, gradient in targets.items():
                # Average gradient
                avg_gradient = gradient / batch_size
                
                # Calculate update with momentum
                update = (self.learning_rate * avg_gradient + 
                          self.momentum * self.previous_updates[source_id][target_id])
                
                # Update weight (with clipping)
                if target_id in source_node.connections:
                    source_node.connections[target_id] += update
                    
                    # Clip weights to avoid numerical instability
                    source_node.connections[target_id] = max(
                        -self.max_weight, 
                        min(self.max_weight, source_node.connections[target_id])
                    )
                
                # Store update for next iteration
                self.previous_updates[source_id][target_id] = update
    
    def _apply_mycelium_features(self) -> None:
        """Apply mycelium-specific features during training."""
        # 1. Chemical signaling - spread activation signals
        self._process_signals()
        
        # 2. Resource allocation based on usage
        self._allocate_resources()
        
        # 3. Network adaptation - strengthen actively used connections
        self._adapt_connections()
    
    def _dynamic_network_adaptation(self) -> None:
        """Dynamically adapt the network structure during training."""
        # Add new nodes in active regions
        self._grow_network()
        
        # Remove inactive nodes
        self._prune_network()
        
        # Re-initialize connections for new nodes
        self._initialize_new_node_connections()
    
    def _initialize_new_node_connections(self) -> None:
        """Initialize connections for newly added nodes."""
        # Identify nodes without connections to output
        nodes_to_connect = []
        for node_id in self.regular_nodes:
            node = self.nodes[node_id]
            
            # Check if this node has no outgoing connections
            if not node.connections:
                nodes_to_connect.append(node_id)
        
        # Connect these nodes to outputs and inputs
        for node_id in nodes_to_connect:
            node = self.nodes[node_id]
            
            # Connect to output nodes
            for output_id in self.output_nodes:
                # Small initial weight
                weight = random.uniform(-0.1, 0.1)
                node.connect_to(self.nodes[output_id], strength=weight)
            
            # Connect from randomly selected input nodes
            num_inputs = min(2, len(self.input_nodes))
            for input_id in random.sample(self.input_nodes, num_inputs):
                input_node = self.nodes[input_id]
                
                # Small initial weight
                weight = random.uniform(-0.1, 0.1)
                input_node.connect_to(node, strength=weight)
    
    def _evaluate(self, X: List[List[float]], y: List[float]) -> Tuple[float, float]:
        """
        Evaluate the network on validation data.
        
        Parameters:
        -----------
        X : List[List[float]]
            Features
        y : List[float]
            Targets
            
        Returns:
        --------
        Tuple[float, float]
            Loss and accuracy
        """
        total_loss = 0.0
        correct = 0
        
        for features, target in zip(X, y):
            # Forward pass (no dropout during evaluation)
            outputs = self.forward(features)
            prediction = 1 if outputs[0] >= 0.5 else 0
            
            # Calculate loss
            error = target - outputs[0]
            loss = 0.5 * (error ** 2)
            total_loss += loss
            
            # Check accuracy
            if prediction == target:
                correct += 1
        
        # Calculate metrics
        avg_loss = total_loss / len(X)
        accuracy = correct / len(X)
        
        return avg_loss, accuracy
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Make predictions for input samples.
        
        Parameters:
        -----------
        X : List[List[float]]
            Features
            
        Returns:
        --------
        List[int]
            Predicted classes
        """
        predictions = []
        
        for features in X:
            outputs = self.forward(features)
            prediction = 1 if outputs[0] >= 0.5 else 0
            predictions.append(prediction)
        
        return predictions
    
    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : List[List[float]]
            Features
            
        Returns:
        --------
        List[List[float]]
            Class probabilities
        """
        probabilities = []
        
        for features in X:
            outputs = self.forward(features)
            # Convert to probability distribution [P(class=0), P(class=1)]
            prob = [1 - outputs[0], outputs[0]]
            probabilities.append(prob)
        
        return probabilities
    
    def forward(self, features: List[float]) -> List[float]:
        """
        Override the forward method with numerically stable implementation.
        
        Parameters:
        -----------
        features : List[float]
            Input features
            
        Returns:
        --------
        List[float]
            Output values
        """
        # Set input node activations
        for i, value in enumerate(features):
            if i < len(self.input_nodes):
                input_id = self.input_nodes[i]
                self.nodes[input_id].activation = value
        
        # Process hidden nodes
        for hidden_id in self.regular_nodes:
            hidden_node = self.nodes[hidden_id]
            
            # Calculate weighted sum of inputs
            weighted_sum = 0.0
            for input_id in self.input_nodes:
                input_node = self.nodes[input_id]
                if hidden_id in input_node.connections:
                    weighted_sum += input_node.activation * input_node.connections[hidden_id]
            
            # Apply activation function (with numerical stability)
            hidden_node.activation = sigmoid(weighted_sum)
        
        # Process output nodes
        outputs = []
        for output_id in self.output_nodes:
            output_node = self.nodes[output_id]
            
            # Calculate weighted sum of hidden nodes
            weighted_sum = 0.0
            for hidden_id in self.regular_nodes:
                hidden_node = self.nodes[hidden_id]
                if output_id in hidden_node.connections:
                    weighted_sum += hidden_node.activation * hidden_node.connections[output_id]
            
            # Apply activation function (with numerical stability)
            output_node.activation = sigmoid(weighted_sum)
            outputs.append(output_node.activation)
        
        return outputs


def load_breast_cancer_data() -> Tuple[List[List[float]], List[int]]:
    """
    Load the Breast Cancer dataset for binary classification.
    
    Returns:
    --------
    Tuple[List[List[float]], List[int]]
        Feature vectors and binary labels
    """
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets/breast_cancer.csv")
    
    features = []
    labels = []
    
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 3 or i == 0:  # Skip header or empty rows
                continue
                
            # Second column is the diagnosis (M=malignant, B=benign)
            label = 1 if row[1] == "M" else 0
            labels.append(label)
            
            # Extract features (skip the ID and diagnosis columns)
            feature_values = [float(x) for x in row[2:]]
            features.append(feature_values)
    
    print(f"Loaded Breast Cancer dataset: {len(features)} samples, {len(features[0])} features")
    print(f"Class distribution: {labels.count(1)} malignant, {labels.count(0)} benign")
    
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
    """Run the hybrid mycelium network example."""
    print("Hybrid Mycelium Network Example (Fixed)")
    print("====================================")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load breast cancer dataset
    features, labels = load_breast_cancer_data()
    
    # Normalize features
    normalized_features = normalize_features(features)
    
    # Split into training and testing sets
    X_train, y_train, X_test, y_test = train_test_split(normalized_features, labels, test_size=0.2)
    
    # Create validation split from training data
    train_size = int(0.8 * len(X_train))
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Create environment
    env = Environment(dimensions=len(features[0]), size=1.0)
    
    # Seed resources in the environment
    for _ in range(10):
        position = tuple(random.random() for _ in range(env.dimensions))
        env.add_resource(position, amount=2.0)
    
    # Create hybrid network
    print("\nInitializing hybrid mycelium network...")
    network = HybridMyceliumNetwork(
        environment=env,
        input_size=len(features[0]),
        output_size=1,
        initial_nodes=20
    )
    
    # Print initial network state
    print(f"Nodes: {len(network.nodes)} ({len(network.input_nodes)} input, {len(network.output_nodes)} output, {len(network.regular_nodes)} regular)")
    
    # Test initial predictions
    print("\nInitial predictions (before training):")
    test_indices = random.sample(range(len(X_test)), min(5, len(X_test)))
    for i in test_indices:
        output = network.forward(X_test[i])
        prediction = 1 if output[0] >= 0.5 else 0
        print(f"Sample {i}: True={y_test[i]}, Predicted={prediction}, Output={output[0]:.4f}")
    
    # Train the network
    print("\nTraining network...")
    history = network.train(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=True
    )
    
    # Test final predictions
    print("\nFinal predictions (after training):")
    for i in test_indices:
        output = network.forward(X_test[i])
        prediction = 1 if output[0] >= 0.5 else 0
        print(f"Sample {i}: True={y_test[i]}, Predicted={prediction}, Output={output[0]:.4f}")
    
    # Evaluate on test set
    predictions = network.predict(X_test)
    correct = sum(1 for p, t in zip(predictions, y_test) if p == t)
    accuracy = correct / len(y_test)
    
    # Calculate confusion matrix
    tp = sum(1 for p, t in zip(predictions, y_test) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(predictions, y_test) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(predictions, y_test) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(predictions, y_test) if p == 0 and t == 1)
    
    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Print final network statistics
    print("\nFinal network statistics:")
    stats = network.get_network_statistics()
    print(f"Nodes: {stats['node_count']} ({stats['input_nodes']} input, {stats['output_nodes']} output, {stats['regular_nodes']} regular)")
    print(f"Connections: {stats['connection_count']} (avg {stats['avg_connections_per_node']:.2f} per node)")
    
    # Highlight unique mycelium network features
    print("\nUnique Mycelium Network Features Used:")
    print("1. Spatial awareness - nodes positioned in multi-dimensional environment")
    print("2. Chemical signaling - activation signals propagate through network")
    print("3. Resource allocation - active nodes receive more resources")
    print("4. Dynamic growth - network structure adapts during training")
    print("5. Hybrid learning - combines gradient-based learning with bio-inspired mechanisms")


if __name__ == "__main__":
    main()
