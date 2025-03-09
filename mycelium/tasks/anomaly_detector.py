"""
Mycelium-based neural network anomaly detector.

This module implements an anomaly detector that leverages the mycelium network's
ability to learn normal patterns and detect deviations.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Union, Optional
from collections import defaultdict

from mycelium.network import AdvancedMyceliumNetwork
from mycelium.environment import Environment


class MyceliumAnomalyDetector:
    """
    Mycelium-based neural network for anomaly detection.
    
    This class adapts the mycelium network for detecting anomalies in data,
    using the network's ability to learn normal patterns and signal unusual behaviors.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_nodes: int = 30,
        contamination: float = 0.1,
        environment: Optional[Environment] = None
    ):
        """
        Initialize the mycelium anomaly detector.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_nodes : int
            Number of initial hidden nodes (default: 30)
        contamination : float
            Expected proportion of anomalies in training data (default: 0.1)
        environment : Environment, optional
            Custom environment for the network
        """
        self.input_size = input_size
        self.contamination = contamination
        
        # Custom environment with obstacles
        if environment is None:
            environment = Environment(dimensions=input_size)
            # Add random obstacles to create irregular patterns
            environment.create_random_obstacles(num_obstacles=3)
        
        self.environment = environment
        
        # Initialize network for reconstruction task
        self.network = AdvancedMyceliumNetwork(
            input_size=input_size,
            output_size=input_size,  # Reconstruction task
            initial_nodes=hidden_nodes,
            environment=environment
        )
        
        # Threshold for anomaly detection
        self.threshold = None
        
        # Training stats
        self.train_history = {
            'reconstruction_errors': []
        }
    
    def fit(
        self,
        X: List[List[float]],
        epochs: int = 20,
        learning_rate: float = 0.1,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the anomaly detector on (mostly) normal data.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors
        epochs : int
            Number of training epochs (default: 20)
        learning_rate : float
            Learning rate for training (default: 0.1)
        verbose : bool
            Whether to print progress (default: True)
            
        Returns:
        --------
        Dict[str, List[float]]
            Training history with reconstruction errors
        """
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        # Train the network to reconstruct normal data
        for epoch in range(epochs):
            epoch_error = 0.0
            
            # Shuffle data
            indices = list(range(len(X_normalized)))
            random.shuffle(indices)
            
            # Process each sample
            for idx in indices:
                # Forward pass
                outputs = self.network.forward(X_normalized[idx])
                
                # Calculate reconstruction error
                errors = [o - x for o, x in zip(outputs, X_normalized[idx])]
                mse = sum(e ** 2 for e in errors) / len(errors)
                epoch_error += mse
                
                # Network adapts internally via signals
            
            # Calculate epoch metrics
            avg_error = epoch_error / len(X_normalized)
            
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - reconstruction_error: {avg_error:.6f}")
            
            # Save metrics
            self.train_history['reconstruction_errors'].append(avg_error)
        
        # Calculate reconstruction errors for all samples
        reconstruction_errors = []
        for features in X_normalized:
            outputs = self.network.forward(features)
            errors = [o - x for o, x in zip(outputs, features)]
            mse = sum(e ** 2 for e in errors) / len(errors)
            reconstruction_errors.append(mse)
        
        # Set threshold based on contamination
        if reconstruction_errors:
            sorted_errors = sorted(reconstruction_errors)
            threshold_idx = int((1 - self.contamination) * len(sorted_errors))
            self.threshold = sorted_errors[threshold_idx]
        else:
            self.threshold = 0.1  # Default threshold
        
        return self.train_history
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Predict if samples are anomalies.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors
            
        Returns:
        --------
        List[int]
            Binary labels (1 for anomalies, 0 for normal samples)
        """
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        predictions = []
        for features in X_normalized:
            # Forward pass
            outputs = self.network.forward(features)
            
            # Calculate reconstruction error
            errors = [o - x for o, x in zip(outputs, features)]
            mse = sum(e ** 2 for e in errors) / len(errors)
            
            # Compare with threshold
            prediction = 1 if mse > self.threshold else 0
            predictions.append(prediction)
        
        return predictions
    
    def decision_function(self, X: List[List[float]]) -> List[float]:
        """
        Calculate anomaly scores for samples.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors
            
        Returns:
        --------
        List[float]
            Anomaly scores (higher values indicate more anomalous samples)
        """
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        scores = []
        for features in X_normalized:
            # Forward pass
            outputs = self.network.forward(features)
            
            # Calculate reconstruction error
            errors = [o - x for o, x in zip(outputs, features)]
            mse = sum(e ** 2 for e in errors) / len(errors)
            
            # Convert to anomaly score
            scores.append(mse)
        
        return scores
    
    def evaluate(self, X: List[List[float]], y: List[int]) -> Dict[str, float]:
        """
        Evaluate the anomaly detector on labeled data.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors
        y : List[int]
            True labels (1 for anomalies, 0 for normal samples)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {}
        
        # Confusion matrix
        confusion = defaultdict(int)
        for pred, true in zip(y_pred, y):
            confusion[(true, pred)] += 1
        
        # Extract counts
        tp = confusion.get((1, 1), 0)  # True positive (anomaly correctly identified)
        fp = confusion.get((0, 1), 0)  # False positive (normal marked as anomaly)
        tn = confusion.get((0, 0), 0)  # True negative (normal correctly identified)
        fn = confusion.get((1, 0), 0)  # False negative (anomaly missed)
        
        # Calculate metrics
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_score'] = (2 * metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        metrics['confusion_matrix'] = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        
        return metrics
    
    def _normalize_features(self, features: List[List[float]]) -> List[List[float]]:
        """
        Normalize features to the [0, 1] range.
        
        Parameters:
        -----------
        features : List[List[float]]
            Feature vectors
            
        Returns:
        --------
        List[List[float]]
            Normalized feature vectors
        """
        if not features:
            return []
        
        normalized = []
        
        # Calculate min and max for each feature
        min_values = [float('inf')] * len(features[0])
        max_values = [float('-inf')] * len(features[0])
        
        for sample in features:
            for i, value in enumerate(sample):
                min_values[i] = min(min_values[i], value)
                max_values[i] = max(max_values[i], value)
        
        # Normalize each sample
        for sample in features:
            normalized_sample = []
            for i, value in enumerate(sample):
                range_value = max_values[i] - min_values[i]
                if range_value == 0:
                    normalized_value = 0.5  # Default value if range is zero
                else:
                    normalized_value = (value - min_values[i]) / range_value
                normalized_sample.append(normalized_value)
            normalized.append(normalized_sample)
        
        return normalized
