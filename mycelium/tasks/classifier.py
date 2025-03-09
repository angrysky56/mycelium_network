"""
Mycelium-based neural network classifier.

This module implements a classifier that leverages the mycelium network
architecture for classification tasks.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Union, Optional
from collections import defaultdict

from mycelium.network import AdvancedMyceliumNetwork
from mycelium.environment import Environment


class MyceliumClassifier:
    """
    Mycelium-based neural network classifier.
    
    This class adapts the mycelium network for classification tasks,
    with specialized methods for training, evaluation, and prediction.
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 2,
        hidden_nodes: int = 20,
        environment: Optional[Environment] = None
    ):
        """
        Initialize the mycelium classifier.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        num_classes : int
            Number of output classes (default: 2 for binary classification)
        hidden_nodes : int
            Number of initial hidden nodes (default: 20)
        environment : Environment, optional
            Custom environment for the network
        """
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Initialize the network
        output_size = num_classes if num_classes > 2 else 1
        self.network = AdvancedMyceliumNetwork(
            input_size=input_size,
            output_size=output_size,
            initial_nodes=hidden_nodes,
            environment=environment
        )
        
        # Training metrics
        self.train_history = {
            'accuracy': [],
            'loss': []
        }
    
    def fit(
        self,
        X: List[List[float]],
        y: List[int],
        epochs: int = 10,
        learning_rate: float = 0.1,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the classifier on labeled data.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors
        y : List[int]
            Target labels (class indices)
        epochs : int
            Number of training epochs (default: 10)
        learning_rate : float
            Learning rate for training (default: 0.1)
        validation_split : float
            Proportion of data to use for validation (default: 0.2)
        verbose : bool
            Whether to print progress (default: True)
            
        Returns:
        --------
        Dict[str, List[float]]
            Training history with accuracy and loss metrics
        """
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        # Convert labels to appropriate format
        y_encoded = self._encode_labels(y)
        
        # Split data if validation is requested
        if validation_split > 0:
            split_idx = int(len(X_normalized) * (1 - validation_split))
            indices = list(range(len(X_normalized)))
            random.shuffle(indices)
            
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            X_train = [X_normalized[i] for i in train_indices]
            y_train = [y_encoded[i] for i in train_indices]
            
            X_val = [X_normalized[i] for i in val_indices]
            y_val = [y_encoded[i] for i in val_indices]
        else:
            X_train, y_train = X_normalized, y_encoded
            X_val, y_val = [], []
        
        # Train the network
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            
            # Create training pairs and shuffle
            train_data = list(zip(X_train, y_train))
            random.shuffle(train_data)
            
            # Process each training sample
            for features, target in train_data:
                # Forward pass
                outputs = self.network.forward(features)
                
                # Calculate prediction
                prediction = self._decode_outputs(outputs)
                
                # Calculate loss and update metrics
                if self.num_classes == 2:
                    # Binary classification
                    target_value = target[0] if isinstance(target, list) else target
                    loss = ((outputs[0] - target_value) ** 2) / 2
                    epoch_loss += loss
                    epoch_correct += 1 if prediction == target_value else 0
                else:
                    # Multi-class classification
                    # Use cross-entropy loss
                    loss = -sum(t * np.log(max(o, 1e-10)) for t, o in zip(target, outputs))
                    epoch_loss += loss
                    epoch_correct += 1 if prediction == target.index(max(target)) else 0
            
            # Calculate epoch metrics
            train_accuracy = epoch_correct / len(X_train) if X_train else 0
            train_loss = epoch_loss / len(X_train) if X_train else 0
            
            # Validate if required
            if X_val:
                val_correct = 0
                val_loss = 0.0
                
                for features, target in zip(X_val, y_val):
                    outputs = self.network.forward(features)
                    prediction = self._decode_outputs(outputs)
                    
                    if self.num_classes == 2:
                        target_value = target[0] if isinstance(target, list) else target
                        loss = ((outputs[0] - target_value) ** 2) / 2
                        val_loss += loss
                        val_correct += 1 if prediction == target_value else 0
                    else:
                        loss = -sum(t * np.log(max(o, 1e-10)) for t, o in zip(target, outputs))
                        val_loss += loss
                        val_correct += 1 if prediction == target.index(max(target)) else 0
                
                val_accuracy = val_correct / len(X_val)
                val_loss = val_loss / len(X_val)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - "
                         f"val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f}")
            
            # Save metrics
            self.train_history['accuracy'].append(train_accuracy)
            self.train_history['loss'].append(train_loss)
        
        return self.train_history
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Make predictions for input samples.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors
            
        Returns:
        --------
        List[int]
            Predicted class indices
        """
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        predictions = []
        for features in X_normalized:
            # Forward pass
            outputs = self.network.forward(features)
            
            # Decode outputs to class prediction
            prediction = self._decode_outputs(outputs)
            predictions.append(prediction)
        
        return predictions
    
    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """
        Predict class probabilities for input samples.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors
            
        Returns:
        --------
        List[List[float]]
            Class probabilities for each sample
        """
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        probabilities = []
        for features in X_normalized:
            # Forward pass
            outputs = self.network.forward(features)
            
            # Convert to probabilities
            if self.num_classes == 2:
                # Binary classification
                binary_probs = [1 - outputs[0], outputs[0]]
                probabilities.append(binary_probs)
            else:
                # Multi-class - ensure outputs sum to 1
                total = sum(outputs)
                if total > 0:
                    probs = [o / total for o in outputs]
                else:
                    probs = [1.0 / len(outputs)] * len(outputs)
                probabilities.append(probs)
        
        return probabilities
    
    def evaluate(self, X: List[List[float]], y: List[int]) -> Dict[str, float]:
        """
        Evaluate the classifier on test data.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors
        y : List[int]
            True labels
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {}
        
        # Accuracy
        correct = sum(1 for pred, true in zip(y_pred, y) if pred == true)
        metrics['accuracy'] = correct / len(y) if len(y) > 0 else 0
        
        # For binary classification, calculate more metrics
        if self.num_classes == 2:
            # Confusion matrix
            confusion = defaultdict(int)
            for pred, true in zip(y_pred, y):
                confusion[(true, pred)] += 1
            
            # Extract counts
            tp = confusion.get((1, 1), 0)  # True positive
            fp = confusion.get((0, 1), 0)  # False positive
            tn = confusion.get((0, 0), 0)  # True negative
            fn = confusion.get((1, 0), 0)  # False negative
            
            # Calculate additional metrics
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
    
    def _encode_labels(self, labels: List[int]) -> List[Union[int, List[float]]]:
        """
        Encode class labels for the network.
        
        Parameters:
        -----------
        labels : List[int]
            Class indices
            
        Returns:
        --------
        List[Union[int, List[float]]]
            Encoded labels appropriate for the network
        """
        if self.num_classes == 2:
            # Binary classification - use 0/1 encoding
            return labels
        else:
            # Multi-class - use one-hot encoding
            encoded = []
            for label in labels:
                one_hot = [0.0] * self.num_classes
                one_hot[label] = 1.0
                encoded.append(one_hot)
            return encoded
    
    def _decode_outputs(self, outputs: List[float]) -> int:
        """
        Decode network outputs to class predictions.
        
        Parameters:
        -----------
        outputs : List[float]
            Network output values
            
        Returns:
        --------
        int
            Predicted class index
        """
        if self.num_classes == 2:
            # Binary classification
            return 1 if outputs[0] >= 0.5 else 0
        else:
            # Multi-class classification
            return outputs.index(max(outputs))
