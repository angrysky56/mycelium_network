"""
Mycelium-based neural network regressor.

This module implements a regressor that leverages the mycelium network
architecture for regression tasks.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Union, Optional
from collections import defaultdict

from mycelium.network import AdvancedMyceliumNetwork
from mycelium.environment import Environment


class MyceliumRegressor:
    """
    Mycelium-based neural network regressor.
    
    This class adapts the mycelium network for regression tasks,
    with specialized methods for training, evaluation, and prediction.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        hidden_nodes: int = 20,
        environment: Optional[Environment] = None
    ):
        """
        Initialize the mycelium regressor.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        output_size : int
            Number of output values to predict (default: 1)
        hidden_nodes : int
            Number of initial hidden nodes (default: 20)
        environment : Environment, optional
            Custom environment for the network
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize the network
        self.network = AdvancedMyceliumNetwork(
            input_size=input_size,
            output_size=output_size,
            initial_nodes=hidden_nodes,
            environment=environment
        )
        
        # Training metrics
        self.train_history = {
            'mse': [],
            'mae': []
        }
        
        # Target scaling
        self.y_min = None
        self.y_max = None
    
    def fit(
        self,
        X: List[List[float]],
        y: List[Union[float, List[float]]],
        epochs: int = 10,
        learning_rate: float = 0.1,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the regressor on labeled data.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors
        y : List[Union[float, List[float]]]
            Target values
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
            Training history with error metrics
        """
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        # Normalize targets
        y_normalized = self._normalize_targets(y)
        
        # Split data if validation is requested
        if validation_split > 0:
            split_idx = int(len(X_normalized) * (1 - validation_split))
            indices = list(range(len(X_normalized)))
            random.shuffle(indices)
            
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            X_train = [X_normalized[i] for i in train_indices]
            y_train = [y_normalized[i] for i in train_indices]
            
            X_val = [X_normalized[i] for i in val_indices]
            y_val = [y_normalized[i] for i in val_indices]
        else:
            X_train, y_train = X_normalized, y_normalized
            X_val, y_val = [], []
        
        # Train the network
        for epoch in range(epochs):
            epoch_mse = 0.0
            epoch_mae = 0.0
            
            # Create training pairs and shuffle
            train_data = list(zip(X_train, y_train))
            random.shuffle(train_data)
            
            # Process each training sample
            for features, target in train_data:
                # Forward pass
                outputs = self.network.forward(features)
                
                # Calculate errors
                if self.output_size == 1:
                    # Single output regression
                    target_value = target[0] if isinstance(target, list) else target
                    error = outputs[0] - target_value
                    mse = error ** 2
                    mae = abs(error)
                else:
                    # Multi-output regression
                    errors = [o - t for o, t in zip(outputs, target)]
                    mse = sum(e ** 2 for e in errors) / len(errors)
                    mae = sum(abs(e) for e in errors) / len(errors)
                
                epoch_mse += mse
                epoch_mae += mae
                
                # Network handles adaptation internally via signals
            
            # Calculate epoch metrics
            train_mse = epoch_mse / len(X_train) if X_train else 0
            train_mae = epoch_mae / len(X_train) if X_train else 0
            
            # Update learning rate over time
            current_lr = learning_rate * (1 / (1 + 0.1 * epoch))
            
            # Validate if required
            if X_val:
                val_mse = 0.0
                val_mae = 0.0
                
                for features, target in zip(X_val, y_val):
                    outputs = self.network.forward(features)
                    
                    if self.output_size == 1:
                        target_value = target[0] if isinstance(target, list) else target
                        error = outputs[0] - target_value
                        mse = error ** 2
                        mae = abs(error)
                    else:
                        errors = [o - t for o, t in zip(outputs, target)]
                        mse = sum(e ** 2 for e in errors) / len(errors)
                        mae = sum(abs(e) for e in errors) / len(errors)
                    
                    val_mse += mse
                    val_mae += mae
                
                val_mse = val_mse / len(X_val)
                val_mae = val_mae / len(X_val)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - mse: {train_mse:.4f} - mae: {train_mae:.4f} - "
                         f"val_mse: {val_mse:.4f} - val_mae: {val_mae:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - mse: {train_mse:.4f} - mae: {train_mae:.4f}")
            
            # Save metrics
            self.train_history['mse'].append(train_mse)
            self.train_history['mae'].append(train_mae)
        
        return self.train_history
    
    def predict(self, X: List[List[float]]) -> Union[List[float], List[List[float]]]:
        """
        Make predictions for input samples.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors
            
        Returns:
        --------
        Union[List[float], List[List[float]]]
            Predicted values
        """
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        predictions = []
        for features in X_normalized:
            # Forward pass
            outputs = self.network.forward(features)
            
            # Denormalize outputs
            if self.output_size == 1:
                # Single output
                denormalized = self._denormalize_output(outputs[0])
                predictions.append(denormalized)
            else:
                # Multi-output
                denormalized = [self._denormalize_output(o, i) for i, o in enumerate(outputs)]
                predictions.append(denormalized)
        
        return predictions
    
    def evaluate(self, X: List[List[float]], y: List[Union[float, List[float]]]) -> Dict[str, float]:
        """
        Evaluate the regressor on test data.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors
        y : List[Union[float, List[float]]]
            True values
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {}
        
        # Mean squared error
        if self.output_size == 1:
            # Single output
            mse = sum((pred - true) ** 2 for pred, true in zip(y_pred, y)) / len(y)
            mae = sum(abs(pred - true) for pred, true in zip(y_pred, y)) / len(y)
            
            # R-squared
            y_mean = sum(y) / len(y)
            ss_total = sum((true - y_mean) ** 2 for true in y)
            ss_residual = sum((true - pred) ** 2 for true, pred in zip(y, y_pred))
            r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        else:
            # Multi-output
            mse = sum(sum((p - t) ** 2 for p, t in zip(pred, true)) / len(pred) 
                      for pred, true in zip(y_pred, y)) / len(y)
            mae = sum(sum(abs(p - t) for p, t in zip(pred, true)) / len(pred)
                      for pred, true in zip(y_pred, y)) / len(y)
            
            # R-squared calculation for multi-output is more complex
            # This is a simplified version
            r2 = 0.0
        
        metrics['mse'] = mse
        metrics['mae'] = mae
        metrics['r2'] = r2
        
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
    
    def _normalize_targets(self, targets: List[Union[float, List[float]]]) -> List[Union[float, List[float]]]:
        """
        Normalize target values to the [0, 1] range.
        
        Parameters:
        -----------
        targets : List[Union[float, List[float]]]
            Target values
            
        Returns:
        --------
        List[Union[float, List[float]]]
            Normalized target values
        """
        if not targets:
            return []
        
        # Determine if targets are scalar or vector
        is_multioutput = isinstance(targets[0], list)
        
        if is_multioutput:
            # Multi-output targets
            output_size = len(targets[0])
            
            # Track min and max for each output
            self.y_min = [float('inf')] * output_size
            self.y_max = [float('-inf')] * output_size
            
            # Find min and max
            for target in targets:
                for i, value in enumerate(target):
                    self.y_min[i] = min(self.y_min[i], value)
                    self.y_max[i] = max(self.y_max[i], value)
            
            # Normalize
            normalized = []
            for target in targets:
                norm_values = []
                for i, value in enumerate(target):
                    range_value = self.y_max[i] - self.y_min[i]
                    if range_value == 0:
                        norm_value = 0.5
                    else:
                        norm_value = (value - self.y_min[i]) / range_value
                    norm_values.append(norm_value)
                normalized.append(norm_values)
        else:
            # Single output targets
            self.y_min = min(targets)
            self.y_max = max(targets)
            
            # Normalize
            range_value = self.y_max - self.y_min
            if range_value == 0:
                normalized = [0.5] * len(targets)
            else:
                normalized = [(y - self.y_min) / range_value for y in targets]
        
        return normalized
    
    def _denormalize_output(self, value: float, output_idx: int = 0) -> float:
        """
        Denormalize a network output back to the original scale.
        
        Parameters:
        -----------
        value : float
            Normalized output value
        output_idx : int
            Index of the output (for multi-output regression)
            
        Returns:
        --------
        float
            Denormalized value
        """
        if self.output_size == 1:
            # Single output
            return value * (self.y_max - self.y_min) + self.y_min
        else:
            # Multi-output
            return value * (self.y_max[output_idx] - self.y_min[output_idx]) + self.y_min[output_idx]
