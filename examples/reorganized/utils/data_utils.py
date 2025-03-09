#!/usr/bin/env python3
"""
Data Utilities for Mycelium Network Examples

This module provides common utility functions for loading datasets,
preprocessing data, and data visualization.
"""

import os
import csv
import random
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional


def load_csv_dataset(
    filename: str, 
    target_col: int = -1,
    header: bool = True,
    delimiter: str = ','
) -> Tuple[List[List[float]], List[Union[float, int]]]:
    """
    Load a dataset from a CSV file.
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file
    target_col : int
        Index of the target column (-1 for last column)
    header : bool
        Whether the file has a header row
    delimiter : str
        Delimiter character for CSV
        
    Returns:
    --------
    Tuple[List[List[float]], List[Union[float, int]]]
        Feature vectors and target values
    """
    features = []
    targets = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        
        # Skip header if present
        if header:
            next(reader, None)
        
        for row in reader:
            if not row:  # Skip empty rows
                continue
            
            # Convert values to float
            try:
                # Extract target
                if target_col < 0:
                    target_col = len(row) + target_col
                
                target = row[target_col]
                
                # Try to convert target to numeric
                try:
                    target = float(target)
                    # Convert to int if it's a whole number (likely a class label)
                    if target.is_integer():
                        target = int(target)
                except ValueError:
                    # Keep as string (could be a class name)
                    pass
                
                # Extract features (all columns except target)
                feature_values = [float(row[i]) for i in range(len(row)) if i != target_col]
                
                features.append(feature_values)
                targets.append(target)
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not process row: {row}. Error: {e}")
    
    return features, targets


def train_test_split(
    features: List[List[float]], 
    targets: List[Any], 
    test_size: float = 0.2,
    stratify: bool = False
) -> Tuple[List[List[float]], List[Any], List[List[float]], List[Any]]:
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    features : List[List[float]]
        Feature vectors
    targets : List[Any]
        Target values or labels
    test_size : float
        Proportion of data to use for testing
    stratify : bool
        Whether to maintain the same class distribution in train and test sets
        
    Returns:
    --------
    Tuple[List[List[float]], List[Any], List[List[float]], List[Any]]
        Train features, train targets, test features, test targets
    """
    if not stratify:
        # Simple random split
        indices = list(range(len(features)))
        random.shuffle(indices)
        
        split_idx = int(len(features) * (1 - test_size))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
    else:
        # Stratified split (for classification)
        class_indices = {}
        
        # Group indices by class
        for i, target in enumerate(targets):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(i)
        
        train_indices = []
        test_indices = []
        
        # Split each class proportionally
        for class_label, indices in class_indices.items():
            random.shuffle(indices)
            
            n_test = max(1, int(len(indices) * test_size))
            n_train = len(indices) - n_test
            
            train_indices.extend(indices[:n_train])
            test_indices.extend(indices[n_train:])
    
    # Extract data
    X_train = [features[i] for i in train_indices]
    y_train = [targets[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    y_test = [targets[i] for i in test_indices]
    
    return X_train, y_train, X_test, y_test


def normalize_features(
    features: List[List[float]],
    return_params: bool = False
) -> Union[List[List[float]], Tuple[List[List[float]], Dict]]:
    """
    Normalize features to the [0, 1] range.
    
    Parameters:
    -----------
    features : List[List[float]]
        Feature vectors
    return_params : bool
        Whether to return normalization parameters
        
    Returns:
    --------
    Union[List[List[float]], Tuple[List[List[float]], Dict]]
        Normalized feature vectors, and optionally normalization parameters
    """
    if not features or not features[0]:
        return features
    
    n_features = len(features[0])
    min_values = [float('inf')] * n_features
    max_values = [float('-inf')] * n_features
    
    # Find min and max values for each feature
    for sample in features:
        for i, value in enumerate(sample):
            min_values[i] = min(min_values[i], value)
            max_values[i] = max(max_values[i], value)
    
    # Normalize
    normalized = []
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
    
    if return_params:
        params = {
            'min_values': min_values,
            'max_values': max_values
        }
        return normalized, params
    
    return normalized


def normalize_with_params(
    features: List[List[float]],
    params: Dict
) -> List[List[float]]:
    """
    Normalize features using the provided parameters.
    
    Parameters:
    -----------
    features : List[List[float]]
        Feature vectors
    params : Dict
        Normalization parameters (min_values, max_values)
        
    Returns:
    --------
    List[List[float]]
        Normalized feature vectors
    """
    min_values = params.get('min_values', [])
    max_values = params.get('max_values', [])
    
    if not min_values or not max_values or len(min_values) != len(features[0]):
        raise ValueError("Invalid normalization parameters")
    
    # Normalize
    normalized = []
    for sample in features:
        normalized_sample = []
        for i, value in enumerate(sample):
            range_value = max_values[i] - min_values[i]
            if range_value == 0:
                normalized_value = 0.5
            else:
                normalized_value = (value - min_values[i]) / range_value
            normalized_sample.append(normalized_value)
        normalized.append(normalized_sample)
    
    return normalized


def generate_synthetic_classification_data(
    n_samples: int = 100, 
    n_features: int = 2,
    n_classes: int = 2,
    noise: float = 0.1,
    random_seed: Optional[int] = None
) -> Tuple[List[List[float]], List[int]]:
    """
    Generate synthetic data for classification tasks.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    n_classes : int
        Number of classes
    noise : float
        Amount of noise to add to the data
    random_seed : Optional[int]
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[List[List[float]], List[int]]
        Feature vectors and class labels
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    features = []
    labels = []
    
    # Generate class centroids
    centroids = []
    for _ in range(n_classes):
        centroid = [random.uniform(0, 1) for _ in range(n_features)]
        centroids.append(centroid)
    
    # Generate samples around centroids
    samples_per_class = n_samples // n_classes
    extra_samples = n_samples % n_classes
    
    for class_idx in range(n_classes):
        n_class_samples = samples_per_class + (1 if class_idx < extra_samples else 0)
        
        for _ in range(n_class_samples):
            # Generate sample near centroid with noise
            sample = []
            for j in range(n_features):
                value = centroids[class_idx][j] + random.uniform(-noise, noise)
                sample.append(max(0, min(1, value)))  # Clip to [0, 1]
            
            features.append(sample)
            labels.append(class_idx)
    
    # Shuffle data
    combined = list(zip(features, labels))
    random.shuffle(combined)
    features, labels = zip(*combined)
    
    return list(features), list(labels)


def generate_synthetic_regression_data(
    n_samples: int = 100,
    n_features: int = 2,
    complexity: float = 0.5,
    noise: float = 0.1,
    random_seed: Optional[int] = None
) -> Tuple[List[List[float]], List[float]]:
    """
    Generate synthetic data for regression tasks.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    complexity : float
        Complexity of the target function (0.0 to 1.0)
    noise : float
        Amount of noise to add to the targets
    random_seed : Optional[int]
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[List[List[float]], List[float]]
        Feature vectors and target values
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    features = []
    targets = []
    
    # Generate random feature vectors
    for _ in range(n_samples):
        feature_vector = [random.uniform(0, 1) for _ in range(n_features)]
        features.append(feature_vector)
    
    # Generate coefficients for the target function
    coefficients = []
    for _ in range(n_features):
        coef = random.uniform(-1, 1)
        coefficients.append(coef)
    
    # Interaction terms (for higher complexity)
    interaction_terms = []
    if complexity > 0.3:
        n_interactions = int(complexity * n_features * (n_features - 1) / 2)
        for _ in range(n_interactions):
            i = random.randint(0, n_features - 1)
            j = random.randint(0, n_features - 1)
            while i == j:
                j = random.randint(0, n_features - 1)
            interaction_terms.append((i, j, random.uniform(-0.5, 0.5)))
    
    # Generate target values
    for feature_vector in features:
        # Linear combination
        target = sum(coef * feature_vector[i] for i, coef in enumerate(coefficients))
        
        # Add non-linear terms based on complexity
        if complexity > 0.1:
            for i, feat_val in enumerate(feature_vector):
                target += complexity * math.sin(feat_val * math.pi) * coefficients[i]
        
        # Add interaction terms
        for i, j, coef in interaction_terms:
            target += coef * feature_vector[i] * feature_vector[j]
        
        # Add noise
        target += random.normalvariate(0, noise)
        
        targets.append(target)
    
    return features, targets
