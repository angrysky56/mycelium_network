"""
Task-specific modules for the mycelium network.

This package contains specialized classes for common machine learning tasks:
- Classification
- Regression
- Anomaly detection
"""

from mycelium.tasks.classifier import MyceliumClassifier
from mycelium.tasks.regressor import MyceliumRegressor
from mycelium.tasks.anomaly_detector import MyceliumAnomalyDetector

__all__ = ['MyceliumClassifier', 'MyceliumRegressor', 'MyceliumAnomalyDetector']
