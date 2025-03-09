#!/usr/bin/env python3
"""
Demonstration of genetic algorithm optimization for mycelium networks.

This script shows how to use the GeneticOptimizer to evolve better
mycelium networks for specific tasks.
"""

import os
import sys
import time
import random
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
from mycelium.enhanced.ml.genetic import NetworkGenome, GeneticOptimizer
from mycelium.visualization.network_visualizer import NetworkVisualizer
from mycelium.visualization.performance_visualizer import PerformanceVisualizer


def generate_classification_dataset(n_samples=200, n_features=2, n_classes=2, random_state=42):
    """Generate a synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features,
        n_classes=n_classes,
        n_redundant=0,
        n_informative=n_features,
        random_state=random_state,
        n_clusters_per_class=1
    )
    
    # Scale features to [0, 1] range
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    return X, y


def classification_fitness(network, X, y, threshold=0.5):
    """
    Fitness function for classification tasks.
    
    Args:
        network: Network to evaluate
        X: Input features
        y: Target labels (0 or 1)
        threshold: Classification threshold
        
    Returns:
        Fitness score (accuracy)
    """
    correct = 0
    
    for features, label in zip(X, y):
        # Forward pass
        output = network.forward(features.tolist())[0]
        
        # Classify
        prediction = 1 if output > threshold else 0
        
        # Check accuracy
        if prediction == label:
            correct += 1
    
    # Return accuracy
    return correct / len(X)


def run_genetic_optimization():
    """Run the genetic optimization demo."""
    print("Genetic Optimization Demo")
    print("=========================")
    
    # Create dataset
    print("\nGenerating classification dataset...")
    X, y = generate_classification_dataset(n_samples=200, n_features=2)
    
    # Split into train/test sets
    train_size = int(0.7 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    print(f"Dataset: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Create environment
    env = RichEnvironment(dimensions=2)
    
    # Create template network
    template = AdaptiveMyceliumNetwork(
        environment=env,
        input_size=2,
        output_size=1,
        initial_nodes=10
    )
    
    # Create genetic optimizer
    print("\nInitializing genetic optimizer...")
    optimizer = GeneticOptimizer(
        environment=env,
        population_size=20,
        mutation_rate=0.2,
        crossover_rate=0.7,
        elite_percentage=0.1
    )
    
    # Initialize population
    optimizer.initialize_population(template)
    
    # Run optimization
    num_generations = 15
    print(f"\nRunning optimization for {num_generations} generations...")
    
    for generation in range(num_generations):
        # Evaluate fitness using training data
        avg_fitness = optimizer.evaluate_fitness(
            lambda net: classification_fitness(net, X_train, y_train)
        )
        
        print(f"Generation {generation + 1}/{num_generations}:")
        print(f"  Average fitness: {avg_fitness:.4f}")
        print(f"  Best fitness: {optimizer.best_fitness:.4f}")
        
        # Evolve to next generation (except for last iteration)
        if generation < num_generations - 1:
            optimizer.evolve()
    
    # Get best network
    best_network = optimizer.get_best_network()
    
    # Evaluate on test set
    test_accuracy = classification_fitness(best_network, X_test, y_test)
    print(f"\nTest accuracy of best network: {test_accuracy:.4f}")
    
    # Visualize optimization progress
    perf_visualizer = PerformanceVisualizer()
    perf_visualizer.plot_genetic_optimization(
        optimizer.fitness_history,
        title="Genetic Optimization Progress",
        save_path="genetic_optimization_progress.png"
    )
    
    # Visualize best network
    net_visualizer = NetworkVisualizer()
    net_visualizer.visualize_network(
        best_network,
        title=f"Best Network (Test Accuracy: {test_accuracy:.4f})",
        show_connections=True,
        save_path="best_network.png"
    )
    
    # Visualize classification decision boundary
    visualize_decision_boundary(best_network, X, y)


def visualize_decision_boundary(network, X, y, resolution=100, save_path="decision_boundary.png"):
    """Visualize the decision boundary of the network."""
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Predict on meshgrid
    Z = np.zeros(xx.shape)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = network.forward([xx[i, j], yy[i, j]])[0]
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    
    # Plot training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Network Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    plt.tight_layout()
    plt.savefig(f"visualizations/{save_path}", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_genetic_optimization()
