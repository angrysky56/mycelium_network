#!/usr/bin/env python3
"""
Demonstration of batch processing optimizations for mycelium networks.

This script compares the performance of standard network processing vs.
optimized batch processing for forward pass and training operations.
"""

import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium.environment import Environment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
from mycelium.optimized.batch_network import BatchProcessingNetwork
from mycelium.visualization.performance_visualizer import PerformanceVisualizer


def generate_moons_dataset(n_samples=200, noise=0.1, random_state=42):
    """Generate two half-moons dataset."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Scale features to [0, 1] range
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    return X, y


def compare_forward_pass(batch_sizes=[1, 5, 10, 20], iterations=100):
    """Compare forward pass performance with different batch sizes."""
    print("Forward Pass Performance Comparison")
    print("===================================")
    
    # Create environment
    env = Environment()
    
    # Create test input
    input_size = 5
    hidden_nodes = 20
    test_inputs = [[random.random() for _ in range(input_size)] for _ in range(iterations)]
    
    # Times for standard network and batch networks
    standard_times = []
    batch_times = []
    
    # Standard network
    standard_network = AdaptiveMyceliumNetwork(
        environment=env,
        input_size=input_size,
        output_size=1,
        initial_nodes=hidden_nodes
    )
    
    # Time standard network
    start_time = time.time()
    standard_outputs = []
    for test_input in test_inputs:
        output = standard_network.forward(test_input)
        standard_outputs.append(output)
    standard_time = time.time() - start_time
    standard_times.append(standard_time)
    
    print(f"Standard network: {standard_time:.4f} seconds")
    
    # Batch networks with different batch sizes
    for batch_size in batch_sizes:
        # Skip batch_size=1 (already measured as standard)
        if batch_size == 1:
            batch_times.append(standard_time)
            continue
            
        # Create network with specified batch size
        batch_network = BatchProcessingNetwork(
            environment=env,
            input_size=input_size,
            output_size=1,
            initial_nodes=hidden_nodes,
            batch_size=batch_size
        )
        
        # Match connections for fair comparison
        for node_id, node in standard_network.nodes.items():
            if node_id in batch_network.nodes:
                batch_network.nodes[node_id].connections = node.connections.copy()
        
        # Time batch network
        start_time = time.time()
        batch_outputs = []
        for test_input in test_inputs:
            output = batch_network.forward(test_input)
            batch_outputs.append(output)
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        
        print(f"Batch network (size={batch_size}): {batch_time:.4f} seconds")
    
    # Visualize results
    visualizer = PerformanceVisualizer()
    
    # Plot times
    plt.figure(figsize=(10, 6))
    bar_width = 0.7
    x = list(range(len(batch_sizes)))
    labels = ['Standard'] + [f'Batch {s}' for s in batch_sizes[1:]]
    
    plt.bar(x, batch_times, bar_width)
    
    plt.xlabel('Network Type')
    plt.ylabel('Time (seconds)')
    plt.title('Forward Pass Performance Comparison')
    plt.xticks(x, labels)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add speedup labels
    for i, (time_val, label) in enumerate(zip(batch_times, labels)):
        if i > 0:  # Skip standard network
            speedup = standard_times[0] / time_val
            plt.text(i, time_val + 0.01, f'{speedup:.2f}x', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/batch_forward_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return standard_times, batch_times


def compare_training_performance(batch_sizes=[1, 5, 10, 20], epochs=5):
    """Compare training performance with different batch sizes."""
    print("\nTraining Performance Comparison")
    print("==============================")
    
    # Create dataset
    X, y = generate_moons_dataset(n_samples=200)
    
    # Format for network training
    inputs = X.tolist()
    targets = [[float(label)] for label in y]
    
    # Create environment
    env = Environment()
    
    # Create networks
    networks = []
    
    # Standard network (batch_size=1)
    standard_network = AdaptiveMyceliumNetwork(
        environment=env,
        input_size=2,
        output_size=1,
        initial_nodes=10
    )
    networks.append(('Standard', standard_network))
    
    # Batch networks with different batch sizes
    for batch_size in batch_sizes[1:]:  # Skip first (standard)
        batch_network = BatchProcessingNetwork(
            environment=env,
            input_size=2,
            output_size=1,
            initial_nodes=10,
            batch_size=batch_size
        )
        networks.append((f'Batch {batch_size}', batch_network))
    
    # Training times and errors
    training_times = []
    final_errors = []
    
    # Train each network
    for name, network in networks:
        # Time training
        start_time = time.time()
        errors = network.train(inputs, targets, epochs=epochs, learning_rate=0.1)
        elapsed_time = time.time() - start_time
        
        # Record results
        training_times.append(elapsed_time)
        final_errors.append(errors[-1])
        
        print(f"{name}: {elapsed_time:.4f} seconds, final error: {errors[-1]:.4f}")
    
    # Visualize results
    visualizer = PerformanceVisualizer()
    
    # Create labels
    labels = ['Standard'] + [f'Batch {s}' for s in batch_sizes[1:]]
    
    # Plot training times
    plt.figure(figsize=(10, 6))
    bar_width = 0.7
    x = list(range(len(labels)))
    
    plt.bar(x, training_times, bar_width)
    
    plt.xlabel('Network Type')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Performance Comparison')
    plt.xticks(x, labels)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add speedup labels
    for i, time_val in enumerate(training_times):
        if i > 0:  # Skip standard network
            speedup = training_times[0] / time_val
            plt.text(i, time_val + 0.01, f'{speedup:.2f}x', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/batch_training_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot errors
    plt.figure(figsize=(10, 6))
    
    plt.bar(x, final_errors, bar_width)
    
    plt.xlabel('Network Type')
    plt.ylabel('Final Error')
    plt.title('Training Error Comparison')
    plt.xticks(x, labels)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    plt.savefig('visualizations/batch_training_error.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return training_times, final_errors


def visualize_decision_boundaries(networks, labels, X, y, resolution=100):
    """Visualize decision boundaries for multiple networks."""
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(networks), figsize=(5 * len(networks), 5))
    
    # Ensure axes is iterable even with one network
    if len(networks) == 1:
        axes = [axes]
    
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Plot each network's decision boundary
    for i, (network, label) in enumerate(zip(networks, labels)):
        # Predict on meshgrid
        Z = np.zeros(xx.shape)
        for j in range(resolution):
            for k in range(resolution):
                Z[j, k] = 1 if network.forward([xx[j, k], yy[j, k]])[0] > 0.5 else 0
        
        # Plot decision boundary
        axes[i].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
        
        # Plot training points
        axes[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
        
        axes[i].set_xlim(xx.min(), xx.max())
        axes[i].set_ylim(yy.min(), yy.max())
        axes[i].set_title(label)
    
    plt.tight_layout()
    plt.savefig('visualizations/decision_boundaries.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_batch_demo():
    """Run the complete batch processing demo."""
    print("Batch Processing Optimization Demo")
    print("=================================")
    
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Compare forward pass performance
    forward_times, batch_times = compare_forward_pass()
    
    # Compare training performance
    training_times, final_errors = compare_training_performance()
    
    # Train networks on two moons dataset for visualization
    print("\nTraining networks on two moons dataset...")
    
    # Generate dataset
    X, y = generate_moons_dataset(n_samples=200)
    inputs = X.tolist()
    targets = [[float(label)] for label in y]
    
    # Create environment
    env = Environment()
    
    # Create networks
    standard_network = AdaptiveMyceliumNetwork(
        environment=env,
        input_size=2,
        output_size=1,
        initial_nodes=10
    )
    
    batch_network = BatchProcessingNetwork(
        environment=env,
        input_size=2,
        output_size=1,
        initial_nodes=10,
        batch_size=10
    )
    
    # Train networks
    print("Training standard network...")
    standard_errors = standard_network.train(inputs, targets, epochs=10, learning_rate=0.1)
    
    print("Training batch network...")
    batch_errors = batch_network.train(inputs, targets, epochs=10, learning_rate=0.1)
    
    # Visualize training errors
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(standard_errors) + 1), standard_errors, 'b-', label='Standard Network')
    plt.plot(range(1, len(batch_errors) + 1), batch_errors, 'r-', label='Batch Network')
    
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training Error Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('visualizations/training_error_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualize decision boundaries
    visualize_decision_boundaries(
        [standard_network, batch_network],
        ['Standard Network', 'Batch Network'],
        X, y
    )


if __name__ == "__main__":
    run_batch_demo()
