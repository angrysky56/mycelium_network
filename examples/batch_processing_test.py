#!/usr/bin/env python3
"""
Simple test for batch processing optimization without visualization.
"""

import os
import sys
import time
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium.environment import Environment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
from mycelium.optimized.batch_network import BatchProcessingNetwork


def test_forward_performance():
    """Test forward pass performance."""
    print("\nTesting Forward Pass Performance")
    print("===============================")
    
    # Create environment
    env = Environment()
    
    # Create test inputs
    input_size = 5
    hidden_nodes = 20
    iterations = 100
    test_inputs = [[random.random() for _ in range(input_size)] for _ in range(iterations)]
    
    # Create networks
    standard_network = AdaptiveMyceliumNetwork(
        environment=env,
        input_size=input_size,
        output_size=1,
        initial_nodes=hidden_nodes
    )
    
    batch_sizes = [5, 10, 20]
    batch_networks = {}
    
    for batch_size in batch_sizes:
        batch_networks[batch_size] = BatchProcessingNetwork(
            environment=env,
            input_size=input_size,
            output_size=1,
            initial_nodes=hidden_nodes,
            batch_size=batch_size
        )
        
        # Match connections for fair comparison
        for node_id, node in standard_network.nodes.items():
            if node_id in batch_networks[batch_size].nodes:
                batch_networks[batch_size].nodes[node_id].connections = node.connections.copy()
    
    # Test standard network
    print("\nTesting standard network...")
    start_time = time.time()
    standard_outputs = []
    for test_input in test_inputs:
        output = standard_network.forward(test_input)
        standard_outputs.append(output)
    standard_time = time.time() - start_time
    print(f"  Time: {standard_time:.4f} seconds")
    
    # Test batch networks
    for batch_size, batch_network in batch_networks.items():
        print(f"\nTesting batch network (size={batch_size})...")
        start_time = time.time()
        batch_outputs = []
        for test_input in test_inputs:
            output = batch_network.forward(test_input)
            batch_outputs.append(output)
        batch_time = time.time() - start_time
        
        # Calculate speedup
        speedup = standard_time / batch_time
        
        print(f"  Time: {batch_time:.4f} seconds")
        print(f"  Speedup: {speedup:.2f}x")


def test_training_performance():
    """Test training performance."""
    print("\nTesting Training Performance")
    print("===========================")
    
    # Create environment
    env = Environment()
    
    # Create networks for training
    input_size = 2  # For XOR problem
    hidden_nodes = 10
    
    standard_network = AdaptiveMyceliumNetwork(
        environment=env,
        input_size=input_size,
        output_size=1,
        initial_nodes=hidden_nodes
    )
    
    batch_sizes = [5, 10, 20]
    batch_networks = {}
    
    for batch_size in batch_sizes:
        batch_networks[batch_size] = BatchProcessingNetwork(
            environment=env,
            input_size=input_size,
            output_size=1,
            initial_nodes=hidden_nodes,
            batch_size=batch_size
        )
    
    # Create simple dataset (XOR-like problem)
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]
    
    # Replicate dataset to make it larger
    X_large = X * 25
    y_large = y * 25
    
    # Train standard network
    print("\nTraining standard network...")
    start_time = time.time()
    standard_errors = standard_network.train(X_large, y_large, epochs=5, learning_rate=0.1)
    standard_time = time.time() - start_time
    print(f"  Time: {standard_time:.4f} seconds")
    print(f"  Final error: {standard_errors[-1]:.4f}")
    
    # Train batch networks
    for batch_size, batch_network in batch_networks.items():
        print(f"\nTraining batch network (size={batch_size})...")
        start_time = time.time()
        batch_errors = batch_network.train(X_large, y_large, epochs=5, learning_rate=0.1)
        batch_time = time.time() - start_time
        
        # Calculate speedup
        speedup = standard_time / batch_time
        
        print(f"  Time: {batch_time:.4f} seconds")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Final error: {batch_errors[-1]:.4f}")


def test_batch_processing():
    """Run all batch processing tests."""
    print("Testing Batch Processing")
    print("=======================")
    
    # Test forward pass performance
    test_forward_performance()
    
    # Test training performance
    test_training_performance()


if __name__ == "__main__":
    test_batch_processing()
