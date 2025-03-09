#!/usr/bin/env python3
"""
Simple test for genetic algorithm optimization without visualization.
"""

import os
import sys
import time
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium.environment import Environment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
from mycelium.enhanced.ml.genetic import NetworkGenome, GeneticOptimizer


def test_genetic_optimization():
    """Test genetic algorithm optimization for XOR problem."""
    print("Testing Genetic Optimization for XOR")
    print("====================================")
    
    # Create environment
    env = Environment()
    
    # Create genetic optimizer with improved parameters
    print("\nInitializing genetic optimizer...")
    optimizer = GeneticOptimizer(
        environment=env,
        population_size=50,  # Increased population size
        mutation_rate=0.3,   # Slightly higher mutation rate
        crossover_rate=0.8,  # Higher crossover rate
        elite_percentage=0.1 # Lower elite percentage to increase diversity
    )
    
    # Initialize population with XOR-specialized template
    print("\nInitializing population with XOR-specific topologies...")
    optimizer.initialize_population(problem_type="xor")
    
    # Define XOR fitness function with improved scoring
    def xor_fitness(network):
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        targets = [0, 1, 1, 0]
        
        # Calculate both accuracy and MSE for better fitness function
        correct = 0
        total_error = 0
        
        for input_vec, target in zip(inputs, targets):
            output = network.forward(input_vec)[0]
            prediction = 1 if output > 0.5 else 0
            
            # Accuracy component
            if prediction == target:
                correct += 1
            
            # MSE component (how close outputs are to ideal values)
            desired_output = 0.0 if target == 0 else 1.0
            error = (output - desired_output) ** 2
            total_error += error
        
        # Combined fitness: 70% accuracy, 30% inverse error
        accuracy = correct / len(inputs)
        mse = total_error / len(inputs)
        error_component = 1.0 / (1.0 + mse)  # Convert error to 0-1 range (higher is better)
        
        # Combined fitness score
        return (0.7 * accuracy) + (0.3 * error_component)
    
    # Run optimization for more generations to allow better learning
    num_generations = 30
    print(f"\nRunning optimization for {num_generations} generations...")
    
    for generation in range(num_generations):
        # Evaluate fitness
        avg_fitness = optimizer.evaluate_fitness(xor_fitness)
        
        print(f"Generation {generation + 1}:")
        print(f"  Average fitness: {avg_fitness:.4f}")
        print(f"  Best fitness: {optimizer.best_fitness:.4f}")
        
        # Evolve to next generation (except for last iteration)
        if generation < num_generations - 1:
            optimizer.evolve()
    
    # Get best network
    best_network = optimizer.get_best_network()
    
    # Test best network on XOR with detailed logging
    print("\nTesting best network on XOR problem:")
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_targets = [0, 1, 1, 0]
    correct = 0
    
    for input_vec, target in zip(inputs, xor_targets):
        # Print network state before forward pass to debug
        print(f"\nInput: {input_vec}, Target: {target}")
        
        # Log the initial state of the network
        print(f"Network has {len(best_network.nodes)} nodes")
        print(f"Input nodes: {best_network.input_nodes}")
        print(f"Output nodes: {best_network.output_nodes}")
        
        # Perform forward pass
        output = best_network.forward(input_vec)[0]
        prediction = 1 if output > 0.5 else 0
        
        # Track accuracy
        if prediction == target:
            correct += 1
        
        # Log results
        print(f"Output: {output:.4f}, Prediction: {prediction}, {'Correct' if prediction == target else 'Incorrect'}")
        
    # Print overall accuracy
    accuracy = correct / len(inputs)
    print(f"\nOverall XOR accuracy: {accuracy:.2f} ({correct}/{len(inputs)})")


if __name__ == "__main__":
    test_genetic_optimization()
