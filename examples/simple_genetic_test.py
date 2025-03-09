#!/usr/bin/env python3
"""
Simplified test for genetic algorithm implementation.
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


def simple_xor_test():
    """Simple XOR test to verify genetic algorithm functionality."""
    print("Simple XOR Genetic Optimization Test")
    print("===================================")
    
    # Create environment
    env = Environment()
    
    # Create genetic optimizer
    print("\nInitializing genetic optimizer...")
    optimizer = GeneticOptimizer(
        environment=env,
        population_size=30,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elite_percentage=0.1
    )
    
    # Initialize population
    print("\nInitializing population...")
    optimizer.initialize_population(problem_type="xor")
    
    # Define XOR fitness function
    def xor_fitness(network):
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        targets = [0, 1, 1, 0]
        
        correct = 0
        total_error = 0
        
        for input_vec, target in zip(inputs, targets):
            output = network.forward(input_vec)[0]
            prediction = 1 if output > 0.5 else 0
            
            if prediction == target:
                correct += 1
            
            desired_output = 0.0 if target == 0 else 1.0
            error = (output - desired_output) ** 2
            total_error += error
        
        # Combined fitness
        accuracy = correct / len(inputs)
        mse = total_error / len(inputs)
        error_component = 1.0 / (1.0 + mse)
        
        return (0.7 * accuracy) + (0.3 * error_component)
    
    # Run optimization
    num_generations = 20
    print(f"\nRunning optimization for {num_generations} generations...")
    
    for generation in range(num_generations):
        # Evaluate fitness
        avg_fitness = optimizer.evaluate_fitness(xor_fitness)
        
        print(f"Generation {generation + 1}/{num_generations}:")
        print(f"  Average fitness: {avg_fitness:.4f}")
        print(f"  Best fitness: {optimizer.best_fitness:.4f}")
        
        # Evolve to next generation (except for last iteration)
        if generation < num_generations - 1:
            optimizer.evolve()
    
    # Test best network
    best_network = optimizer.get_best_network()
    
    print("\nTesting best network:")
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_targets = [0, 1, 1, 0]
    
    for input_vec, target in zip(inputs, xor_targets):
        output = best_network.forward(input_vec)[0]
        prediction = 1 if output > 0.5 else 0
        
        print(f"Input: {input_vec}, Output: {output:.4f}, Prediction: {prediction}, Target: {target}")


if __name__ == "__main__":
    simple_xor_test()
