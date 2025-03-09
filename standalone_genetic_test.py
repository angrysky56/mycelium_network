#!/usr/bin/env python3
"""
Standalone test for genetic algorithm concept.
"""

import random
import copy
import math
from typing import List, Dict, Any, Callable, Tuple, Optional

class SimpleNetwork:
    """A simple feed-forward neural network for XOR."""
    
    def __init__(self, num_inputs=2, num_hidden=2, num_outputs=1):
        self.weights1 = [[random.uniform(-1.0, 1.0) for _ in range(num_hidden)] for _ in range(num_inputs)]
        self.biases1 = [random.uniform(-1.0, 1.0) for _ in range(num_hidden)]
        self.weights2 = [[random.uniform(-1.0, 1.0) for _ in range(num_outputs)] for _ in range(num_hidden)]
        self.biases2 = [random.uniform(-1.0, 1.0) for _ in range(num_outputs)]
        
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + math.exp(-x))
    
    def forward(self, inputs):
        """Forward pass through the network."""
        # Hidden layer
        hidden = []
        for i in range(len(self.biases1)):
            value = self.biases1[i]
            for j in range(len(inputs)):
                value += inputs[j] * self.weights1[j][i]
            hidden.append(self.sigmoid(value))
        
        # Output layer
        outputs = []
        for i in range(len(self.biases2)):
            value = self.biases2[i]
            for j in range(len(hidden)):
                value += hidden[j] * self.weights2[j][i]
            outputs.append(self.sigmoid(value))
            
        return outputs


class NetworkGenome:
    """Genome representation for a simple network."""
    
    def __init__(self, network=None):
        """Initialize from a network or create random genome."""
        if network is None:
            # Create random genome
            self.weights1 = [[random.uniform(-1.0, 1.0) for _ in range(2)] for _ in range(2)]
            self.biases1 = [random.uniform(-1.0, 1.0) for _ in range(2)]
            self.weights2 = [[random.uniform(-1.0, 1.0) for _ in range(1)] for _ in range(2)]
            self.biases2 = [random.uniform(-1.0, 1.0) for _ in range(1)]
        else:
            # Extract from network
            self.weights1 = copy.deepcopy(network.weights1)
            self.biases1 = copy.deepcopy(network.biases1)
            self.weights2 = copy.deepcopy(network.weights2)
            self.biases2 = copy.deepcopy(network.biases2)
    
    def to_network(self):
        """Convert genome back to network."""
        network = SimpleNetwork(
            num_inputs=len(self.weights1),
            num_hidden=len(self.biases1),
            num_outputs=len(self.biases2)
        )
        
        network.weights1 = copy.deepcopy(self.weights1)
        network.biases1 = copy.deepcopy(self.biases1)
        network.weights2 = copy.deepcopy(self.weights2)
        network.biases2 = copy.deepcopy(self.biases2)
        
        return network


class GeneticOptimizer:
    """Genetic algorithm for optimizing networks."""
    
    def __init__(self, population_size=10, mutation_rate=0.2, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population = []
        self.fitness_scores = []
        
    def initialize_population(self):
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            network = SimpleNetwork()
            genome = NetworkGenome(network)
            self.population.append(genome)
            
    def evaluate_fitness(self, fitness_function):
        """Evaluate fitness for all genomes."""
        self.fitness_scores = []
        
        for genome in self.population:
            network = genome.to_network()
            fitness = fitness_function(network)
            self.fitness_scores.append(fitness)
            
        # Sort population by fitness
        combined = list(zip(self.population, self.fitness_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        
        self.population, self.fitness_scores = zip(*combined)
        self.population = list(self.population)
        self.fitness_scores = list(self.fitness_scores)
        
        return sum(self.fitness_scores) / len(self.fitness_scores)
    
    def selection(self):
        """Select genome based on fitness (tournament selection)."""
        tournament_size = max(2, self.population_size // 5)
        tournament = random.sample(range(self.population_size), tournament_size)
        
        best_idx = tournament[0]
        best_fitness = self.fitness_scores[best_idx]
        
        for idx in tournament[1:]:
            if self.fitness_scores[idx] > best_fitness:
                best_idx = idx
                best_fitness = self.fitness_scores[idx]
                
        return copy.deepcopy(self.population[best_idx])
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two genomes."""
        if len(parent1.weights1) != len(parent2.weights1) or len(parent1.weights2) != len(parent2.weights2):
            # Can't cross different structures, just return parents
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1 = NetworkGenome()
        child2 = NetworkGenome()
        
        # Crossover weights1
        crossover_point = random.randint(0, len(parent1.weights1))
        child1.weights1 = parent1.weights1[:crossover_point] + parent2.weights1[crossover_point:]
        child2.weights1 = parent2.weights1[:crossover_point] + parent1.weights1[crossover_point:]
        
        # Crossover biases1
        crossover_point = random.randint(0, len(parent1.biases1))
        child1.biases1 = parent1.biases1[:crossover_point] + parent2.biases1[crossover_point:]
        child2.biases1 = parent2.biases1[:crossover_point] + parent1.biases1[crossover_point:]
        
        # Crossover weights2
        crossover_point = random.randint(0, len(parent1.weights2))
        child1.weights2 = parent1.weights2[:crossover_point] + parent2.weights2[crossover_point:]
        child2.weights2 = parent2.weights2[:crossover_point] + parent1.weights2[crossover_point:]
        
        # Crossover biases2
        crossover_point = random.randint(0, len(parent1.biases2))
        child1.biases2 = parent1.biases2[:crossover_point] + parent2.biases2[crossover_point:]
        child2.biases2 = parent2.biases2[:crossover_point] + parent1.biases2[crossover_point:]
        
        return child1, child2
    
    def mutate(self, genome):
        """Mutate a genome."""
        # Mutate weights1
        for i in range(len(genome.weights1)):
            for j in range(len(genome.weights1[i])):
                if random.random() < self.mutation_rate:
                    genome.weights1[i][j] += random.gauss(0, 0.2)
                    
        # Mutate biases1
        for i in range(len(genome.biases1)):
            if random.random() < self.mutation_rate:
                genome.biases1[i] += random.gauss(0, 0.2)
                
        # Mutate weights2
        for i in range(len(genome.weights2)):
            for j in range(len(genome.weights2[i])):
                if random.random() < self.mutation_rate:
                    genome.weights2[i][j] += random.gauss(0, 0.2)
                    
        # Mutate biases2
        for i in range(len(genome.biases2)):
            if random.random() < self.mutation_rate:
                genome.biases2[i] += random.gauss(0, 0.2)
    
    def evolve(self):
        """Evolve population for one generation."""
        new_population = []
        
        # Keep best genome (elitism)
        new_population.append(copy.deepcopy(self.population[0]))
        
        # Create rest through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.selection()
            parent2 = self.selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
            # Mutation
            self.mutate(child1)
            self.mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
                
        self.population = new_population
        
    def get_best(self):
        """Get the best genome and its fitness."""
        if not self.fitness_scores:
            return None, 0
            
        return self.population[0], self.fitness_scores[0]


def xor_fitness(network):
    """Calculate fitness for XOR problem."""
    # XOR inputs and targets
    xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_targets = [0, 1, 1, 0]
    
    total_error = 0
    for inputs, target in zip(xor_inputs, xor_targets):
        # Get network output for this input
        output = network.forward(inputs)[0]
        
        # Calculate error (squared)
        error = (output - target) ** 2
        total_error += error
        
    # Convert error to fitness (lower error = higher fitness)
    fitness = 1 / (1 + total_error)
    return fitness


def test_genetic_optimization():
    """Test genetic optimization for XOR problem."""
    print("Testing Genetic Optimization for XOR")
    print("====================================\n")
    
    # Create optimizer
    print("Initializing genetic optimizer...")
    optimizer = GeneticOptimizer(
        population_size=20,
        mutation_rate=0.2,
        crossover_rate=0.7
    )
    
    # Initialize population
    print("\nInitializing population...")
    optimizer.initialize_population()
    
    # Evaluate initial fitness
    avg_fitness = optimizer.evaluate_fitness(xor_fitness)
    print(f"Initial average fitness: {avg_fitness:.4f}")
    best_genome, best_fitness = optimizer.get_best()
    print(f"Initial best fitness: {best_fitness:.4f}")
    
    # Run evolution for several generations
    print("\nRunning optimization for 30 generations...")
    best_fitnesses = []
    avg_fitnesses = []
    
    for generation in range(30):
        # Evolve population
        optimizer.evolve()
        
        # Evaluate fitness
        avg_fitness = optimizer.evaluate_fitness(xor_fitness)
        best_genome, best_fitness = optimizer.get_best()
        
        best_fitnesses.append(best_fitness)
        avg_fitnesses.append(avg_fitness)
        
        if generation % 5 == 0 or generation == 29:
            print(f"Generation {generation+1}: Avg fitness = {avg_fitness:.4f}, Best fitness = {best_fitness:.4f}")
    
    # Test best solution
    print("\nTesting best solution:")
    best_network = best_genome.to_network()
    
    xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_targets = [0, 1, 1, 0]
    
    for inputs, target in zip(xor_inputs, xor_targets):
        output = best_network.forward(inputs)[0]
        print(f"Inputs: {inputs}, Target: {target}, Output: {output:.4f}")
    
    print("\nOptimization complete!")


if __name__ == "__main__":
    test_genetic_optimization()
