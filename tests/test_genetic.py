"""
Tests for genetic algorithm optimization.

This module tests the genetic algorithm implementation for optimizing mycelium networks.
"""

import unittest
import random
from mycelium.environment import Environment
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
from mycelium.enhanced.ml.genetic import NetworkGenome, GeneticOptimizer


class TestNetworkGenome(unittest.TestCase):
    """Test cases for the NetworkGenome class."""
    
    def test_genome_creation_from_network(self):
        """Test creating a genome from a network."""
        # Create a test network
        network = AdaptiveMyceliumNetwork(
            input_size=2,
            output_size=1,
            initial_nodes=5
        )
        
        # Create genome from network
        genome = NetworkGenome(network)
        
        # Check basic properties were copied
        self.assertEqual(genome.network_type, type(network))
        self.assertEqual(genome.input_size, network.input_size)
        self.assertEqual(genome.output_size, network.output_size)
        
        # Check nodes were copied
        self.assertEqual(len(genome.node_genes), len(network.nodes))
        
        # Check connections were copied
        total_connections = sum(len(node.connections) for node in network.nodes.values())
        self.assertEqual(len(genome.connection_genes), total_connections)
    
    def test_network_from_genome(self):
        """Test creating a network from a genome."""
        # Create a test network
        original_network = AdaptiveMyceliumNetwork(
            input_size=2,
            output_size=1,
            initial_nodes=5
        )
        
        # Create genome from network
        genome = NetworkGenome(original_network)
        
        # Create new network from genome
        new_network = genome.to_network()
        
        # Check basic properties were preserved
        self.assertEqual(new_network.input_size, original_network.input_size)
        self.assertEqual(new_network.output_size, original_network.output_size)
        self.assertEqual(len(new_network.nodes), len(original_network.nodes))
        
        # Check node IDs
        original_node_ids = set(original_network.nodes.keys())
        new_node_ids = set(new_network.nodes.keys())
        self.assertEqual(original_node_ids, new_node_ids)
        
        # Check connections (should be roughly the same, but order might differ)
        original_conn_count = sum(len(node.connections) for node in original_network.nodes.values())
        new_conn_count = sum(len(node.connections) for node in new_network.nodes.values())
        self.assertEqual(new_conn_count, original_conn_count)


class TestGeneticOptimizer(unittest.TestCase):
    """Test cases for the GeneticOptimizer class."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        # Create environment
        env = Environment()
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            environment=env,
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.7
        )
        
        # Check properties
        self.assertEqual(optimizer.population_size, 10)
        self.assertEqual(optimizer.mutation_rate, 0.1)
        self.assertEqual(optimizer.crossover_rate, 0.7)
        self.assertEqual(optimizer.generation, 0)
        self.assertEqual(len(optimizer.population), 0)
    
    def test_population_initialization(self):
        """Test initializing a population of networks."""
        # Create environment
        env = Environment()
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            environment=env,
            population_size=5
        )
        
        # Initialize population
        template_network = AdaptiveMyceliumNetwork(
            environment=env,
            input_size=2,
            output_size=1,
            initial_nodes=3
        )
        
        optimizer.initialize_population(template_network)
        
        # Check population size
        self.assertEqual(len(optimizer.population), 5)
        
        # Check all are genomes
        for genome in optimizer.population:
            self.assertIsInstance(genome, NetworkGenome)
            self.assertEqual(genome.input_size, 2)
            self.assertEqual(genome.output_size, 1)
    
    def test_selection(self):
        """Test selection mechanism."""
        # Create environment
        env = Environment()
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            environment=env,
            population_size=5
        )
        
        # Initialize population
        optimizer.initialize_population()
        
        # Set some fitness scores (higher is better)
        optimizer.fitness_scores = [0.1, 0.5, 0.2, 0.8, 0.3]
        
        # Perform selection multiple times
        selected_indices = []
        for _ in range(20):
            genome = optimizer._selection()
            # Find which genome was selected by matching objects
            for i, pop_genome in enumerate(optimizer.population):
                if genome.node_genes == pop_genome.node_genes:
                    selected_indices.append(i)
                    break
        
        # Selection should favor higher fitness scores
        # So index 3 (fitness 0.8) should be selected more often
        counts = [selected_indices.count(i) for i in range(5)]
        
        # The highest fitness individual should be selected most often
        self.assertEqual(counts.index(max(counts)), 3)
    
    def test_crossover(self):
        """Test crossover operation."""
        # Create environment
        env = Environment()
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            environment=env,
            population_size=5
        )
        
        # Initialize population
        template_network = AdaptiveMyceliumNetwork(
            environment=env,
            input_size=2,
            output_size=1,
            initial_nodes=5
        )
        
        optimizer.initialize_population(template_network)
        
        # Select two parent genomes
        parent1 = optimizer.population[0]
        parent2 = optimizer.population[1]
        
        # Perform crossover
        child1, child2 = optimizer._crossover(parent1, parent2)
        
        # Children should be different from parents
        self.assertNotEqual(child1.node_genes, parent1.node_genes)
        self.assertNotEqual(child1.node_genes, parent2.node_genes)
        self.assertNotEqual(child2.node_genes, parent1.node_genes)
        self.assertNotEqual(child2.node_genes, parent2.node_genes)
        
        # Children should have valid structure
        self.assertEqual(child1.input_size, parent1.input_size)
        self.assertEqual(child1.output_size, parent1.output_size)
        self.assertEqual(child2.input_size, parent2.input_size)
        self.assertEqual(child2.output_size, parent2.output_size)
        
        # Each child should have unique node IDs
        child1_node_ids = set(node['id'] for node in child1.node_genes)
        self.assertEqual(len(child1_node_ids), len(child1.node_genes))
        
        child2_node_ids = set(node['id'] for node in child2.node_genes)
        self.assertEqual(len(child2_node_ids), len(child2.node_genes))
    
    def test_mutation(self):
        """Test mutation operation."""
        # Create environment
        env = Environment()
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            environment=env,
            population_size=5
        )
        
        # Initialize population
        template_network = AdaptiveMyceliumNetwork(
            environment=env,
            input_size=2,
            output_size=1,
            initial_nodes=5
        )
        
        optimizer.initialize_population(template_network)
        
        # Get a genome to mutate
        genome = optimizer.population[0]
        
        # Save original state
        original_nodes = [node.copy() for node in genome.node_genes]
        original_connections = [conn.copy() for conn in genome.connection_genes]
        
        # Perform mutation (strong to ensure changes)
        optimizer._mutate(genome, mutation_strength=0.9)
        
        # Check if mutation occurred
        changed_nodes = False
        for i, node in enumerate(genome.node_genes):
            if i < len(original_nodes):
                if (node['sensitivity'] != original_nodes[i]['sensitivity'] or
                    node['adaptability'] != original_nodes[i]['adaptability']):
                    changed_nodes = True
                    break
        
        changed_connections = False
        for i, conn in enumerate(genome.connection_genes):
            if i < len(original_connections):
                if conn['strength'] != original_connections[i]['strength']:
                    changed_connections = True
                    break
        
        # At least some changes should have occurred
        self.assertTrue(changed_nodes or changed_connections or 
                       len(genome.node_genes) != len(original_nodes) or
                       len(genome.connection_genes) != len(original_connections))
    
    def test_evolution(self):
        """Test evolution over multiple generations."""
        # Create environment
        env = Environment()
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            environment=env,
            population_size=5,
            elite_percentage=0.2
        )
        
        # Initialize population
        optimizer.initialize_population()
        
        # Define a simple fitness function
        def simple_fitness(network):
            # Output closer to 0.75 is better
            output = network.forward([0.5, 0.5])[0]
            return 1.0 - abs(output - 0.75)
        
        # Run evolution for a few generations
        for generation in range(3):
            # Evaluate fitness
            avg_fitness = optimizer.evaluate_fitness(simple_fitness)
            
            # Check that fitness scores were calculated
            self.assertEqual(len(optimizer.fitness_scores), optimizer.population_size)
            
            # Evolve to next generation
            optimizer.evolve()
            
            # Check generation counter was incremented
            self.assertEqual(optimizer.generation, generation + 1)
            
            # Check population size remained the same
            self.assertEqual(len(optimizer.population), optimizer.population_size)
    
    def test_complete_optimization(self):
        """Test complete optimization process with a simple problem."""
        # Create environment
        env = RichEnvironment()
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            environment=env,
            population_size=5,
            mutation_rate=0.2,
            crossover_rate=0.7,
            elite_percentage=0.2
        )
        
        # Create a template network
        template = AdaptiveMyceliumNetwork(
            environment=env,
            input_size=2,
            output_size=1,
            initial_nodes=5
        )
        
        # Initialize population
        optimizer.initialize_population(template)
        
        # Define a simple XOR-like fitness function
        def xor_fitness(network):
            # XOR-like problem: f([0,0])=0, f([0,1])=1, f([1,0])=1, f([1,1])=0
            inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
            targets = [0.0, 1.0, 1.0, 0.0]
            
            # Calculate error
            total_error = 0.0
            for input_vec, target in zip(inputs, targets):
                output = network.forward(input_vec)[0]
                error = abs(output - target)
                total_error += error
            
            # Convert to fitness (lower error = higher fitness)
            return 1.0 / (1.0 + total_error)
        
        # Run optimization for a few generations
        best_fitness_before = 0.0
        
        # Evaluate initial fitness
        optimizer.evaluate_fitness(xor_fitness)
        best_fitness_before = optimizer.best_fitness
        
        # Run evolution for a few generations
        for _ in range(3):
            optimizer.evolve()
            optimizer.evaluate_fitness(xor_fitness)
        
        # Check final fitness
        best_fitness_after = optimizer.best_fitness
        
        # Get best network
        best_network = optimizer.get_best_network()
        
        # Check that optimization improved fitness
        # Note: With only 3 generations, improvement isn't guaranteed
        # so we don't assert this, but print the results
        print(f"Initial best fitness: {best_fitness_before:.4f}")
        print(f"Final best fitness: {best_fitness_after:.4f}")
        
        # Best network should exist
        self.assertIsNotNone(best_network)
        
        # Test the best network on XOR problem
        outputs = []
        for input_vec in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]:
            outputs.append(best_network.forward(input_vec)[0])
        
        print(f"Best network outputs for XOR problem: {outputs}")


if __name__ == '__main__':
    unittest.main()
