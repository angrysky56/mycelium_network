"""
Genetic algorithm implementation for optimizing mycelium networks.

This module provides genetic optimization capabilities for evolving
populations of mycelium networks to improve performance.
"""

import random
import copy
import math
from typing import List, Dict, Any, Callable, Tuple, Optional

from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
from mycelium.enhanced.rich_environment import RichEnvironment


class NetworkGenome:
    """Represents the genetic information of a mycelium network."""
    
    def __init__(self, network=None):
        """
        Initialize from an existing network or create empty genome.
        
        Args:
            network: MyceliumNetwork instance or None
        """
        if network is None:
            # Empty genome
            self.network_type = AdaptiveMyceliumNetwork
            self.input_size = 2
            self.output_size = 1
            self.node_genes = []
            self.connection_genes = []
            return
            
        self.network_type = type(network)
        self.input_size = network.input_size
        self.output_size = network.output_size
        
        # Extract genetic information
        self.node_genes = []
        self.connection_genes = []
        
        # Extract node genes
        for node_id, node in network.nodes.items():
            node_gene = {
                'id': node_id,
                'position': node.position,
                'type': node.type,
                'sensitivity': node.sensitivity,
                'adaptability': node.adaptability,
                'specializations': node.specializations.copy() if hasattr(node, 'specializations') else {}
            }
            self.node_genes.append(node_gene)
        
        # Extract connection genes
        for node_id, node in network.nodes.items():
            for target_id, strength in node.connections.items():
                connection_gene = {
                    'source': node_id,
                    'target': target_id,
                    'strength': strength
                }
                self.connection_genes.append(connection_gene)
    
    def to_network(self, environment=None):
        """
        Convert genome back to a network.
        
        Args:
            environment: Environment instance
            
        Returns:
            Network instance
        """
        # Create basic network
        network = self.network_type(
            environment=environment,
            input_size=self.input_size,
            output_size=self.output_size,
            initial_nodes=0  # Start with empty network
        )
        
        # Clear initial nodes
        network.nodes = {}
        network.regular_nodes = []
        
        # Add nodes
        for gene in self.node_genes:
            node_id = gene['id']
            
            # Create node
            from mycelium.node import MyceliumNode
            node = MyceliumNode(node_id, gene['position'], gene['type'])
            
            # Set properties
            node.sensitivity = gene['sensitivity']
            node.adaptability = gene['adaptability']
            
            # Add to network
            network.nodes[node_id] = node
            
            # Update collections
            if gene['type'] == 'input':
                network.input_nodes.append(node_id)
            elif gene['type'] == 'output':
                network.output_nodes.append(node_id)
            elif gene['type'] == 'regular':
                network.regular_nodes.append(node_id)
            elif network.specializations is not None:
                # Handle specialized nodes
                node_type = gene['type']
                if node_type not in network.specializations:
                    network.specializations[node_type] = []
                network.specializations[node_type].append(node_id)
        
        # Add connections
        for gene in self.connection_genes:
            source_id = gene['source']
            target_id = gene['target']
            
            # Skip if nodes don't exist
            if source_id not in network.nodes or target_id not in network.nodes:
                continue
                
            # Add connection
            network.nodes[source_id].connections[target_id] = gene['strength']
        
        return network


class GeneticOptimizer:
    """
    Genetic algorithm for optimizing mycelium networks.
    
    Uses evolutionary techniques (mutation, crossover, selection)
    to evolve better networks over generations.
    """
    
    def __init__(
        self,
        environment,
        population_size=10,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elite_percentage=0.2
    ):
        """
        Initialize the genetic optimizer.
        
        Args:
            environment: Environment for the networks
            population_size: Number of networks in the population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_percentage: Percentage of top performers to keep unchanged
        """
        self.environment = environment
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_percentage = elite_percentage
        
        # Population and fitness
        self.population = []  # List of NetworkGenome
        self.fitness_scores = []
        
        # Statistics
        self.generation = 0
        self.best_fitness = 0
        self.best_genome = None
        self.fitness_history = []
    
    def _create_xor_network(self, randomize=False):
        """
        Creates a network with topology suited for solving XOR problems.
        
        Args:
            randomize: Whether to add random variations to the network
            
        Returns:
            Network with XOR-friendly structure
        """
        # Create a new network
        network = AdaptiveMyceliumNetwork(
            environment=self.environment,
            input_size=2,
            output_size=1,
            initial_nodes=4  # We'll manually set up the nodes
        )
        
        # Clear existing nodes
        for node_id in list(network.nodes.keys()):
            if node_id not in network.input_nodes and node_id not in network.output_nodes:
                del network.nodes[node_id]
        network.regular_nodes = []
        
        # Create hidden neurons (one for AND, one for OR)
        h1_id = max(network.nodes.keys()) + 1
        h2_id = h1_id + 1
        
        # Add hidden nodes
        from mycelium.node import MyceliumNode
        
        # First hidden node - will act as AND gate
        h1_node = MyceliumNode(h1_id, (0.5, 0.3), "regular")
        network.nodes[h1_id] = h1_node
        network.regular_nodes.append(h1_id)
        
        # Second hidden node - will act as NAND gate
        h2_node = MyceliumNode(h2_id, (0.5, 0.7), "regular")
        network.nodes[h2_id] = h2_node
        network.regular_nodes.append(h2_id)
        
        # Set up connections for XOR
        # XOR can be built as (A OR B) AND NOT (A AND B)
        input_ids = network.input_nodes
        output_id = network.output_nodes[0]
        
        # Clear existing connections
        for node_id in network.nodes:
            network.nodes[node_id].connections = {}
            
        # Set up connections to implement XOR
        # Input -> Hidden 1 (AND gate)
        network.nodes[input_ids[0]].connections[h1_id] = 0.5 * (1.0 + random.uniform(-0.1, 0.1) if randomize else 1.0)
        network.nodes[input_ids[1]].connections[h1_id] = 0.5 * (1.0 + random.uniform(-0.1, 0.1) if randomize else 1.0)
        
        # Input -> Hidden 2 (OR-like gate)
        network.nodes[input_ids[0]].connections[h2_id] = 1.0 * (1.0 + random.uniform(-0.1, 0.1) if randomize else 1.0)
        network.nodes[input_ids[1]].connections[h2_id] = 1.0 * (1.0 + random.uniform(-0.1, 0.1) if randomize else 1.0)
        
        # Hidden 1 (AND) -> Output (negative)
        network.nodes[h1_id].connections[output_id] = -1.0 * (1.0 + random.uniform(-0.1, 0.1) if randomize else 1.0)
        
        # Hidden 2 (OR) -> Output (positive)
        network.nodes[h2_id].connections[output_id] = 1.0 * (1.0 + random.uniform(-0.1, 0.1) if randomize else 1.0)
        
        # Add some random connections if specified
        if randomize and random.random() < 0.5:
            # Maybe add direct connections
            if random.random() < 0.3:
                network.nodes[input_ids[0]].connections[output_id] = random.uniform(-0.5, 0.5)
            if random.random() < 0.3:
                network.nodes[input_ids[1]].connections[output_id] = random.uniform(-0.5, 0.5)
        
        return network