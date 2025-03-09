#!/usr/bin/env python3
"""
Test script for the adaptive mycelium network implementation.

This script creates an adaptive network within a rich environment and tests:
- Specialization of nodes
- Environmental adaptation
- Enhanced growth patterns
- Resource processing efficiency
"""

import os
import sys
import random
import time
import math
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the standard network for comparison
from mycelium.network import AdvancedMyceliumNetwork

# Import the enhanced components
from mycelium.enhanced.resource import ResourceType, Environmental_Factors
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork


def test_basic_functionality():
    """Test the basic functionality of the adaptive network."""
    print("\n=== Testing Basic Functionality ===")
    
    # Create a rich environment
    env = RichEnvironment(dimensions=3, size=1.0, name="Test Environment")
    
    # Add some resources
    print("Adding resources to environment...")
    env.add_nutrient_cluster((0.5, 0.5, 0.6), 0.2, ResourceType.CARBON, 2.0)
    env.add_nutrient_cluster((0.3, 0.3, 0.6), 0.15, ResourceType.WATER, 1.5)
    env.add_nutrient_cluster((0.7, 0.7, 0.6), 0.1, ResourceType.NITROGEN, 1.0)
    
    # Create an adaptive network
    print("\nInitializing adaptive network...")
    network = AdaptiveMyceliumNetwork(
        environment=env,
        input_size=2,
        output_size=1,
        initial_nodes=10
    )
    
    # Print initial state
    print(f"Network initialized with {len(network.nodes)} total nodes")
    print(f"Input nodes: {len(network.input_nodes)}")
    print(f"Output nodes: {len(network.output_nodes)}")
    print(f"Regular nodes: {len(network.regular_nodes)}")
    
    # Check for specialized nodes
    spec_count = sum(len(nodes) for nodes in network.specializations.values())
    print(f"Specialized nodes: {spec_count}")
    for spec_type, nodes in network.specializations.items():
        print(f"  {spec_type}: {len(nodes)}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    inputs = [0.7, 0.3]
    outputs = network.forward(inputs)
    print(f"Inputs: {inputs}")
    print(f"Outputs: {outputs}")
    
    # Check node stats
    stats = network.get_specialization_statistics()
    print("\nNode specialization statistics:")
    print(f"Temperature adaptation: {stats['adaptation']['temperature_adaptation']:.3f}")
    print(f"Moisture adaptation: {stats['adaptation']['moisture_adaptation']:.3f}")
    
    print("\nBasic functionality test completed successfully!")
    return network, env


def test_environmental_adaptation(network, env):
    """Test the adaptive network's response to environmental changes."""
    print("\n=== Testing Environmental Adaptation ===")
    
    # Store initial state
    initial_stats = network.get_specialization_statistics()
    
    # Run simulation for several steps
    print("Running simulation with changing environment...")
    steps = 10
    
    # Original environmental conditions
    original_temp = env.factors.temperature
    original_moisture = env.factors.moisture
    
    # Record adaptation metrics
    temp_adaptations = []
    moisture_adaptations = []
    growth_rates = []
    node_counts = []
    specialization_counts = {}
    
    for i in range(steps):
        # Update environment - cycle through different conditions
        cycle_phase = i / steps
        
        # Temperature cycles from cool to hot
        env.factors.temperature = 0.2 + 0.6 * math.sin(cycle_phase * math.pi)
        
        # Moisture cycles from wet to dry (opposite of temperature)
        env.factors.moisture = 0.2 + 0.6 * math.sin((cycle_phase + 0.5) * math.pi)
        
        print(f"\nStep {i+1}:")
        print(f"  Temperature: {env.factors.temperature:.2f}")
        print(f"  Moisture: {env.factors.moisture:.2f}")
        
        # Run a few iterations of the network
        for _ in range(3):
            inputs = [random.random(), random.random()]
            outputs = network.forward(inputs)
        
        # Get updated statistics
        stats = network.get_specialization_statistics()
        
        # Record metrics
        temp_adaptations.append(stats['adaptation']['temperature_adaptation'])
        moisture_adaptations.append(stats['adaptation']['moisture_adaptation'])
        growth_rates.append(network.growth_rate)
        node_counts.append(len(network.nodes))
        
        # Specialized node counts
        for spec_type in network.specializations.keys():
            if spec_type not in specialization_counts:
                specialization_counts[spec_type] = []
            
            count = len(network.specializations[spec_type])
            specialization_counts[spec_type].append(count)
        
        # Print current adaptations
        print(f"  Temperature adaptation: {stats['adaptation']['temperature_adaptation']:.3f}")
        print(f"  Moisture adaptation: {stats['adaptation']['moisture_adaptation']:.3f}")
        print(f"  Growth rate: {network.growth_rate:.3f}")
        print(f"  Total nodes: {len(network.nodes)}")
        
        # Specialization breakdown
        spec_str = ", ".join(f"{spec}: {len(nodes)}" for spec, nodes in network.specializations.items())
        print(f"  Specializations: {spec_str}")
    
    # Restore original environment
    env.factors.temperature = original_temp
    env.factors.moisture = original_moisture
    
    print("\nFinal adaptation statistics:")
    stats = network.get_specialization_statistics()
    print(f"Initial temperature adaptation: {initial_stats['adaptation']['temperature_adaptation']:.3f}")
    print(f"Final temperature adaptation: {stats['adaptation']['temperature_adaptation']:.3f}")
    print(f"Initial moisture adaptation: {initial_stats['adaptation']['moisture_adaptation']:.3f}")
    print(f"Final moisture adaptation: {stats['adaptation']['moisture_adaptation']:.3f}")
    
    print("\nEnvironmental adaptation test completed!")
    return temp_adaptations, moisture_adaptations, growth_rates, node_counts, specialization_counts


def test_resource_processing():
    """Test the adaptive network's ability to process different resource types."""
    print("\n=== Testing Resource Processing ===")
    
    # Create a rich environment with diverse resources
    env = RichEnvironment(dimensions=2, size=1.0)
    
    # Add different resource types in clusters
    env.add_nutrient_cluster((0.2, 0.2), 0.1, ResourceType.CARBON, 2.0)
    env.add_nutrient_cluster((0.5, 0.5), 0.1, ResourceType.WATER, 1.5)
    env.add_nutrient_cluster((0.8, 0.2), 0.1, ResourceType.NITROGEN, 1.0)
    env.add_nutrient_cluster((0.2, 0.8), 0.1, ResourceType.SUGAR, 1.2)
    env.add_nutrient_cluster((0.8, 0.8), 0.1, ResourceType.PHOSPHORUS, 0.8)
    
    # Create an adaptive network
    network = AdaptiveMyceliumNetwork(
        environment=env,
        input_size=2,
        output_size=1,
        initial_nodes=15
    )
    
    # Record initial resource levels
    initial_resources = {}
    for r_type in ResourceType:
        initial_resources[r_type] = env.get_total_resources(r_type)
    
    print("Initial resource levels:")
    for r_type, amount in initial_resources.items():
        if amount > 0:
            print(f"  {r_type.name}: {amount:.2f}")
    
    # Run network for several iterations
    print("\nRunning network for 15 iterations...")
    for i in range(15):
        # Generate input
        inputs = [random.random(), random.random()]
        
        # Forward pass
        outputs = network.forward(inputs)
        
        # Every 5 iterations, check resource levels
        if (i + 1) % 5 == 0:
            current_resources = {}
            for r_type in ResourceType:
                current_resources[r_type] = env.get_total_resources(r_type)
            
            print(f"\nAfter {i + 1} iterations:")
            for r_type, initial in initial_resources.items():
                if initial > 0:
                    current = current_resources[r_type]
                    change = current - initial
                    print(f"  {r_type.name}: {current:.2f} ({change:+.2f})")
            
            # Node statistics
            print(f"  Total nodes: {len(network.nodes)}")
            print(f"  Regular nodes: {len(network.regular_nodes)}")
            
            # Specialization breakdown
            spec_str = ", ".join(f"{spec}: {len(nodes)}" for spec, nodes in network.specializations.items())
            print(f"  Specializations: {spec_str}")
            
            # Resource efficiency
            efficiencies = network.resource_efficiency
            print(f"  Resource efficiencies:")
            for r_type, efficiency in efficiencies.items():
                if initial_resources.get(r_type, 0) > 0:
                    print(f"    {r_type.name}: {efficiency:.2f}")
    
    print("\nResource processing test completed!")


def compare_with_standard_network():
    """Compare the adaptive network with the standard network."""
    print("\n=== Comparing with Standard Network ===")
    
    # Create a common environment
    env = RichEnvironment(dimensions=2, size=1.0)
    
    # Add resources
    env.add_nutrient_cluster((0.5, 0.5), 0.25, ResourceType.CARBON, 3.0)
    env.add_nutrient_cluster((0.3, 0.7), 0.2, ResourceType.WATER, 2.0)
    
    # Create networks
    std_network = AdvancedMyceliumNetwork(
        environment=env,
        input_size=2,
        output_size=1,
        initial_nodes=10
    )
    
    adpt_network = AdaptiveMyceliumNetwork(
        environment=env,
        input_size=2,
        output_size=1,
        initial_nodes=10
    )
    
    print("Networks initialized:")
    print(f"  Standard Network: {len(std_network.nodes)} nodes")
    print(f"  Adaptive Network: {len(adpt_network.nodes)} nodes")
    
    # Run both networks for several iterations
    print("\nRunning networks for 10 iterations with identical inputs...")
    
    std_outputs = []
    adpt_outputs = []
    std_node_counts = []
    adpt_node_counts = []
    
    for i in range(10):
        # Both networks receive identical inputs
        inputs = [random.random(), random.random()]
        
        # Standard network forward pass
        std_output = std_network.forward(inputs)
        std_outputs.append(std_output[0])
        std_node_counts.append(len(std_network.nodes))
        
        # Adaptive network forward pass
        adpt_output = adpt_network.forward(inputs)
        adpt_outputs.append(adpt_output[0])
        adpt_node_counts.append(len(adpt_network.nodes))
    
    print("\nFinal network states:")
    print(f"  Standard Network: {len(std_network.nodes)} nodes ({std_node_counts[-1] - std_node_counts[0]:+d})")
    print(f"  Adaptive Network: {len(adpt_network.nodes)} nodes ({adpt_node_counts[-1] - adpt_node_counts[0]:+d})")
    
    # Compare outputs
    std_mean = sum(std_outputs) / len(std_outputs)
    adpt_mean = sum(adpt_outputs) / len(adpt_outputs)
    
    std_variance = sum((o - std_mean) ** 2 for o in std_outputs) / len(std_outputs)
    adpt_variance = sum((o - adpt_mean) ** 2 for o in adpt_outputs) / len(adpt_outputs)
    
    print("\nOutput statistics:")
    print(f"  Standard Network: mean={std_mean:.4f}, variance={std_variance:.4f}")
    print(f"  Adaptive Network: mean={adpt_mean:.4f}, variance={adpt_variance:.4f}")
    
    # Compare network structure
    std_stats = std_network.get_network_statistics()
    adpt_stats = adpt_network.get_specialization_statistics()
    
    print("\nNetwork structure comparison:")
    print(f"  Standard Network:")
    print(f"    Average connections per node: {std_stats['avg_connections_per_node']:.2f}")
    print(f"    Average resources per node: {std_stats.get('avg_resource_level', 0):.2f}")
    
    print(f"  Adaptive Network:")
    print(f"    Average connections per node: {std_stats['avg_connections_per_node']:.2f}")
    print(f"    Specialized node types: {len(adpt_network.specializations)}")
    for spec_type, nodes in adpt_network.specializations.items():
        print(f"      {spec_type}: {len(nodes)}")
    
    print("\nComparison completed!")


def main():
    """Run all tests."""
    print("=== Adaptive Mycelium Network Tests ===")
    
    # Run basic functionality test
    network, env = test_basic_functionality()
    
    # Run environmental adaptation test
    adaptation_data = test_environmental_adaptation(network, env)
    
    # Test resource processing
    test_resource_processing()
    
    # Compare with standard network
    compare_with_standard_network()
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
