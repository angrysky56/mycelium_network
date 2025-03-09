#!/usr/bin/env python3
"""
Enhanced Mycelium Network Demo

This script demonstrates the key features of the enhanced environment
and adaptive network implementation.
"""

import os
import sys
import random
import math
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the enhanced components
from mycelium.enhanced.resource import ResourceType, Environmental_Factors
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork


def print_separator(title):
    """Print a section separator with title."""
    width = 60
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def demo_rich_environment():
    """Demonstrate the rich environment features."""
    print_separator("Rich Environment Demo")
    
    # Create a 3D environment
    env = RichEnvironment(dimensions=3, size=1.0, name="Demo Environment")
    print(f"Created environment: {env.name}")
    print(f"Dimensions: {env.dimensions}")
    print(f"Layers: {len(env.layers)}")
    
    # List the terrain layers
    print("\nTerrain Layers:")
    for i, layer in enumerate(env.layers):
        print(f"  {i+1}. {layer.name} ({layer.height_range[0]:.1f}-{layer.height_range[1]:.1f})")
        print(f"     Conductivity: {layer.conductivity:.1f}, Density: {layer.density:.1f}")
    
    # Add diverse resources
    print("\nAdding resources:")
    resources = [
        ((0.5, 0.5, 0.6), 0.2, ResourceType.CARBON, 2.0),
        ((0.3, 0.3, 0.65), 0.15, ResourceType.WATER, 1.5),
        ((0.7, 0.7, 0.6), 0.1, ResourceType.NITROGEN, 1.0),
        ((0.2, 0.8, 0.6), 0.12, ResourceType.SUGAR, 0.8),
        ((0.8, 0.2, 0.5), 0.1, ResourceType.MINERAL, 0.7)
    ]
    
    for center, radius, resource_type, amount in resources:
        env.add_nutrient_cluster(center, radius, resource_type, amount)
        print(f"  Added {resource_type.name} cluster at {center}, amount: {amount}")
    
    # Check total resources
    print("\nTotal resources:")
    for resource_type in ResourceType:
        total = env.get_total_resources(resource_type)
        if total > 0:
            print(f"  {resource_type.name}: {total:.2f}")
    
    # Demonstrate environmental factors
    print("\nInitial environmental factors:")
    print(f"  Temperature: {env.factors.temperature:.2f}")
    print(f"  Moisture: {env.factors.moisture:.2f}")
    print(f"  Light level: {env.factors.light_level:.2f}")
    print(f"  pH: {env.factors.ph:.1f}")
    
    # Add organisms
    print("\nAdding organisms:")
    env.add_organism("plant1", (0.4, 0.6, 0.7), "plant", {"size": 0.8, "productivity": 0.1})
    env.add_organism("plant2", (0.6, 0.4, 0.7), "plant", {"size": 0.5, "productivity": 0.08})
    env.add_organism("herbivore1", (0.2, 0.2, 0.7), "herbivore", {"speed": 0.1, "energy": 0.7})
    print(f"  Added {len(env.organisms)} organisms")
    
    # Set up seasonal cycles
    print("\nCreating seasonal cycle (accelerated time):")
    env.create_seasonal_cycle(year_length=24.0, intensity=0.7)  # 1 day = 1 year
    print("  Season cycle created with 24.0 time units per year")
    
    # Run environment for several steps to show dynamics
    print("\nRunning environment simulation for 12 steps:")
    for i in range(12):
        # Update environment
        env.update(delta_time=2.0)  # 2 time units per step = 1/12 of a year
        
        # Calculate season
        year_phase = (env.time % env.year_length) / env.year_length
        season_idx = int(year_phase * 4) % 4
        season_names = ["Spring", "Summer", "Fall", "Winter"]
        current_season = season_names[season_idx]
        
        if i % 3 == 0 or i == 11:  # Print every 3rd step and last step
            print(f"\nStep {i+1} - Time: {env.time:.1f}, Season: {current_season}")
            print(f"  Temperature: {env.factors.temperature:.2f}")
            print(f"  Moisture: {env.factors.moisture:.2f}")
            print(f"  Light level: {env.factors.light_level:.2f}")
            
            # Check how resources have changed
            carbon = env.get_total_resources(ResourceType.CARBON)
            water = env.get_total_resources(ResourceType.WATER)
            sugar = env.get_total_resources(ResourceType.SUGAR)
            print(f"  Carbon: {carbon:.2f}, Water: {water:.2f}, Sugar: {sugar:.2f}")
            
            # Check organisms
            live_organisms = [org for org in env.organisms.values() if org["alive"]]
            print(f"  Live organisms: {len(live_organisms)}")
    
    # Get final state snapshot
    snapshot = env.get_state_snapshot()
    print("\nFinal environment state snapshot:")
    print(f"  Time: {snapshot['time']:.1f}")
    print(f"  Global factors: {', '.join(f'{k}: {v:.2f}' for k, v in snapshot['global_factors'].items())}")
    print(f"  Resources: {', '.join(f'{k}: {v:.2f}' for k, v in snapshot['resources'].items() if v > 0)}")
    print(f"  Organisms: {snapshot['organisms']['count']} total")
    for org_type, count in snapshot['organisms'].get('by_type', {}).items():
        print(f"    {org_type}: {count}")
    
    return env


def demo_adaptive_network(environment):
    """Demonstrate the adaptive network features."""
    print_separator("Adaptive Network Demo")
    
    # Create adaptive network
    network = AdaptiveMyceliumNetwork(
        environment=environment,
        input_size=2,
        output_size=1,
        initial_nodes=10
    )
    
    print("Created adaptive network:")
    print(f"  Total nodes: {len(network.nodes)}")
    print(f"  Input nodes: {len(network.input_nodes)}")
    print(f"  Output nodes: {len(network.output_nodes)}")
    print(f"  Regular nodes: {len(network.regular_nodes)}")
    
    # Check for specialized nodes
    spec_count = sum(len(nodes) for nodes in network.specializations.values())
    print(f"\nSpecialized nodes: {spec_count} total")
    for spec_type, nodes in network.specializations.items():
        print(f"  {spec_type}: {len(nodes)}")
    
    # Get initial adaptation state
    initial_stats = network.get_specialization_statistics()
    print("\nInitial adaptation state:")
    print(f"  Temperature adaptation: {initial_stats['adaptation']['temperature_adaptation']:.3f}")
    print(f"  Moisture adaptation: {initial_stats['adaptation']['moisture_adaptation']:.3f}")
    print(f"  Drought resistance: {initial_stats['adaptation']['drought_resistance']:.3f}")
    
    # Run network with different environmental conditions
    print("\nRunning network with changing environment:")
    
    # Track adaptation metrics
    steps = 8
    metrics = {
        'temperatures': [],
        'moistures': [],
        'temp_adaptations': [],
        'moisture_adaptations': [],
        'node_counts': [],
        'output_values': []
    }
    
    for i in range(steps):
        # Change environmental conditions dramatically
        if i < steps/2:
            # First half: hot and dry
            environment.factors.temperature = 0.8
            environment.factors.moisture = 0.2
        else:
            # Second half: cool and wet
            environment.factors.temperature = 0.3
            environment.factors.moisture = 0.8
        
        # Update environment
        environment.update(delta_time=1.0)
        
        # Run network for 3 iterations at each step
        for _ in range(3):
            inputs = [0.7, 0.3]  # consistent inputs to see adaptation effects
            outputs = network.forward(inputs)
            
            # Record metrics
            metrics['temperatures'].append(environment.factors.temperature)
            metrics['moistures'].append(environment.factors.moisture)
            stats = network.get_specialization_statistics()
            metrics['temp_adaptations'].append(stats['adaptation']['temperature_adaptation'])
            metrics['moisture_adaptations'].append(stats['adaptation']['moisture_adaptation'])
            metrics['node_counts'].append(len(network.nodes))
            metrics['output_values'].append(outputs[0])
            
        # Print adaptation progress
        stats = network.get_specialization_statistics()
        print(f"\nStep {i+1}:")
        print(f"  Environment: Temp={environment.factors.temperature:.2f}, Moisture={environment.factors.moisture:.2f}")
        print(f"  Adaptation: Temp={stats['adaptation']['temperature_adaptation']:.3f}, Moisture={stats['adaptation']['moisture_adaptation']:.3f}")
        print(f"  Nodes: {len(network.nodes)} total, {sum(len(nodes) for nodes in network.specializations.values())} specialized")
        print(f"  Output for [0.7, 0.3]: {outputs[0]:.4f}")
    
    # Show final adaptation state
    final_stats = network.get_specialization_statistics()
    print("\nAdaptation results:")
    print(f"  Initial temperature adaptation: {initial_stats['adaptation']['temperature_adaptation']:.3f}")
    print(f"  Final temperature adaptation: {final_stats['adaptation']['temperature_adaptation']:.3f}")
    print(f"  Initial moisture adaptation: {initial_stats['adaptation']['moisture_adaptation']:.3f}")
    print(f"  Final moisture adaptation: {final_stats['adaptation']['moisture_adaptation']:.3f}")
    
    # Specialization changes
    print("\nNode specialization changes:")
    print(f"  Initial specialized nodes: {spec_count}")
    final_spec_count = sum(len(nodes) for nodes in network.specializations.values())
    print(f"  Final specialized nodes: {final_spec_count} ({final_spec_count - spec_count:+d})")
    for spec_type, nodes in network.specializations.items():
        print(f"  {spec_type}: {len(nodes)}")
    
    return network, metrics


def main():
    """Run the enhanced demo."""
    print("Enhanced Mycelium Network Demo")
    print("==============================")
    
    # Demo the rich environment
    environment = demo_rich_environment()
    
    # Demo the adaptive network
    network, metrics = demo_adaptive_network(environment)
    
    print_separator("Demo Completed")
    print("\nSuccessfully demonstrated the enhanced environment and adaptive network!")
    print("Key features demonstrated:")
    print("- Multi-layered terrain in 3D environments")
    print("- Multiple resource types with interactions")
    print("- Seasonal cycles and environmental dynamics")
    print("- Organism simulation and ecosystem effects")
    print("- Network adaptation to environmental conditions")
    print("- Node specialization based on environmental factors")
    print("- Resource distribution influencing growth patterns")
    
    print("\nFor more information, see ENHANCED_ENVIRONMENT_SETUP.md")


if __name__ == "__main__":
    main()
