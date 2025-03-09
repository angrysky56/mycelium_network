#!/usr/bin/env python3
"""
Ecosystem Demo for the Enhanced Mycelium Network

This script demonstrates the enhanced ecosystem features with plants,
herbivores, decomposers, and complex environmental interactions.
"""

import os
import sys
import random
import time
import math
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.resource import ResourceType, Environmental_Factors
from mycelium.enhanced.ecosystem.ecosystem import Ecosystem
from mycelium.enhanced.ecosystem.organisms import Plant, Herbivore, Decomposer
from mycelium.enhanced.ecosystem.organisms.plant import Plant


def print_separator(title):
    """Print a section separator with title."""
    width = 60
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def demo_ecosystem_creation():
    """Create and set up the ecosystem for demonstration."""
    print_separator("Ecosystem Creation")
    
    # Create a 3D environment
    environment = RichEnvironment(dimensions=3, size=1.0, name="Ecosystem Demo Environment")
    print(f"Created environment: {environment.name}")
    print(f"Dimensions: {environment.dimensions}")
    
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
        environment.add_nutrient_cluster(center, radius, resource_type, amount)
        print(f"  Added {resource_type.name} cluster at {center}, amount: {amount}")
    
    # Set up seasonal cycles
    environment.create_seasonal_cycle(year_length=24.0, intensity=0.7)  # 1 day = 1 year
    print("Seasonal cycle created with 24.0 time units per year")
    
    # Create the ecosystem
    ecosystem = Ecosystem(environment)
    print("\nCreated ecosystem with:")
    print(f"  Environment size: {environment.size}")
    print(f"  Terrain layers: {len(environment.layers)}")
    
    # Populate with organisms
    print("\nPopulating ecosystem:")
    counts = ecosystem.populate_randomly(
        num_plants=15,
        num_herbivores=6,
        num_decomposers=4
    )
    
    for org_type, count in counts.items():
        print(f"  Added {count} {org_type}s")
    
    return ecosystem


def run_ecosystem_simulation(ecosystem, steps=60, time_per_step=0.5):
    """Run the ecosystem simulation for several steps."""
    print_separator("Ecosystem Simulation")
    
    # Collect data for plotting
    times = []
    populations = {
        "plant": [],
        "herbivore": [],
        "decomposer": []
    }
    energy_flows = {
        "photosynthesis": [],
        "consumption": [],
        "decomposition": []
    }
    
    # Run simulation steps
    print(f"Running simulation for {steps} steps:")
    print(f"{'Step':^5} | {'Time':^5} | {'Plants':^6} | {'Herbivores':^10} | {'Decomposers':^10} | {'Season':^8}")
    print("-" * 60)
    
    for step in range(steps):
        # Update ecosystem
        stats = ecosystem.update(time_per_step)
        
        # Calculate season
        env = ecosystem.environment
        year_phase = (env.time % env.year_length) / env.year_length
        season_idx = int(year_phase * 4) % 4
        season_names = ["Spring", "Summer", "Fall", "Winter"]
        current_season = season_names[season_idx]
        
        # Print status
        pop = stats["population"]["by_type"]
        print(f"{step+1:^5} | {env.time:5.1f} | {pop.get('plant', 0):^6} | {pop.get('herbivore', 0):^10} | {pop.get('decomposer', 0):^10} | {current_season:^8}")
        
        # Collect data for plots
        times.append(env.time)
        for org_type in populations:
            populations[org_type].append(pop.get(org_type, 0))
        
        # Energy flows
        energy_flows["photosynthesis"].append(ecosystem.energy_flow["photosynthesis"])
        energy_flows["consumption"].append(ecosystem.energy_flow["consumption"])
        energy_flows["decomposition"].append(ecosystem.energy_flow["decomposition"])
    
    print("\nSimulation completed!")
    
    # Print final ecosystem statistics
    stats = ecosystem.get_ecosystem_stats()
    
    print("\nFinal ecosystem state:")
    print(f"  Total biomass: {stats['biomass']['total']:.2f}")
    print(f"  Population: {stats['population']['total']} organisms")
    for org_type, count in stats['population']['by_type'].items():
        print(f"    {org_type}s: {count}")
    
    print(f"\nEnergy flows:")
    print(f"  Photosynthesis: {stats['energy']['flow']['photosynthesis']:.2f}")
    print(f"  Consumption: {stats['energy']['flow']['consumption']:.2f}")
    print(f"  Decomposition: {stats['energy']['flow']['decomposition']:.2f}")
    
    print(f"\nEnvironmental factors:")
    print(f"  Temperature: {stats['environmental_factors']['temperature']:.2f}")
    print(f"  Moisture: {stats['environmental_factors']['moisture']:.2f}")
    print(f"  Light level: {stats['environmental_factors']['light_level']:.2f}")
    
    return {
        "times": times,
        "populations": populations,
        "energy_flows": energy_flows
    }


def plot_ecosystem_data(data):
    """Plot the ecosystem simulation data."""
    print_separator("Ecosystem Data Visualization")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Population dynamics
    ax1.set_title("Population Dynamics")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Population")
    
    ax1.plot(data["times"], data["populations"]["plant"], 'g-', label="Plants")
    ax1.plot(data["times"], data["populations"]["herbivore"], 'b-', label="Herbivores")
    ax1.plot(data["times"], data["populations"]["decomposer"], 'r-', label="Decomposers")
    
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Energy flows
    ax2.set_title("Energy Flows")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Cumulative Energy")
    
    ax2.plot(data["times"], data["energy_flows"]["photosynthesis"], 'g-', label="Photosynthesis")
    ax2.plot(data["times"], data["energy_flows"]["consumption"], 'b-', label="Consumption")
    ax2.plot(data["times"], data["energy_flows"]["decomposition"], 'r-', label="Decomposition")
    
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the figure to the visualizations directory in the repository
    import os
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vis_dir = os.path.join(repo_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)  # Ensure directory exists
    save_path = os.path.join(vis_dir, "ecosystem_simulation.png")
    plt.savefig(save_path)
    print(f"Plots saved to '{save_path}'")
    
    # Display if in an interactive environment
    try:
        plt.show()
    except:
        pass


def main():
    """Run the ecosystem demonstration."""
    print("Enhanced Mycelium Network Ecosystem Demo")
    print("====================================")
    
    # Make sure we have random imported
    import random
    
    # Create the ecosystem
    ecosystem = demo_ecosystem_creation()
    
    # Add some dead organisms for the decomposers
    dead_plants = []
    for i in range(3):
        # Create a dead plant
        position = tuple(random.random() * ecosystem.environment.size for _ in range(ecosystem.environment.dimensions))
        dead_plant = Plant(
            organism_id=f"dead_plant_{i+1}",
            position=position,
            energy=0.1,
            size=random.uniform(0.5, 1.0)
        )
        dead_plant.alive = False  # Set as dead
        dead_plant_id = ecosystem.add_organism(dead_plant)
        dead_plants.append(dead_plant_id)
    
    print(f"Added {len(dead_plants)} dead plants for decomposers")
    
    # Let's manually position herbivores and plants close to each other to force interactions
    herbivores = ecosystem.get_organisms_by_type("herbivore")
    plants = ecosystem.get_organisms_by_type("plant")
    
    # Position each herbivore near a plant
    for i, herbivore in enumerate(herbivores):
        if i < len(plants):
            plant = plants[i]
            # Move herbivore closer to plant
            plant_pos = plant.position
            new_pos = []
            for j in range(len(plant_pos)):
                # Position slightly offset from plant
                offset = 0.05  # Small distance
                new_pos.append(plant_pos[j] + random.uniform(-offset, offset))
            herbivore.position = tuple(new_pos)
            print(f"Positioned herbivore {herbivore.id} near plant {plant.id}")
    
    # Position decomposers near dead plants
    decomposers = ecosystem.get_organisms_by_type("decomposer")
    for i, decomposer in enumerate(decomposers):
        if i < len(dead_plants):
            dead_plant_id = dead_plants[i]
            dead_plant = ecosystem.organisms[dead_plant_id]
            # Move decomposer right to the dead plant
            decomposer.position = dead_plant.position
            # Set it as attached
            decomposer.attached_to = dead_plant_id
            print(f"Positioned decomposer {decomposer.id} on dead plant {dead_plant_id}")
    
    # Run simulation
    simulation_data = run_ecosystem_simulation(ecosystem, steps=60, time_per_step=0.5)
    
    # Plot data
    try:
        plot_ecosystem_data(simulation_data)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print_separator("Demo Completed")
    print("\nSuccessfully demonstrated the enhanced ecosystem!")
    print("Key features demonstrated:")
    print("- Complex ecosystem with multiple organism types")
    print("- Population dynamics and interactions")
    print("- Energy flows and nutrient cycling")
    print("- Environmental effects on organisms")
    print("- Seasonal cycles and adaptive behaviors")


if __name__ == "__main__":
    main()
