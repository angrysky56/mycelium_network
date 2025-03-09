#!/usr/bin/env python3
"""
Test script for the enhanced environment implementation.

This script creates a rich environment and demonstrates its features:
- Multi-layered terrain
- Dynamic environmental conditions
- Resource interactions
- Organism simulation
"""

import os
import sys
import random
import time
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import standard environment for comparison
from mycelium.environment import Environment

# Import the enhanced environment components
from mycelium.enhanced.resource import ResourceType, Environmental_Factors
from mycelium.enhanced.environment import TerrainLayer
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.environment_utils import (
    count_resources_in_layer,
    apply_seasonal_spring_effect,
    apply_seasonal_summer_effect,
    apply_seasonal_fall_effect,
    apply_seasonal_winter_effect
)


def test_basic_functionality():
    """Test the basic functionality of the rich environment."""
    print("\n=== Testing Basic Functionality ===")
    
    # Create a rich environment
    env = RichEnvironment(dimensions=3, size=1.0, name="Test Environment")
    
    # Print initial state
    print(f"Environment created: {env.name}")
    print(f"Dimensions: {env.dimensions}")
    print(f"Number of layers: {len(env.layers)}")
    
    # Add some resources
    print("\nAdding resources...")
    env.add_resource((0.5, 0.5, 0.6), 1.0, ResourceType.CARBON)
    env.add_resource((0.3, 0.3, 0.6), 0.8, ResourceType.WATER)
    env.add_resource((0.7, 0.7, 0.6), 0.5, ResourceType.NITROGEN)
    
    # Add a nutrient cluster
    env.add_nutrient_cluster((0.2, 0.8, 0.6), 0.1, ResourceType.SUGAR, 2.0)
    
    # Check resources
    carbon_total = env.get_total_resources(ResourceType.CARBON)
    water_total = env.get_total_resources(ResourceType.WATER)
    nitrogen_total = env.get_total_resources(ResourceType.NITROGEN)
    sugar_total = env.get_total_resources(ResourceType.SUGAR)
    
    print(f"Total Carbon: {carbon_total:.2f}")
    print(f"Total Water: {water_total:.2f}")
    print(f"Total Nitrogen: {nitrogen_total:.2f}")
    print(f"Total Sugar: {sugar_total:.2f}")
    
    # Test environment factors
    print("\nEnvironmental Factors:")
    print(f"Temperature: {env.factors.temperature:.2f}")
    print(f"Moisture: {env.factors.moisture:.2f}")
    print(f"Light Level: {env.factors.light_level:.2f}")
    print(f"pH: {env.factors.ph:.2f}")
    
    # Check layer specific resources
    if env.layers:
        print("\nLayer Resources:")
        for i, layer in enumerate(env.layers):
            layer_resources = sum(sum(res.values()) for res in layer.resources.values())
            print(f"  Layer {i} ({layer.name}): {layer_resources:.2f} total resources")
    
    print("\nBasic functionality test completed successfully!")


def test_environment_update():
    """Test the environment update functionality."""
    print("\n=== Testing Environment Update ===")
    
    # Create a rich environment
    env = RichEnvironment(dimensions=3, size=1.0)
    
    # Add initial resources
    print("Adding initial resources...")
    env.add_nutrient_cluster((0.5, 0.5, 0.6), 0.2, ResourceType.CARBON, 2.0)
    env.add_nutrient_cluster((0.5, 0.5, 0.6), 0.15, ResourceType.WATER, 1.5)
    
    # Get initial state
    initial_carbon = env.get_total_resources(ResourceType.CARBON)
    initial_water = env.get_total_resources(ResourceType.WATER)
    initial_sugar = env.get_total_resources(ResourceType.SUGAR)
    
    print(f"Initial Carbon: {initial_carbon:.2f}")
    print(f"Initial Water: {initial_water:.2f}")
    print(f"Initial Sugar: {initial_sugar:.2f}")
    print(f"Initial Temperature: {env.factors.temperature:.2f}")
    print(f"Initial Light Level: {env.factors.light_level:.2f}")
    
    # Run simulation for several steps
    print("\nRunning simulation for 5 steps...")
    for i in range(5):
        print(f"\nStep {i+1}:")
        
        # Update environment
        env.update(delta_time=1.0)
        
        # Get updated state
        current_carbon = env.get_total_resources(ResourceType.CARBON)
        current_water = env.get_total_resources(ResourceType.WATER)
        current_sugar = env.get_total_resources(ResourceType.SUGAR)
        
        print(f"  Carbon: {current_carbon:.2f} ({current_carbon - initial_carbon:+.2f})")
        print(f"  Water: {current_water:.2f} ({current_water - initial_water:+.2f})")
        print(f"  Sugar: {current_sugar:.2f} ({current_sugar - initial_sugar:+.2f})")
        print(f"  Temperature: {env.factors.temperature:.2f}")
        print(f"  Light Level: {env.factors.light_level:.2f}")
        print(f"  Time: {env.time:.2f}, Day Phase: {(env.time % env.day_length) / env.day_length:.2f}")
    
    print("\nEnvironment update test completed successfully!")


def test_organism_simulation():
    """Test the organism simulation functionality."""
    print("\n=== Testing Organism Simulation ===")
    
    # Create a rich environment
    env = RichEnvironment(dimensions=2, size=1.0)
    
    # Add some resources
    print("Adding resources...")
    env.add_nutrient_cluster((0.5, 0.5), 0.3, ResourceType.CARBON, 3.0)
    env.add_nutrient_cluster((0.3, 0.7), 0.25, ResourceType.WATER, 2.0)
    
    # Add organisms
    print("\nAdding organisms...")
    env.add_organism("plant1", (0.4, 0.6), "plant", {"size": 0.8, "productivity": 0.1})
    env.add_organism("plant2", (0.6, 0.4), "plant", {"size": 0.5, "productivity": 0.08})
    env.add_organism("herbivore1", (0.2, 0.2), "herbivore", {"speed": 0.1, "energy": 0.7})
    
    print(f"Initial organisms: {len(env.organisms)}")
    
    # Run simulation for several steps
    print("\nRunning simulation for 10 steps...")
    for i in range(10):
        # Update environment
        env.update(delta_time=0.5)
        
        # Count live organisms
        live_organisms = [org for org in env.organisms.values() if org["alive"]]
        
        # Check resource changes
        carbon = env.get_total_resources(ResourceType.CARBON)
        water = env.get_total_resources(ResourceType.WATER)
        sugar = env.get_total_resources(ResourceType.SUGAR)
        
        if i % 2 == 0 or i == 9:
            print(f"\nStep {i+1}:")
            print(f"  Live organisms: {len(live_organisms)}")
            print(f"  Carbon: {carbon:.2f}")
            print(f"  Water: {water:.2f}")
            print(f"  Sugar: {sugar:.2f}")
            
            # Show organism positions
            positions = {
                org_id: org["position"] 
                for org_id, org in env.organisms.items() 
                if org["alive"]
            }
            print(f"  Organism positions: {positions}")
    
    print("\nOrganism simulation test completed successfully!")


def test_seasonal_effects():
    """Test the seasonal effects functionality."""
    print("\n=== Testing Seasonal Effects ===")
    
    # Create a rich environment with accelerated seasons
    env = RichEnvironment(dimensions=3, size=1.0)
    year_length = 24.0  # One day = one year for testing
    env.create_seasonal_cycle(year_length=year_length, intensity=0.7)
    
    # Add initial resources
    initial_carbon = 2.0
    initial_water = 1.5
    print(f"Adding initial resources (Carbon: {initial_carbon}, Water: {initial_water})...")
    env.add_nutrient_cluster((0.5, 0.5, 0.6), 0.2, ResourceType.CARBON, initial_carbon)
    env.add_nutrient_cluster((0.5, 0.5, 0.6), 0.15, ResourceType.WATER, initial_water)
    
    # Run simulation for a full year
    steps = 12
    step_size = year_length / steps
    
    for i in range(steps):
        # Update environment
        env.update(delta_time=step_size)
        env.apply_seasonal_effects(delta_time=step_size)
        
        # Calculate season
        year_phase = (env.time % year_length) / year_length
        season_idx = int(year_phase * 4) % 4
        season_names = ["Spring", "Summer", "Fall", "Winter"]
        current_season = season_names[season_idx]
        
        # Get resource levels
        carbon = env.get_total_resources(ResourceType.CARBON)
        water = env.get_total_resources(ResourceType.WATER)
        nitrogen = env.get_total_resources(ResourceType.NITROGEN)
        
        print(f"\nStep {i+1} - Time: {env.time:.1f}, Season: {current_season} ({year_phase*100:.0f}%):")
        print(f"  Temperature: {env.factors.temperature:.2f}")
        print(f"  Moisture: {env.factors.moisture:.2f}")
        print(f"  Light: {env.factors.light_level:.2f}")
        print(f"  Carbon: {carbon:.2f} ({carbon - initial_carbon:+.2f})")
        print(f"  Water: {water:.2f} ({water - initial_water:+.2f})")
        print(f"  Nitrogen: {nitrogen:.2f}")
    
    print("\nSeasonal effects test completed successfully!")


def compare_with_standard_environment():
    """Compare the rich environment with the standard environment."""
    print("\n=== Comparing with Standard Environment ===")
    
    # Create a standard environment
    std_env = Environment(dimensions=2, size=1.0)
    
    # Create a rich environment
    rich_env = RichEnvironment(dimensions=2, size=1.0)
    
    # Add similar resources to both
    print("Adding resources to both environments...")
    
    # Standard environment - using single resource values
    std_env.add_resource((0.5, 0.5), 1.0)
    std_env.add_resource((0.3, 0.7), 0.8)
    std_env.add_resource((0.7, 0.3), 0.5)
    
    # Rich environment - using typed resources
    rich_env.add_resource((0.5, 0.5), 1.0, ResourceType.CARBON)
    rich_env.add_resource((0.3, 0.7), 0.8, ResourceType.WATER)
    rich_env.add_resource((0.7, 0.3), 0.5, ResourceType.NITROGEN)
    
    # Add obstacles to both
    std_env.add_obstacle((0.2, 0.2), 0.1)
    rich_env.add_obstacle((0.2, 0.2), 0.1)
    
    # Compare resource retrieval
    std_center_resources = sum(std_env.get_resources_in_range((0.5, 0.5), 0.3).values())
    rich_center_resources = sum(
        sum(res.values()) 
        for res in rich_env.get_resources_in_range((0.5, 0.5), 0.3).values()
    )
    
    print(f"\nResources near center:")
    print(f"  Standard Environment: {std_center_resources:.2f}")
    print(f"  Rich Environment: {rich_center_resources:.2f}")
    
    # Compare position validity check
    test_positions = [
        (0.5, 0.5),  # center, should be valid
        (0.2, 0.2),  # in obstacle, should be invalid
        (1.1, 0.5),  # out of bounds, should be invalid
    ]
    
    print("\nPosition validity checks:")
    for pos in test_positions:
        std_valid = std_env.is_position_valid(pos)
        rich_valid = rich_env.is_position_valid(pos)
        print(f"  Position {pos}: Standard={std_valid}, Rich={rich_valid}")
    
    # Compare performance
    print("\nPerformance comparison (1000 resource lookups):")
    
    # Standard environment
    start_time = time.time()
    for _ in range(1000):
        x, y = random.random(), random.random()
        std_env.get_resources_in_range((x, y), 0.2)
    std_time = time.time() - start_time
    
    # Rich environment
    start_time = time.time()
    for _ in range(1000):
        x, y = random.random(), random.random()
        rich_env.get_resources_in_range((x, y), 0.2)
    rich_time = time.time() - start_time
    
    print(f"  Standard Environment: {std_time:.4f} seconds")
    print(f"  Rich Environment: {rich_time:.4f} seconds")
    print(f"  Ratio: {rich_time/std_time:.2f}x")
    
    print("\nComparison completed!")


def main():
    """Run all tests."""
    print("=== Enhanced Environment Tests ===")
    
    # Run tests
    test_basic_functionality()
    test_environment_update()
    test_organism_simulation()
    test_seasonal_effects()
    compare_with_standard_environment()
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
