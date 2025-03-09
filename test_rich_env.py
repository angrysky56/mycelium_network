#!/usr/bin/env python3
"""
Quick test for the rich environment.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.resource import ResourceType

def test_rich_environment():
    """Test basic functionality of the rich environment."""
    print("\n=== Testing Rich Environment ===")
    
    # Create a rich environment
    env = RichEnvironment(dimensions=3, size=1.0, name="Test Environment")
    
    # Print initial state
    print(f"Environment created: {env.name}")
    print(f"Dimensions: {env.dimensions}")
    
    if hasattr(env, 'layers'):
        print(f"Number of layers: {len(env.layers)}")
    
    # Add some resources
    print("\nAdding resources...")
    env.add_resource((0.5, 0.5, 0.6), 1.0, ResourceType.CARBON)
    env.add_resource((0.3, 0.3, 0.6), 0.8, ResourceType.WATER)
    
    # Check resources
    carbon_total = env.get_total_resources(ResourceType.CARBON)
    water_total = env.get_total_resources(ResourceType.WATER)
    
    print(f"Total Carbon: {carbon_total:.2f}")
    print(f"Total Water: {water_total:.2f}")
    
    print("\nRich environment test completed successfully!")

if __name__ == "__main__":
    try:
        test_rich_environment()
        print("Test succeeded!")
    except Exception as e:
        print(f"Test failed with error: {e}")
