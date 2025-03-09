#!/usr/bin/env python3
"""
A simple test script for the enhanced environment implementation.
"""

import os
import sys
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Imported paths:")
print(sys.path)

try:
    # Import the enhanced environment components
    from mycelium.enhanced.resource import ResourceType, Environmental_Factors
    from mycelium.enhanced.layers import TerrainLayer
    
    print("\nSuccessfully imported enhanced environment modules!")
    
    # Test ResourceType enum
    print("\nResource Types:")
    for resource in ResourceType:
        print(f"  {resource.name}")
    
    # Test Environmental_Factors
    factors = Environmental_Factors(
        temperature=0.7,
        moisture=0.5,
        light_level=0.8
    )
    
    print("\nEnvironmental Factors:")
    print(f"  Temperature: {factors.temperature}")
    print(f"  Moisture: {factors.moisture}")
    print(f"  Light Level: {factors.light_level}")
    
    # Test TerrainLayer
    topsoil = TerrainLayer(
        name="Topsoil",
        height_range=(0.6, 0.7),
        conductivity=0.9,
        density=0.8,
        resource_affinity={
            ResourceType.CARBON: 1.5,
            ResourceType.NITROGEN: 1.2,
            ResourceType.WATER: 1.0,
        },
        description="Rich topsoil with organic matter"
    )
    
    print("\nTerrain Layer:")
    print(f"  Name: {topsoil.name}")
    print(f"  Height Range: {topsoil.height_range}")
    print(f"  Conductivity: {topsoil.conductivity}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
