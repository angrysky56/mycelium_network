# Enhanced Environment for Mycelium Networks

This document provides instructions for activating and using the enhanced environment features developed for the Mycelium Network.

## Overview

The enhanced environment adds several significant capabilities to the basic Mycelium Network environment:

1. **Multi-layered terrain** - 3D environments with different soil/air layers having unique properties
2. **Dynamic environmental factors** - Temperature, moisture, pH, light levels that change over time
3. **Resource diversity** - Multiple resource types (carbon, water, nitrogen, etc.) with interactions
4. **Seasonal cycles** - Simulated seasons that affect growth patterns and resource availability
5. **Organism simulation** - Simple simulation of other organisms that interact with resources

The adaptive network implementation complements these features by:

1. **Environmental adaptation** - Networks adapt to temperature, moisture, and other conditions
2. **Node specialization** - Different specialized node types with unique properties
3. **Resource efficiency** - Varying efficiency in processing different resource types
4. **Enhanced growth patterns** - Growth influenced by environmental conditions

## Setup Instructions

The enhanced environment modules have been added to the project structure. To use these features:

1. **Activate the virtual environment**:
   ```bash
   # Navigate to the project directory
   cd /home/ty/Repositories/ai_workspace/mycelium_network

   # Activate the virtual environment
   source venv/bin/activate
   ```

2. **Verify dependencies** (already in requirements.txt):
   ```bash
   # Install or update dependencies
   pip install -r requirements.txt
   ```

3. **Import the enhanced modules** in your code:
   ```python
   from mycelium.enhanced.resource import ResourceType, Environmental_Factors
   from mycelium.enhanced.environment import RichEnvironment
   from mycelium.enhanced.layers import TerrainLayer
   from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
   ```

## Example Usage

### Creating a Rich Environment

```python
# Create a 3D environment
env = RichEnvironment(dimensions=3, size=1.0, name="Test Environment")

# Add different resource types
env.add_nutrient_cluster((0.5, 0.5, 0.6), 0.2, ResourceType.CARBON, 2.0)
env.add_nutrient_cluster((0.3, 0.3, 0.6), 0.15, ResourceType.WATER, 1.5)
env.add_nutrient_cluster((0.7, 0.7, 0.6), 0.1, ResourceType.NITROGEN, 1.0)

# Create seasonal cycles
env.create_seasonal_cycle(year_length=100.0, intensity=0.7)
```

### Creating an Adaptive Network

```python
# Create an adaptive network
network = AdaptiveMyceliumNetwork(
    environment=env,
    input_size=2,
    output_size=1,
    initial_nodes=15
)

# Run the network
for i in range(10):
    # Update environment conditions
    env.update(delta_time=1.0)
    
    # Run network forward pass
    inputs = [0.7, 0.3]
    outputs = network.forward(inputs)
    
    # Check adaptation metrics
    stats = network.get_specialization_statistics()
    print(f"Temperature adaptation: {stats['adaptation']['temperature_adaptation']:.3f}")
    print(f"Moisture adaptation: {stats['adaptation']['moisture_adaptation']:.3f}")
```

## Running the Example Scripts

We've included example scripts to demonstrate the enhanced environment:

```bash
# Run the enhanced environment test
python examples/enhanced_environment_test.py

# Run the adaptive network test
python examples/adaptive_network_test.py
```

## Understanding the Enhanced Environment Architecture

### Resource Types

The `ResourceType` enum defines various resource types that can exist in the environment:

- CARBON - Basic carbon resources
- NITROGEN - Nitrogen compounds
- PHOSPHORUS - Phosphorus compounds
- WATER - Water resources
- SUGAR - Processed carbon resources
- PROTEIN - Complex organic compounds
- MINERAL - Inorganic minerals
- LIGHT - Light resource (mostly for plants)

### Environmental Factors

The `Environmental_Factors` class contains environmental conditions:

- temperature (0-1) - Environmental temperature
- moisture (0-1) - Water content in the environment
- ph (0-14) - pH level
- light_level (0-1) - Amount of light
- toxicity (0-1) - Level of toxins
- oxygen (0-1) - Oxygen level
- wind (0-1) - Wind strength
- gravity (relative) - Gravity strength
- season (0-3) - Current season (spring, summer, fall, winter)

### Terrain Layers

In 3D environments, layers represent different strata of the environment:

- Air layer - Top layer with light and gases
- Topsoil - Rich soil layer with organic matter
- Subsoil - Middle soil layer with some nutrients
- Bedrock - Bottom layer with minerals

### Adaptive Network Specializations

The adaptive network can develop specialized node types:

- Regular nodes - Basic processing nodes
- Storage nodes - Higher resource capacity but slower response
- Processing nodes - Faster processing but higher energy usage
- Sensor nodes - Highly sensitive to inputs but energy inefficient
