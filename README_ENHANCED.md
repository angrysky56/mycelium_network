# Enhanced Mycelium Network

> **Note:** This is an extension to the original Mycelium Network project that adds advanced environmental simulation and adaptive network capabilities.

## 🌿 Enhanced Features

The Enhanced Mycelium Network builds upon the biomimetic neural network foundation with these advanced features:

### Rich Environment
- **Multi-layered terrain** - 3D environments with different soil/air layers
- **Resource diversity** - Multiple interacting resource types (carbon, water, nitrogen, etc.)
- **Dynamic environmental conditions** - Temperature, moisture, pH, and light changes
- **Seasonal cycles** - Simulated seasons affecting growth patterns
- **Organism simulation** - Simple ecosystem with plants and herbivores

### Adaptive Network
- **Environmental adaptation** - Networks adapt to local conditions over time
- **Node specialization** - Specialized node types with different properties:
  - **Storage nodes** - Higher resource capacity for stability
  - **Processing nodes** - Enhanced computational abilities
  - **Sensor nodes** - Increased sensitivity to inputs
- **Resource efficiency** - Varying efficiency for different resource types
- **Complex growth patterns** - Growth influenced by environmental conditions

## 🚀 Getting Started with Enhanced Features

### Environment Setup

```python
from mycelium.enhanced.resource import ResourceType, Environmental_Factors
from mycelium.enhanced.environment import RichEnvironment

# Create a 3D environment
env = RichEnvironment(dimensions=3, size=1.0)

# Add different resource types
env.add_nutrient_cluster((0.5, 0.5, 0.6), 0.2, ResourceType.CARBON, 2.0)
env.add_nutrient_cluster((0.3, 0.3, 0.6), 0.15, ResourceType.WATER, 1.5)

# Create seasonal cycles
env.create_seasonal_cycle(year_length=100.0, intensity=0.7)

# Run environment simulation
for i in range(10):
    env.update(delta_time=1.0)
    print(f"Temperature: {env.factors.temperature:.2f}")
    print(f"Moisture: {env.factors.moisture:.2f}")
```

### Adaptive Network

```python
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork

# Create an adaptive network
network = AdaptiveMyceliumNetwork(
    environment=env,
    input_size=2,
    output_size=1,
    initial_nodes=15
)

# Run network with adaptive properties
for i in range(10):
    inputs = [0.7, 0.3]
    outputs = network.forward(inputs)
    
    # Check adaptation
    stats = network.get_specialization_statistics()
    print(f"Temperature adaptation: {stats['adaptation']['temperature_adaptation']:.2f}")
    print(f"Specialized nodes: {sum(len(nodes) for nodes in network.specializations.values())}")
```

## 📊 Environmental Visualization

The enhanced environment supports rich visualizations of its state:

```python
# Get environment snapshot
snapshot = env.get_state_snapshot()

# Resource distribution information
print(f"Total carbon: {snapshot['resources']['carbon']}")
print(f"Total water: {snapshot['resources']['water']}")

# Layer information (3D environments)
for layer in snapshot.get('layers', []):
    print(f"Layer {layer['name']}: {layer['resources']}")

# Organism information
print(f"Total organisms: {snapshot['organisms']['count']}")
for org_type, count in snapshot['organisms'].get('by_type', {}).items():
    print(f"  {org_type}: {count}")
```

## 🌱 Complete Example Workflow

The following example shows how to use the enhanced environment for training a specialized adaptive network:

```python
from mycelium.enhanced.resource import ResourceType
from mycelium.enhanced.environment import RichEnvironment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork

# Create environment with seasonal cycles
env = RichEnvironment(dimensions=3)
env.create_seasonal_cycle(year_length=24.0)  # One day = one year for rapid testing

# Add diverse resources
for resource_type in [ResourceType.CARBON, ResourceType.WATER, ResourceType.NITROGEN]:
    env.add_nutrient_cluster(
        center=(random.random(), random.random(), 0.6), 
        radius=0.15, 
        resource_type=resource_type,
        amount=1.5
    )

# Create adaptive network
network = AdaptiveMyceliumNetwork(
    environment=env,
    input_size=4,
    output_size=2,
    initial_nodes=20
)

# Training loop with environmental adaptation
for epoch in range(50):
    # Update environment - progress through seasons
    env.update(delta_time=0.5)
    
    # Training data - could be from any source
    X_batch = [[random.random() for _ in range(4)] for _ in range(10)]
    y_batch = [[random.random() for _ in range(2)] for _ in range(10)]
    
    # Train on batch with adaptation to current conditions
    for X, y in zip(X_batch, y_batch):
        outputs = network.forward(X)
        # Normally would do backprop here
    
    # Check adaptation progress periodically
    if epoch % 10 == 0:
        stats = network.get_specialization_statistics()
        print(f"Epoch {epoch} - Nodes: {stats['node_counts']['total']}")
        print(f"Temperature: {env.factors.temperature:.2f}, Adaptation: {stats['adaptation']['temperature_adaptation']:.2f}")
        print(f"Specialized nodes: {sum(count for type, count in stats['node_counts'].items() if type not in ['input', 'output', 'regular'])}")
```

For more information, see the complete documentation in the `ENHANCED_ENVIRONMENT_SETUP.md` file.
