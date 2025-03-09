# Mycelium Network Examples

This directory contains example scripts that demonstrate the features and capabilities of the Mycelium Network project. Each example focuses on a specific aspect of the system.

## Prerequisites

Before running any examples, make sure you have:

1. Activated the virtual environment:
   ```bash
   cd /path/to/mycelium_network
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Installed all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the health check to verify your setup:
   ```bash
   python tools/health_check.py
   ```

## Running Examples

All examples should be run from the project root directory:

```bash
cd /path/to/mycelium_network
python examples/batch_processing_demo.py
```

## Available Examples

| Example File | Description | Components Demonstrated |
|--------------|-------------|-------------------------|
| `adaptive_network_test.py` | Tests adaptive network functionality | AdaptiveMyceliumNetwork, RichEnvironment |
| `batch_processing_demo.py` | Demonstrates batch processing performance | BatchProcessingNetwork |
| `batch_processing_test.py` | Tests batch processing functionality | BatchProcessingNetwork |
| `ecosystem_demo.py` | Demonstrates ecosystem simulation | RichEnvironment, organisms, interactions |
| `enhanced_demo.py` | Demonstrates enhanced features | AdaptiveMyceliumNetwork, RichEnvironment |
| `enhanced_environment_test.py` | Tests rich environment functionality | RichEnvironment |
| `genetic_optimization_demo.py` | Demonstrates genetic algorithm optimization | GeneticOptimizer |
| `genetic_test.py` | Tests genetic optimization | GeneticOptimizer |
| `ml_integration_demo.py` | Demonstrates machine learning integration | AdaptiveMyceliumNetwork |
| `simple_environment_test.py` | Tests basic environment functionality | Environment |
| `simple_genetic_test.py` | Simplified genetic optimization | GeneticOptimizer |
| `spatial_index_test.py` | Tests spatial indexing | SpatialIndex |
| `spatial_indexing_demo.py` | Demonstrates spatial indexing performance | SpatialIndex |

## Recommended Learning Path

If you're new to the Mycelium Network project, we recommend exploring the examples in this order:

1. **Basic Concepts**:
   - `simple_environment_test.py` - Basic environment
   - `quick_test.py` - (In project root) - Basic functionality

2. **Enhanced Features**:
   - `enhanced_environment_test.py` - Rich environment
   - `adaptive_network_test.py` - Network adaptation

3. **Performance Optimizations**:
   - `spatial_indexing_demo.py` - Spatial indexing
   - `batch_processing_demo.py` - Batch processing

4. **Advanced Features**:
   - `genetic_test.py` - Genetic optimization
   - `ecosystem_demo.py` - Ecosystem simulation

## Example Output

Each example will print out information about its operation. For instance, running `batch_processing_demo.py` will display performance comparisons between standard and batch optimized networks:

```
Batch Processing Optimization Demo
=================================
Forward Pass Performance Comparison
===================================
Standard network: 0.0094 seconds
Batch network (size=5): 0.0035 seconds
Batch network (size=10): 0.0029 seconds
Batch network (size=20): 0.0028 seconds

Training Performance Comparison
==============================
Standard: 0.1585 seconds, final error: 0.2500
Batch 5: 0.0981 seconds, final error: 0.2446
Batch 10: 0.0459 seconds, final error: 0.2439
Batch 20: 0.0391 seconds, final error: 0.2412
```

## Troubleshooting

If you experience issues with an example:

1. Make sure you're running from the project root directory
2. Verify the virtual environment is activated
3. Check for import errors (see USAGE_GUIDE.md)
4. Run the health check script: `python tools/health_check.py`

## Creating Your Own Examples

To create your own examples:

1. Start by copying and modifying an existing example
2. Make sure to import the correct modules
3. Follow the patterns shown in the existing examples
4. Use appropriate docstrings to explain what your example demonstrates

See the `USAGE_GUIDE.md` in the root directory for more detailed usage patterns.
