# Contributing to Mycelium Network

Thank you for your interest in contributing to the Mycelium Network project! This document provides guidelines and instructions for contributing.

## Project Structure

The project is organized as follows:

```
mycelium_network/
├── mycelium/                  # Core package
│   ├── __init__.py            # Package initialization
│   ├── environment.py         # Basic environment
│   ├── network.py             # Base network implementation
│   ├── node.py                # Node implementation
│   ├── enhanced/              # Enhanced components
│   │   ├── __init__.py
│   │   ├── adaptive_network.py # Adaptive network implementation
│   │   ├── rich_environment.py # Rich environment implementation
│   │   ├── resource.py        # Resource types and handling
│   │   ├── environment.py     # Environment related utilities
│   │   ├── ml/                # Machine learning components
│   │   │   ├── __init__.py
│   │   │   ├── genetic.py     # Genetic optimization
│   ├── optimized/             # Performance optimized components
│   │   ├── __init__.py
│   │   ├── batch_network.py   # Batch processing network
│   ├── spatial/               # Spatial indexing components
│   │   ├── __init__.py
│   │   ├── quadtree.py        # Quadtree implementation
│   │   ├── octree.py          # Octree implementation
│   ├── tasks/                 # Task-specific implementations
│       ├── __init__.py
│       ├── classifier.py
│       ├── regressor.py
│       ├── anomaly_detector.py
├── examples/                  # Example scripts
├── tests/                     # Unit tests
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mycelium_network.git
   cd mycelium_network
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run tests to verify everything is working:
   ```bash
   python -m pytest tests/
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines for Python code
- Use 4 spaces for indentation (no tabs)
- Keep line length to 88 characters or less
- Use descriptive variable and function names
- Add docstrings to all functions, classes, and modules

### Documentation

- Every function should have a docstring describing:
  - What the function does
  - Parameters and their types
  - Return values and their types
  - Any exceptions that might be raised
- Use Google style docstrings
- Example:
  ```python
  def calculate_distance(point1, point2):
      """
      Calculate Euclidean distance between two points.
      
      Args:
          point1: Tuple with coordinates of first point
          point2: Tuple with coordinates of second point
          
      Returns:
          float: The Euclidean distance between the points
      """
  ```

### Testing

- Write unit tests for all new functionality
- Put tests in the `tests/` directory
- Use pytest for testing
- Run tests before submitting a pull request:
  ```bash
  python -m pytest tests/
  ```

### Git Workflow

- Create a new branch for each feature or bugfix
- Use descriptive branch names (e.g., `feature/adaptive-growth` or `fix/resource-allocation`)
- Make small, focused commits with clear messages
- Submit a pull request when ready for review

## Common Pitfalls and Solutions

### Import Issues

- When importing from enhanced modules, make sure to use the correct path:
  - Use `from mycelium.enhanced.rich_environment import RichEnvironment` (not `from mycelium.enhanced.environment import RichEnvironment`)

### Network Initialization

- Make sure the input and output sizes match when testing:
  ```python
  # Create network with 2 inputs
  network = AdaptiveMyceliumNetwork(input_size=2, output_size=1)
  
  # Must provide exactly 2 inputs
  outputs = network.forward([0.5, 0.7])  # Correct
  outputs = network.forward([0.5])       # Error: Expected 2 inputs, got 1
  ```

### Running Examples

- Many examples need to run from the project root directory to ensure correct imports
- Activate the virtual environment before running examples:
  ```bash
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  python examples/batch_processing_demo.py
  ```

## Key Concepts

### Environment

- **Environment**: Spatial context for the network
- **RichEnvironment**: Enhanced environment with more features (terrain layers, seasonal effects, etc.)

### Network

- **AdvancedMyceliumNetwork**: Core network implementation
- **AdaptiveMyceliumNetwork**: Network that adapts to environmental conditions
- **BatchProcessingNetwork**: Optimized network for batch processing

### Nodes

- **MyceliumNode**: Base node implementation
- Specialized nodes: storage, processing, sensor

### Optimization

- **GeneticOptimizer**: Evolves networks using genetic algorithms
- **BatchProcessing**: Improves performance through batch operations
- **SpatialIndex**: Optimizes spatial queries

## Getting Help

If you're stuck or have questions:
1. Check the example scripts in the `examples/` directory
2. Look at the unit tests for usage examples
3. Open an issue on GitHub with a clear description of the problem

Thank you for contributing to the Mycelium Network project!
