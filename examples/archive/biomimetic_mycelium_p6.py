    def consume_resource(self, resource_id: int, amount: float) -> float:
        """
        Consume some amount of a resource.
        
        Args:
            resource_id: ID of the resource
            amount: Amount to consume
            
        Returns:
            Amount actually consumed
        """
        for resource in self.resources:
            if resource['id'] == resource_id:
                consumed = min(resource['level'], amount)
                resource['level'] -= consumed
                return consumed
        return 0.0
        
    def is_position_valid(self, position: Tuple[float, ...]) -> bool:
        """
        Check if a position is valid (inside bounds and not inside an obstacle).
        
        Args:
            position: Position to check
            
        Returns:
            True if valid, False otherwise
        """
        # Check boundaries
        if not all(0 <= p <= self.size for p in position):
            return False
            
        # Check obstacles
        for obstacle in self.obstacles:
            distance = self.calculate_distance(position, obstacle['position'])
            if distance <= obstacle['size']:
                return False
                
        return True
        
    def get_environmental_conditions(self, position: Tuple[float, ...]) -> Dict[str, float]:
        """
        Get environmental conditions at a specific position.
        
        Args:
            position: Position to check
            
        Returns:
            Dictionary of conditions
        """
        # Base conditions
        conditions = {
            'moisture': self.moisture_level,
            'temperature': self.temperature,
            'ph': self.ph_level,
            'light': self.light_level
        }
        
        # Add spatial variations based on position
        x_factor = position[0] / self.size
        conditions['moisture'] = max(0.1, min(1.0, conditions['moisture'] * (1.0 - 0.3 * x_factor)))
        conditions['temperature'] = max(0.1, min(1.0, conditions['temperature'] * (1.0 + 0.2 * x_factor)))
        
        return conditions
        
    def register_node(self, node_id: int, node) -> None:
        """
        Register a node in the environment.
        
        Args:
            node_id: Node ID
            node: Node object
        """
        self.nodes[node_id] = node
        
    def get_highest_node_id(self) -> int:
        """
        Get the highest node ID in the environment.
        
        Returns:
            Highest node ID or 0 if no nodes
        """
        if not self.nodes:
            return 0
        return max(self.nodes.keys())


class BiomimeticNetwork(AdvancedMyceliumNetwork):
    """
    Enhanced mycelium network with more biologically accurate features and behaviors.
    """
    
    def __init__(
        self, 
        environment: EnhancedEnvironment = None,
        input_size: int = 3, 
        output_size: int = 1, 
        initial_nodes: int = 20
    ):
        """
        Initialize the biomimetic network.
        
        Args:
            environment: Enhanced environment in which the network exists
            input_size: Number of input nodes
            output_size: Number of output nodes
            initial_nodes: Initial number of regular nodes
        """
        # Create environment if not provided
        if environment is None:
            environment = EnhancedEnvironment()
            
        # Initialize parent
        super().__init__(environment, input_size, output_size, initial_nodes)
        
        # Enhanced properties
        self.anastomosis_count = 0
        self.enzymatic_activity = defaultdict(float)  # {enzyme_type: activity_level}
        self.network_age = 0
        self.average_stress = 0.0
        self.resource_efficiency = 1.0
        self.adaptation_history = []
        
        # Replace regular nodes with biomimetic nodes
        self._convert_to_biomimetic_nodes()
        
        # Additional connections based on anastomosis
        self._initialize_hyphal_connections()