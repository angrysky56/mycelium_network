 obstacle.
        
        Args:
            obstacle_type: Type of obstacle
            
        Returns:
            Effectiveness against this obstacle (0.0 to 1.0)
        """
        if not self.active:
            return 0.0
            
        # Match enzyme to obstacle type
        effectiveness = {
            'cellulose': {'cellulase': 0.8, 'beta_glucosidase': 0.5},
            'lignin': {'lignin_peroxidase': 0.7, 'manganese_peroxidase': 0.6},
            'protein': {'protease': 0.9},
            'lipid': {'lipase': 0.8}
        }
        
        return effectiveness.get(obstacle_type, {}).get(self.type, 0.1) * self.strength


class BiomimeticNode(MyceliumNode):
    """
    Enhanced mycelium node with more biologically accurate behaviors.
    """
    
    def __init__(self, node_id: int, position: Tuple[float, ...], node_type: str = 'regular'):
        """
        Initialize a biomimetic node.
        
        Args:
            node_id: Unique identifier for the node
            position: Spatial coordinates in the environment
            node_type: Type of node ('input', 'hidden', 'output', or 'regular')
        """
        super().__init__(node_id, position, node_type)
        
        # Enhanced biological properties
        self.hyphal_branches = []  # Sub-branches from this node
        self.anastomosis_targets = set()  # Nodes that this node has fused with
        self.spatial_memory = {}  # {position_hash: (resource_level, timestamp)}
        
        # Enzymatic capabilities
        self.enzymes = {
            'cellulase': EnzymaticAbility('cellulase', random.uniform(0.1, 0.4)),
            'lignin_peroxidase': EnzymaticAbility('lignin_peroxidase', random.uniform(0.0, 0.3)),
            'protease': EnzymaticAbility('protease', random.uniform(0.0, 0.2))
        }
        
        # Stress responses
        self.stress_level = 0.0
        self.stress_memory = []  # [(stress_type, level, timestamp),...]
        
        # Enhanced metabolic properties
        self.nutrient_storage = defaultdict(float)  # {nutrient_type: amount}
        self.metabolic_rate = random.uniform(0.05, 0.15)  # Base metabolic rate
        self.growth_potential = random.uniform(0.5, 1.5)  # Growth capacity
        
        # Circadian-like rhythm
        self.cycle_phase = random.uniform(0, 2 * math.pi)  # Random starting phase
        self.cycle_duration = random.randint(20, 40)  # Cycle length in iterations
        
    def sense_environment(self, environment: Environment, radius: float = 0.2) -> Dict:
        """
        Sense the surrounding environment for resources, obstacles, and signals.
        
        Args:
            environment: Environment object
            radius: Sensing radius
            
        Returns:
            Dictionary with sensed information
        """
        sensed_info = {
            'resources': [],
            'obstacles': [],
            'signals': [],
            'nodes': []
        }
        
        # Sense resources
        resources = environment.get_resources_in_radius(self.position, radius)
        sensed_info['resources'] = resources
        
        # Sense obstacles
        obstacles = environment.get_obstacles_in_radius(self.position, radius)
        sensed_info['obstacles'] = obstacles
        
        # Update spatial memory
        for resource in resources:
            pos_hash = hash(tuple(resource['position']))
            self.spatial_memory[pos_hash] = (resource['level'], time.time())
            
        # Record stress from obstacles
        for obstacle in obstacles:
            if obstacle['distance'] < 0.1:  # Close obstacle causes stress
                self.add_stress('obstacle', 0.2 * (1 - obstacle['distance'] / 0.1))
                
        return sensed_info