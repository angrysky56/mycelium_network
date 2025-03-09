class EnhancedEnvironment(Environment):
    """
    Enhanced environment with more biologically relevant features.
    """
    
    def __init__(self, dimensions: int = 2, size: float = 1.0):
        """
        Initialize the enhanced environment.
        
        Args:
            dimensions: Number of spatial dimensions
            size: Size of the environment (0.0 to size)
        """
        super().__init__(dimensions, size)
        
        # Environmental factors
        self.moisture_level = 0.7  # 0.0 (dry) to 1.0 (saturated)
        self.temperature = 0.5  # 0.0 (cold) to 1.0 (hot)
        self.ph_level = 0.5  # 0.0 (acidic) to 1.0 (alkaline)
        self.light_level = 0.6  # 0.0 (dark) to 1.0 (bright)
        
        # Resources and obstacles
        self.resources = []  # [{id, position, type, level},...]
        self.obstacles = []  # [{id, position, type, size},...]
        
        # Resource generation parameters
        self.resource_depletion_rate = 0.01
        self.resource_regeneration_rate = 0.005
        self.resource_clustering = 0.7  # Tendency of resources to cluster
        
        # Node registry
        self.nodes = {}  # {node_id: node}
        
        # Initialize environment
        self._initialize_environment()
        
    def _initialize_environment(self) -> None:
        """Initialize the environment with resources and obstacles."""
        # Create resource clusters
        self._create_resource_clusters(4, 5, 10)
        
        # Create obstacles
        self._create_obstacles(10)
        
    def _create_resource_clusters(self, num_clusters: int, resources_per_cluster: int, radius: float = 0.1) -> None:
        """
        Create clusters of resources.
        
        Args:
            num_clusters: Number of clusters to create
            resources_per_cluster: Resources per cluster
            radius: Cluster radius
        """
        resource_types = ['carbon', 'nitrogen', 'phosphorus', 'general']
        resource_id = 0
        
        for _ in range(num_clusters):
            # Cluster center
            center = self.get_random_position()
            main_type = random.choice(resource_types)
            
            # Create resources around center
            for _ in range(resources_per_cluster):
                # Random position within radius of center
                offset = [random.uniform(-radius, radius) for _ in range(self.dimensions)]
                position = tuple(min(self.size, max(0, c + o)) for c, o in zip(center, offset))
                
                # Resource properties
                if random.random() < 0.7:
                    resource_type = main_type  # Most resources in a cluster are the same type
                else:
                    resource_type = random.choice(resource_types)
                    
                level = random.uniform(0.3, 1.0)
                
                # Add resource
                self.resources.append({
                    'id': resource_id,
                    'position': position,
                    'type': resource_type,
                    'level': level,
                    'created_at': time.time()
                })
                resource_id += 1
    
    def _create_obstacles(self, num_obstacles: int) -> None:
        """
        Create obstacles in the environment.
        
        Args:
            num_obstacles: Number of obstacles to create
        """
        obstacle_types = ['cellulose', 'lignin', 'protein']
        obstacle_id = 0
        
        for _ in range(num_obstacles):
            position = self.get_random_position()
            obstacle_type = random.choice(obstacle_types)
            size = random.uniform(0.05, 0.15)
            
            self.obstacles.append({
                'id': obstacle_id,
                'position': position,
                'type': obstacle_type,
                'size': size
            })
            obstacle_id += 1
            
    def update(self) -> None:
        """Update the environment (resource regeneration, depletion, etc.)."""
        # Update environmental conditions
        self._update_environment_conditions()
        
        # Update resources
        self._update_resources()
        
    def _update_environment_conditions(self) -> None:
        """Update environmental conditions with small fluctuations."""
        # Small random fluctuations
        self.moisture_level = max(0.1, min(1.0, self.moisture_level + random.uniform(-0.03, 0.03)))
        self.temperature = max(0.1, min(1.0, self.temperature + random.uniform(-0.02, 0.02)))
        self.ph_level = max(0.1, min(1.0, self.ph_level + random.uniform(-0.01, 0.01)))
        
        # Light follows a day/night cycle
        day_cycle = (time.time() % 100) / 100  # 0.0 to 1.0
        self.light_level = 0.5 + 0.4 * math.sin(day_cycle * 2 * math.pi)
        
    def _update_resources(self) -> None:
        """Update resources (depletion, regeneration)."""
        # Deplete resources
        for resource in self.resources:
            resource['level'] = max(0.0, resource['level'] - self.resource_depletion_rate)
            
        # Remove depleted resources
        self.resources = [r for r in self.resources if r['level'] > 0.1]
        
        # Regenerate resources with small probability
        if random.random() < 0.05:
            self._create_resource_clusters(1, random.randint(2, 5))