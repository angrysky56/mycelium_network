    def _select_enzyme_for_obstacle(self, obstacle_type: str) -> Optional[str]:
        """
        Select the most appropriate enzyme for an obstacle.
        
        Args:
            obstacle_type: Type of obstacle
            
        Returns:
            Enzyme type or None if no suitable enzyme
        """
        enzyme_mapping = {
            'cellulose': 'cellulase',
            'lignin': 'lignin_peroxidase',
            'protein': 'protease'
        }
        
        return enzyme_mapping.get(obstacle_type)
        
    def update_circadian_rhythm(self, external_cue: float = None) -> None:
        """
        Update the node's circadian rhythm, optionally synchronized to external cues.
        
        Args:
            external_cue: Optional external synchronization signal (0.0 to 1.0)
        """
        # Natural phase progression
        self.cycle_phase = (self.cycle_phase + 2 * math.pi / self.cycle_duration) % (2 * math.pi)
        
        # External synchronization if provided
        if external_cue is not None:
            # Convert external cue to target phase
            target_phase = external_cue * 2 * math.pi
            # Gradually adjust phase (entrainment)
            phase_difference = target_phase - self.cycle_phase
            # Normalize to -π to π
            if phase_difference > math.pi:
                phase_difference -= 2 * math.pi
            elif phase_difference < -math.pi:
                phase_difference += 2 * math.pi
                
            # Gradual adjustment
            adjustment = phase_difference * 0.1
            self.cycle_phase = (self.cycle_phase + adjustment) % (2 * math.pi)


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