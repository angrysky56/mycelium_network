    def _convert_to_biomimetic_nodes(self) -> None:
        """Convert all nodes to biomimetic nodes."""
        for node_id, node in list(self.nodes.items()):
            # Convert only if not already biomimetic
            if not isinstance(node, BiomimeticNode):
                new_node = BiomimeticNode(node_id, node.position, node.type)
                
                # Transfer properties
                new_node.connections = node.connections.copy()
                new_node.activation = node.activation
                new_node.resource_level = node.resource_level
                new_node.energy = node.energy
                
                # Replace in network
                self.nodes[node_id] = new_node
                
                # Register with environment
                self.environment.register_node(node_id, new_node)
                
    def _initialize_hyphal_connections(self) -> None:
        """Initialize hyphal connections including anastomosis."""
        for node_id, node in self.nodes.items():
            # Skip input and output nodes
            if node_id in self.input_nodes or node_id in self.output_nodes:
                continue
                
            # Attempt anastomosis with nearby nodes
            for other_id, other_node in self.nodes.items():
                if (other_id not in self.input_nodes and 
                    other_id not in self.output_nodes and
                    other_id != node_id):
                    
                    # Check if close enough for anastomosis
                    if isinstance(node, BiomimeticNode) and isinstance(other_node, BiomimeticNode):
                        distance = node.calculate_distance(other_node)
                        
                        if distance < 0.15:  # Close enough for anastomosis
                            # Higher chance for nearby nodes
                            probability = 0.5 * (1 - distance / 0.15)
                            
                            if node.attempt_anastomosis(other_node, probability):
                                self.anastomosis_count += 1
    
    def forward(self, inputs: List[float]) -> List[float]:
        """
        Enhanced forward pass with environmental sensing and adaptation.
        
        Args:
            inputs: Input values
            
        Returns:
            Output values
        """
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        
        # Update environment
        self.environment.update()
        
        # Environmental sensing for all nodes
        self._sense_environment()
        
        # Enhanced forward pass
        outputs = super().forward(inputs)
        
        # Additional biomimetic behaviors
        self._perform_anastomosis()
        self._grow_hyphal_branches()
        self._update_enzyme_activities()
        self._process_circadian_rhythms()
        
        # Update network age
        self.network_age += 1
        
        return outputs