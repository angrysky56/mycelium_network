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
        
    def _sense_environment(self) -> None:
        """Have all nodes sense the environment."""
        for node_id, node in self.nodes.items():
            if isinstance(node, BiomimeticNode):
                sensed_info = node.sense_environment(self.environment)
                
                # Process resources
                for resource in sensed_info['resources']:
                    if resource['distance'] < 0.05:  # Close enough to consume
                        consumed = self.environment.consume_resource(
                            resource['id'], 
                            min(0.2, node.energy * 0.5)
                        )
                        
                        if consumed > 0:
                            node.allocate_resources(consumed, resource['type'])
                
    def _perform_anastomosis(self) -> None:
        """Perform anastomosis between close nodes."""
        # Only attempt occasionally
        if random.random() > 0.1:
            return
            
        # Choose a random regular node
        if not self.regular_nodes:
            return
            
        node_id = random.choice(self.regular_nodes)
        node = self.nodes[node_id]
        
        if not isinstance(node, BiomimeticNode):
            return
            
        # Find nearby nodes
        for other_id, other_node in self.nodes.items():
            if (other_id != node_id and 
                other_id not in self.input_nodes and 
                other_id not in self.output_nodes and
                isinstance(other_node, BiomimeticNode)):
                
                distance = node.calculate_distance(other_node)
                
                if distance < 0.1 and random.random() < 0.3:
                    if node.attempt_anastomosis(other_node):
                        self.anastomosis_count += 1