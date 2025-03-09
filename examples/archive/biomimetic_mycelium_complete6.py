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
                        
    def _grow_hyphal_branches(self) -> None:
        """Grow new hyphal branches from existing nodes."""
        # Only attempt occasionally
        if random.random() > 0.05:
            return
            
        # Only grow if sufficient network resources
        if self.total_resources < 8.0:
            return
            
        # Choose a node with high resources
        candidates = []
        for node_id in self.regular_nodes:
            node = self.nodes[node_id]
            if (isinstance(node, BiomimeticNode) and 
                node.resource_level > 1.2 and 
                node.energy > 0.6):
                candidates.append(node_id)
                
        if not candidates:
            return
            
        # Select a node to grow from
        node_id = random.choice(candidates)
        node = self.nodes[node_id]
        
        # Determine growth direction (towards resources if possible)
        direction = None
        resources = self.environment.get_resources_in_radius(node.position, 0.3)
        
        if resources:
            # Growth towards richest nearby resource
            resource = max(resources, key=lambda r: r['level'])
            # Calculate direction vector
            p1 = node.position
            p2 = resource['position']
            # Normalize direction vector
            magnitude = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
            if magnitude > 0:
                direction = tuple((a - b) / magnitude for a, b in zip(p2, p1))
        
        # Grow new branch
        new_id = node.grow_hyphal_branch(self.environment, direction)
        
        if new_id is not None:
            # Add to network
            new_node = self.environment.nodes.get(new_id)
            if new_node:
                self.nodes[new_id] = new_node
                self.regular_nodes.append(new_id)
                self.total_resources -= 0.5
                
    def _update_enzyme_activities(self) -> None:
        """Update enzymatic activities across the network."""
        # Reset activity counters
        self.enzymatic_activity = defaultdict(float)
        
        # Sum up all enzyme activities
        for node_id, node in self.nodes.items():
            if isinstance(node, BiomimeticNode):
                for enzyme_type, enzyme in node.enzymes.items():
                    if enzyme.active:
                        self.enzymatic_activity[enzyme_type] += enzyme.strength
        
    def _process_circadian_rhythms(self) -> None:
        """Process circadian rhythms across the network."""
        # Get light level as external synchronization cue
        light_level = self.environment.light_level
        
        # Update each node's circadian rhythm
        for node_id, node in self.nodes.items():
            if isinstance(node, BiomimeticNode):
                node.update_circadian_rhythm(light_level)