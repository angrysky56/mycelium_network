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