    def get_network_statistics(self) -> Dict:
        """
        Get enhanced network statistics.
        
        Returns:
            Dictionary of network statistics
        """
        # Get basic statistics
        stats = super().get_network_statistics()
        
        # Add biomimetic statistics
        stats['anastomosis_count'] = self.anastomosis_count
        stats['network_age'] = self.network_age
        
        # Calculate average stress
        stress_sum = 0.0
        stress_count = 0
        
        for node_id, node in self.nodes.items():
            if isinstance(node, BiomimeticNode):
                stress_sum += node.stress_level
                stress_count += 1
                
        stats['average_stress'] = stress_sum / max(1, stress_count)
        
        # Enzymatic activities
        stats['enzymatic_activities'] = dict(self.enzymatic_activity)
        
        # Adaptation metrics
        if self.adaptation_history:
            stats['adaptation_progress'] = self.adaptation_history[-1]['error'] / max(0.001, self.adaptation_history[0]['error'])
            
        return stats
        
    def visualize_network(self, filename: str = None) -> Dict:
        """
        Enhanced visualization data for the network.
        
        Args:
            filename: If provided, save visualization data to this file
            
        Returns:
            Dictionary with visualization data
        """
        # Get basic visualization data
        vis_data = super().visualize_network(None)
        
        # Add biomimetic features
        for i, node_data in enumerate(vis_data['nodes']):
            node_id = node_data['id']
            node = self.nodes.get(node_id)
            
            if isinstance(node, BiomimeticNode):
                # Add biomimetic properties
                vis_data['nodes'][i].update({
                    'stress_level': node.stress_level,
                    'hyphal_branches': node.hyphal_branches,
                    'anastomosis_targets': list(node.anastomosis_targets),
                    'circadian_phase': node.cycle_phase
                })
                
        # Add environmental data
        vis_data['environment'] = {
            'moisture': self.environment.moisture_level,
            'temperature': self.environment.temperature,
            'ph': self.environment.ph_level,
            'light': self.environment.light_level,
            'resources': len(self.environment.resources),
            'obstacles': len(self.environment.obstacles)
        }
        
        # Save to file if specified
        if filename:
            import json
            with open(filename, 'w') as f:
                json.dump(vis_data, f, indent=2)
        
        return vis_data