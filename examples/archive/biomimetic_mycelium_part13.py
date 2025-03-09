    def get_network_insights(self) -> Dict:
        """
        Get biological insights about the network's adaptations.
        
        Returns:
        --------
        Dict
            Dictionary with network insights
        """
        stats = self.network.get_network_statistics()
        
        # Calculate meaningful biological metrics
        insights = {
            'adaptive_capacity': stats.get('adaptation_progress', 1.0),
            'network_complexity': stats['connection_count'] / max(1, stats['node_count']),
            'resource_efficiency': stats['total_resources'] / max(1, stats['node_count']),
            'stress_resilience': 1.0 - stats.get('average_stress', 0.0),
            'growth_pattern': len(stats.get('enzymatic_activities', {})),
            'connectivity': stats.get('anastomosis_count', 0) / max(1, stats['node_count'])
        }
        
        # Overall resilience score
        insights['resilience_score'] = (
            0.3 * insights['adaptive_capacity'] +
            0.2 * insights['network_complexity'] +
            0.2 * insights['resource_efficiency'] +
            0.2 * insights['stress_resilience'] +
            0.1 * insights['connectivity']
        )
        
        return insights


def generate_complex_data(n_samples: int = 200, noise: float = 0.1) -> Tuple[List[List[float]], List[int]]:
    """
    Generate a more complex classification dataset with non-linear decision boundary.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Amount of noise to add
        
    Returns:
    --------
    Tuple[List[List[float]], List[int]]
        Feature vectors and binary labels
    """
    features = []
    labels = []
    
    for _ in range(n_samples):
        # Generate points in a 2D space
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        # Add noise
        x += random.uniform(-noise, noise)
        y += random.uniform(-noise, noise)
        
        # Keep within bounds
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        
        features.append([x, y])
        
        # Complex decision boundary: circular pattern
        distance_from_center = math.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        
        # Alternating concentric circles
        if (0.15 < distance_from_center < 0.25) or (0.35 < distance_from_center < 0.45):
            labels.append(1)
        else:
            labels.append(0)
    
    # Shuffle data
    combined = list(zip(features, labels))
    random.shuffle(combined)
    features, labels = zip(*combined)
    
    return list(features), list(labels)