    def _mutate(self, genome, mutation_strength=0.2):
        """
        Mutate a genome with adaptive mutation strength.
        
        Args:
            genome: Genome to mutate
            mutation_strength: Base strength of mutations (0-1)
        """
        # Node mutations
        for node in genome.node_genes:
            # Mutate position with Gaussian noise
            if random.random() < mutation_strength:
                # Perturb position with more controlled gaussian noise
                pos = list(node['position'])
                for i in range(len(pos)):
                    # Use gaussian distribution for more natural mutations
                    pos[i] += random.gauss(0, 0.1)
                    pos[i] = max(0, min(1, pos[i]))  # Keep within bounds
                node['position'] = tuple(pos)
            
            # Mutate sensitivity with adaptive rate
            if random.random() < mutation_strength:
                # More aggressive mutation for sensitivity
                if random.random() < 0.3:  # 30% chance of larger mutation
                    node['sensitivity'] *= random.uniform(0.5, 1.5)  # Wider range
                else:
                    node['sensitivity'] *= random.uniform(0.9, 1.1)  # Smaller tweaks
                node['sensitivity'] = max(0.1, min(3.0, node['sensitivity']))  # Wider bounds
            
            # Mutate adaptability with similar approach
            if random.random() < mutation_strength:
                if random.random() < 0.3:  # 30% chance of larger mutation
                    node['adaptability'] *= random.uniform(0.5, 1.5)
                else:
                    node['adaptability'] *= random.uniform(0.9, 1.1)
                node['adaptability'] = max(0.1, min(3.0, node['adaptability']))
            
            # Mutate type (rarely)
            if random.random() < mutation_strength * 0.2:
                # Don't change input/output nodes
                if node['type'] not in ['input', 'output']:
                    node['type'] = random.choice(['regular', 'storage', 'processing', 'sensor'])
        
        # Connection mutations
        for conn in genome.connection_genes:
            # Mutate strength
            if random.random() < mutation_strength:
                conn['strength'] *= random.uniform(0.8, 1.2)
                conn['strength'] = max(0.1, min(2.0, conn['strength']))
        
        # Add/remove nodes (rarely)
        if random.random() < mutation_strength * 0.3:
            if random.random() < 0.5 and len(genome.node_genes) > 3:
                # Remove a random node (not input/output)
                removable_indices = [
                    i for i, node in enumerate(genome.node_genes)
                    if node['type'] not in ['input', 'output']
                ]
                
                if removable_indices:
                    idx = random.choice(removable_indices)
                    removed_id = genome.node_genes[idx]['id']
                    genome.node_genes.pop(idx)
                    
                    # Remove affected connections
                    genome.connection_genes = [
                        conn for conn in genome.connection_genes
                        if conn['source'] != removed_id and conn['target'] != removed_id
                    ]
            else:
                # Add a new node
                new_id = max([node['id'] for node in genome.node_genes], default=0) + 1
                
                # Random position
                dimensions = len(genome.node_genes[0]['position']) if genome.node_genes else 2
                position = tuple(random.random() for _ in range(dimensions))
                
                # Create new node
                new_node = {
                    'id': new_id,
                    'position': position,
                    'type': random.choice(['regular', 'storage', 'processing', 'sensor']),
                    'sensitivity': random.uniform(0.8, 1.2),
                    'adaptability': random.uniform(0.5, 1.5),
                    'specializations': {}
                }
                
                genome.node_genes.append(new_node)
                
                # Add some connections to existing nodes
                if genome.node_genes:
                    num_connections = random.randint(1, min(3, len(genome.node_genes)))
                    for _ in range(num_connections):
                        target_node = random.choice(genome.node_genes)
                        
                        # Skip self-connections
                        if target_node['id'] == new_id:
                            continue
                        
                        # Determine source and target IDs first
                        is_new_node_source = random.random() < 0.5
                        source_id = new_id if is_new_node_source else target_node['id']
                        target_id = target_node['id'] if is_new_node_source else new_id
                        
                        # Add connection
                        new_conn = {
                            'source': source_id,
                            'target': target_id,
                            'strength': random.uniform(0.1, 0.5)
                        }
                        
                        genome.connection_genes.append(new_conn)
        
        # Add/remove connections (sometimes)
        if random.random() < mutation_strength * 0.5:
            if random.random() < 0.5 and genome.connection_genes:
                # Remove a random connection
                idx = random.randrange(len(genome.connection_genes))
                genome.connection_genes.pop(idx)
            else:
                # Add a new connection
                if len(genome.node_genes) >= 2:
                    # Select random source and target
                    source = random.choice(genome.node_genes)
                    target = random.choice(genome.node_genes)
                    
                    # Avoid self-connections
                    while target['id'] == source['id']:
                        target = random.choice(genome.node_genes)
                    
                    # Check if connection already exists
                    connection_exists = any(
                        conn['source'] == source['id'] and conn['target'] == target['id']
                        for conn in genome.connection_genes
                    )
                    
                    if not connection_exists:
                        # Add new connection
                        new_conn = {
                            'source': source['id'],
                            'target': target['id'],
                            'strength': random.uniform(0.1, 0.5)
                        }
                        
                        genome.connection_genes.append(new_conn)
    
    def get_best_network(self):
        """
        Get the best network found so far.
        
        Returns:
            Best network instance
        """
        if self.best_genome is None:
            return None
            
        return self.best_genome.to_network(self.environment)
    
    def get_statistics(self):
        """
        Get optimization statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'generation': self.generation,
            'population_size': self.population_size,
            'best_fitness': self.best_fitness,
            'current_avg_fitness': sum(self.fitness_scores) / len(self.fitness_scores) if self.fitness_scores else 0,
            'fitness_history': self.fitness_history
        }


# Example fitness functions

def classification_fitness(network, X, y, threshold=0.5):
    """
    Fitness function for classification tasks.
    
    Args:
        network: Network to evaluate
        X: Input features
        y: Target labels (0 or 1)
        threshold: Classification threshold
        
    Returns:
        Fitness score (accuracy)
    """
    correct = 0
    
    for features, label in zip(X, y):
        # Forward pass
        output = network.forward(features)[0]
        
        # Classify
        prediction = 1 if output > threshold else 0
        
        # Check accuracy
        if prediction == label:
            correct += 1
    
    # Return accuracy
    return correct / len(X)


def regression_fitness(network, X, y):
    """
    Fitness function for regression tasks.
    
    Args:
        network: Network to evaluate
        X: Input features
        y: Target values
        
    Returns:
        Fitness score (1 / (1 + MSE))
    """
    total_error = 0
    
    for features, target in zip(X, y):
        # Forward pass
        output = network.forward(features)[0]
        
        # Calculate error
        error = (output - target) ** 2
        total_error += error
    
    # Mean squared error
    mse = total_error / len(X)
    
    # Convert to fitness (higher is better)
    return 1 / (1 + mse)


def resource_gathering_fitness(network, env, steps=100):
    """
    Fitness function for resource gathering behavior.
    
    Args:
        network: Network to evaluate
        env: Environment with resources
        steps: Number of simulation steps
        
    Returns:
        Fitness score based on resources gathered
    """
    # Record initial resource amounts
    initial_resources = {}
    for resource_type in env.get_resource_types():
        initial_resources[resource_type] = env.get_total_resources(resource_type)
    
    # Track resources processed by the network
    resources_processed = 0
    
    # Run simulation
    for _ in range(steps):
        # Generate random input
        input_vec = [random.random() for _ in range(network.input_size)]
        
        # Forward pass
        network.forward(input_vec)
        
        # Resources are processed during forward pass
        # We can estimate the resources processed by changes in network
        resources_processed += network.total_resources - network.nodes[0].resource_level
    
    # Return fitness based on resources processed
    return resources_processed