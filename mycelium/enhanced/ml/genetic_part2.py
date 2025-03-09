    def initialize_population(self, template_network=None, problem_type=None):
        """
        Initialize a population of networks with possible problem-specific templates.
        
        Args:
            template_network: Optional network to use as a template
            problem_type: Optional string indicating problem type for specialized initialization
        """
        self.population = []
        
        # Problem-specific initializations
        if problem_type == "xor":
            print("Using specialized XOR network initialization")
            # Create at least one network with a proper XOR structure
            xor_network = self._create_xor_network()
            self.population.append(NetworkGenome(xor_network))
            
            # Create other variations
            for i in range(1, min(int(self.population_size * 0.3), 10)):
                variant = self._create_xor_network(randomize=True)
                self.population.append(NetworkGenome(variant))
        
        # Fill remaining population
        remaining = self.population_size - len(self.population)
        for i in range(remaining):
            if template_network is not None and i == 0:
                # Keep one copy of template network
                genome = NetworkGenome(template_network)
            elif template_network is not None:
                # Create variation of template
                genome = NetworkGenome(template_network)
                self._mutate(genome, mutation_strength=0.5)
            else:
                # Create a new network
                network = AdaptiveMyceliumNetwork(
                    environment=self.environment,
                    input_size=random.randint(1, 4),
                    output_size=random.randint(1, 2),
                    initial_nodes=random.randint(5, 15)
                )
                genome = NetworkGenome(network)
            
            self.population.append(genome)
        
        # Reset statistics
        self.generation = 0
        self.fitness_scores = [0] * self.population_size
        self.best_fitness = 0
        self.best_genome = None
        self.fitness_history = []
    
    def evaluate_fitness(self, fitness_function, *args, **kwargs):
        """
        Evaluate fitness for all networks in the population.
        
        Args:
            fitness_function: Function that takes a network and returns a fitness score
            *args, **kwargs: Additional arguments for the fitness function
            
        Returns:
            Average fitness
        """
        total_fitness = 0
        
        for i, genome in enumerate(self.population):
            # Convert genome to network
            network = genome.to_network(self.environment)
            
            # Evaluate fitness
            fitness = fitness_function(network, *args, **kwargs)
            self.fitness_scores[i] = fitness
            
            # Update best network
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = copy.deepcopy(genome)
                
            total_fitness += fitness
        
        # Sort population by fitness
        combined = list(zip(self.population, self.fitness_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        
        self.population, self.fitness_scores = zip(*combined)
        self.population = list(self.population)
        self.fitness_scores = list(self.fitness_scores)
        
        # Track history
        avg_fitness = total_fitness / len(self.population)
        self.fitness_history.append({
            'generation': self.generation,
            'average': avg_fitness,
            'best': self.best_fitness
        })
        
        return avg_fitness
    
    def evolve(self):
        """
        Evolve the population for one generation.
        
        Returns:
            The new population
        """
        new_population = []
        
        # Determine number of elites to keep
        elite_count = max(1, int(self.population_size * self.elite_percentage))
        
        # Keep elite individuals
        for i in range(elite_count):
            new_population.append(copy.deepcopy(self.population[i]))
        
        # Fill the rest of the population
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._selection()
            parent2 = self._selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutation
            if random.random() < self.mutation_rate:
                self._mutate(child1)
            if random.random() < self.mutation_rate:
                self._mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Update population
        self.population = new_population
        self.fitness_scores = [0] * self.population_size
        self.generation += 1
        
        return self.population
    
    def _selection(self):
        """
        Select a genome from the population based on fitness.
        Uses an improved selection strategy with a mix of tournament selection
        and occasional random selection to maintain diversity.
        
        Returns:
            Selected genome
        """
        # Occasionally (10% chance) select completely random individual to maintain diversity
        if random.random() < 0.1:
            random_index = random.randrange(len(self.population))
            return copy.deepcopy(self.population[random_index])
        
        # 90% of the time use tournament selection with adaptive sizing
        # Use larger tournaments when population is more converged to increase selection pressure
        avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores) if self.fitness_scores else 0
        fitness_diversity = sum((f - avg_fitness)**2 for f in self.fitness_scores) / len(self.fitness_scores)
        
        # Adaptive tournament size - smaller when diversity is low
        base_size = max(2, self.population_size // 5)
        if fitness_diversity < 0.01:  # Low diversity
            tournament_size = max(2, base_size // 2)  # Smaller tournaments
        else:
            tournament_size = base_size + int(fitness_diversity * 10)  # Larger tournaments
            
        tournament_size = min(tournament_size, len(self.population) // 2)  # Cap size
        
        # Select random individuals for tournament
        tournament = random.sample(range(len(self.population)), tournament_size)
        
        # Find the best in the tournament
        best_index = tournament[0]
        best_fitness = self.fitness_scores[best_index]
        
        for idx in tournament[1:]:
            if self.fitness_scores[idx] > best_fitness:
                best_index = idx
                best_fitness = self.fitness_scores[idx]
        
        return copy.deepcopy(self.population[best_index])
    
    def _crossover(self, parent1, parent2):
        """
        Perform advanced crossover between two genomes.
        Uses more intelligent crossover strategies that better preserve
        beneficial traits from parents.
        
        Args:
            parent1, parent2: Parent genomes
            
        Returns:
            Two child genomes
        """
        child1 = NetworkGenome()
        child2 = NetworkGenome()
        
        # Copy basic properties
        child1.network_type = parent1.network_type
        child1.input_size = parent1.input_size
        child1.output_size = parent1.output_size
        
        child2.network_type = parent2.network_type
        child2.input_size = parent2.input_size
        child2.output_size = parent2.output_size
        
        # Use a more sophisticated crossover method with some randomness
        # to determine which nodes to keep from each parent
        
        # Option 1: Traditional crossover point (70% chance)
        if random.random() < 0.7:
            node_crossover_point = random.randint(0, min(len(parent1.node_genes), len(parent2.node_genes)))
            
            child1.node_genes = (
                parent1.node_genes[:node_crossover_point] + 
                parent2.node_genes[node_crossover_point:]
            )
            
            child2.node_genes = (
                parent2.node_genes[:node_crossover_point] + 
                parent1.node_genes[node_crossover_point:]
            )
        # Option 2: Uniform crossover for more diversity (30% chance)
        else:
            child1.node_genes = []
            child2.node_genes = []
            
            # Go through each node position and randomly select from either parent
            for i in range(max(len(parent1.node_genes), len(parent2.node_genes))):
                # If we've exhausted nodes from one parent, take from the other
                if i >= len(parent1.node_genes):
                    child1.node_genes.append(copy.deepcopy(parent2.node_genes[i]))
                    child2.node_genes.append(copy.deepcopy(parent2.node_genes[i]))
                elif i >= len(parent2.node_genes):
                    child1.node_genes.append(copy.deepcopy(parent1.node_genes[i]))
                    child2.node_genes.append(copy.deepcopy(parent1.node_genes[i]))
                # Otherwise randomly select from either parent
                elif random.random() < 0.5:
                    child1.node_genes.append(copy.deepcopy(parent1.node_genes[i]))
                    child2.node_genes.append(copy.deepcopy(parent2.node_genes[i]))
                else:
                    child1.node_genes.append(copy.deepcopy(parent2.node_genes[i]))
                    child2.node_genes.append(copy.deepcopy(parent1.node_genes[i]))
        
        # Ensure node IDs are unique
        self._ensure_unique_node_ids(child1)
        self._ensure_unique_node_ids(child2)
        
        # More intelligent connection crossover
        # We'll match connection strategy to node strategy for consistency
        if random.random() < 0.7:  # Standard crossover point
            conn_crossover_point = random.randint(0, min(len(parent1.connection_genes), len(parent2.connection_genes)))
            
            child1.connection_genes = (
                parent1.connection_genes[:conn_crossover_point] + 
                parent2.connection_genes[conn_crossover_point:]
            )
            
            child2.connection_genes = (
                parent2.connection_genes[:conn_crossover_point] + 
                parent1.connection_genes[conn_crossover_point:]
            )
        else:  # Uniform crossover for connections
            child1.connection_genes = []
            child2.connection_genes = []
            
            # Get all connections from both parents
            all_connections = parent1.connection_genes + parent2.connection_genes
            
            # Remove duplicates based on source and target
            unique_connections = {}
            for conn in all_connections:
                key = (conn['source'], conn['target'])
                if key not in unique_connections or random.random() < 0.5:  # 50% chance to override
                    unique_connections[key] = conn
            
            # Distribute to children
            for key, conn in unique_connections.items():
                if random.random() < 0.5:
                    child1.connection_genes.append(copy.deepcopy(conn))
                else:
                    child2.connection_genes.append(copy.deepcopy(conn))
        
        # Fix connections to match nodes
        self._fix_connections(child1)
        self._fix_connections(child2)
        
        return child1, child2
    
    def _ensure_unique_node_ids(self, genome):
        """Ensure all node IDs are unique."""
        node_ids = set()
        next_id = 0
        
        # Find highest ID
        for node in genome.node_genes:
            next_id = max(next_id, node['id'] + 1)
        
        # Assign new IDs to duplicates
        for node in genome.node_genes:
            if node['id'] in node_ids:
                # Assign new ID
                old_id = node['id']
                node['id'] = next_id
                
                # Update connections
                for conn in genome.connection_genes:
                    if conn['source'] == old_id:
                        conn['source'] = next_id
                    if conn['target'] == old_id:
                        conn['target'] = next_id
                
                next_id += 1
            
            node_ids.add(node['id'])
    
    def _fix_connections(self, genome):
        """Fix connections to ensure they reference valid nodes."""
        # Get valid node IDs
        valid_ids = set(node['id'] for node in genome.node_genes)
        
        # Filter connections
        valid_connections = []
        for conn in genome.connection_genes:
            if conn['source'] in valid_ids and conn['target'] in valid_ids:
                # Make sure we're not creating self-connections
                if conn['source'] != conn['target']:
                    valid_connections.append(conn)
        
        # Make sure connections have reasonable strengths
        for conn in valid_connections:
            # Ensure connection strengths are within reasonable bounds
            if abs(conn['strength']) < 0.1:
                # Strengthen weak connections
                conn['strength'] = 0.1 if conn['strength'] >= 0 else -0.1
            elif abs(conn['strength']) > 2.0:
                # Cap very strong connections
                conn['strength'] = 2.0 if conn['strength'] > 0 else -2.0
        
        genome.connection_genes = valid_connections