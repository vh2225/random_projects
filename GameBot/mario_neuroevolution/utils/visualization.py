import matplotlib.pyplot as plt
import neat
import numpy as np
import os
from datetime import datetime


class Visualizer:
    """Visualization utilities for NEAT evolution."""
    
    def __init__(self, output_dir='outputs'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectory for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(output_dir, f'run_{timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)
        
    def plot_fitness_history(self, stats, save=True, show=False):
        """
        Plot fitness history over generations.
        
        Args:
            stats: Dictionary containing fitness statistics
            save: Whether to save the plot
            show: Whether to display the plot
        """
        generations = range(len(stats['max_fitness']))
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, stats['max_fitness'], 'b-', label='Max Fitness')
        plt.plot(generations, stats['avg_fitness'], 'r-', label='Average Fitness')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution Over Generations')
        plt.legend()
        plt.grid(True)
        
        if save:
            plt.savefig(os.path.join(self.run_dir, 'fitness_history.png'))
        if show:
            plt.show()
        plt.close()
    
    def plot_species_history(self, pop, save=True, show=False):
        """
        Plot species diversity over time.
        
        Args:
            pop: NEAT population
            save: Whether to save the plot
            show: Whether to display the plot
        """
        if hasattr(pop, 'reporters'):
            stats = None
            for reporter in pop.reporters:
                if hasattr(reporter, 'generation_species'):
                    stats = reporter
                    break
            
            if stats and hasattr(stats, 'generation_species'):
                generations = range(len(stats.generation_species))
                num_species = [len(species) for species in stats.generation_species]
                
                plt.figure(figsize=(10, 6))
                plt.plot(generations, num_species, 'g-')
                plt.xlabel('Generation')
                plt.ylabel('Number of Species')
                plt.title('Species Diversity Over Generations')
                plt.grid(True)
                
                if save:
                    plt.savefig(os.path.join(self.run_dir, 'species_history.png'))
                if show:
                    plt.show()
                plt.close()
    
    def visualize_network(self, config, genome, filename=None, node_names=None):
        """
        Visualize a neural network from a genome.
        
        Args:
            config: NEAT configuration
            genome: Genome to visualize
            filename: Output filename
            node_names: Dictionary mapping node IDs to names
        """
        if filename is None:
            filename = os.path.join(self.run_dir, 'network.png')
        
        # Create default node names if not provided
        if node_names is None:
            node_names = {}
            
            # Input nodes (game state features)
            for i in range(config.genome_config.num_inputs):
                node_names[-i-1] = f'In{i}'
            
            # Output nodes (actions)
            action_names = ['NOOP', 'Right', 'Right+A', 'Right+B', 
                          'Right+A+B', 'A', 'Left']
            for i in range(config.genome_config.num_outputs):
                if i < len(action_names):
                    node_names[i] = action_names[i]
                else:
                    node_names[i] = f'Out{i}'
        
        # Use NEAT's built-in visualization if available
        try:
            import graphviz
            import neat.visualize
            neat.visualize.draw_net(config, genome, True, node_names=node_names,
                                  filename=filename)
        except ImportError:
            print("Graphviz not available. Skipping network visualization.")
    
    def plot_game_metrics(self, stats, save=True, show=False):
        """
        Plot game-specific metrics like distance and score.
        
        Args:
            stats: Dictionary containing game statistics
            save: Whether to save the plot
            show: Whether to display the plot
        """
        generations = range(len(stats['max_distance']))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Distance plot
        ax1.plot(generations, stats['max_distance'], 'b-', label='Max Distance')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Distance')
        ax1.set_title('Maximum Distance Reached')
        ax1.grid(True)
        ax1.legend()
        
        # Score plot
        ax2.plot(generations, stats['max_score'], 'g-', label='Max Score')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Score')
        ax2.set_title('Maximum Score Achieved')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.run_dir, 'game_metrics.png'))
        if show:
            plt.show()
        plt.close()
    
    def save_best_genome(self, genome, generation):
        """
        Save the best genome to a file.
        
        Args:
            genome: Best genome to save
            generation: Current generation number
        """
        filename = os.path.join(self.run_dir, f'best_genome_gen_{generation}.pkl')
        
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(genome, f)
        
        print(f"Saved best genome to {filename}")
    
    def create_summary_plots(self, evaluator):
        """
        Create all summary plots for the evolution run.
        
        Args:
            evaluator: FitnessEvaluator instance with statistics
        """
        print("Creating summary plots...")
        
        # Plot fitness history
        self.plot_fitness_history(evaluator.generation_stats, save=True)
        
        # Plot game metrics
        self.plot_game_metrics(evaluator.generation_stats, save=True)
        
        print(f"All plots saved to {self.run_dir}")


class StatsReporter(neat.StatisticsReporter):
    """Extended statistics reporter for NEAT that tracks species."""
    
    def __init__(self):
        super().__init__()
        self.generation_species = []
    
    def end_generation(self, config, population, species_set):
        """Called at the end of each generation."""
        super().end_generation(config, population, species_set)
        
        # Track species information
        species_info = []
        for sid, species in species_set.species.items():
            species_info.append({
                'id': sid,
                'size': len(species.members),
                'fitness': species.fitness,
                'age': getattr(species, 'age', 0)
            })
        self.generation_species.append(species_info)