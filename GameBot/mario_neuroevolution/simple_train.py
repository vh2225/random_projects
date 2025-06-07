#!/usr/bin/env python3
"""
Simple training script with just the original CPU implementation but better startup.
"""

import os
import neat
from utils.fitness import FitnessEvaluator
from datetime import datetime

def main():
    """Simple training with original CPU implementation."""
    print("Starting Mario Neuroevolution training (CPU version)...")
    
    # Load configuration
    config_path = 'configs/neat_config.txt'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Create population
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    
    # Create checkpoint directory
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_prefix = os.path.join(checkpoint_dir, f'neat-checkpoint-{timestamp}-')
    pop.add_reporter(neat.Checkpointer(5, filename_prefix=checkpoint_prefix))
    
    # Create fitness evaluator
    evaluator = FitnessEvaluator(level='1-1', render=False)
    
    # Define evaluation function
    def eval_genomes(genomes, config):
        evaluator.evaluate_genomes(genomes, config)
    
    try:
        # Run for specified generations
        print("Starting evolution...")
        winner = pop.run(eval_genomes, 50)
        
        print(f'\nBest genome fitness: {winner.fitness}')
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

if __name__ == '__main__':
    main()