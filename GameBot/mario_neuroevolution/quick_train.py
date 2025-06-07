#!/usr/bin/env python3
"""
Quick training test with GPU support.
"""

import os
import neat
import torch
from utils.fitness_gpu import FitnessEvaluatorGPU

def quick_train():
    """Quick training test."""
    print("Starting quick GPU training test...")
    
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'configs/neat_config.txt')
    
    # Create small population
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    
    # Create GPU evaluator
    evaluator = FitnessEvaluatorGPU(level='1-1', render=False, device='cuda', batch_size=1)
    
    # Define evaluation function
    def eval_genomes(genomes, config):
        print(f"Evaluating {len(genomes)} genomes...")
        evaluator.evaluate_genomes(genomes, config)
    
    # Run for 1 generation
    print("Running 1 generation with GPU acceleration...")
    try:
        winner = pop.run(eval_genomes, 1)
        print(f"Training completed! Best fitness: {winner.fitness}")
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        return False

if __name__ == '__main__':
    success = quick_train()
    if success:
        print("✓ GPU training test PASSED")
    else:
        print("✗ GPU training test FAILED")