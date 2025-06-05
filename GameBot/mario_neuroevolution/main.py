#!/usr/bin/env python3
"""
Main training script for Mario Neuro-Evolution using NEAT.

This script evolves neural networks to play Super Mario Bros using
the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.
"""

import os
import neat
import pickle
import argparse
from datetime import datetime
from utils.fitness import FitnessEvaluator
from utils.visualization import Visualizer, StatsReporter
from agents.mario_agent import MarioAgent
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


def run_best_genome(config_path, genome_path, level='1-1', render=True):
    """
    Run the best evolved genome.
    
    Args:
        config_path: Path to NEAT configuration file
        genome_path: Path to saved genome file
        level: Mario level to play
        render: Whether to render the game
    """
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Load genome
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    # Create neural network
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Create environment and agent
    env = SuperMarioBrosEnv()
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    agent = MarioAgent(genome, config, 'simple')
    
    # Play multiple episodes
    for episode in range(5):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        
        while True:
            if render:
                env.render()
            
            # Get action from neural network
            action = agent.get_action(net, state)
            
            # Take action
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                print(f"Episode finished: Reward={total_reward:.2f}, "
                      f"Distance={info.get('x_pos', 0)}, "
                      f"Score={info.get('score', 0)}")
                break
    
    env.close()


def train_neat(config_path, generations=100, checkpoint_freq=10, 
               level='1-1', render=False, restore_checkpoint=None):
    """
    Train NEAT algorithm to play Mario.
    
    Args:
        config_path: Path to NEAT configuration file
        generations: Number of generations to evolve
        checkpoint_freq: How often to save checkpoints
        level: Mario level to train on
        render: Whether to render during training
        restore_checkpoint: Path to checkpoint file to restore from
    """
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Create or restore population
    if restore_checkpoint:
        print(f"Restoring from checkpoint: {restore_checkpoint}")
        pop = neat.Checkpointer.restore_checkpoint(restore_checkpoint)
    else:
        pop = neat.Population(config)
    
    # Add reporters
    pop.add_reporter(neat.StdOutReporter(True))
    stats = StatsReporter()
    pop.add_reporter(stats)
    
    # Create checkpoint directory
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_prefix = os.path.join(checkpoint_dir, f'neat-checkpoint-{timestamp}-')
    pop.add_reporter(neat.Checkpointer(checkpoint_freq, filename_prefix=checkpoint_prefix))
    
    # Create fitness evaluator and visualizer
    evaluator = FitnessEvaluator(level=level, render=render)
    visualizer = Visualizer()
    
    # Run evolution
    try:
        # Define evaluation function
        def eval_genomes(genomes, config):
            evaluator.evaluate_genomes(genomes, config)
        
        # Run for specified generations
        winner = pop.run(eval_genomes, generations)
        
        # Save the winner
        winner_path = os.path.join(visualizer.run_dir, 'winner_genome.pkl')
        with open(winner_path, 'wb') as f:
            pickle.dump(winner, f)
        print(f"\nSaved winning genome to {winner_path}")
        
        # Create visualizations
        visualizer.create_summary_plots(evaluator)
        visualizer.visualize_network(config, winner)
        
        # Display winner information
        print(f'\nBest genome:\n{winner}')
        print(f'\nFitness: {winner.fitness}')
        
        return winner
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
        # Still create visualizations with data collected so far
        visualizer.create_summary_plots(evaluator)
        
        # Save the best genome so far
        best_genome = pop.best_genome
        if best_genome:
            interrupted_path = os.path.join(visualizer.run_dir, 'best_genome_interrupted.pkl')
            with open(interrupted_path, 'wb') as f:
                pickle.dump(best_genome, f)
            print(f"\nSaved best genome so far to {interrupted_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train NEAT to play Super Mario Bros')
    parser.add_argument('--mode', choices=['train', 'play'], default='train',
                       help='Mode to run: train or play')
    parser.add_argument('--config', default='configs/neat_config.txt',
                       help='Path to NEAT configuration file')
    parser.add_argument('--generations', type=int, default=100,
                       help='Number of generations for training')
    parser.add_argument('--level', default='1-1',
                       help='Mario level (e.g., 1-1, 2-1, etc.)')
    parser.add_argument('--render', action='store_true',
                       help='Render the game during training/playing')
    parser.add_argument('--checkpoint', help='Path to checkpoint file to restore')
    parser.add_argument('--genome', help='Path to genome file for play mode')
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                       help='How often to save checkpoints')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting NEAT training for Super Mario Bros")
        print(f"Configuration: {args.config}")
        print(f"Generations: {args.generations}")
        print(f"Level: {args.level}")
        print(f"Render: {args.render}")
        print("-" * 50)
        
        train_neat(
            config_path=args.config,
            generations=args.generations,
            checkpoint_freq=args.checkpoint_freq,
            level=args.level,
            render=args.render,
            restore_checkpoint=args.checkpoint
        )
        
    elif args.mode == 'play':
        if not args.genome:
            print("Error: --genome required for play mode")
            return
            
        print("Playing Super Mario Bros with evolved genome")
        print(f"Genome: {args.genome}")
        print(f"Level: {args.level}")
        print("-" * 50)
        
        run_best_genome(
            config_path=args.config,
            genome_path=args.genome,
            level=args.level,
            render=True  # Always render in play mode
        )


if __name__ == '__main__':
    main()