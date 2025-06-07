#!/usr/bin/env python3
"""
GPU-accelerated training script for Mario Neuro-Evolution using NEAT.

This script evolves neural networks to play Super Mario Bros using
the NEAT algorithm with GPU acceleration for preprocessing and inference.
"""

import os
import neat
import pickle
import argparse
import torch
from datetime import datetime
from utils.fitness_gpu import FitnessEvaluatorGPU
from utils.visualization import Visualizer, StatsReporter
from agents.mario_agent_gpu import MarioAgentGPU
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


def check_gpu_availability():
    """Check and display GPU availability."""
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        return True
    else:
        print("No GPU available. Running on CPU.")
        return False


def run_best_genome_gpu(config_path, genome_path, level='1-1', render=True, device=None):
    """
    Run the best evolved genome with GPU acceleration.
    
    Args:
        config_path: Path to NEAT configuration file
        genome_path: Path to saved genome file
        level: Mario level to play
        render: Whether to render the game
        device: PyTorch device to use
    """
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Set device
    if device is None:
        device = torch.device('cuda' if gpu_available else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Load genome
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    # Create neural network
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Create environment and GPU-accelerated agent
    env = SuperMarioBrosEnv()
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    agent = MarioAgentGPU(genome, config, 'simple', device)
    
    # Play multiple episodes
    for episode in range(5):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        
        while True:
            if render:
                env.render()
            
            # Get action from neural network with GPU preprocessing
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


def train_neat_gpu(config_path, generations=100, checkpoint_freq=10, 
                   level='1-1', render=False, restore_checkpoint=None,
                   device=None, batch_size=4):
    """
    Train NEAT algorithm to play Mario with GPU acceleration.
    
    Args:
        config_path: Path to NEAT configuration file
        generations: Number of generations to evolve
        checkpoint_freq: How often to save checkpoints
        level: Mario level to train on
        render: Whether to render during training
        restore_checkpoint: Path to checkpoint file to restore from
        device: PyTorch device to use
        batch_size: Number of genomes to evaluate in parallel
    """
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Set device
    if device is None:
        device = torch.device('cuda' if gpu_available else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Training will use device: {device}")
    print("-" * 50)
    
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
    checkpoint_prefix = os.path.join(checkpoint_dir, f'neat-checkpoint-gpu-{timestamp}-')
    pop.add_reporter(neat.Checkpointer(checkpoint_freq, filename_prefix=checkpoint_prefix))
    
    # Create GPU-accelerated fitness evaluator and visualizer
    evaluator = FitnessEvaluatorGPU(level=level, render=render, device=device, batch_size=batch_size)
    visualizer = Visualizer()
    
    # Run evolution
    try:
        # Define evaluation function
        def eval_genomes(genomes, config):
            evaluator.evaluate_genomes(genomes, config)
        
        # Run for specified generations
        winner = pop.run(eval_genomes, generations)
        
        # Save the winner
        winner_path = os.path.join(visualizer.run_dir, 'winner_genome_gpu.pkl')
        with open(winner_path, 'wb') as f:
            pickle.dump(winner, f)
        print(f"\nSaved winning genome to {winner_path}")
        
        # Create visualizations
        visualizer.create_summary_plots(evaluator)
        visualizer.visualize_network(config, winner)
        
        # Display winner information
        print(f'\nBest genome:\n{winner}')
        print(f'\nFitness: {winner.fitness}')
        
        # Display final GPU stats
        if device.type == 'cuda':
            print(f"\nFinal GPU Memory Usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"  Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
            print(f"  Max Allocated: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
        
        return winner
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
        # Still create visualizations with data collected so far
        visualizer.create_summary_plots(evaluator)
        
        # Save the best genome so far
        best_genome = pop.best_genome
        if best_genome:
            interrupted_path = os.path.join(visualizer.run_dir, 'best_genome_gpu_interrupted.pkl')
            with open(interrupted_path, 'wb') as f:
                pickle.dump(best_genome, f)
            print(f"\nSaved best genome so far to {interrupted_path}")
            
        # Display GPU stats
        if device.type == 'cuda':
            print(f"\nGPU Memory Usage at interruption:")
            print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"  Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train NEAT to play Super Mario Bros with GPU acceleration')
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
    parser.add_argument('--device', choices=['cuda', 'cpu'], 
                       help='Device to use (defaults to cuda if available)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Number of genomes to evaluate in parallel on GPU')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting GPU-Accelerated NEAT training for Super Mario Bros")
        print(f"Configuration: {args.config}")
        print(f"Generations: {args.generations}")
        print(f"Level: {args.level}")
        print(f"Render: {args.render}")
        print(f"Batch Size: {args.batch_size}")
        print("-" * 50)
        
        train_neat_gpu(
            config_path=args.config,
            generations=args.generations,
            checkpoint_freq=args.checkpoint_freq,
            level=args.level,
            render=args.render,
            restore_checkpoint=args.checkpoint,
            device=args.device,
            batch_size=args.batch_size
        )
        
    elif args.mode == 'play':
        if not args.genome:
            print("Error: --genome required for play mode")
            return
            
        print("Playing Super Mario Bros with evolved genome (GPU-accelerated)")
        print(f"Genome: {args.genome}")
        print(f"Level: {args.level}")
        print("-" * 50)
        
        run_best_genome_gpu(
            config_path=args.config,
            genome_path=args.genome,
            level=args.level,
            render=True,  # Always render in play mode
            device=args.device
        )


if __name__ == '__main__':
    main()