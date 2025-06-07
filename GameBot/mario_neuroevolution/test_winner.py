#!/usr/bin/env python3
"""
Simple script to test the winner genome directly without GUI.
"""

import neat
import pickle
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from agents.mario_agent import MarioAgent

def test_winner():
    """Test the winner genome."""
    
    # Load config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'configs/neat_config.txt')
    
    # Load winner genome (use the most recent run)
    winner_path = 'outputs/run_20250606_141942/winner_genome.pkl'
    
    try:
        with open(winner_path, 'rb') as f:
            winner = pickle.load(f)
        print(f"✓ Loaded winner genome from {winner_path}")
    except Exception as e:
        print(f"✗ Failed to load winner: {e}")
        return
    
    # Create neural network
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    print("✓ Created neural network")
    
    # Create environment
    env = gym_super_mario_bros.SuperMarioBrosEnv()
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    print("✓ Created Mario environment")
    
    # Create agent
    agent = MarioAgent(winner, config, 'simple')
    print("✓ Created Mario agent")
    
    # Test for a few episodes
    episodes = 3
    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")
        
        state = env.reset()
        total_reward = 0
        steps = 0
        max_distance = 0
        
        while True:
            # Render the game
            env.render()
            
            # Get action from neural network
            action = agent.get_action(net, state)
            
            # Take step
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            current_distance = info.get('x_pos', 0)
            max_distance = max(max_distance, current_distance)
            
            if done:
                break
        
        print(f"Episode {episode + 1} Results:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Max Distance: {max_distance}")
        print(f"  Score: {info.get('score', 0)}")
        print(f"  Steps: {steps}")
        print(f"  Time: {info.get('time', 0)}")
    
    env.close()
    print("\n✓ Testing complete!")

if __name__ == '__main__':
    test_winner()