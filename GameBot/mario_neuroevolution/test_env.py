#!/usr/bin/env python3
"""
Test Mario environment setup.
"""

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

def test_mario_env():
    """Test Mario environment creation and basic functionality."""
    try:
        print("Creating Mario environment...")
        env = gym_super_mario_bros.SuperMarioBrosEnv()
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        print("Environment created successfully!")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        print("Testing environment reset...")
        state = env.reset()
        print(f"Initial state shape: {state.shape}")
        
        print("Testing a few random actions...")
        for i in range(5):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            print(f"Step {i+1}: action={action}, reward={reward}, done={done}")
            if done:
                state = env.reset()
                break
        
        env.close()
        print("✓ Mario environment test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Mario environment test FAILED: {e}")
        return False

if __name__ == '__main__':
    test_mario_env()