#!/usr/bin/env python3
"""
Simple test script to verify GPU functionality.
"""

import torch
import numpy as np
from agents.mario_agent_gpu import MarioAgentGPU
import neat

def test_gpu_setup():
    """Test basic GPU setup."""
    print("Testing GPU setup...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Current device: {torch.cuda.current_device()}")
        
        # Test GPU memory
        device = torch.device('cuda')
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.mm(x, y)
        print(f"GPU matrix multiplication test: PASSED")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        torch.cuda.empty_cache()
        
    return torch.cuda.is_available()

def test_mario_agent_gpu():
    """Test Mario agent with GPU."""
    print("\nTesting Mario Agent GPU...")
    
    # Create a dummy genome and config
    config_path = 'configs/neat_config.txt'
    
    try:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
        
        # Create a simple genome
        genome = neat.DefaultGenome(1)
        genome.fitness = 0
        
        # Test agent creation
        agent = MarioAgentGPU(genome, config, 'simple')
        print(f"Agent created successfully on device: {agent.device}")
        
        # Test preprocessing with dummy state
        dummy_state = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
        print(f"Dummy state shape: {dummy_state.shape}")
        
        if agent.device.type == 'cuda':
            processed = agent.preprocess_state_gpu(dummy_state)
            print(f"GPU preprocessing successful: {processed.shape}")
        else:
            processed = agent.preprocess_state_cpu(dummy_state)
            print(f"CPU preprocessing successful: {processed.shape}")
            
        print("Mario Agent GPU test: PASSED")
        return True
        
    except Exception as e:
        print(f"Mario Agent GPU test: FAILED - {e}")
        return False

def main():
    """Main test function."""
    print("=" * 50)
    print("GPU FUNCTIONALITY TEST")
    print("=" * 50)
    
    gpu_available = test_gpu_setup()
    agent_test = test_mario_agent_gpu()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"GPU Available: {'✓' if gpu_available else '✗'}")
    print(f"Mario Agent GPU: {'✓' if agent_test else '✗'}")
    
    if gpu_available and agent_test:
        print("\n✓ All tests passed! GPU training should work.")
    else:
        print("\n✗ Some tests failed. Check the errors above.")

if __name__ == '__main__':
    main()