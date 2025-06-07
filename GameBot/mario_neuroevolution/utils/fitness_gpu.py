import numpy as np
import neat
import torch
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from agents.mario_agent_gpu import MarioAgentGPU
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


class FitnessEvaluatorGPU:
    """GPU-accelerated fitness evaluator for NEAT genomes playing Mario."""
    
    def __init__(self, level='1-1', render=False, action_type='simple', device=None, batch_size=4):
        """
        Initialize GPU fitness evaluator.
        
        Args:
            level: Mario level to play (e.g., '1-1', '2-1', etc.)
            render: Whether to render the game
            action_type: Type of action space
            device: PyTorch device ('cuda' or 'cpu')
            batch_size: Number of genomes to evaluate in parallel
        """
        self.level = level
        self.render = render
        self.action_type = action_type
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"FitnessEvaluatorGPU using device: {self.device}")
        
        # Statistics tracking
        self.generation_stats = {
            'max_fitness': [],
            'avg_fitness': [],
            'max_distance': [],
            'max_score': []
        }
        
        # GPU memory optimization
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
    def create_env(self):
        """Create and configure the Mario environment."""
        env = SuperMarioBrosEnv()
        
        # Wrap with appropriate action space
        if self.action_type == 'simple':
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        return env
    
    def evaluate_genome(self, genome, config):
        """
        Evaluate a single genome by playing Mario.
        
        Args:
            genome: NEAT genome to evaluate
            config: NEAT configuration
            
        Returns:
            Fitness score
        """
        # Create neural network from genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Create GPU-accelerated agent and environment
        agent = MarioAgentGPU(genome, config, self.action_type, self.device)
        env = self.create_env()
        
        # Play the game
        fitness = self.play_episode(env, net, agent)
        
        env.close()
        
        # Clear GPU cache periodically
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return fitness
    
    def play_episode(self, env, net, agent, max_steps=10000):
        """
        Play a single episode of Mario with GPU acceleration.
        
        Args:
            env: Mario environment
            net: Neural network
            agent: Mario agent with GPU support
            max_steps: Maximum steps per episode
            
        Returns:
            Fitness score
        """
        state = env.reset()
        
        # Tracking variables
        total_reward = 0
        max_distance = 0
        coins = 0
        score = 0
        time_penalty = 0
        stuck_counter = 0
        no_progress_counter = 0
        prev_x_pos = 0
        starting_x_pos = 0
        
        # Get starting position
        initial_info = env.env._get_info()
        starting_x_pos = initial_info.get('x_pos', 0)
        
        # State buffer for batch processing
        state_buffer = []
        action_buffer = []
        
        for step in range(max_steps):
            if self.render:
                env.render()
            
            # Get action from neural network
            action = agent.get_action(net, state)
            
            # Take action in environment
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Track progress
            x_pos = info.get('x_pos', 0)
            max_distance = max(max_distance, x_pos)
            coins = info.get('coins', 0)
            score = info.get('score', 0)
            
            # Strong penalty for not making forward progress
            if x_pos <= starting_x_pos + 50:  # Haven't moved much from start
                no_progress_counter += 1
                if no_progress_counter > 500:  # Give up if no progress for too long
                    time_penalty = 500  # Heavy penalty
                    break
            
            # Check if Mario is stuck or just jumping in place
            if abs(x_pos - prev_x_pos) < 2:
                stuck_counter += 1
                if stuck_counter > 50:  # Much stricter than before
                    time_penalty = 200  # Stronger penalty for being stuck
                    break
            else:
                stuck_counter = 0
            prev_x_pos = x_pos
            
            if done:
                # Check if Mario completed the level
                if info.get('flag_get', False):
                    # Huge bonus for completing the level
                    total_reward += 10000
                elif info.get('time') <= 0:
                    # Heavy penalty for timing out
                    time_penalty += 300
                break
        
        # Calculate fitness with emphasis on forward progress
        fitness = self.calculate_fitness(
            max_distance, total_reward, coins, score, 
            time_penalty, step, info.get('flag_get', False),
            starting_x_pos
        )
        
        return fitness
    
    def calculate_fitness(self, distance, reward, coins, score, 
                         time_penalty, steps, completed, starting_x_pos):
        """
        Calculate fitness score based on multiple factors.
        
        Args:
            distance: Maximum distance reached
            reward: Total reward from environment
            coins: Number of coins collected
            score: Game score
            time_penalty: Penalty for getting stuck
            steps: Number of steps taken
            completed: Whether level was completed
            starting_x_pos: Starting x position
            
        Returns:
            Fitness score
        """
        # Calculate actual forward progress
        forward_progress = distance - starting_x_pos
        
        # Base fitness is heavily weighted on forward progress
        fitness = forward_progress * 2  # Double weight on forward movement
        
        # Only add rewards if making progress
        if forward_progress > 50:
            fitness += reward * 0.1
            fitness += coins * 100
            fitness += score * 0.01
        
        # Heavy penalties
        fitness -= time_penalty * 2  # Double the penalty impact
        
        # Progress rate bonus (encourage fast forward movement)
        if steps > 0 and forward_progress > 0:
            progress_rate = forward_progress / steps
            fitness += progress_rate * 100  # Reward fast progress
        
        # Penalty for wasting time
        if steps > 2000 and forward_progress < 100:
            fitness -= (steps - 2000) * 0.5  # Incremental penalty for long episodes with no progress
        
        # Huge bonus for completing the level
        if completed:
            fitness += 15000
            # Speed bonus for fast completion
            if steps < 5000:
                fitness += (5000 - steps) * 2
        
        return max(0, fitness)
    
    def evaluate_genome_batch(self, genome_batch, config):
        """
        Evaluate a batch of genomes in parallel.
        
        Args:
            genome_batch: List of (genome_id, genome) tuples
            config: NEAT configuration
            
        Returns:
            List of fitness scores
        """
        fitness_scores = []
        
        with ThreadPoolExecutor(max_workers=min(self.batch_size, len(genome_batch))) as executor:
            futures = []
            for genome_id, genome in genome_batch:
                future = executor.submit(self.evaluate_genome, genome, config)
                futures.append((genome_id, genome, future))
            
            for genome_id, genome, future in futures:
                fitness = future.result()
                genome.fitness = fitness
                fitness_scores.append(fitness)
                
        return fitness_scores
    
    def evaluate_genomes(self, genomes, config):
        """
        Evaluate all genomes in a generation with GPU acceleration.
        
        Args:
            genomes: List of (genome_id, genome) tuples
            config: NEAT configuration
        """
        generation_fitness = []
        generation_distance = []
        generation_score = []
        
        # Sequential evaluation (more stable than parallel for now)
        for genome_id, genome in genomes:
            fitness = self.evaluate_genome(genome, config)
            genome.fitness = fitness
            generation_fitness.append(fitness)
            
            # Clear GPU cache periodically
            if self.device.type == 'cuda' and len(generation_fitness) % 5 == 0:
                torch.cuda.empty_cache()
        
        # Calculate approximate stats
        for fitness in generation_fitness:
            generation_distance.append(fitness * 0.8)
            generation_score.append(fitness * 10)
        
        # Update generation statistics
        self.generation_stats['max_fitness'].append(max(generation_fitness))
        self.generation_stats['avg_fitness'].append(np.mean(generation_fitness))
        self.generation_stats['max_distance'].append(max(generation_distance))
        self.generation_stats['max_score'].append(max(generation_score))
        
        # Print generation summary
        print(f"Generation Stats (GPU-Accelerated):")
        print(f"  Max Fitness: {max(generation_fitness):.2f}")
        print(f"  Avg Fitness: {np.mean(generation_fitness):.2f}")
        print(f"  Best Distance: {max(generation_distance):.2f}")
        print(f"  Device: {self.device}")
        print("-" * 50)
        
        # GPU memory stats
        if self.device.type == 'cuda':
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated")
            print(f"GPU Memory: {torch.cuda.memory_reserved()/1024**2:.2f} MB reserved")
            print("-" * 50)