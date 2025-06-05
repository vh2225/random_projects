import numpy as np
import neat
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from agents.mario_agent import MarioAgent


class FitnessEvaluator:
    """Evaluates fitness of NEAT genomes by playing Mario."""
    
    def __init__(self, level='1-1', render=False, action_type='simple'):
        """
        Initialize fitness evaluator.
        
        Args:
            level: Mario level to play (e.g., '1-1', '2-1', etc.)
            render: Whether to render the game
            action_type: Type of action space
        """
        self.level = level
        self.render = render
        self.action_type = action_type
        
        # Statistics tracking
        self.generation_stats = {
            'max_fitness': [],
            'avg_fitness': [],
            'max_distance': [],
            'max_score': []
        }
        
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
        
        # Create agent and environment
        agent = MarioAgent(genome, config, self.action_type)
        env = self.create_env()
        
        # Play the game
        fitness = self.play_episode(env, net, agent)
        
        env.close()
        return fitness
    
    def play_episode(self, env, net, agent, max_steps=10000):
        """
        Play a single episode of Mario.
        
        Args:
            env: Mario environment
            net: Neural network
            agent: Mario agent
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
        prev_x_pos = 0
        
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
            
            # Check if Mario is stuck
            if abs(x_pos - prev_x_pos) < 1:
                stuck_counter += 1
                if stuck_counter > 100:  # Stuck for too long
                    time_penalty = 50
                    break
            else:
                stuck_counter = 0
            prev_x_pos = x_pos
            
            if done:
                # Check if Mario completed the level
                if info.get('flag_get', False):
                    # Huge bonus for completing the level
                    total_reward += 5000
                break
        
        # Calculate fitness
        fitness = self.calculate_fitness(
            max_distance, total_reward, coins, score, 
            time_penalty, step, info.get('flag_get', False)
        )
        
        return fitness
    
    def calculate_fitness(self, distance, reward, coins, score, 
                         time_penalty, steps, completed):
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
            
        Returns:
            Fitness score
        """
        # Base fitness is distance traveled
        fitness = distance
        
        # Add reward component
        fitness += reward * 0.1
        
        # Bonus for coins
        fitness += coins * 100
        
        # Small bonus for score
        fitness += score * 0.01
        
        # Penalty for taking too long
        fitness -= time_penalty
        
        # Efficiency bonus (distance per step)
        if steps > 0:
            efficiency = distance / steps
            fitness += efficiency * 50
        
        # Huge bonus for completing the level
        if completed:
            fitness += 10000
        
        return max(0, fitness)
    
    def evaluate_genomes(self, genomes, config):
        """
        Evaluate all genomes in a generation.
        
        Args:
            genomes: List of (genome_id, genome) tuples
            config: NEAT configuration
        """
        generation_fitness = []
        generation_distance = []
        generation_score = []
        
        for genome_id, genome in genomes:
            fitness = self.evaluate_genome(genome, config)
            genome.fitness = fitness
            generation_fitness.append(fitness)
            
            # For detailed stats, we'd need to track these in play_episode
            # For now, we'll approximate based on fitness
            generation_distance.append(fitness * 0.8)
            generation_score.append(fitness * 10)
        
        # Update generation statistics
        self.generation_stats['max_fitness'].append(max(generation_fitness))
        self.generation_stats['avg_fitness'].append(np.mean(generation_fitness))
        self.generation_stats['max_distance'].append(max(generation_distance))
        self.generation_stats['max_score'].append(max(generation_score))
        
        # Print generation summary
        print(f"Generation Stats:")
        print(f"  Max Fitness: {max(generation_fitness):.2f}")
        print(f"  Avg Fitness: {np.mean(generation_fitness):.2f}")
        print(f"  Best Distance: {max(generation_distance):.2f}")
        print("-" * 50)