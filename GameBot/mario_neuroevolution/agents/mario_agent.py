import numpy as np
import cv2
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


class MarioAgent:
    """Neural network agent for playing Super Mario Bros using NEAT."""
    
    def __init__(self, genome, config, action_type='simple'):
        """
        Initialize the Mario agent with a NEAT genome.
        
        Args:
            genome: NEAT genome representing the neural network
            config: NEAT configuration
            action_type: Type of action space ('simple', 'complex', or 'right_only')
        """
        self.genome = genome
        self.config = config
        self.action_type = action_type
        
        # Set up action space
        if action_type == 'simple':
            self.actions = SIMPLE_MOVEMENT
        elif action_type == 'complex':
            self.actions = COMPLEX_MOVEMENT
        elif action_type == 'right_only':
            self.actions = RIGHT_ONLY
        else:
            self.actions = SIMPLE_MOVEMENT
            
        self.num_actions = len(self.actions)
        
    def preprocess_state(self, state):
        """
        Preprocess the game state for the neural network.
        
        Args:
            state: Raw game state (RGB image)
            
        Returns:
            Flattened, normalized state vector
        """
        # Convert to grayscale
        gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        
        # Resize to smaller dimensions
        resized = cv2.resize(gray, (84, 84))
        
        # Crop to focus on Mario's area (remove score/status bar)
        cropped = resized[10:74, :]  # 64x84
        
        # Further downsample for NEAT
        downsampled = cv2.resize(cropped, (20, 12))
        
        # Normalize to [0, 1]
        normalized = downsampled / 255.0
        
        # Flatten to 1D array
        flattened = normalized.flatten()
        
        return flattened
    
    def get_action(self, net, state):
        """
        Get action from the neural network given a game state.
        
        Args:
            net: NEAT neural network
            state: Current game state
            
        Returns:
            Action index
        """
        # Preprocess the state
        processed_state = self.preprocess_state(state)
        
        # Get network output
        output = net.activate(processed_state)
        
        # Select action with highest activation
        action_idx = np.argmax(output[:self.num_actions])
        
        return action_idx
    
    def get_action_probabilities(self, net, state):
        """
        Get action probabilities from the neural network.
        
        Args:
            net: NEAT neural network
            state: Current game state
            
        Returns:
            Array of action probabilities
        """
        # Preprocess the state
        processed_state = self.preprocess_state(state)
        
        # Get network output
        output = net.activate(processed_state)
        
        # Apply softmax to get probabilities
        action_outputs = output[:self.num_actions]
        exp_outputs = np.exp(action_outputs - np.max(action_outputs))
        probabilities = exp_outputs / np.sum(exp_outputs)
        
        return probabilities