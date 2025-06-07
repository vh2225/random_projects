import numpy as np
import torch
import torch.nn as nn
import cv2
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


class NeatNetworkGPU(nn.Module):
    """PyTorch implementation of NEAT network for GPU acceleration."""
    
    def __init__(self, connections, node_evals, num_inputs, num_outputs):
        super().__init__()
        self.connections = connections
        self.node_evals = node_evals
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
    def forward(self, inputs):
        """Forward pass through the NEAT network."""
        # Initialize node values
        node_values = {}
        
        # Set input values
        for i, value in enumerate(inputs):
            node_values[i] = value
            
        # Evaluate nodes in the order specified by node_evals
        for node_id, bias, aggregation, activation, inputs_list in self.node_evals:
            if node_id in node_values:
                continue
                
            # Aggregate inputs
            node_sum = bias
            for i, w in inputs_list:
                node_sum += node_values.get(i, 0) * w
                
            # Apply activation
            if activation == 'sigmoid':
                node_values[node_id] = torch.sigmoid(torch.tensor(node_sum))
            elif activation == 'tanh':
                node_values[node_id] = torch.tanh(torch.tensor(node_sum))
            elif activation == 'relu':
                node_values[node_id] = torch.relu(torch.tensor(node_sum))
            else:  # identity
                node_values[node_id] = torch.tensor(node_sum)
                
        # Extract output values
        outputs = []
        for i in range(self.num_inputs, self.num_inputs + self.num_outputs):
            outputs.append(node_values.get(i, 0))
            
        return torch.stack(outputs)


class MarioAgentGPU:
    """GPU-accelerated neural network agent for playing Super Mario Bros using NEAT."""
    
    def __init__(self, genome, config, action_type='simple', device=None):
        """
        Initialize the GPU-accelerated Mario agent with a NEAT genome.
        
        Args:
            genome: NEAT genome representing the neural network
            config: NEAT configuration
            action_type: Type of action space ('simple', 'complex', or 'right_only')
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.genome = genome
        self.config = config
        self.action_type = action_type
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
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
        
        # Create GPU preprocessing layers
        self.preprocess_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)
        
        # Calculate preprocessing output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 84).to(self.device)
            preprocess_output_size = self.preprocess_net(dummy_input).shape[1]
            
        # Create adapter layer to match NEAT input size
        self.input_adapter = nn.Linear(preprocess_output_size, 240).to(self.device)
        
    def preprocess_state_gpu(self, state):
        """
        GPU-accelerated preprocessing of the game state.
        
        Args:
            state: Raw game state (RGB image)
            
        Returns:
            Preprocessed state tensor on GPU
        """
        # Convert to grayscale using GPU (make a copy to handle negative strides)
        state_copy = np.array(state, copy=True)
        state_tensor = torch.from_numpy(state_copy).float().to(self.device)
        
        # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        if len(state_tensor.shape) == 3:
            gray = (0.299 * state_tensor[:,:,0] + 
                   0.587 * state_tensor[:,:,1] + 
                   0.114 * state_tensor[:,:,2])
        else:
            gray = state_tensor
            
        # Resize using interpolation
        gray = gray.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        resized = torch.nn.functional.interpolate(gray, size=(84, 84), mode='bilinear', align_corners=False)
        
        # Crop to focus on Mario's area
        cropped = resized[:, :, 10:74, :]  # 64x84
        
        # Normalize
        normalized = cropped / 255.0
        
        # Apply CNN preprocessing
        features = self.preprocess_net(normalized)
        
        # Adapt to NEAT input size
        adapted = self.input_adapter(features)
        
        return adapted.squeeze(0)  # Remove batch dimension
        
    def preprocess_state_cpu(self, state):
        """
        CPU fallback preprocessing (original method).
        
        Args:
            state: Raw game state (RGB image)
            
        Returns:
            Flattened, normalized state vector
        """
        # Convert to grayscale
        gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        
        # Resize to smaller dimensions
        resized = cv2.resize(gray, (84, 84))
        
        # Crop to focus on Mario's area
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
        # Preprocess the state on GPU
        if self.device.type == 'cuda':
            with torch.no_grad():
                processed_state = self.preprocess_state_gpu(state)
                processed_state_np = processed_state.cpu().numpy()
        else:
            processed_state_np = self.preprocess_state_cpu(state)
        
        # Get network output (NEAT runs on CPU)
        output = net.activate(processed_state_np)
        
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
        # Preprocess the state on GPU
        if self.device.type == 'cuda':
            with torch.no_grad():
                processed_state = self.preprocess_state_gpu(state)
                processed_state_np = processed_state.cpu().numpy()
        else:
            processed_state_np = self.preprocess_state_cpu(state)
        
        # Get network output
        output = net.activate(processed_state_np)
        
        # Apply softmax to get probabilities
        action_outputs = output[:self.num_actions]
        exp_outputs = np.exp(action_outputs - np.max(action_outputs))
        probabilities = exp_outputs / np.sum(exp_outputs)
        
        return probabilities
    
    def batch_preprocess_states(self, states):
        """
        Preprocess multiple states in a batch for efficiency.
        
        Args:
            states: List of raw game states
            
        Returns:
            Batch of preprocessed states
        """
        if self.device.type == 'cuda':
            # Stack states into a batch (make copies to handle negative strides)
            batch = torch.stack([torch.from_numpy(np.array(s, copy=True)).float() for s in states]).to(self.device)
            
            # Convert to grayscale
            gray_batch = (0.299 * batch[:,:,:,0] + 
                         0.587 * batch[:,:,:,1] + 
                         0.114 * batch[:,:,:,2])
            
            # Add channel dimension
            gray_batch = gray_batch.unsqueeze(1)
            
            # Resize batch
            resized_batch = torch.nn.functional.interpolate(gray_batch, size=(84, 84), mode='bilinear', align_corners=False)
            
            # Crop batch
            cropped_batch = resized_batch[:, :, 10:74, :]
            
            # Normalize
            normalized_batch = cropped_batch / 255.0
            
            # Apply CNN preprocessing
            features_batch = self.preprocess_net(normalized_batch)
            
            # Adapt to NEAT input size
            adapted_batch = self.input_adapter(features_batch)
            
            return adapted_batch.cpu().numpy()
        else:
            # Fallback to CPU processing
            return np.array([self.preprocess_state_cpu(s) for s in states])