import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

# Set the device to CPU for leaderboard submission as required
device = torch.device("cpu")

def weight_init(m):
    """Custom weight initialization for better training convergence"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class Actor(nn.Module):
    """Actor network for the SAC algorithm"""
    def __init__(self, state_dim, action_dim, max_action=1.0, hidden_dim=256, 
                 log_std_min=-20, log_std_max=2, epsilon=1e-6):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon
        
        # Larger network for more complex policy function
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.apply(weight_init)
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        
        # Mean and log_std as separate outputs
        mean = self.mean(a)
        log_std = self.log_std(a)
        
        # Constrain log_std to prevent numerical instability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        
        # Re-parameterization trick
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()  # Sample with reparameterization
        
        # Use tanh to constrain actions between -max_action and max_action
        action = torch.tanh(x)
        
        # Log probability calculation with correction for tanh squashing
        log_prob = normal.log_prob(x)
        # Correction for tanh squashing (from the original SAC paper)
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return self.max_action * action, log_prob
    
    def get_action(self, state):
        """Deterministic action for evaluation"""
        mean, _ = self.forward(state)
        return self.max_action * torch.tanh(mean)

class Critic(nn.Module):
    """Critic network for the SAC algorithm"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 architecture - larger for more expressive value function
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.l5 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, hidden_dim)
        self.l7 = nn.Linear(hidden_dim, hidden_dim)
        self.l8 = nn.Linear(hidden_dim, 1)
        
        self.apply(weight_init)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        
        # Q1 value
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        
        # Q2 value
        q2 = F.relu(self.l5(sa))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        
        return q1, q2

class Agent:
    """SAC agent implementation for Humanoid Walk environment"""
    def __init__(self):
        # Humanoid environment dimensions
        self.state_dim = 67  # Observation space dimension for humanoid-walk
        self.action_dim = 21  # Action space dimension for humanoid-walk
        self.max_action = 1.0
        
        # Network parameters
        self.hidden_dim = 512  # Larger network for the complex humanoid control
        
        # Initialize networks
        self.actor = Actor(
            self.state_dim, 
            self.action_dim, 
            self.max_action,
            hidden_dim=self.hidden_dim
        ).to(device)
        
        # Load pre-trained model if available
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humanoid_sac_best_model.pt")
        if os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, path):
        """Load pre-trained model weights"""
        try:
            checkpoint = torch.load(path, map_location=device)
            self.actor.load_state_dict(checkpoint['actor'])
            print(f"Successfully loaded model from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def act(self, observation):
        """Return action based on observation"""
        # Convert observation to tensor
        state = torch.FloatTensor(observation).unsqueeze(0).to(device)
        
        # Set to evaluation mode
        self.actor.eval()
        
        with torch.no_grad():
            action = self.actor.get_action(state).cpu().numpy().flatten()
        
        # Ensure action is within the valid range
        action = np.clip(action, -self.max_action, self.max_action)
        
        return action