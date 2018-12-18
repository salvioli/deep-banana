import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_size=64, fc2_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(state_size, fc1_size),
            torch.nn.ReLU()
        )
        
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(fc1_size, fc2_size),
            torch.nn.ReLU()
        )
        
        self.fc3 = torch.nn.Linear(fc2_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        action = self.fc1(state)
        action = self.fc2(action)
        action = self.fc3(action)
        
        return action
