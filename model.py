import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.out = nn.Linear(fc2_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action = self.out(x)
        return action


class DuelingDQN(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_size=64, fc2_size=64):
        super(DuelingDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)

        self.fc1_value = nn.Linear(fc1_size, fc2_size)
        self.fc2_value = nn.Linear(fc2_size, fc2_size)
        self.out_value = nn.Linear(fc2_size, 1)

        self.fc1_action = nn.Linear(fc1_size, fc2_size)
        self.fc2_action = nn.Linear(fc2_size, fc2_size)
        self.out_action = nn.Linear(fc2_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))

        x_value = F.relu(self.fc1_value(x))
        x_value = F.relu(self.fc2_value(x_value))
        x_value = self.out_value(x_value)

        x_action = F.relu(self.fc1_action(x))
        x_action = F.relu(self.fc2_action(x_action))
        x_action = self.out_action(x_action)

        return x_value + x_action - x_action.mean()