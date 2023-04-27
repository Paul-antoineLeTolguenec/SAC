import torch 
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, env, state_dim, action_dim, hidden_dim, device):
        super(Actor, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.min_log_std = -5
        self.max_log_std = 2
        self.action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32).to(self.device)
        self.action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32).to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.tanh(log_std)
        # Clip log_std to avoid numerical issues
        log_std=self.min_log_std + 0.5 * (self.max_log_std - self.min_log_std) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
