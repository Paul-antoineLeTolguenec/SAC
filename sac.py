import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F
from utils.replay_buffer import ReplayBuffer
import ray 
import logging
import gym

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device):
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



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device):
        super(Critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # Concaténer les états et les actions en entrée
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class SAC:
    def __init__(self, env, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, 
                batch_size=256, max_size=int(1e6), num_wokers=2, num_steps=500, num_epochs=100):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.env = env
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        action_dim = env.action_space.shape[0]  
        state_dim  = env.observation_space.shape[0]
        self.batch_size = batch_size
        self.policy_frequency = 2
        self.target_update_frequency = 1
        self.num_workers = num_wokers
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        # replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size)
        # Initialize actor network and its target
        self.actor = Actor(state_dim, action_dim, hidden_dim,self.device).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim,self.device).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Initialize critic networks and their targets
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim,self.device).to(self.device)
        self.critic_target_1 = Critic(state_dim, action_dim, hidden_dim,self.device).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim,self.device).to(self.device)
        self.critic_target_2 = Critic(state_dim, action_dim, hidden_dim,self.device).to(self.device)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer_1 = Adam(self.critic_1.parameters(), lr=lr)
        self.critic_optimizer_2 = Adam(self.critic_2.parameters(), lr=lr)

    @ray.remote
    def rollout_worker(self, num_steps):
        state, done = self.env.reset(), False
        episode_reward = 0
        full_episode_reward = 0
        nb_episode = 0
        episode_timesteps = 0
        rollout_data = []

        for step in range(num_steps):
            episode_timesteps += 1

            # Convert state to tensor and get action from agent
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _, _ = self.actor.get_action(state_tensor)
            action = action.cpu().numpy().squeeze()
            action_played = action if self.env.action_space.shape[0] > 1 else np.array([action])
            # Step in the environment
            next_state, reward, done, _ = self.env.step(action_played)
            done_bool = 1.0 if episode_timesteps == self.env._max_episode_steps else float(done)
            episode_reward += reward

            # Save transition data
            rollout_data.append((state, action, next_state, reward, done_bool))

            # Update state
            state = next_state

            if done:
                state, done = self.env.reset(), False
                full_episode_reward += episode_reward
                nb_episode += 1
                episode_reward = 0
                episode_timesteps = 0
        return rollout_data, full_episode_reward/nb_episode

    @ray.remote
    def compute_gradients(self, critic=True):
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)
        critic_1_gradients, critic_2_gradients, actor_gradients = None, None, None
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        not_done = not_done.to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.get_action(next_state)
            target_Q1, target_Q2 = self.critic_target_1(next_state, next_action), self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + self.gamma * not_done * target_Q

        if critic:
            # Critic loss
            current_Q1, current_Q2 = self.critic_1(state, action), self.critic_2(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Compute critic gradients
            self.critic_optimizer_1.zero_grad()
            self.critic_optimizer_2.zero_grad()
            critic_loss.backward()
            # Store gradients for the critic
            critic_1_gradients = [param.grad.clone() for param in self.critic_1.parameters()]
            critic_2_gradients = [param.grad.clone() for param in self.critic_2.parameters()]
        else:
            # Actor loss
            new_action, log_prob, _ = self.actor.get_action(state)
            Q1_new, Q2_new = self.critic_1(state, new_action), self.critic_2(state, new_action)
            Q_new = torch.min(Q1_new, Q2_new)
            actor_loss = (self.alpha * log_prob - Q_new).mean()

            # Compute actor gradients
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Store gradients for the actor
            actor_gradients = [param.grad.clone() for param in self.actor.parameters()]

        return critic_1_gradients, critic_2_gradients, actor_gradients
    

    def mean_gradients(self, list_critic_1_gradients, list_critic_2_gradients, list_actor_gradients):
        num_gradients = len(list_critic_1_gradients)
        mean_actor_gradients, mean_critic_1_gradients, mean_critic_2_gradients = None, None, None
        
        # Calculate mean critic_1 gradients
        if None not in list_critic_1_gradients:
            mean_critic_1_gradients = [
                sum([critic_grads[i] for critic_grads in list_critic_1_gradients]) / num_gradients
                for i in range(len(list_critic_1_gradients[0]))
            ]

        # Calculate mean critic_2 gradients
        if None not in list_critic_2_gradients:
            mean_critic_2_gradients = [
                sum([critic_grads[i] for critic_grads in list_critic_2_gradients]) / num_gradients
                for i in range(len(list_critic_2_gradients[0]))
            ]

        # Calculate mean actor gradients
        if None not in list_actor_gradients:
            mean_actor_gradients = [
                sum([actor_grads[i] for actor_grads in list_actor_gradients]) / num_gradients
                for i in range(len(list_actor_gradients[0]))
            ]

        return mean_critic_1_gradients, mean_critic_2_gradients, mean_actor_gradients


    def update_model_parameters(self, critic_1_gradients, critic_2_gradients, actor_gradients):
        if critic_1_gradients!=None:
            # Update critic_1 parameters
            for param, grad in zip(self.critic_1.parameters(), critic_1_gradients):
                param.grad = grad
            self.critic_optimizer_1.step()
        if critic_2_gradients!=None:
            # Update critic_2 parameters
            for param, grad in zip(self.critic_2.parameters(), critic_2_gradients):
                param.grad = grad
            self.critic_optimizer_2.step()
        if actor_gradients!=None:
            # Update actor parameters
            for param, grad in zip(self.actor.parameters(), actor_gradients):
                param.grad = grad
            self.actor_optimizer.step()

    def update(self, critic=True):
        # Compute gradients
        list_gradients = ray.get([self.compute_gradients.remote(self,critic) for _ in range(self.num_workers)])
        list_critic_1_gradients , list_critic_2_gradients, list_actor_gradients = zip(*list_gradients)
        # Calculate mean gradients
        critic_1_gradients, critic_2_gradients, actor_gradients = self.mean_gradients(list_critic_1_gradients, list_critic_2_gradients, list_actor_gradients)
        # Update model parameters
        self.update_model_parameters(critic_1_gradients, critic_2_gradients, actor_gradients)
    
    def update_target_networks(self):
        for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self):
        for epoch in range(self.num_epochs):
            # Sample data from rollout worker
            res = ray.get([self.rollout_worker.remote(self, self.num_steps) for _ in range(self.num_workers)])
            rollout_data, rollout_reward = zip(*res)
            # Store data in replay buffer
            for data in rollout_data: self.replay_buffer.add_rollout_data_to_buffer(data)
            # Update model
            self.update(critic=True) if epoch%self.policy_frequency==0 else [self.update(critic=False) for _ in range(self.policy_frequency)]
            # Update target networks
            self.update_target_networks() if epoch%self.target_update_frequency==0 else None
            # logs
            print(f"Epoch: {epoch+1}/{self.num_epochs}")
            print(f"Rollout reward: {np.mean(rollout_reward)}")

if __name__=="__main__":
    env=gym.make('InvertedPendulum-v2')
    ray.init()
    logger = logging.getLogger("ray")
    logger.setLevel(logging.CRITICAL)
    sac=SAC(env, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, num_epochs=10)
    sac.train()
    
    