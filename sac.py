import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F
from utils.replay_buffer import ReplayBuffer
from utils.actor import Actor
from utils.critic import Critic
import ray 
import logging
import gym


class SAC:
    def __init__(self, env, hidden_dim=256, plr=3e-4, qlr=1e-3,gamma=0.99, tau=0.005, alpha=0.2, 
                batch_size=256, max_size=int(1e6), num_wokers=2, num_steps=500, num_epochs=100, autonune=True):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.env = env
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.autotune = autonune
        if autonune:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = Adam([self.log_alpha], lr=qlr)

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
        self.actor = Actor(env,state_dim, action_dim, hidden_dim,self.device).to(self.device)
        self.actor_target = Actor(env,state_dim, action_dim, hidden_dim,self.device).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Initialize critic networks and their targets
        self.critic_1 = Critic(env,state_dim, action_dim, hidden_dim,self.device).to(self.device)
        self.critic_target_1 = Critic(env,state_dim, action_dim, hidden_dim,self.device).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = Critic(env,state_dim, action_dim, hidden_dim,self.device).to(self.device)
        self.critic_target_2 = Critic(env,state_dim, action_dim, hidden_dim,self.device).to(self.device)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=plr)
        self.critic_optimizer_1 = Adam(self.critic_1.parameters(), lr=qlr)
        self.critic_optimizer_2 = Adam(self.critic_2.parameters(), lr=qlr)

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
    

    # def mean_gradients(self, list_critic_1_gradients, list_critic_2_gradients, list_actor_gradients):
    #     num_gradients = len(list_critic_1_gradients)
    #     mean_actor_gradients, mean_critic_1_gradients, mean_critic_2_gradients = None, None, None
        
    #     # Calculate mean critic_1 gradients
    #     if None not in list_critic_1_gradients:
    #         mean_critic_1_gradients = [
    #             sum([critic_grads[i] for critic_grads in list_critic_1_gradients]) / num_gradients
    #             for i in range(len(list_critic_1_gradients[0]))
    #         ]

    #     # Calculate mean critic_2 gradients
    #     if None not in list_critic_2_gradients:
    #         mean_critic_2_gradients = [
    #             sum([critic_grads[i] for critic_grads in list_critic_2_gradients]) / num_gradients
    #             for i in range(len(list_critic_2_gradients[0]))
    #         ]

    #     # Calculate mean actor gradients
    #     if None not in list_actor_gradients:
    #         mean_actor_gradients = [
    #             sum([actor_grads[i] for actor_grads in list_actor_gradients]) / num_gradients
    #             for i in range(len(list_actor_gradients[0]))
    #         ]

    #     return mean_critic_1_gradients, mean_critic_2_gradients, mean_actor_gradients


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
        critic_1_gradients, critic_2_gradients, actor_gradients = self.compute_gradients(critic)
        # list_critic_1_gradients , list_critic_2_gradients, list_actor_gradients = zip(*list_gradients)
        # Calculate mean gradients
        # critic_1_gradients, critic_2_gradients, actor_gradients = self.mean_gradients(list_critic_1_gradients, list_critic_2_gradients, list_actor_gradients)
        # Update model parameters
        self.update_model_parameters(critic_1_gradients, critic_2_gradients, actor_gradients)
        self.update_entropy(critic)
    
    def update_entropy(self, critic):
        if self.autotune and not critic :
            with torch.no_grad():
                state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)
                _, log_pi, _ = self.actor.get_action(state)
            alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()
            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            alpha = self.log_alpha.exp().item()
        

    def update_target_networks(self):
        for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self):
        for epoch in range(self.num_epochs):

            state, done = self.env.reset(), False
            episode_reward = 0
            full_episode_reward = 0
            nb_episode = 0
            episode_timesteps = 0

            for step in range(1000):
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
                # rollout_data.append((state, action, next_state, reward, done_bool))
                self.replay_buffer.add(state, action, next_state, reward, done_bool)

                # Update state
                state = next_state

                if done:
                    state, done = self.env.reset(), False
                    full_episode_reward += episode_reward
                    nb_episode += 1
                    episode_reward = 0
                    episode_timesteps = 0



                # Sample data from rollout worker
                # res = ray.get([self.rollout_worker.remote(self, self.num_steps) for _ in range(self.num_workers)])
                # rollout_data, rollout_reward = zip(*res)
                # # Store data in replay buffer
                # for data in rollout_data: self.replay_buffer.add_rollout_data_to_buffer(data)
                if epoch > 0 :
                    # Update model
                    self.update(critic=True) if epoch%self.policy_frequency==0 else [self.update(critic=False) for _ in range(self.policy_frequency)]
                    # Update target networks
                    self.update_target_networks() if epoch%self.target_update_frequency==0 else None
                    # # Update model
                    # self.update(critic=True) if epoch%self.policy_frequency==0 else [self.update(critic=False) for _ in range(self.policy_frequency)]
                    # # Update target networks
                    # self.update_target_networks() if epoch%self.target_update_frequency==0 else None
            # logs
            print(f"Epoch: {epoch+1}/{self.num_epochs}")
            print(f"Rollout reward: {full_episode_reward/nb_episode}")
            if full_episode_reward/nb_episode > 500:
                break
        self.eval()
    
    def eval(self):
        s,d=self.env.reset(),False
        r=0
        while not d:
            s_t=torch.FloatTensor(s).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _,_,a=self.actor.get_action(s_t)
            a=a.cpu().numpy().squeeze()
            ap = a if self.env.action_space.shape[0] > 1 else np.array([a])
            s,r,d,_=self.env.step(ap)
            self.env.render()
        self.env.close()

if __name__=="__main__":
    env=gym.make('InvertedPendulum-v2')
    ray.init()
    logger = logging.getLogger("ray")
    logger.setLevel(logging.CRITICAL)
    sac=SAC(env, hidden_dim=256, plr=3e-4,qlr=5e-4, gamma=0.99, tau=0.005, alpha=0.2, num_epochs=100, num_wokers=1)
    sac.train()
    
    