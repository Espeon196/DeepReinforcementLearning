if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import numpy as np
import gym
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from common.utils import plot_total_reward
from replay_buffer import ReplayBuffer


def init_weight(size):
    f = size[0]
    v = 1. / np.sqrt(f)
    return torch.tensor(np.random.uniform(low=-v, high=v, size=size), dtype=torch.float)


class ActorNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden1_size=400, hidden2_size=300, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state[0], hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_action[0])

        self.num_state = num_state
        self.num_action = num_action

        self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = torch.tanh(self.fc3(h)) * 2
        return y


class CriticNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden1_size=400, hidden2_size=300, init_w=3e-4):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state[0], hidden1_size)
        self.fc2 = nn.Linear(hidden1_size+num_action[0], hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)

        self.num_state = num_state
        self.num_action = num_action

        self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, action):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(torch.cat([h, action], dim=1)))
        y = self.fc3(h)
        return y


class OrnsteinUhlenbeckProcess:
    def __init__(self, theta=0.15, mu=0.0, sigma=0.2, dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.num_steps = 0

        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0
            self.c = sigma
            self.sigma_min = sigma

    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.num_steps) + self.c)
        return sigma

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma() * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.num_steps += 1
        return x


class DDPGAgent:
    def __init__(self, actor, critic, optimizer_actor, optimizer_critic, batch_size=64, device=torch.device('cpu')):
        self.actor = actor
        self.critic = critic
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.replay_buffer = ReplayBuffer(100000, batch_size=batch_size)
        self.device = device

        self.gamma = 0.99
        self.tau = 1e-3
        self.batch_size = batch_size
        self.random_process = OrnsteinUhlenbeckProcess(size=actor.num_action[0])

        self.num_state = actor.num_state
        self.num_action = actor.num_action

    def add_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def reset_memory(self):
        self.replay_buffer.reset()

    def get_action(self, state, greedy=False):
        state_tensor = torch.tensor(state[np.newaxis, :], dtype=torch.float, device=self.device)
        action = self.actor(state_tensor)
        if not greedy:
            action += torch.tensor(self.random_process.sample(), dtype=torch.float, device=self.device)

        return action.squeeze(0).detach().cpu().numpy()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        q_value = self.critic(state, action)
        next_q_value = self.critic_target(next_state, self.actor_target(next_state))
        target_q_value = reward + (self.gamma * next_q_value * (1 - done))

        critic_loss = F.mse_loss(q_value, target_q_value)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        actor_loss = - self.critic(state, self.actor(state)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1. - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1. - self.tau) + param.data * self.tau)


if __name__ == '__main__':
    runs = 1
    episodes = 300
    all_rewards = np.zeros((runs, episodes))

    for run in range(runs):
        env = gym.make('Pendulum-v1')
        num_state = env.observation_space.shape
        num_action = env.action_space.shape
        max_steps = env.spec.max_episode_steps

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        actorNet = ActorNetwork(num_state, num_action).to(device)
        criticNet = CriticNetwork(num_state, num_action).to(device)
        optimizer_actor = optim.Adam(actorNet.parameters(), lr=1e-4)
        optimizer_critic = optim.Adam(criticNet.parameters(), lr=1e-3, weight_decay=1e-2)
        agent = DDPGAgent(actorNet, criticNet, optimizer_actor, optimizer_critic)

        reward_history = []
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                agent.add_memory(state, action, reward, next_state, done)

                agent.update()

                state = next_state

            reward_history.append(total_reward)

            if episode % 50 == 0:
                print(f'episode: {episode}, total_reward: {total_reward}')

        all_rewards[run] = reward_history

        if run == 0:
            for episode in range(3):
                done = False
                state = env.reset()
                while not done:
                    action = agent.get_action(state, greedy=True)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    env.render()

        env.close()

    avg_reward_history = np.average(all_rewards, axis=0)
    plot_total_reward(avg_reward_history)