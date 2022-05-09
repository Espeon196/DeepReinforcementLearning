if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from common.utils import plot_total_reward


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3, 64)
        self.l2 = nn.Linear(64, 64)

        self.l3 = nn.Linear(64, 64)
        self.l4 = nn.Linear(64, 1)

        self.l5 = nn.Linear(64, 64)
        self.l6 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        m = F.relu(self.l3(x))
        m = self.l4(m)

        v = F.relu(self.l5(x))
        v = torch.exp(self.l6(v))
        return m, v


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005

        self.pi = PolicyNet()
        self.v = ValueNet()

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :])
        probs = self.pi(state)
        m = Normal(probs[0][0], probs[1][0])
        ms = m.sample()
        action = torch.special.expit(ms).item() * 4 - 2
        return [action], m.log_prob(ms)

    def update(self, state, log_action_prob, reward, next_state, done):
        state = torch.tensor(state[np.newaxis, :])
        next_state = torch.tensor(next_state[np.newaxis, :])

        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.detach()
        v = self.v(state)
        loss_fn = nn.MSELoss()
        loss_v = loss_fn(v, target)

        delta = target - v
        loss_pi = - log_action_prob * delta.item()

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    runs = 1
    episodes = 3000
    all_reward_history = np.zeros((runs, episodes))

    for run in range(runs):
        reward_history = []
        agent = Agent()
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, log_prob = agent.get_action(state)
                next_state, reward, done, info = env.step(action)

                agent.update(state, log_prob, reward, next_state, done)
                state = next_state
                total_reward += reward

            reward_history.append(total_reward)

            if episode % 100 == 0:
                print(f"run: {run}, episode: {episode}, total reward: {total_reward}")

        all_reward_history[run] = reward_history

    avg_reward_history = np.average(all_reward_history, axis=0)
    plot_total_reward(avg_reward_history)

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        env.render()
    print(f'Total reward: {total_reward}')
