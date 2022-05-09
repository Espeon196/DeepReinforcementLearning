import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from common.utils import plot_total_reward


class Policy(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimier = optim.Adam(self.pi.parameters(), lr=self.lr)


    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :])
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G

        for reward, prob in self.memory:
            loss += - torch.log(prob) * G

        self.optimier.zero_grad()
        loss.backward()
        self.optimier.step()
        self.memory = []


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    runs = 10
    episodes = 5000
    all_reward_history = np.zeros((runs, episodes))

    for run in range(runs):
        reward_history = []
        agent = Agent()
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, prob = agent.get_action(state)
                next_state, reward, done, info = env.step(action)

                agent.add(reward, prob)
                state = next_state
                total_reward += reward

            agent.update()
            reward_history.append(total_reward)

            if episode % 100 == 0:
                print(f"episode: {episode}, total reward: {total_reward}")

        all_reward_history[run] = reward_history

    avg_reward_history = np.average(all_reward_history, axis=0)
    plot_total_reward(avg_reward_history)
