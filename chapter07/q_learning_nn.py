import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from common.gridworld import GridWorld
import seaborn as sns

HEIGHT, WIDTH = 3, 4


def one_hot(state):
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return torch.tensor(vec[np.newaxis, :])


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(HEIGHT * WIDTH, 100)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.l2(x)
        return x


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optim.SGD(self.qnet.parameters(), self.lr)
        self.loss = nn.MSELoss()

    def get_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state_vec)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = torch.zeros(1)
        else:
            next_qs = self.qnet(next_state)
            next_q = torch.max(next_qs, dim=1)[0]
            next_q.detach()

        target = self.gamma * next_q + reward
        qs = self.qnet(state)
        q = qs[:, action]
        output = self.loss(target, q)

        self.optimizer.zero_grad()
        output.backward()
        self.optimizer.step()

        return output.item()


if __name__ == '__main__':
    """
    qnet = QNet()
    state = (2, 0)
    state = one_hot(state)

    qs = qnet(state)
    print(qs.shape)
    """

    env = GridWorld()
    agent = QLearningAgent()

    episodes = 1000
    loss_history = []

    for episode in range(episodes):
        state = env.reset()
        state = one_hot(state)
        total_loss, cnt = 0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = one_hot(next_state)

            loss = agent.update(state, action, reward, next_state, done)
            total_loss += loss
            cnt += 1
            state = next_state

        average_loss = total_loss / cnt
        loss_history.append(average_loss)

    sns.set()
    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.plot(range(len(loss_history)), loss_history)
    plt.show()

    Q = {}
    for state in env.states():
        for action in env.action_space:
            q = agent.qnet(one_hot(state))[:, action]
            Q[state, action] = q.item()
    env.render_q(Q)