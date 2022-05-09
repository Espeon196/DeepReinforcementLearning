import copy
import numpy as np
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer

sns.set()


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(2, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 3

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :])
            qs = self.qnet(state)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(len(action)), action]

        next_qs_target = self.qnet_target(next_state)
        next_qs = self.qnet(next_state)
        max_action = next_qs.argmax(axis=1)
        next_q = next_qs_target[np.arange(len(max_action)), max_action]

        next_q.detach()
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.HuberLoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def set_epsilon(self, episode):
        self.epsilon = 0.001 + 0.9 / (1.0 + episode)


def energy(state):
    x = state[0]
    g = 0.0009
    v = state[1]

    c = 1 / (g * 2 + 0.5*0.07*0.07)

    return c * (g * (np.sin(x / 0.5 * (np.pi / 2)) + 1) + 0.5 * v * v)


if __name__ == '__main__':
    runs = 30
    episodes = 300
    all_rewards = np.zeros((runs, episodes))
    all_cnt = np.full((runs, episodes), fill_value=200)
    env = gym.make('MountainCar-v0')
    best_agent = None
    best_cnt = 201

    for run in range(runs):
        sync_interval = 20
        agent = DQNAgent()
        reward_history = []

        for episode in range(episodes):
            state = env.reset()
            done = False
            agent.set_epsilon(episode)
            total_reward = 0
            cnt = 0

            while not done:
                cnt += 1
                action = agent.get_action(state)
                next_state, _, done, info = env.step(action)
                reward = energy(next_state) - energy(state)
                if done:
                    reward += 10

                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    all_cnt[run, episode] = cnt

                if done and cnt < best_cnt:
                    cnt = best_cnt
                    best_agent = copy.deepcopy(agent)

            if episode % sync_interval == 0:
                agent.sync_qnet()

            reward_history.append(total_reward)
            if episode % 50 == 0:
                print(f'run: {run}, episode: {episode}, total_reward: {total_reward}, epsilon: {agent.epsilon}')

        all_rewards[run] = reward_history

    avg_reward = np.average(all_rewards, axis=0)

    fig = plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(avg_reward)), avg_reward)
    plt.show()
    plt.close()
    # fig.savefig('dqn_rmse.png')

    avg_cnt = np.average(all_cnt, axis=0)

    fig = plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Total cnt')
    plt.plot(range(len(avg_cnt)), avg_cnt)
    plt.show()

    best_agent.epsilon = 0
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = best_agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        env.render()
    print(f'Total reward: {total_reward}')