from collections import deque
import random
import numpy as np
import gym
import torch


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.stack([x[1] for x in data]), dtype=torch.float)
        reward = torch.tensor(np.stack([[x[2]] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.stack([[x[4]] for x in data]).astype(np.int32))
        return state, action, reward, next_state, done

    def reset(self):
        self.buffer.clear()


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

    for episode in range(3):
        state = env.reset()
        done = False

        while not done:
            action = [np.random.rand() * 4 - 2]
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

    state, action, reward, next_state, done = replay_buffer.get_batch()
    print("state-shape", state.shape)
    print("action-shape", action.shape)
    print("action-dtype", action.dtype)
    print("reward-shape", reward.shape)
    print("next_state-shape", next_state.shape)
    print("done-shape", done.shape)
