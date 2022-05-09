import numpy as np
import gym
import matplotlib.pyplot as plt
import seaborn as sns


def energy(state):
    x = state[0]
    g = 0.0009
    v = state[1]

    c = 1 / (g * 1 + 0.5*0.07*0.07)

    return g * (np.sin(x / 0.5 * (np.pi / 2)) + 1) + 0.5 * v * v


def potential_energy(state):
    x = state[0]
    g = 0.0009
    return (np.sin(x / 0.5 * (np.pi / 2)) + 1) * g


def kinetic_energy(state):
    v = state[1]
    return 0.5 * v * v


env = gym.make('MountainCar-v0')
state = env.reset()
print(state)
print(energy(state))
action_space = env.action_space
print(action_space)
done = False

action = 0
next_state, reward, done, info = env.step(action)
print(next_state)

state = env.reset()
done = False


cnt = 0
energy_history = []
potential_history = []
kinetic_history = []
while not done:
    env.render()
    #action = np.random.choice([0, 1])
    if cnt <= 30:
        action = 2
    else:
        action = 1
    state, reward, done, info = env.step(action)
    e = energy(state)
    energy_history.append(e)
    potential_history.append(potential_energy(state))
    kinetic_history.append(kinetic_energy(state))
    cnt += 1
    if cnt % 10 == 0:
        print(reward)
env.close()

sns.set()
plt.plot(potential_history, label='potential')
plt.plot(kinetic_history, label='kinetic')
plt.plot(energy_history, label='total')
plt.legend()
plt.show()