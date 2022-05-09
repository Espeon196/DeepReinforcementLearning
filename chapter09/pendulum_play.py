import numpy as np
import gym


env = gym.make('Pendulum-v1')
state = env.reset()
print(state)
action_space = env.action_space
print(action_space)
done = False

action = [0.43]
next_state, reward, done, info = env.step(action)
print(next_state)

state = env.reset()
done = False


cnt = 0

while not done:
    env.render()
    #action = [np.random.rand() * 4 - 2]
    action = [2]
    state, reward, done, info = env.step(action)

    cnt += 1
    if cnt % 10 == 0:
        print(reward)
env.close()
