import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit, Agent


np.random.seed(0)
sns.set()

for epsilon in [0.0001, 0.001, 0.1, 0.3, 0.5, 0.8, 1]:
    runs = 200
    steps = 1000
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)
    plt.plot(avg_rates, label=epsilon)


plt.ylabel('Rates')
plt.xlabel('Steps')
plt.legend(loc='best')
plt.show()