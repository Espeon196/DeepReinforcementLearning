import numpy as np

x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])


e = np.sum(x * pi)
print(f'E_pi[x] = {e}')


n = 100
samples = []
for _ in range(n):
    s = np.random.choice(x, p=pi)
    samples.append(s)
print('MC: {:.2f} (var: {:.2f})'.format(np.mean(samples), np.var(samples)))


b = np.array([0.2, 0.2, 0.6])
#b = np.array([1/3, 1/3, 1/3])
samples = []
for _ in range(n):
    idx = np.arange(len(b))
    i = np.random.choice(idx, p=b)
    s = x[i]
    rho = pi[i] / b[i]
    samples.append(rho * s)
print(f'IS: {np.mean(samples):.2f} (var: {np.var(samples):.2f})')