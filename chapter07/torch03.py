import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns


np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x = torch.tensor(x, requires_grad=False)
y = torch.tensor(y, requires_grad=False)

W = torch.tensor(np.zeros((1, 1)), requires_grad=True)
b = torch.tensor(np.zeros(1), requires_grad=True)


def predict(x):
    y = torch.matmul(x, W) + b
    return y


def mean_squared_error(x0, x1):
    diff = x0 - x1
    return torch.sum(diff ** 2) / len(diff)


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    if W.grad:
        W.grad.zero_()
    if b.grad:
        b.grad.zero_()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if i % 10 == 0:
        print(loss.data)

print('----')
print(f'W = {W.data}')
print(f'b = {b.data}')

sns.set()
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
t = torch.tensor(t)
y_pred = predict(t)
plt.plot(t.data, y_pred.data, color='r')
plt.show()