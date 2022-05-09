import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import seaborn as sns

np.random.seed(0)
x = np.random.rand(100, 1)
x = torch.tensor(x, requires_grad=False).float()
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
y = y.float()

lr = 0.2
iters = 10000


class TwoLayerNet(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = nn.Linear(1, hidden_size)
        self.l2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        y = torch.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


model = TwoLayerNet(10, 1)
optimizer = SGD(model.parameters(), lr)
loss = nn.MSELoss()

for i in range(iters):
    y_pred = model(x)
    output = loss(y, y_pred)

    optimizer.zero_grad()
    output.backward()

    optimizer.step()
    if i % 1000 == 0:
        print(output.data)

sns.set()
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.linspace(0, 1, 1000)[:, np.newaxis]
t = torch.tensor(t).float()
y_pred = model(t)
plt.plot(t.data, y_pred.data, color='r')
plt.show()