import numpy as np
import torch


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)

lr = 0.001
iters = 10000

for i in range(iters):
    y = rosenbrock(x0, x1)

    if x0.grad:
        x0.grad.zero_()
    if x1.grad:
        x1.grad.zero_()
    y.backward()

    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data

print(x0, x1)