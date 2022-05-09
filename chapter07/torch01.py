import numpy as np
from torch.autograd import Variable
import torch

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a, b = Variable(torch.tensor(a)), Variable(torch.tensor(b))
c = torch.matmul(a, b)
print(c)

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
a, b = torch.tensor(a), torch.tensor(b)
c = torch.matmul(a, b)
print(c)