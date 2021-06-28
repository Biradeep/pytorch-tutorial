import torch
import torch.nn as nn
import torch.nn.functional as F
a_data = [[1., 2., 4., 1.], [1., 2., 1., 0.], [0., 2., 0., 1.], [1., 2., 8., 5.]]
a = torch.tensor(a_data,dtype=torch.float32)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)


    def forward(self, x):
        return self.pool(x)


net = Net()


a_new = a.unsqueeze(0).unsqueeze(0)
print(a_new.shape)

print(net.forward(a_new))