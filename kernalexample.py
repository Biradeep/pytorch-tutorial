import torch
import torch.nn as nn
import torch.nn.functional as F
a_data = [[1., 2., 4., 1.], [1., 2., 1., 0.], [0., 2., 0., 1.], [1., 2., 8., 5.]]
a = torch.tensor(a_data)
b_data = [[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]]
b = torch.tensor(b_data)
#b.requires_grad = True
#print(a)
#print(b)
B = torch.nn.Parameter( b )
#B = b.view(-1, 3*3)
class Net(nn.Module):
    def __init__(self):
       super().__init__()
       self.conv = nn.Conv2d(4*4, 2*2, 3,stride=1)
       #with torch.no_grad():
       self.conv.weight = B

    def forward(self, x):
        return self.conv(x)


net = Net()
#net.conv.weight = B
output = net(a)
print(output)


