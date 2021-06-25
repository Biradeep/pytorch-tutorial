import torch
import torchvision
from torchvision import transforms, datasets
train = datasets.MNIST("", train=True, download=True,transform= transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,transform= transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

#super runs initiallisation for nn.moudle as well anything else we put in init
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64) #input is the flatned image
        self.fc2 = nn.Linear(64, 64) #input for fc2 is output of fc1 and output is a choice
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) #10 because there is 10 classes 0-9

    def forward(self, x):
        x = F.relu(self.fc1(x)) #relu is the activation function used
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) #for the output we dont want relu we want some function that constrains to one of the neurons to be "fired" for example a prob dist function
        return F.log_softmax(x, dim=1) #dim=1 is similar to axis; which thing is the prob dist which we want to sum to 1; we are distributing amognst outplayer tensor

net = Net()
print(net)

X = torch.rand((28,28))
X = X.view(-1, 28*28) #the -1 specifies that this input will be of an unknown shape; our actual shape is 1 28 28
print(X)

output = net(X)
print(output)
