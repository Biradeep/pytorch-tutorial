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
#print(net)

X = torch.rand((28,28))
X = X.view(-1, 28*28) #the -1 specifies that this input will be of an unknown shape; our actual shape is 1 28 28
#print(X)

output = net(X)
#print(output)

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.001) # this corresponds to everything that are adjustable in the model; lr dictates the size of the step that opt will take to get to local/global minimum loss; usually we decay lr but for this problem simple

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset: # data is a batch of featuresets and labels
        X, y = data
        net.zero_grad()
        #2 reasons we batch data: 1) reduces training time 2) there is a law of diminishing returns; helps to generalise
        output = net (X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward() #backward propogate
        optimizer.step()
    print(loss)

correct = 0
total = 0

# check how correct model was
with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print("Accuracy: ", round(correct/total, 3))

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
plt.imshow(X[1].view(28,28))
plt.show() #visualisation of input number/what we want to the model to predict

print(torch.argmax(net(X[1].view(-1,28*28))[0])) #output of model








