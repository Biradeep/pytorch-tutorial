import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA = False
#do not need class here but probably would need steps for this case
class DogsVSCats():
    IMG_SIZE = 50
    #turning all images to 50 by 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    TESTING = "PetImages/Testing"
    LABELS = {CATS: 0, DOGS: 1} #one hot vector format and labelled
    training_data = []
    catcount = 0 #pay attention to balance
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            #we want to iterate over the images within the directory; f file name
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #CONVERTING TO GREYSCALE; is colour a parameter for determining cat or dog
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE)) #want to resize images
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]]) #np.eye converts one hot vector

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass
                    # some images may have error hence accounting for that print(str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) # first number input, outputs 32 conv features then kernel size in this 5by5 kernal
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        #print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
# determining what input size is after conv to fc layers
#
    def forward(self, x):
        x = self.convs(x) #pass through all conv layers
        x = x.view(-1, self._to_linear) #flaten it
        x = F.relu(self.fc1(x)) #pass through first fc layer
        x = self.fc2(x) #should have activation layer
        return F.softmax(x, dim=1) # the x is a batch of xes, we want it to be distributed against cats and dogs dim1

if REBUILD_DATA:
    dogsvscats = DogsVSCats()
    dogsvscats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)

print(len(training_data))

#print(training_data[1])
#import matplotlib.pyplot as plt
#plt.imshow(training_data[0][0], cmap="gray")
#plt.show()

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0 #SCALING IMAGERY PIXEL VALUE BETWEEN 0-255 THIS MAKES IT 0-1
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size)
#training and testing data
train_X = X[:-val_size] #up to negative valsize
train_y = y[:-val_size]

test_X = X[-val_size:] #negative val size onwards
test_y = y[-val_size:]

#print(len(train_X)) to check slicing worked
#print(len(test_X))

def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs,y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

def test(size=32):
    random_start = np.random.randint(len(test_X)-size)
    X, y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1,1,50,50), y)
    return val_acc, val_loss

val_acc, val_loss = test(size=32)
print(val_acc, val_loss)

import time

MODEL_NAME = f"model-{int(time.time())}"

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

print(MODEL_NAME)

def train():
    BATCH_SIZE = 100 #if memory error lower batch size
    EPOCHS = 5
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
                batch_y = train_y[i:i+BATCH_SIZE]

                acc, loss = fwd_pass(batch_X, batch_y, train=True)
                if i % 50 == 0: #every 50 steps this will be true
                    val_acc, val_loss = test(size=100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")
train()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

model_name = MODEL_NAME

def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split('\n')

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)

    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2)

    ax2.plot(times, losses, label="loss")
    ax2.plot(times, val_losses, label="val_loss")
    ax2.legend(loc=2)

    plt.show()

create_acc_loss_graph(model_name)



