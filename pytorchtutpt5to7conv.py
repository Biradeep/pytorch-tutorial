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

if REBUILD_DATA:
    dogsvscats = DogsVSCats()
    dogsvscats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)

print(len(training_data))

#print(training_data[1])
#import matplotlib.pyplot as plt
#plt.imshow(training_data[0][0], cmap="gray")
#plt.show()



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

        print(x[0].shape)
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

BATCH_SIZE = 100 #IF MEMORY ERROR OCCURS LOWER BATCH SIZE MIN 8
EPOCHS = 1
#epochs refers to one cycle through the full training dataset
for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        #print(i, i+BATCH_SIZE)
        batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_y = train_y[i:i+BATCH_SIZE]
        #NOW FITMENT BY ZERO GRAD; there might be cases where to nn build one network then you would use specific network in question
        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

print(loss)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1,1,50,50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy:", round(correct/total,3))

torch.cuda.is_available()

