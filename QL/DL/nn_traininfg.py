## Loss: How wrong is a model. Should decrease with time
## Optimizer: It optimizes all the possible weights to reduce the loss slowly over the time (learning rate)
import torch
import torchvision

from torchvision import datasets, transforms

train = datasets.MNIST("", train = True, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train = False, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
trainset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

## batch size: How many at a time do we want to pass to a model

for data in trainset:
    print(data)
    break

x, y = data[0][0], data[1][0]

print(y)

import torch.nn as nn   ## for creation

import torch.nn.functional as F ## for the functions we are going to create
## Both are very similar

class Net(nn.Module): ## Inherits from the nn.module 
    def __init__(self):
        super().__init__() ## corresponds to nn.module and runs the initialization for the nn.module
## Defining Layers
## Target : 3 Layers fo 64 Neurons for Hidden Layers
        self.fc1 = nn.Linear(28*28, 64)
##  self.fc1 = nn.Linear(input, output) 
## Here the input is what we are feeding to the model and the output can be anything we want, for instance - 64
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        ## How do we want the data to pass through here, well here we go!      
        x = F.relu(self.fc1(x))
        # RELU : whether the neuron is active or not; runs on the output, and not on the input data
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim = 1)

net = Net()
X = torch.rand((28,28))
X = X.view(1,28*28)
# output = net(X)
# print(output)

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr = 0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        ## data is a batch of featuresets and Labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss  = F.nll_loss(output,y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X,y = data
        output = net(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy : ", round(correct/total, 3))

import matplotlib.pyplot as plt

print(torch.argmax(net(X[0].view(-1,28*28))))

plt.imshow(X[0].view(28,28))
plt.show()

