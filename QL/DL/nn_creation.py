import torch
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
output = net(X)
print(output)
