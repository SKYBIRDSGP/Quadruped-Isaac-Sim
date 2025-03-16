### This consists the basic functions of the Pytorch 
## You can check if the pytorch is rightly installed, the version and the availablity of CUDA 
import torch
print(torch.__version__)
print(torch.cuda.is_available())

### Now let us see some of the basic functions of the pytorch

### 1 aray multiplicn
print("### 1 aray multiplicn")
x =torch.Tensor([5,3])
y =torch.Tensor([2,1])

print(x*y)

# // Tensor is an array, sounds scary nut is simple

### 2 Get the shape
print("### 2 Get the shape")
x = torch.zeros([2,5])
print(x.shape)

### 3 Random initializn
print("### 3 Random initializn")
y = torch.rand([2,5])
print(y.shape)
print(y)

### 4 Reshaping
print("### 4 Reshaping")
# In case of images to be fed to the NN, we need to flatten it, so we use the reshaping to accomplish it
# for eg, if we have 2x5, then we flatten it to 1x10
y = y.view([1,10])
print(y)

