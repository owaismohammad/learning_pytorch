#imports
from random import shuffle
from h11 import Data
import torch
import torch.nn as nn #all the nn modules
import torch.optim as optim #optimization algo sgd, adam
import torch.nn.functional as F #activation fn relu, tanh
import torch.utils.data as DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.fc1 = nn.Linear(input, 50)
        self.fc2 = nn.Linear(50, output)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
model = NN(784, 10)
x = torch.randn(64, 784)
print(model(x).shape) 
    
#set Device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

#hyperParameter
input_size = 784
output = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 1

#Load Data
train_dataset = datasets.MNIST(root='dataset/', train = True, transform=transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset = train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root = 'dataset/',
                              train = False, 
                              transform= transforms.ToTensor(),
                              download=True)
test_loader = DataLoader(dataset = test_dataset, shuffle= True, batch_size= batch_size)

#initialize network

model = NN(input_size, output).to(device)

##loss and optimizer

crieterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

# Train Network

for epoch in range (num_epochs):
    for batch_idx, (data , targets) in enumerate(train_loader):
        