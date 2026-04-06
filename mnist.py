import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time

from train_function import train

torch.manual_seed(42)

# Downloading training and test datasets
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Transformations applied on each image
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std dev
])

train_dataset.transform = transform
test_dataset.transform = transform

# Data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda: 168.66, cpu: 170.79, collab cuda: 181.86, collab cpu: 180.45
# device = torch.device("cpu")
print(device)
print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.version.cuda)
# print(torch.__version__)
# exit()

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 320)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

model = ConvNet()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4) # lr = 0.001
epochs = 10 # epochs = 10
i = time.perf_counter()
train(model, train_loader, test_loader, optimizer, epochs)
print(f"elapsed time on {device}: {time.perf_counter() - i}")