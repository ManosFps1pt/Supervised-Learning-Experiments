# %%
from torchvision import datasets, transforms as T, models
import numpy as np
from torchinfo import summary
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
transform_train = T.Compose([
    T.RandomCrop(32, padding=4),   # Native CIFAR-100 resolution
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
])
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
])
train_ds = datasets.CIFAR100("./data/", train=True, transform=transform_train, download=True)
test_ds = datasets.CIFAR100("./data/", train=False, transform=transform_test, download=True)
print(train_ds.data.shape, test_ds.data.shape)
print(len(np.unique(test_ds.classes)))

# %%
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Adapt for 32x32 CIFAR images (from the original ResNet paper for CIFAR):
# The default 7x7 conv (stride=2) + maxpool would shrink 32x32 -> 8x8 immediately,
# destroying spatial info. Replace with a 3x3 conv (stride=1) and remove the maxpool.
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

# Freeze backbone for a warm-up phase — only train the head first
for param in model.parameters():
    param.requires_grad = False
# Replace head: pretrained ResNet fc outputs 1000 (ImageNet), we need 100
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)
summary(model)

# %%
import copy

def train(model, epochs, optimizer, train_loader, test_loader, criterion, device):
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            # batch_X = batch_X.reshape(batch_X.size(0), -1)
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0   
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        scheduler.step()
        val_loss = val_loss / len(test_loader) # Replace with len(val_loader)
        val_accuracy = correct / total
        
        # --- Save Best Model ---
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
    
        print(f"Epoch: {epoch+1}/{epochs}\tTrain Loss: {train_loss:.4f}\tVal Loss: {val_loss:.4f}\tVal Acc: {val_accuracy:.4f}")
    model.load_state_dict(best_model_wts)
    print(f"Training complete. Best Validation Accuracy: {best_accuracy:.4f}")

# %%
batch_size = 128
epochs = 30

# %%
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# %%
train(model, epochs, optimizer, train_loader, test_loader, criterion, device)

# %%
for param in model.parameters():
    param.requires_grad = True

# %%
train(model, epochs, optimizer, train_loader, test_loader, criterion, device)


