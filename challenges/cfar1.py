# %%
from torchvision import datasets
import torch
import torchvision.transforms as T
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchinfo

# %%
transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
train_ds = datasets.CIFAR10("./data", train=True, transform=transform_train, download=True)
test_ds = datasets.CIFAR10("./data", train=False, transform=transform_test, download=True)

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = torch.relu(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = torch.relu(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        # print(x.shape)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        x = torch.relu(x)
        # print(x.shape)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

# %%
import copy

def train(model, epochs, optimizer, train_loader, test_loader, criterion, device):
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            # batch_X = batch_X.reshape(batch_X.size(0), -1)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
batch_size = 256
epochs = 60


# %%
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# %%
from torchinfo import summary
summary(model)

# %%
train(model, epochs, optimizer, train_loader, test_loader, criterion, device)


