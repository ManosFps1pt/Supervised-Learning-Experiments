# %%
from torchvision import datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# %%
transform = T.ToTensor()
train_ds = datasets.MNIST(root="./data", download=True, transform=transform, train=True)
test_ds = datasets.MNIST(root="./data", download=True, transform=transform, train=False)

# %%
t_ds = datasets.MNIST(root="./data", download=True, transform=transform, train=False)

# %%
X = train_ds.data[:5000].numpy()
y = train_ds.targets[:5000].numpy()

X: np.ndarray = X.reshape(5000, -1)
X = X / 255.0
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# %%
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='tab10', alpha=0.7)
plt.title("MNIST dataset crushed from 784D down to 2D using PCA")
plt.xlabel("Principal Component 1 (Best Angle)")
plt.ylabel("Principal Component 2 (2nd Best Angle)")
# This handy little trick pulls the legend out automatically from your scatter plot
plt.legend(*scatter.legend_elements(), title="Digits")
plt.show()

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Take the labels (y) and our 2D summary (X_pca) 
# and split them into a training and testing set
split_index = 4000
X_train, X_test = X_pca[:split_index], X_pca[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 2. Build the exact classifier you just mentioned
clf = DecisionTreeClassifier(max_depth=10, random_state=42)

# 3. Fit it using ONLY the 2 lines of PCA data!
clf.fit(X_train, y_train)

# 4. Predict and see how well it works
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"{accuracy * 100:.2f}%")


# %%
kmeans = KMeans(n_clusters=10)
cluster_labels = kmeans.fit_predict(X_pca)

# %%
plt.figure(figsize=(10, 8))

# Notice the change here: c=cluster_labels
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)

plt.title("MNIST Clusters Invented by K-Means (No Human Labels!)")
plt.xlabel("Principal Component 1 (Best Angle)")
plt.ylabel("Principal Component 2 (2nd Best Angle)")

# This adds a legend matching the colors to the clusters
plt.legend(*scatter.legend_elements(), title="Invented Clusters")

plt.show()


# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

# %%
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.conv1(x)
        x = torch.relu(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = torch.relu(x)
        x = self.pool2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1) # x.size(0) is the batch size
        # print(x.shape)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

# %%


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
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        val_loss = val_loss / len(train_loader) # Replace with len(val_loader)
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
trainLoader = DataLoader(train_ds, batch_size=32)
testLoader = DataLoader(test_ds, batch_size=32)
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 10
criterion = nn.CrossEntropyLoss()

# %%
train(model, epochs, optimizer, trainLoader, testLoader, criterion, device)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
trainLoader = DataLoader(train_ds, batch_size=32)
testLoader = DataLoader(test_ds, batch_size=32)
model = Net2().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 10
criterion = nn.CrossEntropyLoss()

# %%
train(model, epochs, optimizer, trainLoader, testLoader, criterion, device)


