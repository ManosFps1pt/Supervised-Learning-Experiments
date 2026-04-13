# %%
from torchvision import datasets, transforms as T, models
import numpy as np
from torchinfo import summary
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import copy

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Richer augmentation: ColorJitter + RandomRotation give the model more
# varied views of each image, which is critical for a 100-class problem.
transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.ToTensor(),
    T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
])
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
])
train_ds = datasets.CIFAR100("./data/", train=True,  transform=transform_train, download=True)
test_ds  = datasets.CIFAR100("./data/", train=False, transform=transform_test,  download=True)
print(train_ds.data.shape, test_ds.data.shape)
print(len(np.unique(test_ds.classes)))

# %%
# Freeze all pre-trained weights first, then attach new (trainable) layers.
# Because conv1 and fc are assigned AFTER the freeze loop, they are new
# modules that default to requires_grad=True — only the backbone is frozen.
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)
summary(model)

# %%
def train(model, epochs, optimizer, scheduler, train_loader, test_loader, criterion, device):
    """
    scheduler is an explicit argument so each training phase can have its own
    fresh schedule — avoids the 'dead scheduler' bug where LR is already ~0
    when Phase 2 starts.
    """
    best_accuracy = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())  # guard: always initialized
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
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
        val_loss     = val_loss / len(test_loader)
        val_accuracy = correct / total

        if val_accuracy > best_accuracy:
            best_accuracy   = val_accuracy
            best_model_wts  = copy.deepcopy(model.state_dict())

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch: {epoch+1}/{epochs}\tTrain Loss: {train_loss:.4f}\tVal Loss: {val_loss:.4f}\tVal Acc: {val_accuracy:.4f}\tLR: {lr:.2e}")

    model.load_state_dict(best_model_wts)
    print(f"Training complete. Best Validation Accuracy: {best_accuracy:.4f}")

# %%
batch_size      = 128
warmup_epochs   = 10   # Phase 1: only conv1 + fc — quick head alignment
finetune_epochs = 50   # Phase 2: all layers — needs more epochs at lower LR

train_loader = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=8, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

# Label smoothing (0.1) is one of the highest-ROI tricks for many-class problems.
# It prevents overconfident softmax outputs and consistently adds +1-2% accuracy.
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ── Phase 1: Warm-up ──────────────────────────────────────────────────────────
print("\n=== Phase 1: Warm-up (conv1 + fc only) ===")
optimizer_warmup  = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4
)
scheduler_warmup  = optim.lr_scheduler.CosineAnnealingLR(optimizer_warmup, T_max=warmup_epochs)
train(model, warmup_epochs, optimizer_warmup, scheduler_warmup,
      train_loader, test_loader, criterion, device)

# ── Phase 2: Full Fine-tune ───────────────────────────────────────────────────
# Fresh optimizer + scheduler — warmup schedule is exhausted by now.
# Lower LR (1e-4) is crucial: too high will overwrite the pretrained backbone.
print("\n=== Phase 2: Full Fine-tune (all layers) ===")
for param in model.parameters():
    param.requires_grad = True
optimizer_finetune = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_finetune = optim.lr_scheduler.CosineAnnealingLR(optimizer_finetune, T_max=finetune_epochs)
train(model, finetune_epochs, optimizer_finetune, scheduler_finetune,
      train_loader, test_loader, criterion, device)
