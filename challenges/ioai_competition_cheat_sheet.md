# 🚀 IOAI Competition Ultimate Cheat Sheet

*This is your survival guide for the competition. It contains rapid-deployment code snippets, essential theory, and plotting templates so you can copy-paste and tweak instead of writing from scratch.*

---

## 1. 📊 Data Processing & EDA (Pandas & Scikit-learn)

### Quick Exploratory Data Analysis (EDA)

```python
import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv("data.csv")

# Quick look
display(df.head())
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True) # For numerical
df.fillna(df.mode().iloc[0], inplace=True)            # For categorical

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)
```

### Visualizations with Matplotlib/Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Distribution of a variable
plt.figure(figsize=(8, 5))
sns.histplot(df['target'], bins=30, kde=True, color='purple')
plt.title('Target Variable Distribution')
plt.show()

# 2. Correlation Matrix
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# 3. Scatter Plot for classification
plt.figure(figsize=(8, 5))
sns.scatterplot(x='feature_1', y='feature_2', hue='target', data=df, palette='Set1')
plt.title('Feature Scatter Plot')
plt.show()
```

### Prep & Pipelines

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # NEVER fit on test data!
```

---

## 2. 🌳 Classical Machine Learning (Scikit-Learn)

**Theory:**

- **Random Forest:** Good baseline, robust to overfitting, no scaling needed.
- **Gradient Boosting (XGBoost/LightGBM):** Usually wins tabular competitions. Prone to overfit if not tuned.
- **Logistic Regression:** Good for fast, linear baselines.

### Training & Evaluation Template

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize and Train
model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
preds = model.predict(X_test_scaled)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print("Classification Report:\n", classification_report(y_test, preds))

# Confusion Matrix Plot
conf_mat = confusion_matrix(y_test, preds)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

---

## 3. 🔥 PyTorch Essentials (Deep Learning)

**Theory:**

- Always verify your tensor shapes `(Batch_Size, Channels, Height, Width)` for vision.
- `loss.backward()` computes gradients, `optimizer.step()` updates weights, `optimizer.zero_grad()` clears old gradients.

### The Golden PyTorch Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example simple model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleMLP(input_dim=X_train_scaled.shape[1], output_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Dataloaders (assuming X_train, y_train are numpy arrays)
train_dataset = TensorDataset(torch.tensor(X_train_scaled).float(), torch.tensor(y_train.values).long())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train() # Set to training mode (enables Dropout/BatchNorm)
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
      
        optimizer.zero_grad()           # 1. Zero gradients
        outputs = model(batch_X)        # 2. Forward pass
        loss = criterion(outputs, batch_y) # 3. Compute loss
        loss.backward()                 # 4. Backward pass
        optimizer.step()                # 5. Update weights
      
        running_loss += loss.item()
  
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")
```

---

## 4. 👁️ Computer Vision (Torchvision)

**Theory:**

- CNNs capture spatial hierarchies.
- Transfer learning is crucial: start with a pre-trained `ResNet`, replace the final layer to match your classes.
- **Freezing:** If your dataset is tiny, freeze the backbone (the early layers) so it retains its general image-understanding capabilities.

### Image Transforms & Augmentation

```python
from torchvision import transforms

# Augmentations fight overfitting
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet stats
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Transfer Learning (ResNet)

```python
from torchvision import models

# Load pretrained ResNet
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze early layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer (unfrozen by default)
num_ftrs = resnet.fc.in_features
num_classes = 10 # Change for your dataset
resnet.fc = nn.Linear(num_ftrs, num_classes)

resnet = resnet.to(device)
# -> Train using the Golden Loop
```

---

## 5. 🗣️ NLP & Transformers (Hugging Face)

**Theory:**

- Text must be tokenized before passing to a model. The tokenizer and model MUST match exactly.
- BERT is great for classification. Generative (causal) models like GPT are for text generation.

### Quick Pipeline (Zero-resource)

```python
from transformers import pipeline

# Sentiment Analysis / Text Classification pipeline out-of-the-box
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print(classifier("This competition is awesome!"))
```

### Fine-Tuning Setup Snippet

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_link = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_link)
model = AutoModelForSequenceClassification.from_pretrained(model_link, num_labels=2).to(device)

text = ["I love AI", "Debugging is hard"]
# Padding and truncation are essential for batches
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)

outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
```

---

## 6. 🎧 Audio & Advanced (Hugging Face)

### Whisper (Audio Transcription)

```python
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
result = transcriber("audio_file.wav")
print(result["text"])
```

### CLIP (Zero-Shot Image Classification)

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("image.jpg")
labels = ["a cat", "a dog", "a machine learning model"]

# Predict which text matches the image
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1) 

print(probs) # Probabilities corresponding to labels
```

---

## 🛠️ Debugging Checklist for the Competition

- [ ] **Loss isn't decreasing?** Check your Learning Rate. It might be too high (exploding) or too low (frozen). Try $1e-3$ or $1e-4$.
- [ ] **Shape Mismatch Error?** Print `.shape` on your inputs right before the forward pass.
- [ ] **CUDA Out of Memory?** Reduce your `batch_size`.
- [ ] **Accuracy at 0 or terrible?** Check if you forgot `optimizer.zero_grad()`.
- [ ] **Validation Loss rising while Train Loss falls?** You are overfitting! Add `nn.Dropout()`, use Data Augmentations, or apply L2 Regularization (`weight_decay` in optimizer).
