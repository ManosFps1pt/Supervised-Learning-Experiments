# IOAI 7-Day Intensive Study Plan

Goal: Build a **personal AI competition manual** while learning the core
concepts required for IOAI-style problems.

Workflow for every topic:

    learn concept → implement it → test on dataset → write 1 page of notes

After one week you should have **15--20 pages of practical notes**
containing: - formulas - when to use each algorithm - minimal code
templates - common pitfalls

------------------------------------------------------------------------

# Day 1 --- Python ML Pipeline (Foundation)

Dataset: Iris Dataset

Learn: - pandas basics - train/test split - normalization - training a
model - evaluation metrics

Libraries: - pandas - scikit-learn - matplotlib

Example pipeline:

``` python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

Train workflow:

    dataset → split → train → predict → evaluate

Notes to write: - ML pipeline template - accuracy / F1 formulas -
train_test_split example

------------------------------------------------------------------------

# Day 2 --- Classical ML Algorithms

Dataset: Wine Dataset

Algorithms to test: - Logistic Regression - K-Nearest Neighbors -
Decision Trees - Random Forest

Experiment: Train all models and compare accuracy.

Example:

``` python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
```

Concepts to understand: - how hyperparameters affect results - when tree
models outperform linear models

Notes: - Logistic regression → linear decision boundary - Random forest
→ strong for tabular data - KNN → sensitive to scaling

------------------------------------------------------------------------

# Day 3 --- Data Processing & Feature Engineering

Dataset: Titanic Dataset

Skills: - handling missing data - encoding categorical variables -
normalization - feature engineering

Important tools:

    StandardScaler
    OneHotEncoder
    SimpleImputer

Experiment: Compare three setups:

    raw data
    processed data
    feature engineered data

Notes: - preprocessing pipeline - normalization rules - one-hot encoding
examples

------------------------------------------------------------------------

# Day 4 --- Clustering & Dimensionality Reduction

Dataset: MNIST

Algorithms: - K-Means - PCA - t-SNE

Experiment:

    MNIST → PCA → visualize 2D

Then cluster digits using K-Means.

Goals: - understand embeddings - visualize high-dimensional data

Notes: - PCA formula - clustering intuition - limitations of clustering

------------------------------------------------------------------------

# Day 5 --- Neural Networks

Library: PyTorch

Dataset: MNIST

Learn: - tensors - autograd - training loop

Typical training loop:

``` python
for batch in data:
    optimizer.zero_grad()
    output = model(batch)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

Important optimizer:

    Adam

Notes: - PyTorch training loop template - loss functions - activation
functions

------------------------------------------------------------------------

# Day 6 --- Computer Vision

Dataset: CIFAR-10

Learn: - CNN basics - image preprocessing - data augmentation

Libraries:

    torchvision

Experiment: Train a simple CNN classifier.

Notes: - CNN architecture - convolution → pooling → classifier - image
preprocessing pipeline

------------------------------------------------------------------------

# Day 7 --- Competition Skills

Dataset: Fashion-MNIST

Practice competition tasks:

1.  Generate adversarial examples
2.  Tune hyperparameters
3.  Train models quickly on small datasets

Workflow:

    baseline model
    improve accuracy
    analyze mistakes

Notes: - hyperparameter tuning tricks - debugging checklist - model
selection guide

------------------------------------------------------------------------

# Final Notes Structure

Your manual should contain:

## 1. ML Pipeline

    load data
    preprocess
    train model
    evaluate

## 2. Algorithms Cheat Sheet

  Problem        Model
  -------------- ---------------
  tabular data   random forest
  images         CNN
  text           transformer

## 3. Metrics Cheat Sheet

-   accuracy
-   precision
-   recall
-   F1 score

## 4. PyTorch Template

Training loop skeleton.

## 5. Data Preprocessing Tricks

    normalization
    one-hot encoding
    missing value handling

------------------------------------------------------------------------

# Important Advice

During the week:

**Spend \~70% of time coding and experimenting, not reading.**

Competitions reward **speed of experimentation and debugging**, not
memorization.
