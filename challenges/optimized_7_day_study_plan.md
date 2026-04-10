# Optimized IOAI 7-Day Intensive Study Plan (Fast-Track)

This optimized plan is designed for a fast-learning 16-year-old with 1 week to prepare. It compresses the basics and leaves crucial time for the **Neural Networks, Advanced Computer Vision, NLP, and Audio** sections heavily emphasized in the official IOAI Syllabus.

**Goal:** Build a personal AI competition manual consisting of rapid deployment templates and cheat sheets.
**Rule of Thumb:** Spend ~80% of your time coding and debugging, 20% skimming theory. Do NOT memorize—build a repository of reusable training loops and pipelines.

---

## Day 1 — Foundation Compression: ML Pipeline & Classical Algorithms
*Combine old Days 1 & 2 to rapidly cover tabular data basics.*

- **Dataset:** Iris & Wine Datasets
- **Concepts:** pandas basics, train/test split, handling missing data, One-Hot Encoding, normalization. 
- **Algorithms:** Logistic Regression, Decision Trees, Random Forests, XGBoost/LightGBM. 
- **Action:** Build a single script that loads data, preprocesses it, trains a Random Forest, and outputs Accuracy/F1-Score.
- **Notes to write:** Scikit-learn tabular pipeline template, metrics cheat sheet.

## Day 2 — Unsupervised Learning & Deep Learning Basics
*Deep Learning is paramount. Start PyTorch early.*

- **Dataset:** MNIST
- **Concepts:** 
  - *Unsupervised:* K-Means, PCA, embeddings.
  - *PyTorch Fundamentals:* Tensors, autograd, Datasets & DataLoaders.
- **Action:** 
  - 1. Reduce MNIST dimensionality with PCA and plot it. 
  - 2. Write a clean, reusable PyTorch training loop for a basic Multi-Layer Perceptron (MLP) using Adam optimizer and CrossEntropyLoss.
- **Notes to write:** Reusable PyTorch training & validation loop workflow.

## Day 3 — Computer Vision & CNNs
- **Dataset:** CIFAR-10
- **Concepts:** Convolutional layers, Max Pooling, image augmentations (flipping, cropping), CNN architecture.
- **Action:** Train a small CNN from scratch on CIFAR-10 using `torchvision`. Experiment with data augmentation to prevent overfitting.
- **Notes to write:** Image preprocessing pipeline, CNN PyTorch class template.

## Day 4 — Advanced Computer Vision (Transfer Learning)
*Critical for IOAI Syllabus Sections 2 & 3.*

- **Dataset:** Custom small image dataset (e.g., ants vs. bees) or CIFAR-100
- **Concepts:** Pre-trained Vision Encoders (ResNet), Model Finetuning, Object Detection intuitions (YOLO), Segmentation intuitions (U-Net).
- **Action:** Load a pre-trained ResNet model from `torchvision.models`, replace the final fully connected layer, and fine-tune it on a new dataset.
- **Notes to write:** Transfer learning code template, when to freeze vs. unfreeze layers.

## Day 5 — NLP & Transformers (Crucial Syllabus Addition)
*NLP is heavily tested in IOAI, missing from the original plan.*

- **Library:** Hugging Face `transformers`
- **Concepts:** Tokenization, Embeddings, Attention Mechanism, Pre-trained Text Encoders (BERT). 
- **Action:** Use a pre-trained Hugging Face BERT model to classify text (e.g., IMDB sentiment analysis).
- **Notes to write:** Hugging Face loading and fine-tuning pipeline.

## Day 6 — Generative AI & Audio Basics
- **Concepts:** 
  - *Generative Models:* Autoencoders, understanding diffusion models and GANs (Theory).
  - *Vision-Language & Audio:* CLIP embeddings, Whisper for audio.
- **Action:** 
  - 1. Build a simple Autoencoder in PyTorch. 
  - 2. Use a Hugging Face API or pre-trained model to run basic Audio transcription (Whisper) or zero-shot image classification (CLIP).
- **Notes to write:** API/Pre-trained model loading snippet for Whisper/CLIP.

## Day 7 — Competition Strategy & Speed Run
- **Dataset:** Fashion-MNIST or a Kaggle Playground tabular dataset.
- **Concepts:** Hyperparameter tuning, generating adversarial examples, rapid iterations.
- **Action:** Do a complete timed "mock competition." Give yourself 4 hours to build a baseline, tune it, submit, and analyze mistakes.
- **Notes to write:** Debugging checklist, hyperparameter tuning tricks, final Model Selection cheat sheet.
