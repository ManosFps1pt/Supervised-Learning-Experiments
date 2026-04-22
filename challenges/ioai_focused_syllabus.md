# IOAI Competition Focused Syllabus

Based on your requirements, here is the streamlined syllabus covering exactly what you need to know: basic `sklearn` models (trees, regression), `xgboost`, neural networks, optimizers, and embeddings.

---

## 1. Basic Scikit-Learn Models

### Regression

- **Linear Regression**: Predicts a continuous value by fitting a straight line (or hyperplane) through the data points that minimizes the mean squared error. Fast but assumes a linear relationship.
- **Logistic Regression**: Despite the name, it's used for **classification** (usually binary). It works by mapping the output of a linear equation to a probability between 0 and 1 using the Sigmoid function.

### Decision Trees

- **Concept**: A flowchart-like model that splits data based on feature values. It asks yes/no questions (e.g., "Is age > 30?") to partition the data until it reaches a prediction leaf.
- **Pros/Cons**: Very interpretable and requires little data preprocessing. Prone to overfitting on training data (building deep, complex trees).
- **Random Forest**: An ensemble method that builds multiple decision trees on random subsets of data and features, then averages their predictions. This significantly reduces the overfitting of single trees.

---

## 2. XGBoost (Extreme Gradient Boosting)

- **Concept**: Another ensemble method built on decision trees. Unlike Random Forest (which builds trees independently), XGBoost builds trees **sequentially**. Each new tree focuses on correcting the errors (residuals) made by the previous trees.
- **Why it wins competitions**: It's highly optimized, handles missing data gracefully, and provides state-of-the-art results on tabular data.
- **Key Hyperparameters**:
  - `n_estimators`: Number of trees.
  - `learning_rate` (alpha): How much each new tree contributes (lower requires more estimators).
  - `max_depth`: Maximum depth of a tree (controls overfitting).

---

## 3. Neural Networks

- **Architecture**: Composed of layers of artificial neurons.
  - **Input Layer**: Takes the features.
  - **Hidden Layers**: Perform computations using weights, biases, and activation functions (like ReLU) to capture non-linear relationships.
  - **Output Layer**: Produces the final prediction (e.g., a single value for regression, or probabilities using Softmax for classification).
- **Forward Pass**: The process of passing data through the network to get a prediction.
- **Backward Pass (Backpropagation)**: Calculating the loss (error) and passing it backwards through the network to compute the gradient of the loss with respect to each weight.

---

## 4. Optimizers

Optimizers are the algorithms that actually update the neural network's weights to minimize the loss.

- **Gradient Descent**: The fundamental idea of taking steps in the opposite direction of the gradient (slope) of the loss function.
- **Stochastic Gradient Descent (SGD)**: Instead of calculating the gradient on the entire dataset, it calculates it on a small batch of data. Much faster but "noisier" steps.
- **Adam (Adaptive Moment Estimation)**: The most popular modern optimizer. It adapts the learning rate for each weight individually and keeps an exponentially decaying average of past gradients (momentum). Generally converges faster and more reliably than standard SGD.
- **Learning Rate**: The most important hyperparameter. If too high, the model "bounces around" and never settles on the minimum. If too low, training takes forever or gets stuck.

---

## 5. Embeddings

Embeddings are a way to represent categorical data (especially text) as dense mathematical vectors (arrays of numbers) so that neural networks can process them.

- **The Problem**: Neural networks only understand numbers, not words like "apple". One-hot encoding creates massive, sparse, and inefficient vectors where every word is orthogonal (no relationship).
- **The Solution (Embeddings)**: Words are mapped to a dense vector (e.g., 300 dimensions). Words with similar meanings end up closer to each other in this mathematical space (e.g., the vector for "king" is closer to "queen" than to "car").
- **How they are used**:
  - They are often the very first layer of a Neural Network for text (`nn.Embedding` in PyTorch).
  - They are learned just like any other weights during training.
  - Pre-trained embeddings (like Word2Vec, GloVe, or those from models like BERT) can be used to give a network an initial understanding of language.

---

*Focus on understanding these concepts intuitively and knowing how to import/call the standard PyTorch or Scikit-Learn functions for them.*
