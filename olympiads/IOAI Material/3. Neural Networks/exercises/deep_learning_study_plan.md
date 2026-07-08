# Deep Learning Study Plan

Target notebook: `solution3.ipynb`

Target duration: about 5 hours.

Main goal: build one reusable Deep Learning notebook that proves you can train,
compare, debug, and choose neural-network methods under contest pressure.

This plan is not broad reading. Every section should leave visible notebook
evidence: executed cells, metrics, tables, plots, predictions, or short
markdown conclusions.

## What This Lesson Must Teach

By the end, the notebook should prove three things:

1. You can train and debug a PyTorch MLP without guessing.
2. You understand optimization vs regularization vs architecture choices.
3. You can recognize which deep-learning architecture fits which problem.

Core distinction to keep repeating:

- Architecture: what pattern family the model can represent.
- Loss: what objective the model is trying to minimize.
- Optimizer: how the weights are updated.
- Scheduler: how the learning rate changes during training.
- Regularization: how you reduce memorization and improve validation behavior.
- Metric: how the trained model is judged outside the loss function.

## 0. Setup And Baseline

Time: 30 minutes.

Notebook section: `0. Baseline MLP`

Use the existing two-moons or small classification setup in `solution3.ipynb`.

Create:

- one clean MLP baseline
- train/validation loss curve
- train/validation accuracy
- one final markdown note: underfit, fit, or overfit?

Concept focus:

- MLP is the architecture.
- Loss is the training objective.
- Optimizer is the parameter update rule.
- Epoch and batch size are training procedure choices.
- Metric is the external judgment.

Do not move on until the baseline runs top-to-bottom.

Expected notebook evidence:

```text
baseline model definition
training loop output
train/validation metrics
loss curve or compact metric table
one markdown diagnosis
```

## 1. Optimization

Time: 60 minutes.

Notebook section: `1. Optimizers And Learning Rates`

Run the same model and data with:

- SGD
- SGD + momentum
- Adam
- AdamW
- one learning-rate scheduler, preferably `StepLR` or `ReduceLROnPlateau`

Comparison table:

```text
optimizer | lr | scheduler | final_train_loss | final_val_loss | val_acc | observed_behavior
```

Understanding target:

- SGD is simple but sensitive to learning rate.
- Momentum smooths updates across steps.
- Adam uses adaptive per-parameter learning rates.
- AdamW is Adam with cleaner weight decay behavior.
- A scheduler changes the learning rate during training, not the model.

Key distinction:

- Optimizer changes how weights move.
- Scheduler changes how aggressively the optimizer moves.
- Neither changes the network architecture.

Minimum conclusion cell:

```text
Best optimizer in this experiment:
Most unstable setting:
What learning-rate behavior I observed:
What I would try first in a contest:
```

## 2. Regularization

Time: 60 minutes.

Notebook section: `2. Regularization`

Use the best or simplest optimizer from section 1, then compare:

- no regularization
- dropout
- weight decay
- early stopping
- optionally batch normalization

Comparison table:

```text
method | train_acc | val_acc | train_val_gap | visual_behavior | verdict
```

Understanding target:

- Dropout randomly hides activations during training.
- Weight decay penalizes large weights.
- Early stopping stops before memorization dominates.
- Batch normalization stabilizes activation distributions and often speeds training.

Key distinction:

- Regularization fights overfitting.
- Optimization fights bad or slow training.
- Architecture controls what patterns the model can represent.

Minimum conclusion cell:

```text
Which setting overfit most:
Which setting generalized best:
What I would try first if validation accuracy is worse than training accuracy:
```

## 3. Architecture Recognition

Time: 75 minutes.

Notebook section: `3. Architecture Routing Table`

Create this table inside the notebook. If useful, also save it as
`dl_architecture_recognition_table.md` in this folder.

```text
architecture | best input type | output type | core idea | when to use | failure mode
MLP | 2 Moons - like | logits for classification - regression | multiple linear layers connected by activation functions | when the features and independent from each other | they treat the input as a single dimentional vector (not ideal for images)
CNN | images | embeddings | they connect with a MLP and classify images | image classification | They don't remember
RNN  | sequential data | classification | it is something like a recursive function that remembers as it executes | with sequential data like telemetry speech to text e.t.c.| gradients vanish on large documents->network forgets, math collapse 
LSTM | sequential data | classification | updataed RNN that remembers better | When I could use an RNN | sequential training,large computation power
GRU|sequential data|classification|it is the evolution of LSTM|text classification|training still sequential
Transformer|sequential data|classification|it looks at all the words of a sentence at the same time and it uses self attention matricies to turn any token into an embedding|on real NLP applications|expensive training
BERT|text|embeddings|an encoder model that gets the meaning of sentences|classification|is not generative
GPT|text|text|decoder-only network that predicts the next word using only the words it had generated|chatbot|expensive
ViT
Autoencoder
VAE
GAN|prompts|images|minimax-like algorithm. A model generates images and another one compares them with real ones. They involve together in training|high quality image generation|very hard to train beacuse the generator could only generate a single type on images and the detector can get perfect, so it doesn't give good feedback to the generator 
Diffusion
```

Understanding target:

- MLP: fixed-size feature vectors; tabular baselines or flattened simple inputs.
- CNN: local spatial patterns; images and grid-like data.
- RNN: sequence state; older sequence baseline.
- LSTM: sequence model with better long-range memory than vanilla RNN.
- GRU: simpler LSTM-style sequence model.
- Transformer: attention over tokens, patches, or sequence elements.
- BERT: encoder model for representations and classification.
- GPT: decoder model for generation.
- ViT: image patches treated like tokens.
- Autoencoder: compress and reconstruct.
- VAE: probabilistic latent space and generation.
- GAN: generator vs discriminator.
- Diffusion: generate by iterative denoising.

Contest target:

Do not try to deeply implement every architecture today. The useful skill is
fast routing: given a task, choose a reasonable model family and know the first
failure mode to check.

## 4. One Compact Implementation Drill

Time: 60 minutes.

Notebook section: `4. Architecture Mini Drill`

Pick one implementation drill, not all of them.

Recommended drill: small CNN on MNIST or Fashion-MNIST.

Reason: you already have MLP experience, and CNNs are essential for both Deep
Learning and Computer Vision.

Task:

- load the dataset
- inspect one batch shape
- train a tiny CNN
- compare against an MLP baseline if quick
- write one markdown cell explaining why CNN fits images better than MLP

Expected notebook evidence:

```text
input batch shape
model output shape
loss curve or epoch metrics
final validation accuracy
one prediction example
```

Minimum conclusion cell:

```text
Why CNN fits this data:
What shape the model receives:
What shape the model outputs:
What I would debug first if loss does not decrease:
```

## 5. Final Synthesis

Time: 30 minutes.

Notebook section: `5. Contest Reflexes`

Write short answers:

```text
When my model underfits, I try:
When my model overfits, I try:
When training is unstable, I inspect:
When validation is worse than training, I suspect:
When I see image data, first model choice:
When I see text sequence data, first model choice:
When I need generation, possible model families:
```

This section is what makes the notebook reusable under pressure.

## Pass Condition

The lesson counts only if `solution3.ipynb` has executed evidence for:

- baseline MLP results
- optimizer comparison
- regularization comparison
- architecture routing table
- one CNN or architecture mini-drill
- final contest-reflex markdown

Fail conditions:

- broad reading without saved evidence
- another generic prompt-only exercise
- only polishing previous MLP code
- implementing standard internals from scratch instead of learning correct use
- architecture notes without problem-routing examples

## Suggested Notebook Order

Use this top-to-bottom structure:

```text
0. Baseline MLP
1. Optimizers And Learning Rates
2. Regularization
3. Architecture Routing Table
4. Architecture Mini Drill
5. Contest Reflexes
```

## Efficient Debugging Checks

Before changing the model, inspect:

```text
type(x_batch), x_batch.shape, x_batch.dtype
type(y_batch), y_batch.shape, y_batch.dtype
model(x_batch).shape
loss.item()
first few predictions
train/validation split sizes
device of model parameters and tensors
```

Common PyTorch reflexes:

- Classification with class indices usually wants `CrossEntropyLoss`.
- Binary classification often wants `BCEWithLogitsLoss`.
- `CrossEntropyLoss` expects raw logits, not softmax probabilities.
- `BCEWithLogitsLoss` expects raw logits, not sigmoid probabilities.
- Shape mismatch beats model cleverness: inspect shapes first.
- If train accuracy rises and validation accuracy falls, suspect overfitting.
- If both train and validation are bad, suspect underfitting, bad learning rate,
  bad labels, bad preprocessing, or a broken loop.

