# CEOAI Competition Notes

Use this during practice like a contest cheat sheet. Goal: valid baseline first, checked output second, improvement third.

Repo evidence behind these notes: `olympiads/ceoai_syllabus.md`, Search/RL notebooks, classical ML sprint notebooks, NLP drills, MNIST/CV benchmark, competition samples, and `olympiads/reviews/error_journal.jsonl`.

## 0. Fast Contest Strategy

Every task starts with five facts: input type, target, metric, output format, restrictions. Do not tune before a valid baseline exists. Always print shapes, dtypes, class balance, missing values, and one output row before submission.

```python
import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(train.shape, test.shape)
print(train.head())
print(train.dtypes)
print(train.isna().sum().sort_values(ascending=False).head(10))
```

### Problem -> First Baseline

| Pattern | First move |
| --- | --- |
| Tabular classification | `RandomForestClassifier` or scaled `LogisticRegression` |
| Tabular regression | `RandomForestRegressor`, `Ridge`, or `HistGradientBoostingRegressor` |
| Text classification | `TfidfVectorizer + LogisticRegression` |
| Embeddings given | scale/check shape, then logistic/ridge/KNN/cosine |
| Image classification | pretrained ResNet head or small CNN |
| Clustering | scale features, PCA inspect, KMeans baseline |
| Gridworld/RL | define states/actions/step, then value/Q table |
| Submission task | copy `sample_submission` shape first |

```python
sample = pd.read_csv("sample_submission.csv")
sub = sample.copy()
print(sub.shape, sub.columns.tolist())
print(sub.head())
# Fill predictions only after format is known.
```

## 1. Reinforcement Learning and AI Search

### 1(a). A* Algorithm and Heuristics

What it is: shortest path search using `f = g + h`. Use when states are graph/grid positions and the task asks for path, cost, or expanded nodes. Contest default: priority queue, visited cost map, Manhattan heuristic for 4-neighbor grids.

```python
from heapq import heappush, heappop

def astar_grid(grid, start, goal):
    R, C = len(grid), len(grid[0])
    def h(p): return abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    pq = [(h(start), 0, start, [start])]
    best = {start: 0}
    expanded = 0
    while pq:
        _, g, p, path = heappop(pq)
        if p == goal:
            return path, g, expanded
        if g != best[p]:
            continue
        expanded += 1
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            q = (p[0] + dr, p[1] + dc)
            if 0 <= q[0] < R and 0 <= q[1] < C and grid[q[0]][q[1]] == 0:
                ng = g + 1
                if ng < best.get(q, float("inf")):
                    best[q] = ng
                    heappush(pq, (ng + h(q), ng, q, path + [q]))
    return None, float("inf"), expanded
```

Debug check: if A* gives non-shortest path, check heuristic admissibility, wall handling, and whether stale queue entries are skipped.

### 1(b). Minimax and Variations

What it is: adversarial search. Use when players alternate and one maximizes while the other minimizes. Contest default: write terminal score first, then minimax, then alpha-beta only if plain minimax works.

```python
def minimax(state, player):
    if terminal(state):
        return score_for_max_player(state)
    vals = []
    for move in legal_moves(state):
        child = apply_move(state, move, player)
        vals.append(minimax(child, other(player)))
    return max(vals) if player == "MAX" else min(vals)

def best_move(state):
    return max(legal_moves(state), key=lambda m: minimax(apply_move(state, m, "MAX"), "MIN"))
```

Debug check: terminal scores must be from one player's viewpoint. Do not flip score meaning between recursive levels.

### 1(c). Monte Carlo Method

What it is: estimate value by random rollouts or repeated sampling. Use when exact transition/value calculation is hard but simulation is easy. Contest default: many episodes, average returns, fixed random seed.

```python
import random
import numpy as np

def rollout(env, policy, gamma=0.99, max_steps=200):
    s, total, disc = env.reset(), 0.0, 1.0
    for _ in range(max_steps):
        a = policy(s)
        s, r, done, info = env.step(a)
        total += disc * r
        disc *= gamma
        if done:
            break
    return total

values = [rollout(env, policy) for _ in range(1000)]
print(np.mean(values), np.std(values))
```

Debug check: high variance is normal. Increase rollouts before changing the algorithm.

### 1(d). Markov Decision Processes

What it is: states, actions, transition rule, rewards, discount. Use when the problem has stochastic/deterministic dynamics and asks for value, policy, or optimal actions. Contest default: explicit state index and transition function.

```python
START, GOAL = (0, 0), (3, 3)
HOLES = {(1, 1), (2, 1)}
ACTIONS = [(-1,0), (0,1), (1,0), (0,-1)]

def state_id(pos): return pos[0] * 4 + pos[1]

def step(pos, a):
    dr, dc = ACTIONS[a]
    nxt = (min(3, max(0, pos[0] + dr)), min(3, max(0, pos[1] + dc)))
    done = nxt == GOAL or nxt in HOLES
    reward = 1.0 if nxt == GOAL else 0.0
    return nxt, reward, done
```

Debug check: keep positions as tuples. Do not compare a list to a NumPy array inside `if`.

### 1(e). Temporal Difference Learning

What it is: update value estimates from one-step bootstrapping. Use when you observe transitions and want values without waiting for full episode returns. Contest default: TD(0) value update.

```python
V = np.zeros(16)
alpha, gamma = 0.1, 0.95

s = START
for t in range(200):
    a = np.random.randint(4)
    ns, r, done = step(s, a)
    i, j = state_id(s), state_id(ns)
    V[i] += alpha * (r + gamma * V[j] - V[i])
    s = START if done else ns
```

Debug check: if values stay zero, check reward reachability, terminal handling, and exploration.

### 1(f). Q-learning

What it is: learn action values without a model. Use when actions matter directly and you need a policy. Contest default: `Q[state, action]`, epsilon-greedy, Bellman max update.

```python
Q = np.zeros((16, 4))
alpha, gamma, eps = 0.1, 0.95, 0.2

for ep in range(1000):
    s = START
    for _ in range(100):
        a = np.random.randint(4) if np.random.rand() < eps else Q[state_id(s)].argmax()
        ns, r, done = step(s, a)
        i, j = state_id(s), state_id(ns)
        Q[i, a] += alpha * (r + gamma * Q[j].max() - Q[i, a])
        s = ns
        if done:
            break
policy = Q.argmax(axis=1).reshape(4, 4)
print(policy)
```

Debug check: `Q.shape` should be `(n_states, n_actions)`. Index Q with `state_id(pos)`, not raw `(row, col)`.

## 2. Machine Learning

### 2(a). Naive Bayes

What it is: probabilistic baseline with strong independence assumptions. Use for text or simple categorical counts. Contest default: `MultinomialNB` after TF-IDF/counts; `GaussianNB` for dense numeric features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
Xtr = vec.fit_transform(train_text)
Xva = vec.transform(val_text)
clf = MultinomialNB()
clf.fit(Xtr, y_train)
print(accuracy_score(y_val, clf.predict(Xva)))
```

Debug check: never `fit_transform` validation/test text.

### 2(a). Linear Regression

What it is: predict a number with a linear function. Use as fast tabular regression baseline. Contest default: impute, one-hot, scale if using linear/ridge/lasso, report MAE/RMSE.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

X_train_enc = pd.get_dummies(X_train, dummy_na=True)
X_val_enc = pd.get_dummies(X_val, dummy_na=True).reindex(columns=X_train_enc.columns, fill_value=0)
model = Ridge(alpha=1.0)
model.fit(X_train_enc, y_train)
pred = model.predict(X_val_enc)
print("MAE", mean_absolute_error(y_val, pred))
print("RMSE", mean_squared_error(y_val, pred, squared=False))
```

Debug check: target must be numeric and row counts must match.

### 2(a). Logistic Regression

What it is: strong linear classifier. Use for tabular, embeddings, TF-IDF, and fast baselines. Contest default: scale dense numeric features, use class weights for imbalance.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight="balanced")
)
clf.fit(X_train, y_train)
pred = clf.predict(X_val)
print(classification_report(y_val, pred))
```

Debug check: scale X only. Do not scale or reshape y.

### 2(a). SVM

What it is: margin classifier; good on smaller, scaled datasets. Use when classes are separated by geometry or features are medium-sized. Contest default: `SVC(kernel="rbf")` after scaling.

```python
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

svm = make_pipeline(StandardScaler(), SVC(C=3, kernel="rbf", gamma="scale"))
svm.fit(X_train, y_train)
print(svm.score(X_val, y_val))
```

Debug check: SVM can be slow on large data. Try `LinearSVC` for big sparse text.

### 2(a). k-NN

What it is: predict from nearest examples. Use as geometry sanity check and small-data baseline. Contest default: scale, try odd `k`, compare against logistic/SVM.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn.fit(X_train, y_train)
print(knn.score(X_val, y_val))
```

Debug check: k-NN is distance-based; unscaled large columns dominate.

### 2(a). Decision Trees

What it is: split-based model, interpretable and handles mixed feature effects. Use for quick tabular logic and feature importance. Contest default: limit depth first; compare to random forest.

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)
print(tree.score(X_train, y_train), tree.score(X_val, y_val))
```

Debug check: train high and val low means overfit; reduce depth or use forest.

### 2(b). K-Means and K-Means++

What it is: distance-based clustering. Use when labels are missing or task asks for grouping/prototypes. Contest default: scale, optionally PCA, `init="k-means++"`, inspect cluster sizes.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Xs = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=10, init="k-means++", n_init=20, random_state=42)
cluster = km.fit_predict(Xs)
print(pd.Series(cluster).value_counts().sort_index())
xy = PCA(n_components=2, random_state=42).fit_transform(Xs)
```

Debug check: cluster IDs are arbitrary. Do not directly compare cluster `0` to class `0`.

### 2(b). DBSCAN

What it is: density clustering with noise points. Use when clusters are irregular and outliers matter. Contest default: scale, tune `eps`, inspect noise count `-1`.

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)
labels = DBSCAN(eps=0.7, min_samples=5).fit_predict(Xs)
print(pd.Series(labels).value_counts().sort_index())
print("noise fraction", np.mean(labels == -1))
```

Debug check: all `-1` means `eps` too small or scaling wrong.

### 2(b). Hierarchical Clustering

What it is: tree of merges. Use for small datasets when cluster count is unclear or dendrogram insight helps. Contest default: scale, `AgglomerativeClustering`, compare several `n_clusters`.

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

Xs = StandardScaler().fit_transform(X)
for k in [2, 3, 4, 5, 8, 10]:
    lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs)
    print(k, silhouette_score(Xs, lab))
```

Debug check: hierarchical clustering is expensive for large `n`.

### 2(c). Random Forests

What it is: many decision trees averaged/voted. Use as reliable tabular default. Contest default: start forest before fancy tuning; inspect validation metric and feature importances.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
print(rf.score(X_val, y_val))
```

Debug check: forests need numeric features; encode strings first.

### 2(c). Voting and Bagging

What it is: combine model predictions. Use only after individual baselines work. Contest default: voting for diverse classifiers; bagging for unstable high-variance models.

```python
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

vote = VotingClassifier([
    ("lr", LogisticRegression(max_iter=1000)),
    ("tree", DecisionTreeClassifier(max_depth=8, random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
], voting="soft")
vote.fit(X_train, y_train)
print(vote.score(X_val, y_val))
```

Debug check: skip voting if it costs time and one model is already strong.

### 2(c). Boosting

What it is: sequentially fix previous errors. Use for tabular tasks with nonlinear signal. Contest default: sklearn histogram gradient boosting if XGBoost is unavailable.

```python
from sklearn.ensemble import HistGradientBoostingClassifier

gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)
print(gb.score(X_val, y_val))
```

Debug check: boosting can overfit; trust validation, not train score.

### 2(d). Dimensionality Reduction

What it is: compress features while keeping useful structure. Use for visualization, denoising, clustering, or speeding models. Contest default: PCA after scaling; keep enough components for variance or use 2D for plots.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=0.95, random_state=42)
Xp = pca.fit_transform(Xs)
print(X.shape, Xp.shape, pca.explained_variance_ratio_.sum())
```

Debug check: fit PCA on train only, then transform val/test.

## 3. Deep Learning

### 3(a). Perceptron and MLP

What it is: stacked linear layers plus nonlinearities. Use for dense tabular features, toy geometry, and as a generic learnable baseline. Contest default: small MLP, Adam, correct loss contract, shape print before training.

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, n_in, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, n_classes))
    def forward(self, x): return self.net(x)

model = MLP(n_in=X_train.shape[1], n_classes=len(set(y_train)))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Debug check: `CrossEntropyLoss` expects logits `[batch, classes]` and labels `Long [batch]`.

### 3(a). Backpropagation

What it is: compute gradients through the graph and update weights. Use for all neural training. Contest default: memorize the order: zero gradients, forward, loss, backward, step.

```python
for batch_x, batch_y in loader:
    optimizer.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    loss.backward()
    optimizer.step()
```

Debug check: if gradients are missing, check `.detach()`, `torch.no_grad()`, and whether the loss uses raw logits.

### 3(b). SGD, Adam, RMSProp

What it is: optimizers decide weight updates. Use Adam as default for fast baselines; SGD for controlled CNN fine-tuning; RMSProp rarely but okay for noisy losses. Contest default: Adam `1e-3`, lower if unstable.

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
opt_sgd = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
opt_rms = torch.optim.RMSprop(model.parameters(), lr=1e-3)
# If loss jumps: try lr=3e-4 or 1e-4 before changing architecture.
```

Debug check: a loss that drops then explodes usually means learning rate too high.

### 3(b). Learning Rate Schedules

What it is: change LR during training. Use when baseline trains but plateaus. Contest default: `ReduceLROnPlateau` for validation loss or cosine for longer runs.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2, factor=0.5)
for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    scheduler.step(val_loss)
```

Debug check: scheduler steps after validation when it needs validation loss.

### 3(b). Regularization, Dropout, BatchNorm

What it is: prevent overfit and stabilize training. Use when train improves but validation stalls. Contest default: weight decay first, dropout in MLPs, BatchNorm in CNN/MLP if batches are stable.

```python
model = nn.Sequential(
    nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(128, n_classes)
)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

Debug check: call `model.train()` for training and `model.eval()` for validation.

### 3(c). CNNs

What it is: local filters for images/grids. Use for image classification, restoration, counting, and feature extraction. Contest default: pretrained ResNet if allowed; otherwise small CNN with shape print.

```python
class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.head = nn.Linear(32 * 7 * 7, n_classes)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)
```

Debug check: preserve batch dimension with `x.view(x.size(0), -1)`.

### 3(c). RNN, LSTM, GRU

What it is: sequence models. Use for time series or token sequences when transformer is unavailable. Contest default: GRU/LSTM over padded sequences; use last hidden state for classification.

```python
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb, hidden, n_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)
    def forward(self, ids):
        _, h = self.gru(self.emb(ids))
        return self.fc(h[-1])
```

Debug check: token IDs must be `Long` and within vocabulary range.

### 3(c). Transformers, Attention, BERT, GPT, ViT

What it is: attention-based pretrained model families. Use by routing task type: BERT encoder for text classification/embeddings, GPT/decoder for generation, ViT for image encoder. Contest default: tokenizer/model contract before fine-tuning.

```python
# Scaled dot-product attention core.
Q, K, V = torch.randn(2, 8, 64), torch.randn(2, 8, 64), torch.randn(2, 8, 64)
scores = Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5)
weights = scores.softmax(dim=-1)
context = weights @ V
print(context.shape)
```

```python
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = AutoModel.from_pretrained("distilbert-base-uncased")
batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
out = enc(**batch)
emb = out.last_hidden_state[:, 0]
print(batch["input_ids"].shape, emb.shape)
```

Debug check: pass `model(**tokenized)`, not the whole `BatchEncoding` as `input_ids`.

### 3(d). Autoencoders and VAEs

What it is: reconstruct inputs through a bottleneck; VAE adds probabilistic latent space. Use for denoising, anomaly detection, compression, representation. Contest default: reconstruction error baseline.

```python
class AE(nn.Module):
    def __init__(self, n_in, z=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, z))
        self.dec = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, n_in))
    def forward(self, x): return self.dec(self.enc(x))

recon = model(x)
loss = nn.MSELoss()(recon, x)
```

```python
# VAE loss shape: reconstruction term + KL penalty.
recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
loss = recon_loss + 0.001 * kl
```

Debug check: reconstruction target shape must equal model output shape.

### 3(d). GANs

What it is: generator vs discriminator. Use when task is generation or recognizing fake/real patterns. Contest default: do not train GAN from scratch unless required; use discriminator-style classifier or pretrained generator outputs.

```python
loss_D = nn.BCEWithLogitsLoss()(D(real), torch.ones_like(D(real))) + \
         nn.BCEWithLogitsLoss()(D(fake.detach()), torch.zeros_like(D(fake)))
loss_G = nn.BCEWithLogitsLoss()(D(fake), torch.ones_like(D(fake)))
```

Debug check: detach fake images when training discriminator.

### 3(d). Diffusion Models

What it is: denoise from noise step by step. Use for generation/editing recognition or pretrained pipelines. Contest default: use provided/pretrained model; inspect input/output image size and constraints.

```python
# Recognition template: score generated images with an existing classifier/encoder.
with torch.no_grad():
    logits = classifier(images.to(device))
    pred = logits.argmax(dim=1)
```

Debug check: for CEOAI, prioritize knowing when diffusion is the right family, not implementing it under time pressure.

## 4. Natural Language Processing

### 4(a). Tokenization

What it is: turn text into tokens or IDs. Use before every text model. Contest default: inspect raw strings, then use tokenizer/vectorizer fitted on train only.

```python
texts = train["text"].astype(str).tolist()
print(type(texts), type(texts[0]), texts[0][:200])
tokens = [t.lower().split() for t in texts[:3]]
print(tokens)
```

Debug check: tokenizer input should be string or list of strings, not nested accidental objects.

### 4(a). Stemming and Basic Preprocessing

What it is: normalize text. Use sparingly for classical models; BERT-style models usually want raw-ish text. Contest default: lowercase, fill missing, maybe remove obvious noise.

```python
train["text"] = train["text"].fillna("").astype(str).str.lower()
test["text"] = test["text"].fillna("").astype(str).str.lower()
```

Debug check: preprocessing train and test differently creates silent distribution shift.

### 4(b). TF-IDF

What it is: sparse word/character features. Use for fast text classification, retrieval, author/style clues. Contest default: word n-grams plus optional char n-grams, logistic regression.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000)
Xtr = vec.fit_transform(train_text)
Xva = vec.transform(val_text)
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Xtr, y_train)
print(clf.score(Xva, y_val))
```

Debug check: `Xtr.shape[1] == Xva.shape[1]` must hold.

### 4(b). Word2Vec and Dense Embeddings

What it is: dense vector per word/document. Use for similarity, clustering, retrieval, or as classifier features. Contest default: average token vectors or use provided embeddings.

```python
from sklearn.metrics.pairwise import cosine_similarity

E = np.load("embeddings.npy")      # rows = items
q = np.load("query_embedding.npy") # shape (dim,)
sims = cosine_similarity(q.reshape(1, -1), E)[0]
top = np.argsort(-sims)[:10]
print(top, sims[top])
```

Debug check: cosine similarity needs matching vector dimensions.

### 4(c). Seq2Seq and T5

What it is: encoder-decoder text-to-text model. Use for translation, summarization, answer generation, reformulation. Contest default: use pretrained pipeline/model if local weights exist; constrain output length.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
batch = tok(prompts, padding=True, truncation=True, max_length=256, return_tensors="pt")
ids = model.generate(**batch, max_new_tokens=32)
print(tok.batch_decode(ids, skip_special_tokens=True))
```

Debug check: generation output must still be converted to required contest format.

### 4(c). LLaMA and Decoder LMs

What it is: autoregressive text generation. Use for prompting, classification-by-generation, clue generation, or reranking if model is allowed. Contest default: small/local model, fixed prompt, parse output defensively.

```python
# Pattern only: after generation, validate labels against allowed set.
allowed = {"yes", "no"}
raw = generated_text.strip().lower().split()[0]
pred = raw if raw in allowed else "no"
```

Debug check: never trust free-form text as a valid label without parsing and fallback.

## 5. Computer Vision

### 5(a). Filtering

What it is: local image operations such as blur, sharpen, threshold. Use for denoising, preprocessing, masks, counting. Contest default: visualize before/after and keep arrays in known range.

```python
import cv2

img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5, 5), 0)
_, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(img.shape, img.dtype, mask.min(), mask.max())
```

Debug check: OpenCV loads BGR, not RGB.

### 5(a). Edge Detection

What it is: find intensity boundaries. Use for segmentation, counting, shape extraction, sanity features. Contest default: Canny or Sobel after grayscale/blur.

```python
gray = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(gray, threshold1=50, threshold2=150)
print(edges.shape, np.unique(edges)[:5])
```

Debug check: threshold sensitivity is high; inspect image, not only metric.

### 5(a). HOG

What it is: histogram of gradient directions. Use as classical image feature for small image classification. Contest default: HOG features plus logistic/SVM.

```python
from skimage.feature import hog

feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
print(feat.shape)
```

Debug check: all images must be resized to same shape before feature extraction.

### 5(b). AlexNet, VGG, ResNet, Inception, EfficientNet

What it is: CNN families. Use as pretrained image encoders/classifiers. Contest default: ResNet18/EfficientNet head replacement, freeze backbone first, train head, then optionally unfreeze.

```python
import torchvision.models as models
from torch import nn

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, n_classes)
opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
```

```python
# Other torchvision families use different classifier attributes.
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg.classifier[-1] = nn.Linear(vgg.classifier[-1].in_features, n_classes)
eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
eff.classifier[-1] = nn.Linear(eff.classifier[-1].in_features, n_classes)
```

Debug check: match input channels and normalization expected by pretrained weights.

### 5(c). YOLO

What it is: object detection model. Use when task asks for boxes, counts, or object locations. Contest default: use provided detector/pretrained model if allowed; otherwise convert detection to counting/features.

```python
# Generic output sanity for any detector.
boxes = np.asarray(boxes)  # [n, 4]
scores = np.asarray(scores)
keep = scores >= 0.25
print(boxes[keep].shape, scores[keep].min() if keep.any() else "none")
```

Debug check: know expected box format: `xyxy`, `xywh`, normalized, or pixels.

### 5(c). Stable Diffusion

What it is: text/image generation by diffusion. Use only if task is generation/editing or fake-image recognition. Contest default: prefer classifier/encoder around provided/generated images; do not build diffusion from scratch.

```python
# Competition recognition pattern: compare image embeddings.
sim = cosine_similarity(query_img_emb.reshape(1, -1), candidate_img_embs)[0]
print(np.argsort(-sim)[:5])
```

Debug check: generation tasks still have exact output constraints.

### 5(c). Vision Transformers

What it is: transformer encoder over image patches. Use as pretrained image classifier/embedding model. Contest default: route like BERT for images: processor + model + classification head/embeddings.

```python
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit = AutoModel.from_pretrained("google/vit-base-patch16-224")
inputs = processor(images=pil_images, return_tensors="pt")
out = vit(**inputs)
emb = out.last_hidden_state[:, 0]
print(emb.shape)
```

Debug check: image size and normalization are handled by the processor; inspect tensors anyway.

## 6. Submission and Output Checks

Use this before every final file. Many contests punish format bugs harder than weak models.

```python
sub = pd.DataFrame({"id": test["id"], "prediction": pred})
assert len(sub) == len(test)
assert sub.isna().sum().sum() == 0
assert sub.columns.tolist() == ["id", "prediction"]
print(sub.head())
sub.to_csv("submission.csv", index=False)
```

For JSONL:

```python
import json
with open("answers.jsonl", "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

## 7. Errors I Already Made

These are grouped from `olympiads/reviews/error_journal.jsonl`. Treat them as contest reflexes.

### PyTorch dtype mismatch: NumPy Double vs model Float

Symptom: `mat1 and mat2 must have the same dtype, but got Double and Float`. Cause: NumPy float64 became torch Double while model weights are Float. Fastest probe: print input, target, and parameter dtypes. Fix reflex: convert tensors with explicit dtype.

```python
X_t = torch.tensor(X_np, dtype=torch.float32)
y_t = torch.tensor(y_np, dtype=torch.float32)
print(X_t.dtype, y_t.dtype, next(model.parameters()).dtype)
```

### PyTorch shape mismatch in `nn.Linear`

Symptom: `mat1 and mat2 shapes cannot be multiplied`. Cause: batch shaped `[batch]` or `[1, batch]` instead of `[batch, features]`. Fastest probe: print shape before model call. Fix reflex: last dimension equals `in_features`.

```python
batch_x = batch_x.reshape(-1, 1)  # scalar feature
out = model(batch_x)
print(batch_x.shape, out.shape)
```

### Regression loss broadcasting

Symptom: MSELoss warning about target shape `[32]` vs input `[32, 1]`. Cause: output and target broadcast silently. Fastest probe: print output and target shapes before loss. Fix reflex: make them intentionally identical.

```python
batch_y = batch_y.reshape(-1, 1).float()
loss = nn.MSELoss()(out, batch_y)
print(out.shape, batch_y.shape)
```

### Adam learning rate too high

Symptom: loss drops then jumps/plateaus. Cause: LR instability, seen with Adam `lr=0.1`. Fastest probe: compare loss curve. Fix reflex: start Adam around `1e-3`, log average epoch loss.

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch_loss = total_loss / len(loader)
```

### Device mismatch

Symptom: CPU tensor with CUDA model. Cause: `.to(device)` called without assignment. Fastest probe: print devices. Fix reflex: reassign every tensor move.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
batch_x = batch_x.to(device)
batch_y = batch_y.to(device)
print(batch_x.device, batch_y.device, next(model.parameters()).device)
```

### BCE metric uses logits as labels

Symptom: accuracy stuck at `0.0` while loss decreases. Cause: comparing raw logits to `0/1` labels. Fastest probe: print logits and sigmoid. Fix reflex: threshold logits at `0` or sigmoid at `0.5`.

```python
logits = model(X)
pred = (logits.squeeze(1) >= 0).long()
acc = (pred.cpu() == y_true.cpu().long()).float().mean()
```

### Decision boundary plot uses flat predictions

Symptom: broken plot or `contourf` shape error. Cause: mesh predictions are flat. Fastest probe: print `xx.shape`, `yy.shape`, `z.shape`. Fix reflex: reshape predictions to mesh.

```python
grid = np.c_[xx.ravel(), yy.ravel()]
z = torch.sigmoid(model(torch.tensor(grid, dtype=torch.float32))).detach().numpy()
z = z.reshape(xx.shape)
plt.contourf(xx, yy, z, levels=20)
```

### Wrong PyTorch tensor constructor

Symptom: `torch.Tensor(..., dtype=...)` TypeError or dtype typo. Cause: wrong constructor API. Fastest probe: inspect construction line. Fix reflex: use `torch.tensor(data, dtype=...)`.

```python
x = torch.tensor(x_np, dtype=torch.float32)
```

### TF-IDF vocabulary mismatch

Symptom: incompatible dimensions in cosine/model prediction. Cause: vectorizer fitted separately on train and test/query. Fastest probe: print second dimension of all matrices. Fix reflex: fit train once, transform everything else.

```python
vec = TfidfVectorizer()
Xtr = vec.fit_transform(train_text)
Xte = vec.transform(test_text)
q = vec.transform([query])
print(Xtr.shape, Xte.shape, q.shape)
```

### Misclassified-example index alignment

Symptom: displayed text, true label, and prediction do not match. Cause: filtered X but unfiltered y/pred. Fastest probe: build one dataframe before filtering. Fix reflex: filter all columns together.

```python
err = pd.DataFrame({"text": X_test, "true": y_test, "pred": y_pred})
err = err[err["true"] != err["pred"]]
print(err.head())
```

### Hugging Face tokenizer input/length/model call

Symptom: tokenizer rejects input, sequence too long, or model gets `BatchEncoding` wrong. Cause: bad input type, no truncation, or passing tokenized dict incorrectly. Fastest probe: print types, keys, shapes. Fix reflex: list of strings, truncation, `model(**batch)`.

```python
texts = [str(x) for x in texts]
batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
print(batch.keys(), batch["input_ids"].shape)
out = model(**batch)
```

### Submission dataframe built with wrong API

Symptom: pandas axis/name error while creating predictions. Cause: using arithmetic-style `.add` instead of rows/columns. Fastest probe: build a tiny 3-row submission first. Fix reflex: construct dataframe from complete columns.

```python
sub = pd.DataFrame({"id": ids, "prediction": preds})
assert len(sub) == len(ids)
print(sub.head())
```

### Gridworld coordinate representation

Symptom: `unhashable type: list` or ambiguous NumPy truth value. Cause: mixing lists, tuples, and arrays for positions. Fastest probe: print `type(pos)` and one equality check. Fix reflex: use tuples for states.

```python
target_pos = tuple(target_pos)
if target_pos in HOLES:
    done = True
if target_pos == GOAL:
    reward = 1.0
```

### Torchvision dataset attribute assumption

Symptom: dataset has no `.labels`. Cause: dataset API/version differs. Fastest probe: inspect one sample and `dir(dataset)`. Fix reflex: use `dataset[0]` contract first.

```python
img, label = train_dataset[0]
print(type(img), getattr(img, "shape", None), label)
```

### CrossEntropyLoss target dtype and logits

Symptom: CUDA NLL loss not implemented for Float, or target/probability error. Cause: labels cast to float or `argmax` applied before loss. Fastest probe: print logits shape and label dtype. Fix reflex: raw logits + long class labels.

```python
logits = model(batch_x)
batch_y = batch_y.long()
loss = nn.CrossEntropyLoss()(logits, batch_y)
pred = logits.argmax(dim=1)
```

### Model called on DataLoader

Symptom: `conv2d()` got `DataLoader`. Cause: `model(test_dataloader)` instead of iterating batches. Fastest probe: print `type(test_dataloader)` and one batch type. Fix reflex: loop over loader.

```python
model.eval()
preds = []
with torch.no_grad():
    for batch_x, batch_y in test_dataloader:
        logits = model(batch_x.to(device))
        preds.append(logits.argmax(dim=1).cpu())
preds = torch.cat(preds)
```

### Sklearn scales or reshapes labels

Symptom: expected 2D array or fit sees one giant sample. Cause: scaling `y_train` or reshaping X to `(1, -1)`. Fastest probe: print fit shapes. Fix reflex: X is `[n_samples, n_features]`, y is `[n_samples]`.

```python
Xtr = scaler.fit_transform(X_train)
Xva = scaler.transform(X_val)
print(Xtr.shape, y_train.shape)
model.fit(Xtr, y_train)
```

### KMeans evaluated like supervised classifier

Symptom: KMeans accuracy near zero. Cause: cluster IDs are arbitrary. Fastest probe: inspect cluster counts and majority labels. Fix reflex: separate clustering evaluation.

```python
cluster = km.fit_predict(Xs)
tab = pd.crosstab(cluster, y_true)
print(tab)
```

### Ragged multi-label text split

Symptom: NumPy inhomogeneous shape error after `np.array([s.split(",") ...])`. Cause: rows have variable number of labels. Fastest probe: split first few rows. Fix reflex: flatten to list/set.

```python
tags = set()
for s in train["Genres"].fillna(""):
    tags.update(t.strip() for t in s.split(",") if t.strip())
print(len(tags), sorted(tags)[:10])
```

### Sklearn notebook HTML display crash

Symptom: `UnicodeDecodeError` inside sklearn `_repr_html`. Cause: Jupyter rich estimator display, not model fit. Fastest probe: traceback points to estimator HTML. Fix reflex: suppress display with semicolon or print metric.

```python
model.fit(X_train, y_train);
print(type(model).__name__, model.score(X_val, y_val))
```

### BCEWithLogitsLoss shape and dtype

Symptom: target size mismatch or float cast to Long error. Cause: binary labels not float or output/target shapes differ. Fastest probe: print prediction/target shape and dtype immediately before loss. Fix reflex: one logit per sample, float target.

```python
logits = model(batch_x).view(-1)
target = batch_y.float().view(-1)
loss = nn.BCEWithLogitsLoss()(logits, target)
print(logits.shape, target.shape, target.dtype)
```

## 8. 60-Second Debug Checklist

Run this before changing the model.

```python
print("X", getattr(X_train, "shape", None), "y", getattr(y_train, "shape", None))
print("missing", getattr(X_train, "isna", lambda: pd.DataFrame())().sum().sum() if hasattr(X_train, "isna") else "na")
print("target counts", pd.Series(y_train).value_counts().head())
```

For PyTorch:

```python
batch_x, batch_y = next(iter(loader))
print(batch_x.shape, batch_x.dtype, batch_x.device)
print(batch_y.shape, batch_y.dtype, batch_y.device)
print(next(model.parameters()).device, next(model.parameters()).dtype)
with torch.no_grad():
    out = model(batch_x.to(next(model.parameters()).device))
print("out", out.shape, out.dtype)
```

For sklearn/text:

```python
print(X_train.shape, y_train.shape)
print(type(X_train), type(y_train))
pred = model.predict(X_val[:5])
print(pred[:5])
```

Final rule: if output format is unchecked, the task is not solved.
