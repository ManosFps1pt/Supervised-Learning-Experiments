# CEOAI Final API Survival Session

Use this on the final morning before CEOAI. Do not solve a new full task unless you are unusually calm and this checklist is already done.

## Verdict

Your remaining bottleneck is not "knowing the idea." It is converting an unfamiliar library object into a working contract quickly.

The final session should train API reconnaissance, not another competition solution.

## Timebox

Total target: 2.5 to 3 hours.

1. 30 minutes: Python introspection reflexes.
2. 45 minutes: sklearn and pandas API contracts.
3. 45 minutes: PyTorch tensor/model/loss contracts.
4. 45 minutes: transformers or torchvision companion-object contracts.
5. 15 minutes: final contest routine and stop rules.

Stop early if fatigue starts to reduce clarity. Sleep beats one more half-absorbed API.

## The Universal API Probe

Use this whenever an object is unfamiliar.

```python
import inspect

def print_dir(obj, name="obj", max_items=80):
    print("TYPE:", type(obj))
    attrs = [a for a in dir(obj) if not a.startswith("_")]
    print(f"{name} public attrs ({len(attrs)}):")
    print(attrs[:max_items])

def probe_callable(fn):
    print("CALLABLE:", fn)
    try:
        print("SIGNATURE:", inspect.signature(fn))
    except Exception as e:
        print("NO SIGNATURE:", type(e).__name__, e)
    doc = getattr(fn, "__doc__", None)
    if doc:
        print(doc[:1200])
```

Contest reflex:

```python
print_dir(obj, "obj")
probe_callable(obj.fit)        # sklearn-style
probe_callable(model.forward)  # torch/transformers-style
```

Do not guess kwargs if `inspect.signature`, `help`, or the object docstring can show them.

## Documentation Reflex

Use docs like an API dictionary, not like a textbook.

Search only for:

- constructor parameters
- method signature
- expected input shape and dtype
- returned object fields
- one minimal example

Ignore tutorials, long theory pages, and tuning advice until the baseline runs.

Useful doc-reading commands inside a notebook:

```python
help(SomeClass)
help(obj.method)
print(obj.__doc__[:1200])
print(type(obj))
print(obj.__dict__.keys())
```

## Pandas And NumPy Drill

Run these from memory on any dataframe.

```python
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
print(df.head())
print(df.isna().sum().sort_values(ascending=False).head(20))
print(df.describe(include="all").T.head(20))
```

Feature contract:

```python
X_train = pd.get_dummies(train_features, dummy_na=True)
X_test = pd.get_dummies(test_features, dummy_na=True).reindex(columns=X_train.columns, fill_value=0)
assert list(X_train.columns) == list(X_test.columns)
print(X_train.shape, X_test.shape)
```

Array contract:

```python
print(type(x), getattr(x, "shape", None), getattr(x, "dtype", None))
print(np.nanmin(x), np.nanmax(x))
```

## Sklearn Drill

Know this pattern, not every model parameter.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight="balanced")
)
clf.fit(X_train, y_train)
pred = clf.predict(X_val)
print(classification_report(y_val, pred))
```

Probe any sklearn object:

```python
print_dir(model, "model")
probe_callable(model.fit)
probe_callable(model.predict)
print(model.get_params().keys())
```

High-value sklearn kwargs:

- `random_state=42`
- `n_jobs=-1`
- `class_weight="balanced"`
- `max_iter=1000` or `2000`
- `n_estimators=200` or `300`
- `min_samples_leaf=2`
- `n_init=20` for `KMeans`

## PyTorch Drill

Before any training loop:

```python
batch_x, batch_y = next(iter(loader))
print(batch_x.shape, batch_x.dtype, batch_x.device)
print(batch_y.shape, batch_y.dtype, batch_y.device)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
batch_x = batch_x.to(device)
batch_y = batch_y.to(device)

out = model(batch_x)
print("out:", out.shape, out.dtype, out.device)
print("param:", next(model.parameters()).shape, next(model.parameters()).dtype, next(model.parameters()).device)
```

Loss contracts:

```python
# Multiclass classification
loss = torch.nn.CrossEntropyLoss()(logits, labels.long())
# logits: [batch, classes], labels: [batch]

# Binary classification
loss = torch.nn.BCEWithLogitsLoss()(logits.view(-1), labels.float().view(-1))
# logits: [batch] or [batch, 1], labels: [batch]

# Regression
loss = torch.nn.MSELoss()(pred.float(), target.float())
# pred and target intentionally same shape
```

Training order:

```python
optimizer.zero_grad()
out = model(batch_x)
loss = criterion(out, batch_y)
loss.backward()
optimizer.step()
```

Never change architecture before this one-batch path works.

## Transformers And Torchvision Drill

Companion object first: tokenizer, processor, image processor, or feature extractor.

```python
print_dir(tokenizer, "tokenizer")
probe_callable(tokenizer.__call__)

batch = tokenizer(texts[:2], padding=True, truncation=True, max_length=256, return_tensors="pt")
print(batch.keys())
for k, v in batch.items():
    print(k, v.shape, v.dtype)

out = model(**batch)
print_dir(out, "outputs")
print(getattr(out, "keys", lambda: [])())
for k in getattr(out, "keys", lambda: [])():
    v = getattr(out, k)
    print(k, getattr(v, "shape", None), getattr(v, "dtype", None))
```

For image processors:

```python
inputs = processor(images=pil_images[:2], return_tensors="pt")
for k, v in inputs.items():
    print(k, v.shape, v.dtype)
out = model(**inputs)
```

Common output choices:

- classifier: `out.logits`
- BERT-style embeddings: `out.last_hidden_state`
- CLS embedding: `out.last_hidden_state[:, 0]`
- mean embedding: mask-aware mean over tokens
- ViT embedding: `out.last_hidden_state[:, 0]`

## Final Submission Reflex

After writing a file, reload it from disk.

```python
sub.to_csv("submission.csv", index=False)
check = pd.read_csv("submission.csv")
print(check.shape)
print(check.columns.tolist())
print(check.head())
print(check.isna().sum())
assert len(check) == expected_rows
```

For zip/npz/pkl/jsonl, inspect members, shapes, line count, or callable fields. A notebook that ran but did not reload the final artifact is not done.

## Last-Morning Rule

If you feel the urge to start a new hard task, do this instead:

1. Open one old solved notebook.
2. Pick one unfamiliar object from it.
3. Run `print_dir`, `probe_callable`, one input example, one output print.
4. Close it after the contract is clear.

The final win condition is not another score. It is walking into CEOAI with a repeatable way to discover APIs under pressure.
