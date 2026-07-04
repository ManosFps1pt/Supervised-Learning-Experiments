# CEOAI ML Guided One-Notebook Sprint

Goal: transfer the useful mental model for classical ML, then build one notebook
that proves you can use the ideas without wasting time on repeated setup.

Structure:

1. Theory transfer: what each concept should feel like in your head.
2. One-notebook drill: one shared setup, one dataset, one split, one evaluation
   helper, many models.

Do not polish. Do not split this into many notebooks. This is a playground with
signs: the path is already marked so your effort goes into recognizing and using
the methods.

## Segment 1: Theory Transfer

### The Root Question

Before any library call, ask:

> What kind of answer am I producing?

- Number: regression.
- Class: classification.
- Group without labels: clustering.
- Smaller view of many features: dimensionality reduction.

If this question is wrong, the notebook can run and still be wrong.

### Linear Regression

Picture: a ruler laid across noisy points.

It predicts a number by finding the straight trend with the smallest average
miss. If the pattern bends, the ruler cannot bend unless you create features
that expose the bend.

Code signs: `LinearRegression()`, `mean_squared_error`.

### Logistic Regression

Picture: a soft gate.

It creates a score, turns it into probability-like confidence, then a threshold
turns that into a class. It is the first clean classification baseline because
it is fast, stable, and easy to compare against.

Code signs: `LogisticRegression(max_iter=3000)`, `confusion_matrix`.

### Naive Bayes

Picture: a pile of weak clues voting very fast.

It pretends clues are independent. That makes it cheap and sometimes useful,
but it can be confidently wrong.

Code signs: `GaussianNB()` for simple numeric features, `MultinomialNB()` for
counts/text.

### k-NN

Picture: ask the nearby examples.

The whole model is the meaning of "nearby." If scales are wrong, distance is
wrong, and the neighbors lie to you.

Code signs: `KNeighborsClassifier(n_neighbors=...)`, `StandardScaler()`.

### Decision Tree

Picture: a flowchart of yes/no questions.

It is readable because each split is a question. It overfits because unlimited
questions let it memorize accidents.

Code signs: `DecisionTreeClassifier(max_depth=...)`, compare shallow vs deep.

### SVM

Picture: the safest fence.

Logistic regression wants a useful probability gate. k-NN asks neighbors. Trees
ask feature questions. SVM asks where to put the boundary so the closest danger
points on each side are as far from the fence as possible.

The RBF kernel is the playground warp: if a straight fence is not enough, the
model behaves as if the ground was warped until a cleaner fence is possible.

Code signs: `SVC(kernel="linear")`, `SVC(kernel="rbf")`, scaling required.

### PCA

Picture: rotate a messy cloud until its longest shadow is visible.

PCA ignores labels. It finds directions where the data varies most. A 2D PCA map
is useful for seeing structure, but it may throw away details a classifier needs.

Code signs: `PCA(n_components=2)`, scatter plot.

### K-Means

Picture: magnets in a point cloud.

Each magnet owns nearby points, then moves to the center of its owned points.
K-Means++ just chooses better starting magnet positions.

Cluster IDs are invented names. Cluster `3` does not mean digit `3`.

Code signs: `KMeans(n_clusters=10, n_init="auto")`.

### DBSCAN

Picture: crowds and loners.

Dense crowds become clusters. Sparse points become noise. You do not choose the
number of clusters; you choose what "close enough" means.

Code signs: `DBSCAN(eps=..., min_samples=...)`, noise label `-1`.

### Hierarchical Clustering

Picture: a family tree of points.

Small groups merge into bigger groups. You can cut the tree at different heights
to get different cluster counts.

Code signs: `AgglomerativeClustering(n_clusters=...)`, use small subsets.

## Segment 2: One-Notebook Drill

Notebook target:

`olympiads/IOAI Material/2. (Mostly) Linear models/exercises/ml_classical_models_one_notebook.ipynb`

### Cell 1: Imports Once

Import everything here and nowhere else:

- `numpy`, `pandas`, `matplotlib.pyplot`
- `load_digits`, `train_test_split`, `StandardScaler`, `Pipeline`
- `LogisticRegression`, `GaussianNB`, `KNeighborsClassifier`
- `DecisionTreeClassifier`, `SVC`
- `PCA`, `KMeans`, `DBSCAN`, `AgglomerativeClustering`
- `accuracy_score`, `confusion_matrix`, `classification_report`

Sign: repeated imports later mean you are drifting from the drill.

### Cell 2: Load One Dataset

Use `load_digits()`.

Create `X`, `y`, `images`, `class_names`.

Print `X.shape`, `y.shape`, class count, and one image.

Sign: no dataset choice tonight.

### Cell 3: Split Once

Create `X_train`, `X_test`, `y_train`, `y_test` with `stratify=y`.

Print train/test class counts.

Sign: every supervised model uses this same split.

### Cell 4: Evaluation Helper

Create `results = []`.

Create `evaluate_classifier(name, model)` that:

- fits on `X_train`, `y_train`,
- predicts `X_test`,
- computes accuracy,
- appends `{"model": name, "accuracy": acc}` to `results`,
- prints a confusion matrix,
- returns predictions.

Sign: after this, each model is one guided lane.

### Cell 5: Logistic Regression

Use a pipeline:

`StandardScaler()` -> `LogisticRegression(max_iter=3000)`

Call the helper.

Answer: which digit pair is confused most?

### Cell 6: Naive Bayes

Use `GaussianNB()`.

Call the helper.

Answer: is it worse than logistic regression, and why might we still keep it?

### Cell 7: k-NN

Use two scaled pipelines:

- `KNeighborsClassifier(n_neighbors=3)`
- `KNeighborsClassifier(n_neighbors=7)`

Answer: which `k` worked better, and what changed?

### Cell 8: Decision Tree

Use:

- `DecisionTreeClassifier(max_depth=3, random_state=42)`
- `DecisionTreeClassifier(max_depth=None, random_state=42)`

Answer: which one memorized more?

### Cell 9: SVM

Use two scaled pipelines:

- `SVC(kernel="linear")`
- `SVC(kernel="rbf")`

Answer: did the RBF playground warp help?

### Cell 10: Results Table

Convert `results` into a dataframe sorted by accuracy.

Answer:

- Which model would you submit first?
- Which model is the simplest honest baseline?

### Cell 11: PCA Map

Create `X_scaled = StandardScaler().fit_transform(X)`.

Create `X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)`.

Plot PCA colored by true digit labels.

Answer: which digits overlap?

### Cell 12: K-Means on PCA

Run `KMeans(n_clusters=10, n_init="auto", random_state=42)` on `X_pca`.

Plot PCA colored by cluster ID.

Answer: why is cluster `3` not digit `3`?

### Cell 13: DBSCAN on PCA

Run DBSCAN twice with two `eps` values.

For each run, print:

- number of clusters excluding noise,
- number of noise points.

Answer: did DBSCAN find crowds or mostly loners?

### Cell 14: Hierarchical on a Subset

Use only the first 300 rows of `X_pca`.

Run `AgglomerativeClustering(n_clusters=10)`.

Plot the subset colored by cluster ID.

Answer: why did we use only a subset?

### Cell 15: Final Reflection Table

Fill this manually:

| Method | Picture in my head | Needs labels? | Needs scaling? | Main danger |
| --- | --- | --- | --- | --- |
| Linear Regression | ruler | yes | often | underfits curves |
| Logistic Regression | soft gate | yes | usually | weak nonlinear boundary |
| Naive Bayes | clue pile | yes | not always | overconfidence |
| k-NN | ask neighbors | yes | yes | bad distance |
| Decision Tree | flowchart | yes | no | memorization |
| SVM | safest fence | yes | yes | bad kernel settings |
| PCA | shadow map | no | yes | discards useful detail |
| K-Means | magnets | no | yes | forced clusters |
| DBSCAN | crowds/loners | no | yes | sensitive `eps` |
| Hierarchical | family tree | no | yes | slow/noisy |

## Completion Check

This counts only if the notebook has:

- one imports/setup cell,
- one dataset,
- one split,
- one evaluation helper,
- supervised results table,
- PCA plot,
- K-Means plot,
- DBSCAN noise counts,
- hierarchical subset plot,
- final reflection table in your own words.

Stop after one pass. No extra datasets, no endless tuning, no visual polish.
