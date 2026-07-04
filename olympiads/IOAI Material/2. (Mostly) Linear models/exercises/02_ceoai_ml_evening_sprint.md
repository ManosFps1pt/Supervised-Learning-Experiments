# CEOAI ML Evening Sprint

Goal: finish the useful part of CEOAI Machine Learning in one evening by building
mental models first and only then making notebook evidence.

This is not a polished lesson. It is a drill map. Each task should produce a
visible artifact when you later implement it: a metric, a plot, a table, or a
short comparison note.

## Scope

Required:

- Linear Regression
- Logistic Regression
- Naive Bayes
- k-NN
- Decision Trees
- SVMs
- K-Means and K-Means++
- DBSCAN
- Hierarchical clustering
- Dimensionality reduction with PCA
- Metrics, scaling, train/test split, and confusion matrix

Not required for this CEOAI sprint:

- full SVM derivations
- implementing SVMs from scratch
- polishing notebooks
- from-scratch algorithm implementations

## Evening Structure

Use two notebooks when you later implement this:

1. `ml_digits_benchmark.ipynb`
   - supervised classical ML and metrics.
2. `ml_clustering_pca_drill.ipynb`
   - PCA, K-Means, DBSCAN, and clustering interpretation.

Do not start by reading broadly. Start from small runnable datasets and compare
models.

## Mental Models

### Linear Regression

Mental model: fit a line or plane that predicts a number.

Use it when:

- the target is continuous,
- errors can be measured by distance from the true value,
- a simple baseline is more useful than a complex model.

Failure mode:

- it underfits curved patterns unless you add better features.

Task:

- Train one linear regression model on a tiny synthetic regression dataset.
- Print train MSE and validation MSE.
- Write one sentence: underfit, overfit, or acceptable baseline.

### Logistic Regression

Mental model: draw a linear boundary and output class probabilities.

Use it when:

- the target is a class,
- you need a fast baseline,
- features are numeric or vectorized text,
- interpretability matters.

Failure mode:

- it misses nonlinear boundaries unless the features already expose the pattern.

Task:

- Train logistic regression on `sklearn.datasets.load_digits()`.
- Use scaling before the model.
- Print accuracy and a confusion matrix.
- Inspect one wrong prediction and write what class it confused.

### Naive Bayes

Mental model: combine simple feature evidence under a strong independence
assumption.

Use it when:

- you need a very fast baseline,
- data is count-like, text-like, or simple tabular,
- the baseline matters more than perfection.

Failure mode:

- correlated features can make its confidence misleading.

Task:

- Train one Naive Bayes classifier on digits or a simple text/count dataset.
- Compare its accuracy against logistic regression.
- Write one sentence explaining whether speed or accuracy is the main reason to
  keep it.

### k-NN

Mental model: classify a point by asking its nearest neighbors.

Use it when:

- nearby examples should have similar labels,
- the dataset is small enough,
- you want a non-parametric baseline.

Failure mode:

- distance becomes weak when features are badly scaled or high-dimensional.

Task:

- Train k-NN on scaled digits features.
- Compare at least two values of `k`.
- Write one sentence explaining the tradeoff between small `k` and large `k`.

### Decision Tree

Mental model: split the data with yes/no feature questions until leaves predict
classes or values.

Use it when:

- you want interpretability,
- features have threshold-like behavior,
- you need the single-model baseline before ensembles.

Failure mode:

- deep trees memorize training data.

Task:

- Train one shallow tree and one deeper tree.
- Compare train accuracy and validation accuracy.
- Write one sentence saying which one overfits more and why.

### SVM

Mental model: find a decision boundary with the widest possible safety margin
between classes. With kernels, the model can behave as if the data was lifted
into a richer feature space without you manually creating all those features.

Why it is special:

- logistic regression asks for a good probability boundary,
- k-NN asks what neighbors are nearby,
- trees ask a sequence of feature questions,
- SVM asks for the most robust separating boundary.

Use it when:

- classes may be separable with a clean margin,
- the dataset is medium-sized,
- scaling is possible,
- you want a strong classical baseline before neural networks.

Failure mode:

- it is sensitive to scaling and kernel/hyperparameter choices.

Task:

- Train a linear SVM and an RBF-kernel SVM on scaled digits features.
- Compare both against logistic regression and k-NN.
- Print accuracy and a confusion matrix for the better SVM.
- Write one sentence explaining whether the kernel helped.

## Dimensionality Reduction

### PCA

Mental model: rotate the data to find the directions with the most variation,
then keep only the strongest directions.

Use it when:

- features are high-dimensional,
- you need a 2D visualization,
- you want compression before clustering or a baseline model.

Failure mode:

- the directions with most variance are not always the most useful for labels.

Repo anchor:

- `challenges/mnist.ipynb` already shows the useful idea: flatten MNIST,
  normalize pixels, reduce with `PCA(n_components=2)`, and plot the result.

Task:

- Run PCA to 2D on digits or MNIST-like data.
- Plot points colored by true labels.
- Write one sentence saying which classes overlap.

## Clustering

### K-Means and K-Means++

Mental model: place `k` centroids, assign each point to the nearest centroid,
then move centroids to the assigned points' average.

K-Means++ mental model: choose better starting centroids so the algorithm is
less likely to start badly.

Use it when:

- you know or can guess the number of clusters,
- clusters are roughly round,
- you want a fast unsupervised grouping baseline.

Failure mode:

- it forces every point into a cluster, even outliers.

Task:

- Run K-Means with `n_clusters=10` on PCA-reduced digits.
- Plot points colored by cluster labels.
- Compare cluster labels to true digit labels informally.
- Write one sentence explaining why cluster IDs are not the same as class names.

### DBSCAN

Mental model: dense neighborhoods become clusters; sparse points become noise.

Use it when:

- clusters may have irregular shapes,
- outliers matter,
- you do not want to choose the number of clusters directly.

Failure mode:

- `eps` is sensitive; a bad value makes everything one cluster or all noise.

Task:

- Run DBSCAN on the 2D PCA projection.
- Count how many points are labeled noise.
- Write one sentence explaining whether DBSCAN found useful structure.

### Hierarchical Clustering

Mental model: build a tree of merges from small groups into bigger groups.

Use it when:

- you want to inspect grouping structure,
- the dataset is small,
- a dendrogram or hierarchy is more useful than one fixed clustering.

Failure mode:

- it becomes slow and visually noisy on large datasets.

Task:

- Run hierarchical clustering on a small subset only.
- Choose a fixed number of clusters from the hierarchy.
- Write one sentence comparing it to K-Means.

## Stop Condition

This sprint is complete when you can answer these from memory:

- Is the target a number, class, or no-label grouping?
- Which baseline do you try first?
- Does the model need scaling?
- Why is SVM different from logistic regression, k-NN, and trees?
- What metric proves it worked?
- What does the confusion matrix reveal?
- Why can PCA help visualization but hurt classification?
- Why are K-Means cluster IDs not class labels?
- Why does DBSCAN sometimes call points noise?

If you cannot answer those, do not move to more theory. Repeat the smallest
failed task.
