# CEOAI Ensemble Delta Drill

Append this to the Lesson 2 notebook after the classical models.

Primary target:

`olympiads/IOAI Material/2. (Mostly) Linear models/exercises/solution.ipynb`

If you make a cleaner copy, use:

`olympiads/IOAI Material/2. (Mostly) Linear models/exercises/ml_digits_benchmark.ipynb`

Syllabus tag: CEOAI `2(c)` Ensemble Methods.

Goal: do not relearn Lesson 2. Use the existing digits setup to see exactly
what ensembles add beyond logistic regression, k-NN, SVM, and one decision tree.

## Starting Assumption

Lesson 2 already gave you:

- `X_train`, `X_test`, `y_train`, `y_test`
- `results`
- `evaluate_classifier(...)`
- logistic regression
- Naive Bayes
- k-NN
- decision tree
- SVM
- PCA / clustering work

Do not reload data. Do not split again. Do not redo PCA. Do not write more
theory about classical models.

## Cell E0: Upgrade The Scoreboard

Your current Lesson 2 helper records only test accuracy. Ensembles need one more
signal: overfitting.

Create a new list:

`ensemble_results = []`

Create a new helper for this section only. It must record:

| Column | Meaning |
| --- | --- |
| `model` | model name |
| `family` | tree, bagging, forest, voting, boosting |
| `train_acc` | memorization signal |
| `test_acc` | contest signal |
| `gap` | `train_acc - test_acc` |
| `fit_seconds` | rough speed signal |

Required behavior:

- fit on `X_train`, `y_train`
- predict on train and test
- append one row to `ensemble_results`
- return test predictions

Do not delete the Lesson 2 helper. This is a second scoreboard for ensemble
behavior.

## Cell E1: Anchor Tree

Run two trees:

- `DecisionTreeClassifier(max_depth=4, random_state=42)`
- `DecisionTreeClassifier(random_state=42)`

Question to answer under the cell:

`The deep tree is the reason ensembles exist because ____.`

Only use the train/test gap. No extra explanation.

## Cell E2: Bagging

Run:

`BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=100, random_state=42, n_jobs=-1)`

Compare only against the deep tree.

Write exactly this sentence:

`Bagging changed the tree from train/test/gap = ____/____/____ to ____/____/____, so it mainly helped ____.`

Expected embedding: bagging is many unstable trees averaged to reduce variance.

## Cell E3: Random Forest

Run:

`RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)`

Compare against:

- deep tree
- bagging
- best Lesson 2 non-ensemble model

Extra probe:

- print top 10 feature importances by feature index

Write:

`Random Forest is the low-cortisol tabular default here because ____. I still would not worship it because ____.`

Expected embedding: forest = bagging plus feature randomness, usually the first
ensemble baseline to try.

## Cell E4: Voting

Build a hard-voting model from already-known Lesson 2 families:

- scaled logistic regression
- scaled k-NN
- scaled RBF SVM

Run it through the ensemble helper.

Then compare against the three individual members from Lesson 2.

Write:

`Voting helped / failed because the members' mistakes were ____.`

Required small probe:

- for 10 test examples where at least one member is wrong, print true label,
  each member prediction, and voting prediction.

Expected embedding: voting is not stronger models; it is error diversity.

## Cell E5: AdaBoost

Run:

`AdaBoostClassifier(random_state=42)`

Compare against:

- shallow tree
- random forest

Write:

`AdaBoost is worth remembering because it tries to ____. On this notebook it is / is not better than Random Forest.`

Expected embedding: boosting reweights attention toward missed examples.

## Cell E6: Gradient Boosting

Run:

`GradientBoostingClassifier(random_state=42)`

Compare against:

- AdaBoost
- Random Forest
- SVM from Lesson 2

Write:

`Gradient boosting is my candidate when ____. Its danger is ____.`

Expected embedding: sequential error repair can be strong, but validation
discipline matters.

## Cell E7: One Noise Stress Test

Create noisy training labels only:

- copy `y_train`
- select 15 percent of training rows with `np.random.default_rng(42)`
- replace those labels with random digits 0 to 9

Train only these on the noisy labels:

- deep tree
- random forest
- gradient boosting

Evaluate on the original clean `X_test`, `y_test`.

Write:

`Under bad labels, ____ was most stable and ____ was most dangerous.`

Expected embedding: ensembles reduce some variance, but they do not make bad
supervision clean.

## Cell E8: Winner Error View

Pick the best ensemble by `test_acc`.

Print:

- model name
- confusion matrix
- classification report
- worst confused digit pair

Write:

`The remaining failure is not "ensemble learning"; it is this specific confusion: ____.`

Expected embedding: a model is not understood until you inspect its errors.

## Cell E9: Final Delta Table

Fill this table from your actual outputs:

| Method | What changed vs Lesson 2 tree? | What it buys | What it cannot fix | Contest use |
| --- | --- | --- | --- | --- |
| Bagging | | lower variance | shared bias | |
| Random Forest | | strong default | not always final best | |
| Voting | | diverse errors | weak voters | |
| AdaBoost | | focuses on misses | noisy misses | |
| Gradient Boosting | | sequential repair | over-tuning | |

## Pass Condition

This counts only if the notebook has:

- deep tree vs bagging comparison
- bagging vs random forest comparison
- voting member disagreement examples
- AdaBoost result
- gradient boosting result
- noisy-label stress test
- winner confusion matrix
- final delta table

## Fail Condition

This does not count if:

- you repeat the full Lesson 2 drill
- you reload/split data again
- you only compare final accuracies
- you skip train/test gap
- you skip disagreement examples for voting
- you skip noisy labels
- you tune instead of learning behavior

Stop after E9. The point is to recognize when an ensemble earns its place.
