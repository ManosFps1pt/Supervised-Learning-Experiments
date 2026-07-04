# CEOAI Ensemble Guided Extension

Goal: extend the same classical-ML notebook with ensembles, without creating a
new setup or a polished side project.

Use this after:

`03_ceoai_ml_guided_one_notebook.md`

Append the cells to:

`ml_classical_models_one_notebook.ipynb`

## Segment 1: Theory Transfer

### Why Ensembles Exist

A single model has a personality.

- A tree is flexible but memorizes.
- Logistic regression is stable but simple.
- k-NN trusts neighborhoods.
- SVM trusts the margin.

An ensemble asks whether several imperfect models can cover each other's
weaknesses.

### Bagging

Picture: many slightly different judges averaging their decisions.

Each judge sees a resampled version of the data. If one tree overreacts, the
group vote calms it down.

Main idea: reduce variance.

### Random Forest

Picture: many trees forced to look through different feature windows.

It is bagging plus feature randomness. This prevents all trees from becoming the
same tree.

Main idea: strong default tabular baseline.

### Voting

Picture: a committee with different personalities.

Voting helps only when members make different mistakes. Weak copies of the same
idea do not become strong by voting.

Main idea: use model diversity.

### Boosting

Picture: a tutor who keeps returning to the examples you missed.

Later models focus more on previous mistakes. This can fix weak models, but it
can also chase noise.

Main idea: attack hard cases.

### Gradient Boosting

Picture: step-by-step error repair.

Each small tree repairs the current prediction errors. This is why gradient
boosting is common in tabular competitions.

Main idea: strong tabular performance with validation discipline.

## Segment 2: One-Notebook Extension

Reuse:

- `X_train`, `X_test`, `y_train`, `y_test`
- `evaluate_classifier(name, model)`
- `results`

### Ensemble Cell 1: Add Imports Only

Import:

- `BaggingClassifier`
- `RandomForestClassifier`
- `VotingClassifier`
- `AdaBoostClassifier`
- `GradientBoostingClassifier`

Sign: do not reload data and do not split again.

### Ensemble Cell 2: Bagging

Use:

`BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=50, random_state=42)`

Call the same helper.

Answer: did many resampled trees beat one tree?

### Ensemble Cell 3: Random Forest

Use:

`RandomForestClassifier(n_estimators=200, random_state=42)`

Call the same helper.

Answer: did feature-random trees beat plain bagging?

### Ensemble Cell 4: Voting

Use three different members:

- logistic regression pipeline,
- scaled k-NN,
- scaled RBF SVM.

Create `VotingClassifier(..., voting="hard")`.

Answer: did the committee beat its members?

### Ensemble Cell 5: Boosting

Use:

`AdaBoostClassifier(random_state=42)`

Call the same helper.

Answer: did focusing on misses help?

### Ensemble Cell 6: Gradient Boosting

Use:

`GradientBoostingClassifier(random_state=42)`

Call the same helper.

Answer: would you submit this or Random Forest first tonight?

### Ensemble Cell 7: Final Ensemble Table

Update the sorted results dataframe.

Then fill:

| Method | Picture in my head | What it fixes | Main danger | Keep for contest? |
| --- | --- | --- | --- | --- |
| Single tree | one flowchart | interpretability | memorization | |
| Bagging | many judges | variance | shared bias | |
| Random Forest | feature-random trees | tree overfit | not final best | |
| Voting | committee | different mistakes | weak voters | |
| Boosting | tutor revisiting misses | hard cases | noise chasing | |
| Gradient Boosting | error repair | tabular performance | over-tuning | |

## Completion Check

This extension counts only if:

- it reuses the same notebook and split,
- Random Forest is compared to a single tree,
- boosting is compared to Random Forest,
- voting uses different model families,
- the final table includes a contest decision.

Stop after one pass.
