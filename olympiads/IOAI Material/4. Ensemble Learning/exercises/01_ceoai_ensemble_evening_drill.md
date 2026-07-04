# CEOAI Ensemble Evening Drill

Goal: cover the remaining ensemble-learning gap efficiently enough for CEOAI
practice.

This drill should come after the single-model benchmark from Lesson 2. Ensembles
only make sense if you first know what they improve over.

## Required Methods

- Random Forests
- Voting
- Bagging
- Boosting
- Gradient Boosting

Do not implement the internals from scratch. Use scikit-learn and focus on when
to use each method, what it improves, and how it fails.

## Mental Models

### Bagging

Mental model: train many unstable models on slightly different bootstrap samples
and average their predictions.

Use it when:

- a single model has high variance,
- a decision tree overfits,
- you want more stability without changing the base model idea.

Failure mode:

- it does not fix a model that is biased in the wrong direction.

Task:

- Compare one decision tree against a bagged decision-tree ensemble.
- Record train accuracy and validation accuracy.
- Write one sentence saying whether bagging reduced overfitting.

### Random Forest

Mental model: bagging plus random feature choices, usually with decision trees.

Use it when:

- you need a strong tabular baseline fast,
- features are mixed or nonlinear,
- you want feature importance as a diagnostic.

Failure mode:

- it can still overfit noisy data and may be weaker than boosting on structured
  tabular competitions.

Task:

- Train a Random Forest on the same dataset as the single tree.
- Compare it against logistic regression, k-NN, and the single tree.
- Print one feature-importance ranking if the dataset has meaningful feature
  names; otherwise skip this and state why.

### Voting

Mental model: combine different model families and let them vote.

Use it when:

- models make different kinds of mistakes,
- no single model is clearly dominant,
- you want a simple ensemble without much tuning.

Failure mode:

- voting weak models together does not magically create a strong model.

Task:

- Combine three different classifiers from the Lesson 2 benchmark.
- Compare the voting classifier to the best individual model.
- Write one sentence saying whether diversity helped.

### Boosting

Mental model: train models sequentially so later models focus more on previous
mistakes.

Use it when:

- single trees are too weak,
- tabular data has nonlinear structure,
- you can tune carefully without overfitting.

Failure mode:

- it can chase noise if learning rate, depth, or number of estimators is wrong.

Task:

- Train one boosting model with conservative defaults.
- Compare it to Random Forest.
- Change exactly one hyperparameter and record whether validation accuracy
  improved or got worse.

### Gradient Boosting

Mental model: boosting as gradient descent in model space; each new tree tries to
correct the current prediction errors.

Use it when:

- tabular performance matters,
- Random Forest is a strong but not final baseline,
- you can validate carefully.

Failure mode:

- it is easy to over-tune on the validation set.

Task:

- Train one gradient boosting classifier.
- Compare against Random Forest using the same split and metric.
- Write one sentence explaining which model you would submit first and why.

## Evening Order

1. Start with the Lesson 2 supervised benchmark results.
2. Add a single decision tree if it is not already there.
3. Add Random Forest.
4. Add Bagging.
5. Add Voting.
6. Add one boosting model.
7. Create one final comparison table.

## Final Comparison Table

Use this exact table shape when you later implement the notebook:

| Method | Main idea | Best use | Main failure | Metric result |
| --- | --- | --- | --- | --- |
| Single tree | yes/no splits | interpretable baseline | overfits | |
| Bagging | average many bootstrapped models | reduce variance | does not fix bias | |
| Random Forest | bagged trees plus random features | fast tabular baseline | less tunable than boosting | |
| Voting | combine different model families | diverse mistakes | weak voters stay weak | |
| Boosting | focus on previous mistakes | stronger tabular model | chases noise | |
| Gradient Boosting | sequential error correction | competition-style tabular baseline | over-tuning | |

## Stop Condition

This drill is complete when you can answer:

- Why is a Random Forest usually better than one tree?
- What problem does bagging solve?
- What problem does boosting solve?
- Why does voting require diverse models?
- Which ensemble would you try first on a tabular classification task?
- Which metric proves the ensemble actually helped?

If the final table is empty or only contains theory, the drill does not count.
