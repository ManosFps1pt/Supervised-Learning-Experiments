# Sprint 01: Linear Models Level Check

## Source

Use:

- `../sources/download (67).pdf` for linear regression, nonlinear features,
  classification, and logistic regression.
- `../sources/download (15).pdf` for the bag-of-words sentiment-analysis
  pipeline.
- `../sources/L02c - Support Vector Machines (TBA).docx` only as a reminder
  that the SVM part is not ready yet, so do not build your plan around it.

## What To Study First

Study only the parts that support a short baseline:

1. linear regression model and loss,
2. nonlinear regression through feature expansion,
3. binary classification framing,
4. logistic regression and probability thresholding,
5. bag-of-words sentiment analysis pipeline.

Do not spend time on SVMs yet. The source note is literally `TBA`.

## Why I Am Telling You To Do This

You need a short diagnostic, not a full lesson rebuild.

This lesson is the cleanest place to check whether you can:

- choose the right target/metric contract,
- separate regression from classification,
- build one small sklearn baseline end to end,
- read a plot and explain what went wrong,
- bridge tabular ML into simple NLP without needing the missing notebooks.

If this feels shaky, then later NLP and generative-model material will become
noise instead of progress.

## Time Box

Target: **60-75 minutes total**.

- **15 minutes** reading the source sections above.
- **20-25 minutes** Exercise A.
- **20-25 minutes** Exercise B.
- **10-15 minutes optional** Exercise C.

If you are busy, do only A and B.

## What Your Jupyter Notebook Should Contain

Create one notebook with these sections:

1. `Source notes`
   - 5-8 bullets from the lesson on regression, logistic regression, and
     bag-of-words.
2. `Exercise A - regression sanity check`
3. `Exercise B - logistic classification sanity check`
4. `Reflection`
   - 4-6 lines: what worked, what confused you, what you would reuse.
5. `Optional Exercise C - text bridge`

Keep each section small. This notebook is for diagnosis, not polish.

## Exercise A: Regression Baseline

Build a tiny regression experiment on synthetic data.

Task:

- generate a 1D dataset with noise,
- fit a plain linear model,
- then fit the same task with polynomial features of degree 2 or 3,
- compare validation MSE for both,
- plot predictions against the data.

Produce:

- printed shapes for `X_train`, `y_train`, `X_val`, `y_val`,
- train and validation MSE,
- one plot for the linear fit,
- one plot for the expanded-feature fit,
- one sentence saying which model underfits less and why.

Reason:

This checks whether you understand the most basic lesson-2 idea: a linear model
can become more expressive through the representation, not by magic.

## Exercise B: Logistic Regression Boundary Check

Build a binary classification baseline on a tiny 2D dataset.

Task:

- create or load a small 2D binary dataset,
- train a logistic-regression classifier,
- compute class probabilities,
- convert probabilities to labels with an explicit threshold,
- visualize the decision boundary or decision regions.

Produce:

- class balance printout,
- accuracy plus confusion matrix,
- one probability example you inspect manually,
- one boundary plot,
- one sentence explaining what the threshold is doing.

Reason:

This is the minimum useful test of whether you actually understand
classification as a probability decision problem rather than just calling
`fit()` and trusting the output.

## Optional Exercise C: Tiny Sentiment Bridge

If you still have time, make a miniature sentiment dataset of your own.

Task:

- write 10-20 short movie-review sentences,
- label them positive or negative,
- vectorize them with `CountVectorizer` or `TfidfVectorizer`,
- train logistic regression,
- inspect the most positive and most negative coefficients.

Produce:

- dataset size,
- vocabulary size,
- accuracy on a tiny held-out split or leave-one-out style check,
- top positive words,
- top negative words.

Reason:

This directly connects the linear-model lesson to NLP and prepares you for
Lesson 5 without needing the missing notebooks first.

## Clear Goals

By the end of this sprint you should be able to:

- explain the difference between a regression target and a classification
  target,
- explain why polynomial features can help a linear model,
- explain why logistic regression outputs probabilities before labels,
- build one complete sklearn baseline with plots and metrics,
- say in plain language whether a failure came from features, thresholding, or
  model mismatch.

## Stop Condition

Stop when:

- Exercises A and B are complete,
- the notebook has the required outputs,
- your reflection clearly names one strength and one weakness.

Do not spend extra time tuning for a prettier score.

## Level Signal

Strong signal:

- you finish A and B cleanly,
- your plots match your metric story,
- your reflection is specific.

Weak signal:

- you mix regression and classification metrics,
- you cannot explain the threshold step,
- you change models before checking shapes, labels, and plots.
