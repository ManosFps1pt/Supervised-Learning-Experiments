# Sprint 01: Manual Forward Pass and Backprop

## Source

Use `../sources/Neural Networks.pptx`, especially the sections on weights,
activations, loss, perceptron updates, gradient descent, and the one-weight
training example.

## Time Box

Target: **25-35 minutes**.

Do this on paper or in a tiny scratch cell. The point is to remove mystery from
`loss.backward()`.

## Goal

Given one input, one target, and a tiny model, compute the prediction, loss,
gradient sign, and one weight update. You should know whether a weight should
increase or decrease before PyTorch tells you.

## Task A: One-Weight Regression

Use:

- input `x = 2`,
- target `y = 1`,
- prediction `y_hat = w * x`,
- initial weight `w = 0.2`,
- loss `0.5 * (y_hat - y)^2`,
- learning rate `eta = 0.1`.

Produce:

- `y_hat`,
- loss,
- gradient of loss with respect to `w`,
- updated weight after one gradient-descent step,
- new prediction after the update.

## Task B: One Perceptron Update

Use a binary classifier:

- `z = w1*x1 + w2*x2 + b`,
- prediction is `1` if `z >= 0`, else `0`,
- update rule: `w_new = w_old + eta * (y - y_hat) * x`,
- `b_new = b_old + eta * (y - y_hat)`.

Choose one example where the model is wrong and perform exactly one update.

## Required Self-Checks

- If the prediction is lower than the target in Task A, the weight update should
  move the prediction upward.
- The loss after the Task A update should be lower than the original loss.
- In Task B, no update should happen when `y == y_hat`.
- You can explain why gradient descent uses `w - eta * gradient`, while the
  perceptron rule uses `+ eta * error * x`.

## Hints

1. Keep all numbers visible; do not hide the calculation inside code.
2. Track signs carefully. Most backprop bugs are sign or shape bugs.
3. For Task A, use the chain rule: loss depends on prediction, prediction
   depends on weight.

## Stop Condition

Stop when you can predict the direction of a one-step update without running
code.
