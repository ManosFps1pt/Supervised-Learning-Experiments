# Overcoming Catastrophic Forgetting in Neural Networks

- Source: https://arxiv.org/abs/1612.00796
- Local source: `paper.pdf`
- Extracted text: `paper_extracted.md`

## What the paper actually says

When a neural network is trained on task B after task A, the gradients for B can overwrite parameters that were important for A. This is catastrophic forgetting. Ordinary L2 regularization treats every parameter as equally important and therefore either protects too little or prevents useful learning on the new task.

The paper proposes Elastic Weight Consolidation (EWC). After training task A, keep a copy of the old parameters and estimate how important each parameter was using the diagonal of the Fisher information matrix. While training task B, optimize the new-task loss plus a quadratic penalty that pulls important parameters toward their old values. The coefficient controls the stability/plasticity tradeoff: retain old behavior versus learn the new task.

The paper demonstrates EWC on sequential permuted-MNIST classification and Atari reinforcement-learning tasks. Its practical lesson is broader than the exact Fisher derivation: when adapting a pretrained model, measure old-task performance, decide which weights may move, and explicitly protect old behavior through replay, distillation, freezing, adapters, or an importance-weighted penalty.

## CEOAI syllabus mapping

- `3(b) Optimization Techniques`: regularization, loss composition, learning-rate and stability tradeoffs.
- `3(a) Neural Network Basics`: gradients, backpropagation, and parameter updates.
- `3(c) Architectures`: practical fine-tuning of a supplied pretrained network.
- `1(d-f) RL`: the paper includes a continual-learning DQN experiment, but this is secondary for the sprint.

Continual learning and EWC are not named explicitly in the CEOAI syllabus. The likely competition expectation is application and debugging, not reproducing the Bayesian derivation.

## What to retain for competition

Know the failure signature: new-class accuracy rises while old-class accuracy collapses. Always keep separate old/new validation metrics. Be able to identify the classifier head, expand it, choose which parameters require gradients, and compare naive fine-tuning against one retention mechanism. Do not spend the sprint deriving the Fisher matrix proof.
