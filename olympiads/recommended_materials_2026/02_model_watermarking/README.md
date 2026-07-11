# Embedding Watermarks into Deep Neural Networks

- Source: https://arxiv.org/abs/1701.04082
- Local source: `paper.pdf`
- Extracted text: `paper_extracted.md`

## What the paper actually says

The paper treats a trained model as intellectual property and proposes embedding a binary watermark directly into selected network parameters. Training uses the original task loss plus a watermark regularizer. A fixed projection maps selected weights to watermark bits, a sigmoid turns the projected values into bit probabilities, and a cross-entropy-like term pushes them toward a secret binary string.

It distinguishes three situations: embed while training from scratch, embed while fine-tuning a pretrained model, and embed while distilling from another model. A useful watermark should preserve task accuracy, be reliably detectable, and survive likely model modifications. The experiments show that the proposed parameter watermark can remain detectable after fine-tuning and substantial pruning; the paper reports full retention even after pruning 65% of parameters in its setup.

The contest-relevant idea is the interface between model parameters and an auxiliary objective: select a layer or tensor, flatten or project its weights, add a differentiable constraint, and verify both the original metric and watermark recovery.

## CEOAI syllabus mapping

- `3(b) Optimization Techniques`: auxiliary regularization terms and multi-objective loss balancing.
- `3(a) Neural Network Basics`: parameters, gradients, backpropagation, and differentiable losses.
- `3(c) Architectures`: locating and manipulating layers in supplied CNNs or other pretrained models.
- `5(b) CNN architectures`: the experiments use convolutional image classifiers.

Watermarking itself is outside the named syllabus. It is probably included because it creates a very competition-friendly task: manipulate provided weights under an accuracy constraint.

## What to retain for competition

Do not memorize the paper's full attack taxonomy. Retain the pattern `task loss + lambda * auxiliary loss`, the need to inspect `named_parameters()`, and the two-metric contract: predictive quality and recovered watermark bits. Pruning is relevant because a hidden evaluator may modify or compress the submitted model before checking the watermark.
