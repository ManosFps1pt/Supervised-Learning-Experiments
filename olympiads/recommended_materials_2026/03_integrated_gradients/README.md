# Axiomatic Attribution for Deep Networks

- Source: https://arxiv.org/abs/1703.01365
- Local source: `paper.pdf`
- Extracted text: `paper_extracted.md`

## What the paper actually says

The paper asks how to attribute one model prediction to its input features. A plain gradient at the input can be zero in a saturated region even when a feature clearly changed the output. The authors propose two requirements: Sensitivity, so a feature that changes the prediction can receive attribution, and Implementation Invariance, so functionally equivalent networks receive the same explanation.

Integrated Gradients chooses a baseline input and follows the straight-line path from that baseline to the real input. It samples gradients along the path, averages them, and multiplies by the input-minus-baseline difference. The attributions satisfy completeness: their sum should approximately equal the difference between the model score at the input and at the baseline. The paper recommends checking this approximation and increasing the number of integration steps when the equality is poor.

The method applies to images, text embeddings, sequence models, and other differentiable inputs. Baseline choice matters: a black image is common for vision, while zero embeddings were used for NLP examples. Attention weights alone are not guaranteed to be complete attributions because information can flow through other model paths.

## CEOAI syllabus mapping

- `3(a) Neural Network Basics`: gradients and backpropagation with respect to inputs.
- `3(c) Architectures`: applying gradient-based probes to CNNs, LSTMs, and Transformers.
- `4(a-c) NLP`: token or embedding attribution for text models.
- `5(a-c) CV`: pixel attribution and saliency for image models.

Attribution is not named explicitly in the syllabus, so expect a practical implementation or interpretation task rather than an axiomatic proof.

## What to retain for competition

You need four decisions: target score/logit, baseline, number of steps, and aggregation dimension. Check tensor shapes and the completeness residual. Do not confuse `argmax` with a differentiable score: select the predicted or requested class logit, then differentiate that scalar with respect to the input.
