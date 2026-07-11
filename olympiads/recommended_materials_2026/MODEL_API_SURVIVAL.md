# Unknown Model API Survival Protocol

The goal is not to memorize every Hugging Face or Torch model. The goal is to reduce any unfamiliar model to the same contract in 15 minutes.

## The 15-minute probe

1. Identify the model and companion object from the imports: tokenizer, feature extractor, image processor, or processor.
2. Inspect `type(model)`, `model.config`, `model.config.id2label`, and `inspect.signature(model.forward)`.
3. Inspect the companion object and its documented sampling rate, image size, maximum length, padding and truncation defaults.
4. Run exactly one or two raw examples through the companion object. Print `batch.keys()`, every tensor's shape and dtype, and the device.
5. Move every tensor in the batch to the model device. Call the model with named dictionary expansion, then inspect `outputs.keys()` and each output shape.
6. Select the actual task tensor: usually `logits`, `last_hidden_state`, `pooler_output`, patch tokens, or a task-specific prediction field.
7. Inspect `model.named_modules()` and `model.named_parameters()` to locate the head and verify which parameters have `requires_grad=True`.
8. Run one loss and one backward pass on a tiny batch. Confirm that the intended parameters receive gradients and frozen parameters do not.
9. Save, reload and repeat the same tiny prediction before starting long training.

## Shape reflexes

- Text classifier logits: `(batch, classes)`.
- Token representations: `(batch, sequence_length, hidden_size)`.
- Attention mask: `(batch, sequence_length)`, aligned with token IDs.
- Image classifier logits: `(batch, classes)`.
- Image features: often `(batch, hidden_size)`.
- ViT patch tokens: often `(batch, patches, hidden_size)`; reshape only after proving the patch-grid dimensions.
- Segmentation output: usually `(batch, classes, height, width)`.

## Stop conditions

Do not write a DataLoader, training loop or submission loop until a two-example forward pass works. Do not guess an output field from memory. Do not use `squeeze()` without naming the dimension you intend to remove. Do not fine-tune before recording a baseline metric and a save/reload check.

## Competition artifact checklist

Every practice block must end with: input keys and shapes, output keys and shapes, trainable-parameter count, one metric, a prediction sample, and exact submission shape/columns. That evidence is the fluency drill.
