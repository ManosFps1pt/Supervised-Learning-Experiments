# AICC Recommended Problem Order

Purpose: choose which imported AICC problems to think about first for baseline-recognition practice before IOAI.

Rule: for each problem, write your own baseline idea before opening the baseline notebook or reference solution.

## Ranked Order

1. **Essay Gap**
   - Why: easiest clean baseline-recognition drill.
   - Main pattern: multiple-choice NLP classification.
   - Good for: task-to-model-head mapping, macro-F1, small fast dataset.

2. **Polarity**
   - Why: tiny data and direct pretrained-text-encoder pattern.
   - Main pattern: lexical binary classification with a restricted pretrained model.
   - Good for: few-shot validation, text-pair encoding, restriction checking.

3. **Word Lookups**
   - Why: sequence tagging and output-length discipline.
   - Main pattern: character-level BMES tagging.
   - Good for: tokenization, per-token outputs, strict submission formatting.

4. **Deceptive Points**
   - Why: classical ML baseline thinking without heavy infrastructure.
   - Main pattern: robust tabular regression.
   - Good for: outliers, validation, simple model comparison.

5. **Find Brain Tumors**
   - Why: CV classification with scarce labels.
   - Main pattern: image classification under limited labels and model restrictions.
   - Good for: image loading, label joins, macro-F1, ResNet-style baselines.

6. **Shuffled**
   - Why: model-internals puzzle that trains tensor/embedding inspection.
   - Main pattern: recovering shuffled CLIP positional embeddings.
   - Good for: tensor shapes, pretrained-model inspection, exact submission contracts.

7. **Buried Fault**
   - Why: hard but valuable time-series plus localization task.
   - Main pattern: sensor classification and weak event localization.
   - Good for: NaN handling, feature extraction, interval outputs, metric decomposition.

8. **Massive Problem**
   - Why: large tabular/memory-pressure practice.
   - Main pattern: high-dimensional classification.
   - Good for: memory-aware loading, dimensionality reduction, macro-F1.

9. **Face Matching**
   - Why: useful CLIP workflow, but already hit CUDA OOM once.
   - Main pattern: image embedding similarity.
   - Good for: processor/model input contracts, batching, memory control.

10. **The Defected Nuts**
    - Why: hard anomaly segmentation with strict output format.
    - Main pattern: one-class/anomaly segmentation.
    - Good for: mask generation, Base85/encoded submission validation, high-resolution image handling.

11. **Oriented Ship**
    - Why: hardest baseline-design task in the current set.
    - Main pattern: rotated object detection.
    - Good for: detection datasets, rotated boxes, mAP, coordinate format discipline.

## Immediate Use

Start with **Essay Gap**. Fill its four lines in `aicc_baseline_recognition_drill.md`, then inspect the baseline notebook only after committing your own baseline idea.
