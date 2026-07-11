# ERASER: A Benchmark to Evaluate Rationalized NLP Models

- Source: https://arxiv.org/abs/1911.03429
- Local source: `paper.pdf`
- Extracted text: `paper_extracted.md`

## What the paper actually says

ERASER combines seven NLP datasets that include human-marked rationales: pieces of text intended to support the gold label. It separates ordinary task performance from explanation quality. A model can predict the right label and still provide a poor or unfaithful rationale.

The benchmark evaluates two different properties. Plausibility asks whether the selected tokens or sentences agree with human rationales, using overlap-oriented measures such as token-level precision/recall/F1 and ranking measures. Faithfulness asks whether the model actually relied on the rationale. Comprehensiveness measures how much the target score falls when the rationale is removed; a larger fall suggests the removed evidence mattered. Sufficiency checks how well the rationale alone preserves the original prediction; a small change is better.

The paper emphasizes that no single metric captures every desirable property and that architectures do not transfer cleanly across datasets with very different document lengths and rationale granularities. Rationales may be extracted spans or generated jointly with predictions, but they must be evaluated independently from label accuracy.

## CEOAI syllabus mapping

- `4(a) NLP preprocessing`: tokenization, spans, masks, and mapping rationale positions back to text.
- `4(b) Embeddings`: token representations used for selection and classification.
- `4(c) Related architectures`: BERT-style encoders and other neural NLP models.
- `3(c) Transformers`: attention masks, logits, and model outputs.

Rationale evaluation is beyond the named syllabus, but it directly exercises the data structures and masking operations that make NLP code slow under competition pressure.

## What to retain for competition

Keep prediction and rationale outputs separate. Validate token indices after tokenization, especially padding and special tokens. Know the difference between human agreement and model faithfulness. If the task removes or retains rationale tokens, verify the resulting attention mask and batch shape before trusting the metric.
