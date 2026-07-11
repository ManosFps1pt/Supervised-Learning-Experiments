# Three-Day CEOAI Coding-Fluency Sprint

Current date: 2026-07-10. CEOAI begins 2026-07-14. Effective study days: three, excluding 2026-07-13.

## Diagnosis

Theory is not the limiting factor. The limiting factor is the delay between seeing an unfamiliar object and proving its input/output contract. The solution is not another broad Python or Transformers course. It is repeated, timed model-interface discovery followed by one valid artifact.

Use `MODEL_API_SURVIVAL.md` on every task. No practice block counts unless it records input/output shapes, trainable parameters, a metric and a submission or saved-model contract.

## Day 1 - Pretrained model adaptation

1. Read the local catastrophic-forgetting and watermarking summaries: 35 minutes total.
2. Attempt English `modellbovites_translated_en.ipynb`: 2.5 hours, closed-book except documentation and the API survival protocol.
3. Attempt Broken BERT: 90 minutes. Stop before reading its repair section.
4. Spend 30 minutes writing one page of reusable facts discovered today: checkpoint format, head path, tokenizer keys, output keys, freeze/unfreeze rule and save/reload method.

Required artifacts: one successful forward/backward probe for each supplied model, separate old/new metrics for model extension, and the exact Broken BERT embedding parameter name plus trainable-parameter count.

## Day 2 - Frozen features and explanations

1. Read the DINOv2 and Integrated Gradients summaries: 30 minutes.
2. Attempt Blind Curator: 2 hours. Produce 300 unique indices and validate the CSV.
3. Implement Integrated Gradients once on a model you already have working: 75 minutes. The artifact must include the baseline, selected class score, attribution shape and completeness residual.
4. Attempt either Borrowing or Pruning: 90 minutes. Prefer Borrowing if its data are available; otherwise use Pruning.

Required artifacts: a valid `submission.csv`, one attribution visualization/table with its numerical completeness check, and one before/after explanation or sparsity metric.

## Day 3 - NLP contract plus full mock

1. Read the DistilBERT, SST-2 and ERASER summaries: 30 minutes.
2. Use the local SST-2 parquet files for a 90-minute DistilBERT pipeline drill. End with tokenizer keys/shapes, logits shape, one validation metric and ten inspected predictions. Do not tune heavily.
3. Attempt Hallucination Detection: 90 minutes. End with ROC-AUC, three errors and the serialized callable contract.
4. Complete the overdue official CEOAI Star Observatory notebook under a strict 2.5-hour mock window. This preserves full-pipeline practice: preprocessing, regression, metric checks and a 600-row submission.
5. Final 45 minutes: run all saved submission validators, confirm environments and condense `MODEL_API_SURVIVAL.md` into a handwritten or memorized checklist.

## Competition execution rule

When a random pretrained model appears, spend the first 15 minutes probing rather than coding the final solution. A tiny working forward pass is progress; a half-written training loop against an assumed API is not. Use the provided processor/tokenizer, pass dictionaries by name, inspect outputs, and only then build batching and training.

## What not to do

Do not read all six papers end to end. Do not learn a new framework. Do not memorize model-specific class names. Do not start long training before a one-batch overfit/probe and a save/reload test. Do not spend more than one timebox rescuing a dataset download; move to the next local task and return later.

## Immediate next artifact

Open `olympiads/competition_samples/raw/hungary-haio-sparse/2026/nyari-tabor/feladatok/modellbovites_translated_en.ipynb` and produce a recorded 30-to-55-class head expansion plus one successful batch while preserving a separate old-class metric.
