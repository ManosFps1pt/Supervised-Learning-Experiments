# SST-2 Dataset

- Source: https://huggingface.co/datasets/stanfordnlp/sst2
- Local card: `dataset_card.md`
- Local data: `train.parquet`, `validation.parquet`, `test.parquet`
- Download manifest: `parquet_manifest.json`
- Competition exercise: `sst2_competition_exercise.ipynb`

## What the source actually says

SST-2 is an English binary sentiment-classification dataset derived from movie-review text. The Hugging Face version has roughly 70,000 rows: about 67,300 train rows, 872 validation rows, and 1,820 test rows. Its relevant columns are `idx`, `sentence`, and `label`. Labels are negative or positive; the public test split may not provide usable gold labels for ordinary evaluation.

The dataset is deliberately simple. Its value here is not sentiment theory but the complete NLP plumbing: loading text, selecting the correct column, tokenizing batches, padding and truncating consistently, pairing labels with encodings, evaluating accuracy or F1, and keeping train/validation/test contracts separate.

All three parquet splits are stored locally so the data can be inspected without downloading it during a timed drill. The model weights are not mirrored; use the checkpoint name from the adjacent DistilBERT folder when online model access is available.

## CEOAI syllabus mapping

- `4(a) Preprocessing`: tokenization, padding, truncation, batching.
- `4(b) Embeddings`: transformer inputs and contextual representations.
- `4(c) Related architectures`: encoder-based text classification.
- `2(a) Classification`: binary labels, validation metrics, and error analysis.

## What to retain for competition

First inspect column names, split sizes, label values and missing labels. Tokenize a two-row batch and print every key and shape. Only then create the full dataset pipeline. A valid baseline artifact is a table containing text, true label, predicted label and confidence for ten validation examples plus one aggregate metric.
