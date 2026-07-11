# DistilBERT Fine-Tuned on SST-2

- Source: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
- Local card: `model_card.md`
- Local configuration: `config.json`
- Local tokenizer configuration: `tokenizer_config.json`

## What the source actually says

This is a 67M-parameter DistilBERT sequence-classification checkpoint fine-tuned for English binary sentiment classification on SST-2. Labels are `0 = NEGATIVE` and `1 = POSITIVE`. The card reports about 91.1% validation accuracy and lists fine-tuning settings of learning rate `1e-5`, batch size `32`, warmup `600`, maximum sequence length `128`, and three epochs.

The normal interface is a tokenizer plus a sequence-classification model. The tokenizer turns strings into a dictionary containing token IDs and an attention mask. That dictionary is passed to the model as named arguments. The relevant output is `logits`, with shape `(batch_size, 2)`; `argmax` gives the class ID and `model.config.id2label` maps IDs to names.

The local config makes the architecture concrete: six Transformer layers, hidden size 768, twelve attention heads, intermediate size 3072, vocabulary size 30,522, maximum position embeddings 512, and a sequence-classification head. The model card also warns that sentiment predictions can reflect demographic or geographic biases unrelated to sentiment.

## CEOAI syllabus mapping

- `3(c) Architectures`: Transformers and BERT.
- `4(a) Preprocessing`: tokenization, truncation, padding, attention masks.
- `4(b) Embeddings`: token and contextual embeddings.
- `4(c) Related architectures`: pretrained language-model use and fine-tuning.

## What to retain for competition

Do not memorize every DistilBERT class. Memorize the contract: texts -> tokenizer dictionary -> move every tensor to the model device -> `model(**batch)` -> inspect `outputs.keys()` and `logits.shape`. Before training, inspect `model.config`, label mappings, tokenizer maximum length, and which parameters require gradients. Verify one example end-to-end before building a DataLoader.
