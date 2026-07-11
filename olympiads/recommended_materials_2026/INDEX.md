# EUROAI/CEOAI Supplementary Materials - Local Index

Source email: `../email.txt`

This folder mirrors every source from the organizer email. The six papers have the original PDF, extracted text and a competition-focused summary. The Hugging Face entries contain their cards/configuration; SST-2 also includes all three parquet splits. Model weights are intentionally not mirrored.

## Sources and syllabus map

| # | Source | Main operational idea | Closest CEOAI syllabus | Three-day priority |
|---|---|---|---|---|
| 1 | [Catastrophic forgetting](01_catastrophic_forgetting/README.md) | Protect old behavior while adapting a supplied model | `3(a-b-c)`, secondary `1(d-f)` | Read summary now |
| 2 | [Model watermarking](02_model_watermarking/README.md) | Add a differentiable constraint over model weights | `3(a-b-c)`, `5(b)` | Read summary; keep as task pattern |
| 3 | [Integrated Gradients](03_integrated_gradients/README.md) | Attribute a class score to inputs through path gradients | `3(a-c)`, `4`, `5` | Read summary and implement once |
| 4 | [ERASER](04_eraser_rationalized_nlp/README.md) | Predict labels plus token/sentence rationales; evaluate faithfulness | `4(a-b-c)`, `3(c)` | Read summary; practice masking |
| 5 | [Explainability disagreement](05_explainability_disagreement/README.md) | Different explainers answer different questions | closest to `2`, `3`, `4`, `5` | Read summary only |
| 6 | [DistilBERT SST-2 model](06_distilbert_sst2/README.md) | Tokenizer dictionary -> model -> logits | `3(c)`, `4(a-b-c)` | Highest API-fluency drill |
| 7 | [SST-2 dataset](07_sst2_dataset/README.md) | Binary text-classification pipeline and metrics | `2(a)`, `4(a-b-c)` | Use local parquet immediately |
| 8 | [DINOv2](08_dinov2/README.md) | Frozen image/patch embeddings for downstream tasks | `2(a-b-d)`, `3(c)`, `5(a-c)` | Highest CV feature drill |

## What the email is signaling

The list clusters into four likely competition patterns:

1. Modify a supplied pretrained model without destroying existing behavior.
2. Manipulate or inspect weights under a second constraint such as watermark recovery or sparsity.
3. Explain predictions with gradients, feature masks or rationales and evaluate the explanation separately from accuracy.
4. Use unfamiliar pretrained NLP/CV interfaces quickly: tokenizer/processor, model inputs, structured outputs, frozen features and small heads.

That is a much narrower target than "learn more theory." The sprint should train interface discovery and end-to-end artifact production.

## Reading rule

Read the eight local `README.md` summaries. Open the papers only to resolve a specific implementation question. Reading all 126 paper pages would consume the time needed to build the actual competition reflex.
