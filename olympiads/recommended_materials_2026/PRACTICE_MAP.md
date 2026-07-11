# Most Relevant Existing Competition Exercises

These are ranked for the organizer email and the current weakness: adapting unfamiliar model APIs under time pressure.

## 1. Hungarian 2026 - Model Extension

English path: `../competition_samples/raw/hungary-haio-sparse/2026/nyari-tabor/feladatok/modellbovites_translated_en.ipynb`

Original path: `../competition_samples/raw/hungary-haio-sparse/2026/nyari-tabor/feladatok/modellbovites.ipynb`

Why first: it directly matches catastrophic forgetting. You receive an ImageNet-pretrained ResNet18 classifier for 30 classes and must extend it to 55 classes using small old/new datasets and partly unlabeled data. It trains checkpoint loading, classifier-head replacement, freezing, fine-tuning, replay/pseudo-labeling, old/new validation, and a 1,100-row submission.

Timebox: 2.5 hours. Evidence: loaded checkpoint, printed old head shape, expanded head shape, one successful batch, separate old/new accuracy, and format-valid `submission.csv`. Do not open `megoldasok/modellbovites-megoldas.ipynb` until the timebox ends.

## 2. NEOAI 2025 - Broken BERT

Path: `../competition_samples/raw/neoai-2025-sparse/5_Broken_BERT/broken_bert_solution_translated_en.ipynb`

Why second: this is the exact unknown-Transformer-interface drill. Only token embeddings are damaged; attention blocks and the classifier must remain frozen. It forces you to use `AutoTokenizer`, inspect the model hierarchy, locate the embedding matrix, control `requires_grad`, batch token IDs and preserve the rest of the checkpoint.

Timebox: 90 minutes. Treat the translated notebook as an exercise: read through model loading, then stop before its repair logic. Evidence: tokenizer keys/shapes, model output keys/shapes, the exact embedding parameter name, trainable-parameter count, and before/after validation metric.

## 3. Hungarian 2026 - Blind Curator

Path: `../competition_samples/raw/hungary-haio-sparse/2026/nyari-tabor/feladatok/vak-kurator.ipynb`

Why third: it uses supplied DINOv2 embeddings rather than requiring DINOv2 training. You select 300 of 15,000 images so a prototype classifier built from the selected samples performs well. It trains embedding normalization, cosine similarity, prototypes, clustering/diversity, array indexing and exact CSV validation.

Timebox: 2 hours. Start with `pool_emb.npy`, `seed_emb.npy` and `seed_labels.npy`; thumbnails are optional. Evidence: shape checks, normalized-vector check, selection strategy, 300 unique indices and `submission.csv`.

## 4. IOAI 2026 Home Task 1 - Operation Night Watch

Path: `../competition_samples/raw/IOAI-2026-sparse/Home Task/Home-Task-1.ipynb`

Why: it is the closest direct match to the EWC paper and the user's API concern. It supplies `ASTFeatureExtractor` and `ASTForAudioClassification` and asks for 16 old plus 13 new classes without forgetting. The full dataset is large, so use this primarily as a 60-minute interface-reading drill unless it is already downloaded.

Evidence: processor output shape, classifier head location, 16-to-29 expansion plan, frozen/trainable parameter list, and old/new validation design.

## 5. Polish OAI 2025 - Hallucination Detection

Path: `../competition_samples/raw/polish-oai-2025-sparse/1_etap/2_wykrywanie_halucynacji/2_wykrywanie_halucynacji_translated_en.ipynb`

Why: it is a compact odd-schema NLP task with questions, answers, alternative generations, token/probability fields, ROC-AUC and a serialized callable. It trains the contest reflex of inspecting nested data and producing the exact evaluator interface.

Timebox: 90 minutes. Evidence: schema table, baseline ROC-AUC, three error examples, and saved callable contract.

## 6. Polish OAI 2025 - Borrowing / Counterfactual Explanations

Path: `../competition_samples/raw/polish-oai-2025-sparse/2_etap/kredytobranie/kredytobranie.ipynb`

Why: it turns model explanation into an optimization problem: change rejected samples enough to cross the classifier boundary while keeping changes small and realistic under a density model. This is closer to explainability than merely plotting SHAP values.

Timebox: 90 minutes if its sparse data assets are available. Evidence: validity, plausibility, average distance and a plotted before/after example.

## 7. Polish OAI 2024 - Pruning

Path: `../competition_samples/raw/polish-oai-2024-sparse/first_stage/pruning/pruning_translated_en.ipynb`

Why: it complements the watermarking paper by requiring direct parameter manipulation while preserving predictive quality. The score combines sparsity and MSE, so it trains layer/weight inspection and metric tradeoffs.

Timebox: 75 minutes. Evidence: before/after nonzero counts, MSE, score, and saved `model_parameters.pkl`.

## Statement-only fallback

`../competition_samples/raw/roai-2026-selection-camp-cpu-practical/too_easy_fairy.md` is the most direct DINOv2 patch-segmentation statement, but its data are not mirrored locally. Use Blind Curator first because it has a notebook with download hooks and a concrete submission route.
