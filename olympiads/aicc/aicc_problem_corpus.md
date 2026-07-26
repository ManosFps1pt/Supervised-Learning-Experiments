# AICC Problem Corpus

Reusable catalog of every task listed by AICC on 2026-07-24 (9 rounds, 27 tasks). This is a routing corpus, not a solution archive: use it to select a task locally, then invoke the `aicc-problem-importer` with the exact task name to create its runnable folder, download its available starter/baseline material, and document dataset access.

## How to use this corpus

- Filter by the **Type** or **IOAI syllabus coverage** text below before recommending non-official practice.
- Use the exact title in `python C:\\Users\\Manos\\.codex\\skills\\aicc-problem-importer\\scripts\\resolve_aicc_problem.py "<title>"` or ask Codex to import that title. Do not download a task until it is selected for study.
- Each **Prompt** is a faithful compact transcription of the official task objective, input/output contract, constraints, and principal metric. Follow the source link for the complete official statement.
- AICC is excellent IOAI-style practice but is not an official IOAI source. During the 2026 emergency sprint, follow `priority.md` before selecting a new AICC task.

## Round 0 — October 2025

### Deceptive Points

- **Type / difficulty:** Classical ML (robust tabular regression) / easy.
- **Prompt:** Train on four numeric features and corrupted target values, then predict the teacher-only true target for each test row. Submit `ID,Target`; lower MSE is better. Any model and preprocessing are allowed, but the hidden solution may not be used.
- **Develops:** baseline regression, detecting outliers/label corruption, validation strategy, robust model comparison, submission contracts.
- **IOAI syllabus coverage:** Supervised Learning—Linear Regression; Model Evaluation Metrics; Underfitting/Overfitting; Feature Engineering; Data Processing; scikit-learn.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/deceptive-points-aicc-round-0).

### Find Brain Tumors

- **Type / difficulty:** CV (semi-supervised image classification) / medium.
- **Prompt:** Classify brain images into no-tumor plus three tumor types when roughly 2% of training images are labelled. Submit `ID,prediction`; macro F1 is the metric.
- **Develops:** image loading and label joins, scarce-label baselines, CNN/transfer-learning decisions, augmentation, macro-F1 debugging.
- **IOAI syllabus coverage:** Computer Vision—Image Classification, Convolutional Layers, Pre-trained Vision Encoders, Image Augmentation; Neural Networks—Loss Functions, Model Finetuning; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/aicc-round-0-brain-tumor).

### Latent Model Classification

- **Type / difficulty:** ML (representation/model-forensics classification) / medium.
- **Prompt:** Given 100-D inputs, observed 5-D logits, and two supplied penultimate PyTorch modules, infer whether hidden model A or B generated each output. No source labels are provided for training; submit `ID,Source` and maximize accuracy.
- **Develops:** PyTorch weight loading, tensor shape contracts, embedding comparison, linear-head reasoning, unsupervised/weak-supervision diagnostics.
- **IOAI syllabus coverage:** PyTorch Basics; Tensor Manipulation; Data Embeddings; Neural Networks; Supervised Learning; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/latent-model-classification-aicc-round-0).

## Round 1 — November 2025

### The Defected Nuts

- **Type / difficulty:** CV (industrial anomaly segmentation) / hard.
- **Prompt:** From 431 clean 1024×1024 hazelnut images, produce a 0–255 pixel anomaly map for each of 70 defect images (cracks, cuts, holes, or contamination). Only ImageNet ResNet18 is permitted; Base85-encode masks in `submission.csv`; AUPRO is the metric.
- **Develops:** one-class/anomaly framing, image preprocessing, feature extraction, pixel masks, strict encoded-submission validation.
- **IOAI syllabus coverage:** Computer Vision—Image Segmentation, Pre-trained Vision Encoders, Image Augmentation; Data Processing; Model Evaluation Metrics; Autoencoders (useful optional baseline family).
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/the-defected-nuts-aicc-round-1-2).

### Autocorrect

- **Type / difficulty:** NLP (character-level sequence correction) / medium.
- **Prompt:** Learn to restore correct text from paired clean and misspelled training strings, then output a headerless corrected row for every test string. No pretrained models or test-set fitting; CPU inference for all 2,643 rows must finish within 250 seconds; lower character error rate is better.
- **Develops:** token/character vocabularies, sequence modelling, edit-distance metrics, runtime limits, leakage prevention.
- **IOAI syllabus coverage:** NLP—Language Modeling, Encoder-Decoder Models; Data Processing—tokenization/vocabulary building; Neural Networks—Embeddings, RNN/transformer-style sequence reasoning; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/autocorrect-aicc-round-1-2).

### Is that audio?

- **Type / difficulty:** Audio (closed-set language/audio classification) / legacy.
- **Prompt:** Predict an anonymized language class for each variable-rate, variable-duration WAV clip using only the supplied training audio and labels. Submit `ID,label` for all test clips.
- **Develops:** waveform inspection, resampling/padding, spectrogram features, audio-classification baseline, noisy-label checks.
- **IOAI syllabus coverage:** Audio Processing; Data Processing—padding and tokenization/features; Data Embeddings; Pre-trained Audio Encoders (optional comparison); Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/is-that-audio-aicc-round-1-2).

## Round 2 — December 2025

### Essay Gap

- **Type / difficulty:** NLP (contextual multiple-choice classification) / easy.
- **Prompt:** For each text gap, select the correct one of four candidate sentences using the surrounding `before` and `after` context. Submit `sampleID,answer` (option 0–3); macro F1 is used.
- **Develops:** text-pair construction, contextual text classification, label encoding, multiclass macro-F1, clean CSV output.
- **IOAI syllabus coverage:** NLP—Text Classification, Pre-trained Text Encoders, Transformers; Data Processing—tokenization; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/essay-gap-aicc-round-2).

### Audio Demixing

- **Type / difficulty:** Audio (source separation) / medium.
- **Prompt:** For every mixed 16,000-sample waveform, reconstruct the two original environmental waveforms. Submit Base85-encoded float32 `sig_1,sig_2`; evaluation is permutation-invariant MSE, so output order does not matter.
- **Develops:** waveform tensors, mixture/separation framing, reconstruction losses, permutation-invariant evaluation, encoded arrays.
- **IOAI syllabus coverage:** Audio Processing; Tensor Manipulation; Neural Networks—Loss Functions and Embeddings; Data Processing; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/audio-demixing-aicc-round-2).

### Face Matching

- **Type / difficulty:** CV (image retrieval / face matching) / easy.
- **Prompt:** For each of 15 reference photos, return all other images showing that celebrity. General pretrained models such as CLIP are allowed, but face-specific models/libraries and manual grouping are forbidden; submit pipe-separated matching IDs and optimize mean F1.
- **Develops:** CLIP image embeddings, similarity search, thresholds, retrieval evaluation, processor/tensor input contracts.
- **IOAI syllabus coverage:** Computer Vision—Pre-trained Vision Encoders; Vision-text Encoders (CLIP); Data Embeddings; Model Evaluation Metrics; PyTorch Basics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/face-matching-aicc-round-2).

## Round 3 — January 2026

### Drawn Apart

- **Type / difficulty:** CV (cross-domain image classification) / hard.
- **Prompt:** Classify unlabeled sketches using labelled photographs and cartoons of the same categories. Submit `filename,class_name`; maximize F1. Only torchvision pretrained models are permitted, with no external data or pseudo-labelling of evaluation sketches.
- **Develops:** domain shift, dataset structure, augmentation, supervised vision baselines, validation discipline.
- **IOAI syllabus coverage:** Computer Vision—Image Classification, Convolutional Layers, Pre-trained Vision Encoders, Image Augmentation; Data Processing; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/drawn-apart-aicc-round-3).

### Sound of Nature

- **Type / difficulty:** Audio (environmental sound classification) / easy.
- **Prompt:** Classify each five-second 44.1 kHz WAV into one of nine animal/nature sounds. Train on 800 labelled clips, predict 200 test IDs in `submission.csv`, and maximize macro F1.
- **Develops:** mel-spectrogram baseline, audio augmentation, class mapping, macro-F1 and confusion-matrix analysis.
- **IOAI syllabus coverage:** Audio Processing; Data Embeddings; Pre-trained Audio Encoders; Data Processing; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/sound-of-nature-aicc-round-3).

### Morty's Time Paradox

- **Type / difficulty:** Classical ML (cyclical-feature classification) / medium.
- **Prompt:** Use eight cyclical watch readings and four allowed candidate labels per training row to predict one of five timeline labels for test rows. Global class totals are known and test data may be used during development; submit `row_id,label`, scored by accuracy.
- **Develops:** circular features (sin/cos), constrained prediction, train/test-distribution reasoning, post-processing, accuracy validation.
- **IOAI syllabus coverage:** Supervised Learning; Feature Engineering; Data Processing; Model Evaluation Metrics; NumPy/Pandas.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/mortys-time-paradox-aicc-round-3).

## Round 4 — February 2026

### Sticky Note Blindness

- **Type / difficulty:** CV (robust CLIP classification) / easy.
- **Prompt:** Recover the true object class from images attacked by a misleading-text sticky note. Use only `openai/clip-vit-base-patch16` and supplied class embeddings; submit `sample_id,label`, scored by accuracy.
- **Develops:** CLIP zero-shot pipeline, multimodal failure analysis, image masking/cropping experiments, embedding similarity.
- **IOAI syllabus coverage:** Vision-text Encoders (CLIP); Pre-trained Vision Encoders; Data Embeddings; Image Preprocessing; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/sticky-note-blindness-aicc-round-4).

### Alchemy

- **Type / difficulty:** NLP (semantic matching under a bijection) / medium.
- **Prompt:** From 150 labelled `item1 + item2 → result` rules, assign each of 70 test pairs its unique result candidate. Only `bert-base-uncased` is allowed; submit `Id,result` and maximize accuracy.
- **Develops:** sentence/phrase embeddings, BERT usage, similarity matrices, one-to-one assignment constraints, accuracy checks.
- **IOAI syllabus coverage:** NLP—Pre-trained Text Encoders, Text Classification; Data Embeddings; Transformers; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/alchemy-aicc-round-4).

### Extreme Condensation

- **Type / difficulty:** ML (dataset distillation) / hard.
- **Prompt:** Design exactly one synthetic 28×28 image and one 10-way soft label. A fixed CNN is retrained from scratch on that lone example and judged on hidden MNIST digits; submit its 784 pixels and ten probabilities, maximizing accuracy.
- **Develops:** soft targets, optimization through a training loop, CNN behaviour, data compression/distillation, submission precision.
- **IOAI syllabus coverage:** Neural Networks—Convolutional Layers, Loss Functions, Gradient Descent, SGD, Softmax-style labels; Tensor Manipulation; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/extreme-condensation-aicc-round-4).

## Round 5 — March 2026

### Watermark Removal

- **Type / difficulty:** CV (image restoration / diffusion) / easy.
- **Prompt:** Learn from 2,000 clean/watermarked 64×64 bird-image pairs to reconstruct clean versions of 200 watermarked test images. Base85-encode each reconstruction in a three-column CSV; lower pixel MSE is better.
- **Develops:** paired image-to-image datasets, reconstruction losses, diffusion/restoration framing, image normalization, encoded image output.
- **IOAI syllabus coverage:** Computer Vision—Diffusion Models; Generating Images with GANs; Convolutional Layers; Image Augmentation; Neural Networks—Loss Functions; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/watermark-removal-aicc-round-5).

### Lost Interpreter

- **Type / difficulty:** ML / program inference (sequence reasoning) / easy.
- **Prompt:** Infer the behaviour of an unknown single-variable assembly-like language from example programs and outputs, then predict the final output of each test program. No pretrained models; submit `ID,output`, minimizing MAE.
- **Develops:** program parsing hypotheses, feature engineering, controlled experiments, sequence/control-flow reasoning, regression metrics.
- **IOAI syllabus coverage:** Programming Fundamentals; Feature Engineering; Supervised Learning; Neural Networks—sequence reasoning (optional); Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/lost-interpreter-aicc-round-5).

### Visual Relations

- **Type / difficulty:** CV (scene-graph / relationship detection) / medium.
- **Prompt:** For each test image, rank subject-box, predicate, object-box relationship triplets from 35 predicates. Only DETR ResNet-50 is permitted; serialize up to 100 predictions per image in `PredictionString`; score is mean AP with IoU ≥ 0.5 for both boxes.
- **Develops:** DETR inference, bounding boxes, relation features, confidence ranking, mAP and strict submission formatting.
- **IOAI syllabus coverage:** Computer Vision—Object Detection, Pre-trained Vision Encoders; Data Embeddings; Model Evaluation Metrics; Tensor Manipulation.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/visual-relations-aicc-round-5).

## Round 6 — April 2026

### Classifier Classifier

- **Type / difficulty:** CV / ML (model classification) / medium.
- **Prompt:** Identify the target digit class of 200 unlabeled small CNNs from their architecture seeds and weights, using 50 labelled CNNs as references. Rebuild the supplied architectures, submit `id,class`, and maximize macro F1; no original data, external data, or pretrained models are allowed.
- **Develops:** PyTorch state dictionaries, model reconstruction, probing trained networks, batch tensor debugging, multiclass F1.
- **IOAI syllabus coverage:** PyTorch Basics; Neural Networks—Convolutional Layers; Tensor Manipulation; Data Embeddings/representations; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/classifier-classifier-aicc-round-6).

### Massive Problem

- **Type / difficulty:** Classical ML (large high-dimensional tabular classification) / medium.
- **Prompt:** Classify cells into one of 12 types from 1,434 gene-expression values per row. Submit `id,label`; macro F1 is scored and memory management matters because the dataset is large.
- **Develops:** memory-aware pandas/NumPy use, preprocessing/scaling, dimensionality handling, baseline classifiers, macro-F1 analysis.
- **IOAI syllabus coverage:** NumPy and Pandas; Supervised Learning; K-NN/Logistic Regression/Model Ensembles; PCA; Feature Engineering; Data Processing; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/massive-problem-aicc-round-6).

### Nuclei Reconstruction

- **Type / difficulty:** CV (medical-image segmentation) / medium.
- **Prompt:** Produce a same-size binary nuclei mask for each H&E histopathology test image, learning from RGB/mask pairs. Submit Base85 PNG masks in a three-column CSV; mean foreground IoU is the metric.
- **Develops:** segmentation data pairing, stain/color preprocessing, U-Net-style baseline selection, resizing without corrupting masks, IoU evaluation.
- **IOAI syllabus coverage:** Computer Vision—Image Segmentation (U-Net), Convolutional Layers, Image Augmentation; Data Processing; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/nuclei-reconstruction-aicc-round-6).

## Round 7 — May 2026

### Polarity

- **Type / difficulty:** NLP (lexical semantic classification) / medium.
- **Prompt:** Predict whether each unseen word pair is synonymous (0) or antonymous (1) from 50 labelled examples. Only `bert-large-uncased` may be downloaded; no external lexicon/corpus; submit `row_id,label`, scored by macro F1.
- **Develops:** pretrained BERT use, lexical-semantic embeddings, few-shot validation, binary macro-F1, restriction checking.
- **IOAI syllabus coverage:** NLP—Text Classification, Pre-trained Text Encoders, Transformers; Data Embeddings; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/polarity-aicc-round-7).

### Oriented Ship

- **Type / difficulty:** CV (oriented object detection) / hard.
- **Prompt:** Detect ships in aerial imagery with rotated boxes `(cx,cy,w,h,theta)`. Standard ImageNet backbones are allowed but no maritime data or oriented-detection pretraining; submit each image's confidence-ranked boxes and score mAP@0.5 with rotated IoU.
- **Develops:** detection datasets, coordinate normalization, rotated geometry, pretrained backbones, detector evaluation and submission syntax.
- **IOAI syllabus coverage:** Computer Vision—Object Detection, Pre-trained Vision Encoders, Image Augmentation; Tensor Manipulation; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/oriented-ship-aicc-round-7).

### Scientific Facts

- **Type / difficulty:** NLP (scientific natural-language inference) / medium.
- **Prompt:** Given a scientific claim and evidence from abstracts, classify it as `SUPPORT`, `CONTRADICT`, or `NOT_ENOUGH_INFO`. Only SciBERT is allowed; submit `datapointID,answer`, scored by macro F1.
- **Develops:** claim-evidence pair encoding, transformer fine-tuning/inference, class imbalance checks, NLI error analysis.
- **IOAI syllabus coverage:** NLP—Text Classification, Pre-trained Text Encoders, Transformers; Model Finetuning; Data Processing; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/scientific-facts-aicc-round-7).

## Round 8 — June 2026 (Nitro Judge)

### Dice Counting

- **Type / difficulty:** CV (visual counting and regression) / hard.
- **Prompt:** From 200 labelled high-resolution dice photos, predict both the number of dice and the sum of up-facing pips for each of 50 deliberately out-of-distribution crowded test photos. Pretrained weights and external images are forbidden; submit both answers per image and minimize capped MAE.
- **Develops:** visual inspection, count/regression heads, out-of-distribution validation, augmentation, multi-output submission validation.
- **IOAI syllabus coverage:** Computer Vision—Image Classification, Convolutional Layers, Image Augmentation; Supervised Learning—Regression; Data Processing; Model Evaluation Metrics.
- **Official source:** [Nitro Judge](https://judge.nitro-ai.org/competitions/aicc/aicc-round-8/1/view).

### Pixel Quest

- **Type / difficulty:** RL + CV (visual odometry and pixel navigation) / medium.
- **Prompt:** (1) From paired 64×64 camera frames, predict the robot's turn; (2) train an agent in `PixelNav-v0` to reach a target from pixel observations, then submit action sequences for 100 hidden episodes. Heading accuracy contributes 40 points and mean episode reward contributes 60.
- **Develops:** Gymnasium API, state/action/reward loops, visual observations, RL rollout validation, action-sequence submission formats.
- **IOAI syllabus coverage:** Reinforcement Learning—states/actions/rewards, MDPs, policy evaluation/improvement, Q-learning/TD learning, exploration; Computer Vision—Convolutional Layers; PyTorch/Tensor Manipulation.
- **Official source:** [Nitro Judge](https://judge.nitro-ai.org/competitions/aicc/aicc-round-8/2/view).

### Archivist's Cards

- **Type / difficulty:** NLP + CV (vision-language retrieval) / medium.
- **Prompt:** For each of 700 long region-description cards, rank the ten most likely matching photos from a 1,500-image gallery. Use only CLIP ViT-B/16 image/text encoders; submit 10 IDs per card, scored by mean reciprocal-rank-squared.
- **Develops:** CLIP text/image embeddings, long-text aggregation, cross-modal retrieval, cosine similarity/ranking, retrieval-metric debugging.
- **IOAI syllabus coverage:** Vision-text Encoders (CLIP); Pre-trained Vision and Text Encoders; Data Embeddings; Transformers; Model Evaluation Metrics.
- **Official source:** [Nitro Judge](https://judge.nitro-ai.org/competitions/aicc/aicc-round-8/3/view).

## Round 9 — July 2026 Surprise Round

### Word Lookups

- **Type / difficulty:** NLP (Chinese word segmentation) / unknown.
- **Prompt:** Given Mandarin character sequences, predict one BMES tag per character for word-boundary segmentation. Submit `id,bio_tags`; boundary F1 is scored. Pretrained models, pretrained embeddings, external dictionaries, and manually labelled external data are forbidden.
- **Develops:** sequence tagging, token/character preprocessing, output-length validation, no-pretrained constraint handling, boundary-F1 error analysis.
- **IOAI syllabus coverage:** NLP Text Classification / sequence labelling; Language Modeling; Data Processing tokenization/vocabulary building; Model Evaluation Metrics.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/aicc-round-9-word-lookups).

### Buried Fault

- **Type / difficulty:** Classical ML / time-series sensor classification and localization / unknown.
- **Prompt:** From six-channel vibration recordings shaped `(6, 2048)` with missing values, predict the fault label and the hidden event interval for each test recording. Submit `recording_id,label,start,end`; score is `0.5 * macro_F1 + 0.5 * mean_IoU`.
- **Develops:** NaN handling, time-series feature engineering, weak localization, machine/site shift validation, interval submission checks.
- **IOAI syllabus coverage:** Supervised Learning; Feature Engineering; Data Processing; Model Evaluation Metrics; NumPy/Pandas; scikit-learn.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/buried-fault-aicc-round-9).

### Shuffled

- **Type / difficulty:** CV + NLP (CLIP positional embedding recovery) / unknown.
- **Prompt:** Given a CLIP ViT-B/16 model whose vision and text positional embedding rows were shuffled, recover the original position of every row using six anchors and optional matched image-caption pairs. Submit `row_id,position`; exact-position accuracy is scored.
- **Develops:** pretrained model inspection, tensor-shape checks, positional embedding reasoning, anchor constraints, exact-submission validation.
- **IOAI syllabus coverage:** Vision-text Encoders (CLIP); Pre-trained Vision Encoders; Pre-trained Text Encoders; Data Embeddings; Tensor Manipulation; Transformers.
- **Official source:** [Kaggle](https://www.kaggle.com/competitions/shuffled-aicc-round-9).

## Catalog integrity

- **Catalog source:** [AICC contests](https://aicc-official.org/contests), previously verified 2026-07-24 for 9 contests and 27 tasks; Round 9 surprise tasks added from the 2026-07-26 announcement and Kaggle task pages.
- **Prompt/evaluation sources:** each linked Kaggle competition or Nitro Judge task page; Round 9 Kaggle pages fetched through the Kaggle CLI on 2026-07-26.
- **Refresh rule:** re-open the AICC contests page before asserting that this is still complete; AICC is live and may change.
