# CEOAI Task Map

This map is CEOAI-first. IOAI-only audio is intentionally not prioritized.

## 1. Reinforcement Learning and AI Search

Strong direct samples are rarer than ML/CV/NLP, so prioritize every usable RL/search task.

- `romania_markov_maze`: MDP/RL notebook from `roai-solved`; maps to CEOAI `1(d)`, `1(e)`, `1(f)`.
- `ceoai_2026_practice1_stochastic_rift`: official CEOAI Practice Round 1 offline stochastic MDP task; maps to CEOAI `1(d)`, `1(e)`, `1(f)`.
- `roai_2026_smart_warehouse`: Nitro ROAI selection-camp CPU practical task; TD(0), Q-learning, SARSA, tabular policy training, and hidden-scenario evaluation; maps to CEOAI `1(d)`, `1(e)`, `1(f)`.
- `hungary_haio_repo`: includes AI olympiad tasks across ML/CV/NLP/RL; inspect the year folders for explicit RL/search tasks.
- Existing local repo artifacts still matter here: your A*, minimax, MDP, Q-learning, and `search_rl_comparison.md` files remain the primary CEOAI Section 1 base.

Expected CEOAI inference: if Search/RL appears, it is more likely to be a compact gridworld, MDP, Q-table, minimax, A*, or recognition/comparison task than a large deep-RL task.

## 2. Machine Learning

Highest-value samples:

- `ioai_2024_help_bobai`: official IOAI tabular/feature-engineering task.
- `ioai_2025_radar` and `ioai_2025_at_home_radar`: official ML workflow tasks.
- `ceoai_2026_practice1_star_observatory`: official CEOAI Practice Round 1 image-plus-regression task; maps to CEOAI `2(a)`.
- `ceoai_2026_practice1_project_kraken`: official CEOAI Practice Round 1 multimodal regression/classification task; maps to CEOAI `2(a)`.
- `kazakhstan_day2_player_clustering`: clustering football players; direct CEOAI `2(b)`.
- `ceoai_2026_practice2_trace_twins`: official CEOAI/EUROAI Practice Round 2 sequence-similarity task; maps to CEOAI `2(d)` through feature design and validation.
- `roai_2026_polyglot`: embedding-space alignment/permutation recovery; secondary fit for CEOAI `2(b)` because it uses similarity geometry and assignment-style matching.
- `neoai_tricy_table`: tabular feature engineering; Kaggle link only.
- `romania_onia_examples`: tabular train/eval task with local data and notebooks.
- `poland_2024_imbalanced_classification`: metric and class-imbalance practice.
- `poland_2025_noisy_labels`: robust training under noisy labels.
- `poland_2025_borrowing`: credit-scoring classifier explanation.
- `china_2024_basketball`: tabular prediction.

Expected CEOAI inference: expect train/test splits, baseline classifiers, metric direction, feature engineering, clustering, dimensionality reduction, and submission-like outputs.

## 3. Deep Learning

Highest-value samples:

- `poland_2024_pruning`: model compression and network-weight reasoning.
- `ceoai_2026_practice2_panda_mnist`: official CEOAI/EUROAI Practice Round 2 small-model CV task with parameter-count penalty; maps to CEOAI `3(b)`, `3(c)`.
- `ceoai_2026_practice1_project_kraken`: official CEOAI Practice Round 1 multimodal DL task; maps to CEOAI `3(c)`.
- `ceoai_2026_practice1_star_observatory`: official CEOAI Practice Round 1 image-feature regression task; maps to CEOAI `3(c)`.
- `roai_2026_too_easy_fairy`: one-shot segmentation from DINOv2 patch features; maps to CEOAI `3(c)` as frozen-model feature use and representation routing.
- `poland_2025_unlearning`: advanced model behavior/editing.
- `neoai_underfitting_cv`: optimizer/regularization debugging in a CV setting.
- `ioai_2025_pixel`: optimization-under-constraints task.
- `poland_2025_ecg_anomaly`: time-series anomaly detection with model selection.
- `poland_2025_data_prototypes`: representation/sample-selection reasoning.

Expected CEOAI inference: expect optimizer choice, overfitting/underfitting diagnosis, dropout/weight decay/lr schedule recognition, and architecture routing more than writing a deep architecture from scratch.

## 4. Natural Language Processing

Highest-value samples:

- `ioai_2025_concepts`: official IOAI NLP/LLM/embedding workflow.
- `ceoai_2026_practice2_trace_twins`: official CEOAI/EUROAI Practice Round 2 transcript-similarity task; maps to CEOAI `4(a)`, `4(b)`.
- `roai_2026_polyglot`: Nitro ROAI selection-camp embedding alignment task; maps to CEOAI `4(b)`, `4(c)`.
- `ioai_2024_lost_hyperspace`: official IOAI NLP adaptation task.
- `poland_2025_hallucination`: hallucination detection.
- `poland_2025_source_extraction`: retrieval/embedding alignment.
- `kazakhstan_day3_code_difficulty`: text/code classification.
- `kazakhstan_day4_masked_word`: masked-word position recovery.
- `neoai_broken_bert`: BERT/embedding repair.
- `neoai_intent_slot`: intent detection and slot filling.
- `china_2024_news_text`: text classification.
- `georgia_2025_dedup_bert`: sentence-pair/BERT deduplication.

Expected CEOAI inference: expect tokenization, TF-IDF/embedding baselines, BERT-style encoders, retrieval, classification, and concise output formatting.

## 5. Computer Vision

Highest-value samples:

- `ioai_2025_chicken_counting`: official CV counting workflow.
- `ceoai_2026_practice2_panda_mnist`: official CEOAI/EUROAI Practice Round 2 scanner/domain-shift digit classification task; maps to CEOAI `5(a)`.
- `ceoai_2026_practice1_star_observatory`: official CEOAI Practice Round 1 star-center and flux task; maps to CEOAI `5(a)`.
- `ceoai_2026_practice1_project_kraken`: official CEOAI Practice Round 1 multimodal task with image slices; maps to CEOAI `5(a)`.
- `roai_2026_too_easy_fairy`: Nitro ROAI selection-camp one-shot segmentation task from DINOv2 features; maps to CEOAI `5(a)`, `5(c)`.
- `ioai_2025_restroom`: visual matching/embedding task.
- `ioai_2025_antique`: image authentication/classification.
- `kazakhstan_day1_image_restoration`: image restoration.
- `poland_2025_coin_counting`: detector/counting style task.
- `poland_2024_color_quantization`: K-means/CV fundamentals.
- `poland_2024_anomaly_detection`: OOD image detection.
- `poland_2025_non_normal_distribution`: denoising/reconstruction.
- `china_2024_real_fake_image`: generated-image detection.
- `neoai_cluster_pictures`: unsupervised image clustering.

Expected CEOAI inference: expect preprocessing, CNN/encoder use, transfer learning, feature extraction, image classification/restoration, and metric-driven improvement.

## Lower Priority or Defer

- IOAI audio tasks: `ioai_2025_gaite_speech_detector`, `poland_2026_whisper_or_shout`.
- Large generative media tasks: `ioai_2024_practical_zip`, `ioai_2024_madarian_cow`, `ioai_2025_at_home_chameleon`, `neoai_hogspell`.
- Login-gated Kaggle/Bohrium tasks: keep them as links until a concrete practice block needs them.
