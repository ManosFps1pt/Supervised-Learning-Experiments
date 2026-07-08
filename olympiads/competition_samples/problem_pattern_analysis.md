# Competition Problem Pattern Analysis for CEOAI Prep

Created: 2026-07-05

This summary is based on the local archive in `olympiads/competition_samples/`:

- `source_index.csv`: 91 indexed tasks/sources.
- Raw local corpus scanned: 555 files from downloaded public sources.
  - 371 notebooks
  - 149 markdown files
  - 32 Python files
- Main sources: official IOAI 2024/2025, Polish OAI, Romania ONIA/ROAI, Kazakhstan IOAI TST, NEOAI, Hungary HAIO, Malaysia/China IOAI training tasks, and indexed-only Kaggle/Bohrium/USA overview sources.

Important: CEOAI has no past contest yet. Anything about "likely CEOAI tasks" below is an inference from the CEOAI syllabus plus IOAI/regional/national task patterns, not an official prediction.

## Executive Verdict

Most tasks are practical modeling tasks, not pure theory questions.

The repeated contest contract is:

1. Load provided train/test/validation data.
2. Understand the target and metric.
3. Build a baseline quickly.
4. Improve with preprocessing, features, pretrained encoders, or model tuning.
5. Produce a correctly formatted submission file or notebook output.
6. Avoid invalid shapes, invalid file names, leakage, and metric-direction mistakes.

For CEOAI prep, your highest-yield habit is not memorizing model internals. It is moving from unfamiliar data to a valid baseline submission fast.

## What Problems Typically Ask For

### 1. Valid Submission Under Strict Format

This is the strongest pattern.

Observed across IOAI 2025, Kazakhstan TST, Polish OAI, Romania ONIA/ROAI, and Georgia-style statements inside the archive.

Typical requirements:

- Create `submission.csv`, `submission.npz`, `submission.zip`, JSONL, or a required notebook.
- Match exact row count, column names, array names, and shapes.
- Use `sample_submission.csv` or an evaluator script.
- Invalid shape or missing file often means zero score.

Concrete examples:

- IOAI 2025 Chicken Counting requires `submission.npz` with arrays `pred_a` and `pred_b` shaped like density maps.
- IOAI 2025 Concepts requires a `submission.zip` containing `clues_a.jsonl` and `clues_b.jsonl`.
- Kazakhstan Day 2 creates a clustering `submission.csv`.
- Romania ONIA examples include train/eval CSVs and an evaluator workflow.

Preparation implication:

- Every practice block must end with a saved output file or checked output shape.
- You should always inspect `sample_submission`, required columns, row count, and metric before model tuning.

### 2. Metric-Driven Modeling

The scan found explicit metric/scoring language in 382 files.

Common metrics:

- Accuracy
- F1 / weighted F1
- AUC
- Mean relative error
- IoU
- PSNR
- RMSE / MAE / MSE
- Silhouette score or clustering-specific scores
- Ranking metrics such as Hits@10 and NDCG@10

Typical twist:

- The public metric may not be ordinary accuracy.
- Some tasks reward threshold crossing, compression, ranking quality, or relative improvement.
- Some tasks weight rare classes or non-background pixels more heavily.

Concrete examples:

- Kazakhstan Day 3 code difficulty uses F1.
- Kazakhstan Day 4 masked-word position uses accuracy.
- Kazakhstan Day 1 image restoration reports PSNR.
- IOAI 2025 Chicken Counting uses mean relative error transformed into a score.
- IOAI 2025 Concepts uses ranking-style clue quality metrics.
- Georgia-style statements in the ROAI solved archive include weighted F1 and score-scaling formulas.

Preparation implication:

- Before training, write down: "What does the metric reward?"
- For classification, always check whether `.predict()` or `.predict_proba()` is needed.
- For imbalanced tasks, accuracy is often the wrong first metric.

### 3. Baseline First, Then Improvement

The word/pattern scan found baseline/starter/reference language in 198 files.

Tasks often provide:

- A starter notebook.
- A baseline model.
- Partial pretrained weights.
- An evaluator.
- A small train/validation split.
- Environment requirements.

Concrete examples:

- IOAI 2025 Chicken Counting provides partial pretrained feature-extractor weights and asks contestants to extend/optimize the model.
- IOAI 2025 individual tasks commonly include "Baseline and Training Set" sections.
- Polish OAI tasks include notebooks plus validation scripts.
- Kazakhstan repos include solution overviews but still follow train/test/sample-submission patterns.

Preparation implication:

- Do not start with the clever model.
- First reproduce the baseline/evaluator/output format.
- Then improve one thing: features, preprocessing, architecture head, threshold, or validation split.

### 4. Preprocessing and Feature Engineering Matter Constantly

The scan found feature/preprocessing/embedding language in 313 files.

Common asks:

- Clean tabular data.
- Impute missing values.
- Scale numeric features.
- Encode categorical variables.
- Engineer meta-features from raw attributes.
- Use TF-IDF or embeddings for text.
- Use PCA or clustering for representation/compression.
- Normalize/resize/augment images.
- Turn raw modality data into model-ready tensors.

Concrete examples:

- Kazakhstan Day 2 clusters football players after meta-feature engineering, imputation, scaling, and cluster-count choice.
- Kazakhstan Day 3 combines handcrafted code features with TF-IDF.
- Romania ONIA examples use tabular feature engineering and train/eval CSV workflows.
- IOAI 2024 Help BOBAI is a feature/model adaptation task around an existing classifier.

Preparation implication:

- Practice `df.shape`, `df.dtypes`, `df.isna().sum()`, `value_counts()`, `train_test_split`, `StandardScaler`, `OneHotEncoder`, `TfidfVectorizer`, and confusion matrices until automatic.
- For CEOAI, this is probably more valuable than deep theory review.

### 5. Pretrained Models Are Tools, Not Magic

The scan found pretrained/model-family language in 244 files.

Common model families:

- BERT / multilingual BERT
- GPT-like or LLM workflows
- ResNet / CNN encoders
- CLIP
- U-Net
- Transformers
- Diffusion/generative models

Typical asks:

- Use or adapt a pretrained encoder.
- Add a task-specific head.
- Fine-tune lightly.
- Extract embeddings.
- Repair or interpret model behavior.
- Respect restrictions on external data/model size.

Concrete examples:

- Kazakhstan Day 4 uses multilingual BERT for masked-word position classification.
- IOAI 2025 Chicken Counting provides partial pretrained weights.
- NEOAI Broken BERT is explicitly about debugging/repairing BERT-like embeddings.
- Romanian MLCompete writeups include CLIP/ResNet usage patterns.
- IOAI 2024/2025 generative CV tasks involve model manipulation or constrained generation.

Preparation implication:

- Know how to load a pretrained model, inspect input/output shape, freeze/unfreeze or add a head, and produce predictions.
- For CEOAI, prioritize recognition and correct use over implementing BERT/CNN/Transformer internals.

### 6. The Data Is Often Small, Weird, or Artificially Constrained

Many tasks are not plain "train a classifier on a clean dataset."

Common twists:

- Very small labeled training sets.
- Hidden validation/test set.
- Missing labels or noisy labels.
- Need to generate training examples.
- Provided model cannot be retrained.
- External data prohibited or limited.
- Submission must include code.
- Hardcoded answers disallowed.
- Kaggle/Bohrium-style mounted input paths.

Concrete examples:

- IOAI 2025 Concepts has only a small training set and asks for clue generation under strict sequence limits.
- IOAI 2024 Help BOBAI asks contestants to work around an already deployed model.
- Polish 2025 Noisy Labels targets robustness under wrong labels.
- Georgia-style statements include restrictions such as no pretrained models or no model retraining in specific tasks.

Preparation implication:

- Practice reading the constraints before coding.
- Make a "rules checklist" for every task: allowed data, allowed libraries, file format, metric, forbidden shortcuts.

## Typical Task Types by CEOAI Syllabus Section

## CEOAI 1. Search / RL

Archive signal: much thinner than ML/CV/NLP, but present.

Typical asks:

- Solve or analyze an MDP/gridworld.
- Produce a policy, value table, or Q-table.
- Apply Markov/TD/Q-learning reasoning.
- Sometimes combine algorithms with notebook-style outputs rather than full RL training.

Representative archive items:

- `romania_markov_maze`
- local existing Search/RL artifacts in `olympiads/IOAI Material/7. Reinforcement Learning and AI Search/`

Likely CEOAI inference:

- Expect compact A*, minimax, MDP, TD, or Q-learning tasks.
- Less likely: large-scale deep reinforcement learning.

Prep target:

- Be able to implement/check A*, minimax, a small MDP, TD update, and Q-table logic under time pressure.

## CEOAI 2. Classical ML

Archive signal: very strong.

Typical asks:

- Tabular classification/regression.
- Clustering.
- Feature engineering.
- Dimensionality reduction.
- Imbalanced classification.
- Noisy labels.
- Model comparison and metric selection.

Representative archive items:

- `romania_onia_examples`
- `ioai_2024_help_bobai`
- `ioai_2025_radar`
- `kazakhstan_day2_player_clustering`
- `poland_2024_imbalanced_classification`
- `poland_2025_noisy_labels`
- `poland_2025_borrowing`
- `neoai_tricy_table`

Likely CEOAI inference:

- This is one of the safest prediction areas for CEOAI.
- Expect train/test CSVs, scikit-learn baselines, metric traps, preprocessing, clustering, and output formatting.

Prep target:

- You should be able to finish a baseline in 20-30 minutes:
  `read_csv -> inspect -> split -> preprocess -> model -> metric -> submission`.

## CEOAI 3. Deep Learning

Archive signal: strong, but usually practical.

Typical asks:

- Use an MLP/CNN/encoder.
- Diagnose overfitting/underfitting.
- Adjust optimizer, learning rate, dropout, weight decay, batch norm.
- Add a task-specific head to pretrained features.
- Model compression/pruning.
- Autoencoder/generative-model recognition or constrained use.

Representative archive items:

- `poland_2024_pruning`
- `neoai_underfitting_cv`
- `ioai_2025_pixel`
- `poland_2025_unlearning`
- `poland_2025_data_prototypes`

Likely CEOAI inference:

- Expect architecture/optimizer recognition plus small implementation changes.
- Less likely: writing a large architecture from scratch.

Prep target:

- Know the PyTorch training loop, `.train()`/`.eval()`, device handling, tensor shapes, loss choice, optimizer choice, and validation curves.

## CEOAI 4. NLP

Archive signal: very strong.

Typical asks:

- Text classification.
- Hallucination / AI-generated text detection.
- Tokenization and cleaning.
- TF-IDF baselines.
- BERT/embedding workflows.
- Retrieval or source matching.
- Sentence-pair similarity.
- Masked word / missing text reasoning.
- Ranking outputs.

Representative archive items:

- `ioai_2025_concepts`
- `ioai_2024_lost_hyperspace`
- `poland_2025_hallucination`
- `poland_2025_source_extraction`
- `kazakhstan_day3_code_difficulty`
- `kazakhstan_day4_masked_word`
- `neoai_broken_bert`
- `neoai_intent_slot`
- `china_2024_news_text`
- `georgia_2025_dedup_bert`

Likely CEOAI inference:

- Expect practical NLP: tokenize, vectorize, use embeddings/encoders, evaluate, inspect mistakes.
- Do not expect a pure essay question about transformers to be enough preparation.

Prep target:

- Be fluent with `TfidfVectorizer`, logistic regression/SVM baselines, Hugging Face tokenizer/model input shapes, embedding similarity, and error inspection.

## CEOAI 5. Computer Vision

Archive signal: very strong.

Typical asks:

- Image classification.
- Counting/detection.
- Restoration/denoising.
- Matching images or icons.
- Generated/real image detection.
- Segmentation.
- Image clustering.
- Transfer learning with CNN/vision encoders.

Representative archive items:

- `ioai_2025_chicken_counting`
- `ioai_2025_restroom`
- `ioai_2025_antique`
- `kazakhstan_day1_image_restoration`
- `poland_2025_coin_counting`
- `poland_2024_color_quantization`
- `poland_2024_anomaly_detection`
- `poland_2025_non_normal_distribution`
- `china_2024_real_fake_image`
- `neoai_cluster_pictures`

Likely CEOAI inference:

- Expect preprocessing, model reuse, metric-aware prediction, and output-shape correctness.
- Full object detection/segmentation may appear as recognition or adaptation, but a simpler classification/counting task is more likely for an early regional contest.

Prep target:

- Be able to load images, batch tensors, use a pretrained CNN or small CNN, inspect predictions, and save the required output format.

## Repeated Contest Workflow

Use this workflow for every practice problem.

1. Read the statement headings first:
   - Task
   - Dataset
   - Submission
   - Scoring
   - Constraints
   - Baseline
2. Inspect the data:
   - shape
   - columns
   - dtypes
   - missing values
   - class balance
   - sample image/text/row
3. Identify the metric:
   - metric name
   - direction: higher or lower
   - weighted classes?
   - threshold/pass-fail?
   - ranking or regression?
4. Build the fastest valid baseline.
5. Generate the submission/evaluation artifact.
6. Only then improve.

## What Your CEOAI Prep Should Depend On

### Highest-Yield Skills

- Reading task statements fast.
- Data inspection.
- Submission-format checking.
- scikit-learn baseline building.
- PyTorch train/validation loops.
- Metric selection and interpretation.
- TF-IDF and embedding baselines.
- CNN/ResNet-style image classification.
- Debugging shape/dtype/path/device errors.
- Preventing data leakage.

### Lower-Yield During Final Sprint

- From-scratch implementation of standard algorithms unless the task explicitly asks for it.
- Broad theory reading without a runnable artifact.
- Audio tasks, because they are IOAI-only relative to the saved CEOAI syllabus.
- Full generative media/team challenge tasks unless all core CEOAI sections are already stable.

## Practical Prediction for CEOAI

Most likely CEOAI problem shape:

- A notebook or programming task with fixed data.
- A visible metric and hidden/private evaluation.
- A `sample_submission` or exact output contract.
- Allowed standard Python/ML libraries.
- A baseline that is easy to beat if you inspect data and metric correctly.
- One or two traps: shape mismatch, wrong metric, leakage, class imbalance, invalid submission file, or using the wrong model output.

Likely domains by probability and value:

1. Classical ML/tabular or clustering.
2. CV classification/counting/restoration.
3. NLP classification/embedding/retrieval.
4. DL optimizer/regularization/architecture-recognition task.
5. Compact Search/RL task because it is explicit in the CEOAI syllabus, even though fewer archived examples exist.

## Immediate Training Rule

For each practice sample, stop only when you have one of these:

- a score,
- a confusion matrix,
- a submission file,
- a prediction table,
- a clustering label output,
- a value/Q/policy table,
- or a short saved error analysis.

If none of those exists, the task did not count as CEOAI preparation.

## Files to Use Next

Start from:

- `practice_queue.md`
- `task_cards/romania_onia_examples.md`
- `task_cards/kazakhstan_day2_player_clustering.md`
- `task_cards/ioai_2024_help_bobai.md`
- `task_cards/ioai_2025_chicken_counting.md`
- `task_cards/ioai_2025_concepts.md`
- `task_cards/romania_markov_maze.md`

Do not open solution notebooks first unless the goal is post-attempt review.
