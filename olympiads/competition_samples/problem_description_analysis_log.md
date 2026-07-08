# Problem Description Analysis Log

Purpose: train the first step of competition problem solving: turn an unfamiliar statement into input, output, metric, baseline, validation, and next action.

Rule: an entry counts only if it includes the student's analysis and a teacher correction. Reading a statement without writing this structure does not count.

## Entry Template

```text
Date:
Problem:
Source path or link:
Syllabus tags:

My interpretation:
Every image/icon has a description. Use a text model such as BERT to generate embeddings for every image description. There are hard parts: not using the images themselves, not knowing how to use them, possible CNN/InfoNCE image embeddings not matching BERT text embeddings, and the need to account for the sequence of clue labels. Sequence models such as RNNs, LSTMs, or transformers might help. The abstract core is to get the embedding of the secret keyword and find a set of clue/icon embeddings whose sum has the maximum dot product with the secret.

Expected input/data:
Icon descriptions, secret target labels, and candidate options/distractors for the guessing game.

Expected output/submission:
Clue sequences that point the guesser toward the secret.

Metric/scoring:
The black-box AI guesser must rank the secret highly. The student did not state the exact metrics.

Baseline idea:
Import BERT, embed icon descriptions and the secret keyword, then choose clue icons whose combined embeddings are closest to the secret embedding by dot product.

Unclear parts:
How to use the images themselves. Whether image embeddings should be trained. How to model the sequence/order of clue labels. How to make embeddings match the black-box guesser's behavior.

First 15 minutes:
Import BERT and compute embeddings for image/icon descriptions and secret labels. Choose clue IDs by maximum dot product between the secret embedding and summed clue embeddings.

Teacher correction:
- Correct:
  - You correctly identified that the usable signal is mostly semantic: icon descriptions, secret words, candidate options, and a black-box guesser.
  - You correctly noticed that the output is not a class prediction; it is a set/sequence of clue icon IDs meant to make another model guess the target.
  - You correctly suspected that sequence/order may matter because the submission format allows multiple clue sequences, not just one unordered bag.
- Missing:
  - You did not state the exact output contract: `clues_a.jsonl`, `clues_b.jsonl`, zipped as `submission.zip`; each row is a list of up to 4 clue sequences, and each sequence has up to 8 marker/icon IDs.
  - You did not state the exact metric: `0.9 * Hits@10 + 0.1 * NDCG@10`.
  - You did not mention the options/distractors enough. The task is not "represent the secret in isolation"; it is "make the guesser prefer the secret over the provided options."
  - You did not include a validation step against the local evaluator/format checks.
- Wrong or risky assumption:
  - Training a CNN with InfoNCE is the wrong first move. The icons already have descriptions, and the contest pressure rewards a fast clue-selection strategy, not a new vision representation pipeline.
  - "Find embeddings whose sum has maximum dot product with the secret" is incomplete. It can choose generic clues that match many options. The score depends on ranking the true label against distractors, so the clue should be discriminative: close to the secret and far from competing options.
  - BERT embeddings alone may not match the black-box guesser. The judge is an LLM-style guesser, so literal token overlap, synonyms, hypernyms, and prompt-like clue combinations may beat pure embedding similarity.
- Why it matters:
  - A valid zip can still score `0.0` if the clue strategy does not move the secret into the guesser's top 10.
  - A generic semantic clue like "animal" may be close to "dog", but if the options include "cat", "wolf", and "pet", it does not discriminate.
  - Spending the first 15 minutes on CNN/InfoNCE would avoid the actual contract: pick valid icon IDs that make the black-box guesser rank the target higher than distractors.
- Better first move:
  - Inspect one dev example: label, options, and available icon descriptions.
  - Print the current baseline clues and the guesser's predictions for 3 weak examples if the local judge is available.
  - Build a text-only discriminative score: score each icon description by similarity to the answer minus similarity to the distractor options.
  - Generate valid `clues_a.jsonl`, `clues_b.jsonl`, and `submission.zip`, then validate row counts, clue limits, and zip names.

Reusable reflex:
For clue/ranking tasks, do not optimize the target alone. Optimize target-vs-distractors under the exact output format and metric.

Next action:
Do a 15-minute Concepts inspection: open one dev item, list its label/options, show 10 candidate icon descriptions, and choose clues by target similarity minus distractor similarity. Then validate the generated JSONL/zip contract before trying model complexity.

## 2026-07-08 - Polish OAI 2025 Hallucination Detection

Date: 2026-07-08
Problem: Polish OAI 2025 Hallucination Detection
Source path or link: `olympiads/competition_samples/raw/polish-oai-2025-sparse/1_etap/2_wykrywanie_halucynacji/2_wykrywanie_halucynacji_translated_en.ipynb`
Syllabus tags: CEOAI `4(b)` embeddings / representations, CEOAI `4(c)` related NLP architectures

My interpretation:
This is binary classification. The data and labels are available, so fit a model to predict whether the model hallucinated or not. The first thought was to try XGBoost, random forest, or logistic regression. Then the student noticed the task is NLP, so ordinary tabular models cannot directly consume raw text. BERT or Word2Vec came to mind, but the contest environment bans heavy deep-learning stacks such as torch and has no GPU. The core question is how to predict hallucination with plain vector math and allowed libraries.

Expected input/data:
`train.json` and `valid.json`. Each item has a factual question, the model's main answer, tokens, four higher-temperature supporting/alternative answers, supporting tokens, supporting probabilities, and label `is_correct`.

Expected output/submission:
A completed notebook containing `predict_hallucinations(sample)`, which returns a probability for one sample. During final evaluation, the function is serialized as `your_model.pkl`.

Metric/scoring:
ROC AUC, converted to points. `ROC AUC <= 0.7` gives 0 points, `ROC AUC >= 0.82` gives 100 points, and values between are scaled linearly.

Baseline idea:
Try XGBoost, random forest, or logistic regression, but uncertainty about how to convert NLP fields into features without BERT/torch.

Unclear parts:
How to represent text with only allowed CPU libraries. How to use the token probabilities. Whether the task predicts hallucination or correctness, since the label is `is_correct` but the algorithm docstring says probability of hallucination.

First 15 minutes:
Initially: fit a standard classifier. Revised concern: first figure out vector features from text/probabilities that allowed models can consume.

Teacher correction:
- Correct:
  - You correctly classified the task as binary classification.
  - You correctly caught the main obstacle: models such as logistic regression, random forest, and XGBoost need numeric features, not raw text.
  - You correctly noticed that BERT/torch is the wrong default because the final environment is CPU-only, offline, and restricted to `xgboost`, `scikit-learn`, `numpy`, `pandas`, and `matplotlib`.
- Missing:
  - The strongest non-text signal is already in the data: `supporting_probabilities`. These are model-confidence traces and can be summarized with plain vector math.
  - You did not yet separate two feature families: cheap text features and probability/token features.
  - You did not mention label direction. The evaluator's `ys` is `1 if is_correct else 0`, so higher predictions should mean "correct", despite comments mentioning hallucination probability.
- Wrong or risky assumption:
  - "NLP means I need BERT" is wrong in this contest setting. Many NLP competition baselines are TF-IDF plus logistic regression, or handcrafted numeric features plus XGBoost.
  - Random forest on sparse text is often weaker than logistic regression for TF-IDF. XGBoost can work better on compact engineered features.
  - Returning hallucination probability may invert ROC AUC if the evaluator expects correctness probability.
- Why it matters:
  - A simple allowed-library baseline can be built fast: `TfidfVectorizer` over question/answer/supporting answers plus logistic regression.
  - The provided probability arrays may reveal uncertainty: low token probabilities, high variance, disagreement among alternative answers, answer length, and `<answer>` span behavior.
  - ROC AUC rewards ranking, so calibrated probability is less important than ordering likely-correct above likely-wrong examples.
- Better first move:
  - Print one sample and list the exact keys, label distribution, and lengths of answers/tokens/probability arrays.
  - Build a tiny feature table: answer length, question length, number of supporting answers, mean/min/std of all supporting probabilities, mean of lowest-k probabilities, token count, and whether supporting answers agree lexically with the main answer.
  - Fit logistic regression or XGBoost on those numeric features first.
  - Then add a TF-IDF baseline using `question + answer + supporting_answers` with logistic regression.

Reusable reflex:
For restricted NLP tasks, translate text into allowed numeric features first: TF-IDF, counts, lengths, overlap, and probability summaries. Do not jump to transformers when the environment forbids them.

Next action:
Open the translated notebook, inspect one train example, and write the first feature table by hand before fitting any model.

## 2026-07-08 - Polish OAI 2025 Source Extraction

Date: 2026-07-08
Problem: Polish OAI 2025 Source Extraction
Source path or link: `olympiads/competition_samples/raw/polish-oai-2025-sparse/2_etap/ekstrakcja_zrodel/ekstrakcja_zrodel_translated_en.ipynb`
Syllabus tags: CEOAI `4(b)` embeddings / representations, CEOAI `4(c)` related NLP architectures

My interpretation:
The problem is about searching a corpus. The task is to create a model that generates high-quality embeddings. The statement says k-nearest neighbours are used for evaluation, and it gives a fine-tuned GPT-2 model. This feels contradictory: they ask contestants to generate embeddings while also giving a model that supposedly calculates good embeddings.

Expected input/data:
`corpus.jsonl`, containing document IDs, titles, and abstracts; and `queries_val.jsonl`, containing validation claims and matching source document IDs.

Expected output/submission:
A completed notebook with an `Embedder` class. The important methods are `encode_queries(queries)` and `encode_corpus(texts)`, each returning 768-dimensional tensors.

Metric/scoring:
nDCG@10. For each query, documents are ranked by cosine similarity to the query embedding. The score is high if the gold source document appears near the top 10. Scores below 0.2 get 0 points; scores above 0.5 get 100 points.

Baseline idea:
Use the provided GPT-2-based model to generate embeddings for queries and corpus documents.

Unclear parts:
If the provided model already gives embeddings, what is left for the contestant to implement?

First 15 minutes:
Understand what `Embedder.encode_queries` and `Embedder.encode_corpus` are supposed to return, then inspect how the provided model outputs hidden states that can be pooled into 768-dimensional embeddings.

Teacher correction:
- Correct:
  - You correctly identified this as a retrieval task, not classification.
  - You correctly noticed that evaluation is nearest-neighbour search using cosine similarity.
  - You correctly noticed that the provided GPT-2 model is central to the task.
- Missing:
  - The provided model does not automatically solve the notebook. You still need to decide how to convert its token-level hidden states into one vector per query/document.
  - You did not mention query/document asymmetry. Many retrieval models use different prompts or formatting for queries and documents.
  - You did not mention pooling. Decoder models output one hidden vector per token; the contest asks for one 768-dimensional vector per whole text.
- Wrong or risky assumption:
  - "They give us a model that calculates embeddings" is only half true. The model produces hidden states. The contestant must choose the embedding recipe: which layer, which token positions, mean pooling vs last-token pooling, normalization, title+text formatting, batching, and truncation length.
  - Treating queries and corpus documents identically may be weaker than adding retrieval-specific prefixes or formatting.
  - Ignoring runtime is risky because the corpus is large and the limit is 10 minutes with GPU.
- Why it matters:
  - In retrieval tasks, small embedding choices can move the gold document from rank 50 to rank 5.
  - nDCG@10 gives zero for a query if the correct document is outside the top 10, so the embedding space must separate near-miss documents well.
  - The real implementation task is not inventing GPT-2; it is using the provided model to build a fast, compatible, rank-effective embedder.
- Better first move:
  - Run the current dummy embedder once to see the baseline nDCG and confirm the evaluation pipeline.
  - Inspect one query and its matching corpus document.
  - Implement mean pooling over token hidden states from the provided model, with attention-mask weighting.
  - Normalize embeddings before cosine search.
  - Try basic formatting: query as raw claim; corpus as `title + "\\n" + text`.

Reusable reflex:
When a task gives a pretrained model, the contest problem is often not "train a model from scratch"; it is "wrap, pool, format, normalize, batch, and validate it under the metric."

Next action:
Open the translated notebook and identify exactly where GPT-2 outputs token hidden states. Then define one pooling rule and one text-formatting rule before thinking about training.

## 2026-07-08 - Polish OAI 2024 Pruning

Date: 2026-07-08
Problem: Polish OAI 2024 Pruning
Source path or link: `olympiads/competition_samples/raw/polish-oai-2024-sparse/first_stage/pruning/pruning_translated_en.ipynb`
Syllabus tags: CEOAI `3(b)` neural-network optimization, CEOAI `3(c)` model architecture / parameters

My interpretation:
The problem is about reducing a trained model while keeping prediction loss low. The loss depends on the input data and the model weights. To keep loss as low as possible, find parameters that affect the output as little as possible and set them to zero. Two possible routes: use backpropagation/gradients to estimate irrelevant parameters, or use the value of each weight directly. A very small weight like `0.01` probably matters less than a large weight like `2`.

Expected input/data:
Training and validation arrays loaded from `.npy` files, plus an already trained MLP with input size 128, hidden size 1024, sigmoid activation, and output size 10.

Expected output/submission:
A pruned model with many zero weights/biases, saved as `model_parameters.pkl` by `save_parameters`.

Metric/scoring:
Score combines sparsity and MSE. More zero parameters is better, but if MSE is too high the score collapses. Architecture cannot be changed.

Baseline idea:
Try magnitude pruning first: zero the smallest weights because they should have the smallest effect on outputs. Possibly explore gradient-based pruning later.

Unclear parts:
How much can be pruned before MSE becomes too large. Whether pruning should affect biases too. Whether to retrain/fine-tune after pruning.

First 15 minutes:
Experiment with the second option: zero small-magnitude weights and measure validation MSE, sparsity, and score.

Teacher correction:
- Correct:
  - You correctly identified the core tradeoff: zero many parameters while preserving model outputs.
  - You correctly identified magnitude pruning as the simplest first baseline.
  - You correctly mentioned gradients as a possible more advanced signal for parameter importance.
- Missing:
  - The score is not just "keep loss low"; it rewards sparsity strongly too. A slightly higher MSE may be worth it if sparsity increases a lot.
  - You should not just "experiment" loosely. The first experiment should be a threshold sweep with a table: threshold/percentile, sparsity, MSE, score.
  - You did not mention saving `model_parameters.pkl`, which is the required artifact.
  - You did not mention that architecture cannot change, so neuron removal is forbidden even if conceptually tempting.
- Wrong or risky assumption:
  - "Small weight means useless" is a good baseline but not always true. A small weight connected to a very large activation can still matter.
  - Looking only at weight size ignores layer scale. A weight of `0.01` in one layer may not mean the same thing as `0.01` in another layer.
  - Gradient pruning can be useful, but it is not the first 15-minute move because it adds complexity before establishing a magnitude baseline.
- Why it matters:
  - This task rewards measured tradeoffs, not a single pruning guess.
  - A table of sparsity/MSE/score immediately tells you how aggressive you can be.
  - Magnitude pruning is fast, legal, easy to validate, and usually strong enough as a first contest baseline.
- Better first move:
  - Compute baseline MSE and baseline sparsity.
  - For several percentiles, zero all weights with absolute value below that percentile.
  - After each prune, compute MSE, sparsity, and score.
  - Pick the best validation score and save `model_parameters.pkl`.
  - Only after that, consider fine-tuning the remaining nonzero weights or trying layer-wise thresholds.

Reusable reflex:
For pruning/compression tasks, start with a measurable sweep: prune by magnitude, validate the task metric, and save the best artifact before adding clever importance estimates.

Next action:
Run a magnitude-threshold sweep on the validation set and record a before/after table with sparsity, MSE, score, and points.

## 2026-07-08 - NEOAI 2025 Broken BERT

Date: 2026-07-08
Problem: NEOAI 2025 Broken BERT
Source path or link: `olympiads/competition_samples/raw/neoai-2025-sparse/5_Broken_BERT/broken_bert_solution_translated_en.ipynb`
Syllabus tags: CEOAI `4(b)` embeddings / representations, CEOAI `4(c)` related NLP architectures, CEOAI `3(c)` transformer/encoder recognition

My interpretation:
The student does not remember the exact BERT layers and what they do. The first instinct is to treat this like a normal supervised learning problem: data, model, labels, train the non-frozen parameters like in an earlier neural-network exercise.

Expected input/data:
Validation/test sentiment-analysis text data and a broken BERT model where only the token embedding matrix is damaged.

Expected output/submission:
A repaired model/notebook submission that improves sentiment-analysis quality while obeying the restrictions.

Metric/scoring:
Not fully stated by the student. The task is evaluated by sentiment-analysis model quality on hidden/test data.

Baseline idea:
Train the parameters that are not frozen using supervised learning.

Unclear parts:
Which BERT layers exist, what each layer does, and which parameters are legal to change.

First 15 minutes:
Try supervised training on the train/validation data and update non-frozen parameters.

Teacher correction:
- Correct:
  - You correctly recognized that there is data, a model, and an evaluation task.
  - You correctly connected the task to supervised learning mechanics.
  - You correctly identified that the trainable part matters.
- Missing:
  - The key restriction is not "train whatever is unfrozen"; it is "only the token embeddings are damaged" and the rest of BERT must remain intact.
  - You need to inspect which parameter names correspond to token embeddings versus position embeddings, segment embeddings, attention blocks, feed-forward blocks, layer norms, and classifier head.
  - You did not mention the ban on fine-tuning/pretraining or extra data.
- Wrong or risky assumption:
  - Treating this as ordinary supervised fine-tuning is risky because the statement says you may not fine-tune/pretrain the model.
  - Updating attention, encoder, or classifier weights would solve the wrong problem and likely violate the task.
  - "Non-frozen parameters" is not precise enough. You must explicitly control which parameters are allowed to change.
- Why it matters:
  - This is a model-repair task, not a normal training task.
  - The intended skill is inspecting a pretrained model and repairing a corrupted embedding matrix under constraints.
  - If you ignore the restriction, you may get local improvement but produce an invalid contest solution.
- Better first move:
  - Print all model parameter names and shapes.
  - Identify `word_embeddings.weight` or the equivalent token embedding matrix.
  - Confirm all non-embedding weights are frozen/unchanged.
  - Measure baseline validation quality of the broken model.
  - Try repairing embeddings using allowed information, such as copying/reinitializing suspicious token vectors, using statistics from intact embeddings, or optimizing only the token embedding matrix if the notebook allows that without violating the stated restriction.

Reusable reflex:
When a task says a pretrained model is "broken," first identify exactly which weights may change. Do not default to full supervised fine-tuning.

Next action:
Open the model, print parameter names, locate the token embedding matrix, and write down which tensors are legal to modify before doing any training.
```

## 2026-07-08 - IOAI 2025 Concepts

Date: 2026-07-08
Problem: IOAI 2025 Concepts
Source path or link: `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts.ipynb`
Syllabus tags: CEOAI `4(b)` embeddings / representations, CEOAI `4(c)` related NLP architectures, CEOAI `3(c)` transformer/encoder recognition

My interpretation:

Expected input/data:

Expected output/submission:

Metric/scoring:

Baseline idea:

Unclear parts:

First 15 minutes:

Teacher correction:
- Correct:
- Missing:
- Wrong or risky assumption:
- Why it matters:
- Better first move:

Reusable reflex:

Next action:
