# Caramanis Generative AI And NLP - 90 Percent Recall Notes

## Purpose

These notes compress the Caramanis `generative_ai_and_nlp` resources into the
version that is useful for fast olympiad preparation.

Use them when you have already watched the lectures but need to recover the
course spine, implementation contracts, and debugging reflexes quickly.

Source set:

- `olympiads/resources/caramanis_colab_notebooks/generative_ai_and_nlp/slides/`
- `olympiads/resources/caramanis_colab_notebooks/generative_ai_and_nlp/*.ipynb`

The level is practical and medium-hard: not theory-only, not full research
training. The teaching style is to start from a visible baseline, expose the
geometry, run small Python experiments, then connect that baseline to modern
retrieval, BERT, and fine-tuning.

## Exam-Mode Warning

If CEOAI is close, do not study this file linearly. Use it only as a reference
for syllabus-tagged tasks.

For the final sprint, the priority is:

1. solve competition-style tasks,
2. produce metrics or submission-like outputs,
3. map each block to the CEOAI/IOAI syllabus,
4. debug fast.

Do not spend a day becoming generally better at NLP. That is useful after the
competition. Before the competition, the useful unit is a covered syllabus item
with a working artifact.

## The Course Spine

### 1. The destination is RAG and fine-tuning

The course is not just "what is generative AI." The two practical endpoints are:

- build a Retrieval Augmented Generation system that uses external documents,
- adapt a pretrained language model or embedding model to a specific task.

Everything earlier in the lectures exists to make those two workflows less
magical.

The basic RAG pipeline is:

1. receive a user query,
2. retrieve relevant documents from a knowledge base,
3. pass query plus retrieved context to a language model,
4. generate an answer grounded in the retrieved material.

This means retrieval quality is not optional. If retrieval fails, generation is
working with bad evidence.

### 2. Transfer learning is the bridge from old ML to GenAI

The lecture path uses image transfer learning before NLP to make one idea stick:
a pretrained model already contains useful representations.

For images:

- train on a huge generic dataset such as ImageNet,
- remove or replace the final classifier,
- reuse the learned representation on a smaller target task.

For text:

- pretrain BERT or a related model on large text corpora,
- reuse the encoder as a text representation engine,
- add a classifier head or fine-tune the embeddings for retrieval.

Contest reflex:

- If data is small, do not start by training a large model from scratch.
- First ask what pretrained representation you can reuse.

### 3. Logistic regression is the simplest neural classifier

Caramanis uses logistic regression and softmax as the first clean model because
it exposes the whole supervised-learning contract.

For a multiclass classifier:

- input features: `x`
- class scores/logits: `z = W x + b`
- probabilities: `softmax(z)`
- prediction: `argmax(probabilities)`
- training loss: cross entropy

Geometric interpretation:

- each class has a weight vector,
- the dot product between input and class vector is a similarity score,
- large dot product means the input points in a similar direction,
- the model learns linear decision boundaries.

Implementation reflex:

- `logits` are raw scores, not probabilities.
- `CrossEntropyLoss` expects logits and integer class labels.
- Do not apply softmax before `CrossEntropyLoss`.
- For plots, check the mesh-grid shape before reshaping predictions.

### 4. Dot products are the recurring primitive

The same idea appears repeatedly:

- logistic regression scores classes with dot products,
- convolution slides a small vector/kernel and computes local dot products,
- Word2Vec scores word-context compatibility with dot products,
- retrieval ranks query/document vectors by dot product or cosine similarity,
- InfoNCE builds a full query-document similarity matrix with dot products.

Fast mental model:

```text
representation -> dot product score -> ranking/classification/loss
```

If you can debug shapes and score matrices, you can debug most of this unit.

### 5. Embeddings mean useful coordinates

An embedding maps an object to a vector such that useful relationships become
geometric.

Examples from the notebooks:

- ResNet image embeddings: images from related classes are near each other.
- Word2Vec word embeddings: semantically related words have high cosine
  similarity.
- BERT sentence/document embeddings: queries and relevant texts should point in
  similar directions.

Embedding is not a label. It is a representation you can compare, cluster,
retrieve with, or feed into a downstream classifier.

### 6. Word2Vec is static word meaning

Word2Vec is taught as the first semantic embedding model.

Core idea:

- train from text without manual labels,
- use context prediction as a self-supervised task,
- learn word vectors because useful vectors help predict nearby words.

What matters:

- the model has a vector per word,
- cosine similarity measures closeness,
- analogies work by vector arithmetic sometimes,
- the same word always has the same embedding.

Limitation:

- static embeddings cannot represent context-dependent meaning.
- The word "bank" or the Greek example around "comma/party" needs context.

This limitation motivates transformers and contextual embeddings.

### 7. Sparse retrieval comes before dense retrieval

The retrieval lectures build from simple to stronger:

1. Bag of Words
2. TF-IDF
3. BM25
4. semantic search with embeddings

Bag of Words:

- tokenize query and documents,
- count overlapping terms,
- rank by shared words.

Failure mode:

- misses synonyms and semantic matches,
- over-rewards common words unless preprocessing is careful.

TF-IDF:

- term frequency rewards words present in a document,
- inverse document frequency rewards words that are rare across documents,
- rare query-specific words matter more than common words.

BM25 improves TF-IDF by:

- saturating term-frequency gains,
- normalizing for document length,
- working well as a practical sparse ranking baseline.

Inverted index:

- map each token to document IDs where it appears,
- avoid scanning every document for every query.

Contest reflex:

- For a text-retrieval task, implement lexical baseline first.
- Measure it before reaching for transformers.

### 8. Retrieval metrics are part of the model

The lectures emphasize that "good result" depends on the user goal.

Recall@K:

- fraction of all relevant documents found in the top K.
- Use it when missing relevant evidence is costly.
- Important for RAG because missing evidence leads to unsupported generation.

NDCG@K:

- rewards relevant documents more when they appear near the top.
- Use it when ranking order matters, not just inclusion.

Practical rule:

- `Recall@K` asks: did we find enough relevant material?
- `NDCG@K` asks: did we rank the best material early?

Debug reflex:

- verify query IDs match qrels IDs,
- verify retrieved document IDs match corpus IDs,
- inspect one query qualitatively before trusting averages,
- compute metrics on a tiny hand-checkable example before full evaluation.

### 9. Dense semantic search replaces token overlap with vector similarity

Dense retrieval pipeline:

1. encode every document into a vector,
2. encode the query into a vector,
3. compute query-document scores with dot product or cosine similarity,
4. sort documents by score,
5. evaluate with Recall/NDCG.

Why it helps:

- can match meaning even without exact word overlap.

Why it can fail:

- generic embeddings may not understand your domain,
- pooling can be wrong,
- vectors may not be normalized when cosine-like scores are expected,
- max sequence length truncation may hide key evidence.

### 10. Transformers create contextual embeddings

Word2Vec gives one vector per word. Transformers update token vectors using the
other tokens in the sequence.

Self-attention idea:

- each token representation can look at other tokens,
- the final token vector depends on the word and its context,
- this creates contextual embeddings.

Encoder vs decoder:

- Encoder: non-causal access to the whole input; good for understanding,
  classification, embeddings, and retrieval. BERT is encoder-style.
- Decoder: causal access to previous tokens; good for next-token generation.
  GPT-like models are decoder-style.

Do not confuse:

- BERT: strong text understanding and embeddings.
- GPT-like model: autoregressive text generation.

### 11. BERT pretraining is self-supervised classification

BERT learns from invented labels on raw text.

Masked Language Modeling:

- choose about 15 percent of tokens,
- replace most selected tokens with `[MASK]`,
- replace some with random tokens,
- leave some unchanged,
- train the model to predict the original selected tokens.

The MLM head:

- maps each selected output token to a distribution over the vocabulary,
- uses cross entropy over the masked/original tokens,
- is mainly a pretraining head.

The "heart" of BERT:

- token embeddings,
- positional embeddings,
- segment embeddings in BERT,
- transformer encoder layers,
- final contextual token vectors.

BERT-base numbers that recur:

- roughly 30k WordPiece vocabulary,
- 768-dimensional hidden vectors,
- 12 encoder layers,
- `[CLS]` at the beginning,
- `[SEP]` at sequence boundaries.

### 12. BERT classification uses the CLS vector

The classification notebook fine-tunes BERT on CoLA.

Pipeline:

1. load dataset,
2. tokenize sentences with padding/truncation,
3. inspect `input_ids` and `attention_mask`,
4. load pretrained BERT encoder,
5. take `last_hidden_state[:, 0, :]` as the CLS representation,
6. pass CLS through a small linear classifier,
7. train with cross entropy and a small learning rate.

Important contracts:

- `input_ids`: `(batch, seq_len)` integer token IDs.
- `attention_mask`: `(batch, seq_len)` with 1 for real tokens, 0 for padding.
- `last_hidden_state`: `(batch, seq_len, hidden_dim)`.
- `CLS`: `last_hidden_state[:, 0, :]`, shape `(batch, hidden_dim)`.
- classifier output: `(batch, num_classes)`.
- labels: `(batch,)` integer class IDs.

Debug reflex:

- if loss does not move, check labels and logits shape first,
- if CUDA/device errors happen, check every tensor in the batch,
- if accuracy is nonsense, check `argmax(dim=1)` and label dtype,
- if memory explodes, reduce `max_length` or batch size.

### 13. BERT embeddings need pooling

For sentence/document embeddings, you need one vector per text.

Two common choices:

- CLS pooling: use `last_hidden_state[:, 0, :]`.
- mean pooling: average token embeddings, ignoring padding with
  `attention_mask`.

Mean pooling contract:

- token embeddings: `(batch, seq_len, hidden_dim)`,
- mask: `(batch, seq_len)`,
- expand mask to `(batch, seq_len, hidden_dim)`,
- zero out padding tokens,
- divide by count of real tokens.

Debug reflex:

- never average padding as if it were real text,
- clamp denominator to avoid division by zero,
- decide whether to normalize embeddings before dot products.

### 14. Fine-tuning embeddings with InfoNCE

The final notebook fine-tunes a MiniLM/BERT-style encoder on SciFact retrieval.

Problem:

- query should be close to relevant documents,
- query should be far from irrelevant documents,
- qrels usually give positives, not explicit negatives.

In-batch negatives:

- build a batch of `(query_i, positive_doc_i)` pairs,
- each other document in the batch acts as a negative for query `i`,
- construct similarity matrix `S = query_embs @ doc_embs.T`,
- diagonal entries are positives,
- off-diagonal entries are negatives.

InfoNCE implementation contract:

- query embeddings: `(B, D)`,
- document embeddings: `(B, D)`,
- normalize along `D`,
- logits: `(query_embs @ doc_embs.T) / temperature`, shape `(B, B)`,
- labels: `torch.arange(B)`,
- loss: cross entropy over rows.

Debug reflex:

- batch size must be at least 2,
- diagonal must be the correct positive pair,
- shuffled docs break the labels unless labels are adjusted,
- temperature too small can make training unstable,
- evaluate before and after with the same metric code.

## The Caramanis Teaching Pattern

The repeated teaching move is:

1. show a concrete problem,
2. reduce it to vectors and scores,
3. interpret the geometry,
4. run a small Python notebook,
5. inspect outputs and plots,
6. scale the same idea to modern pretrained models.

That means your study should mirror this pattern. Reading definitions is low
yield. The high-yield loop is:

1. build a minimal version,
2. print shapes,
3. print one example,
4. compute one score by hand,
5. compare with the library result,
6. break one assumption and debug it.

## What Matters Most For Speed

Focus hard on:

- tokenization consistency,
- vectorizer vocabulary and shape,
- sparse matrix vs dense array behavior,
- logits/probabilities/loss contracts,
- embedding normalization,
- top-k ranking and ID alignment,
- attention-mask-aware pooling,
- evaluation metrics on tiny examples,
- `DataLoader` batch structure,
- device and dtype errors in PyTorch.

Spend less time on:

- memorizing every architecture name,
- training large models,
- full transformer math derivations,
- polished notebooks,
- long theory summaries with no executable check.

## Rusty Programmer Recovery Plan

You have watched the lectures, so the bottleneck is not exposure. The bottleneck
is implementation fluency.

Use this order:

1. lexical retrieval from scratch,
2. TF-IDF/BM25-style ranking,
3. metric implementation,
4. embedding similarity,
5. mean pooling with masks,
6. BERT CLS classification shape checks,
7. InfoNCE similarity matrix.

Stop condition for this unit:

- You can implement a tiny retrieval system without looking at code.
- You can debug a wrong top-k list by inspecting tokens, IDs, and scores.
- You can explain every dimension in a BERT batch.
- You can detect the three common loss-contract mistakes:
  softmax before cross entropy, wrong label shape, wrong label dtype.
