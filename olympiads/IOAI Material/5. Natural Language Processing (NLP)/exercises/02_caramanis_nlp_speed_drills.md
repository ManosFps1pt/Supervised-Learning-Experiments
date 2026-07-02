# Sprint 02: CEOAI NLP Syllabus Sprint

## Read This First

This is exam-mode preparation, not a general NLP course.

The target is direct syllabus coverage in three days. Do not spend the session
proving that you understand the internals of every method. The useful question
is:

```text
Can I recognize the task, build a legal baseline fast, debug it, submit outputs,
and explain what syllabus item I covered?
```

This sprint covers the CEOAI local syllabus section:

- NLP preprocessing: tokenization, stemming, related text cleaning.
- Embeddings: TF-IDF, Word2Vec-style vectors, pretrained text embeddings.
- Related architectures: Seq2Seq, T5, LLaMA-style language models.

It also overlaps the official IOAI 2026 syllabus:

- NLP text classification.
- Pre-trained text encoders such as BERT.
- Language modeling.
- Encoder-decoder models.
- Pre-trained language models.

## Rules For This Sprint

- Use libraries unless the task explicitly says "tiny manual check."
- Do not implement decision trees, BERT, Word2Vec, tokenizers, or transformers
  from scratch.
- Do not do broad theory notes before producing notebook evidence.
- Every section must end with a syllabus tag and a visible artifact.
- If a task takes too long, cut scope and preserve the scored output.

Allowed manual work:

- tiny tokenization checks,
- one hand-computed metric,
- one shape/dtype/debugging check.

Forbidden time sinks:

- from-scratch algorithm reimplementation,
- large model training,
- polishing markdown,
- long conceptual summaries,
- downloading huge models unless already necessary and feasible.

## Required Notebook

Continue or create:

```text
solution2_caramanis_speed_drills.ipynb
```

The notebook must start with a small checklist:

```text
CEOAI NLP syllabus covered:
[ ] preprocessing/tokenization
[ ] TF-IDF or lexical embedding baseline
[ ] text classification
[ ] pretrained text encoder / BERT-style workflow
[ ] language modeling or encoder-decoder / LLM workflow
```

Each completed task must include:

- the syllabus item,
- the dataset/input used,
- the model or library used,
- the metric or output,
- the main bug/debug check.

## Task 1: Text Classification Baseline

Syllabus target:

- CEOAI NLP preprocessing.
- CEOAI embeddings: TF-IDF.
- IOAI NLP text classification.
- IOAI data processing: tokenization and vocabulary building.

Time box: 45-60 minutes.

Use a small built-in or local text dataset. Good options:

- `sklearn.datasets.fetch_20newsgroups` if available,
- a small manually written sentiment/topic dataset if network/data access is
  blocked,
- any local text classification file already in the repo.

Build the fastest useful baseline:

1. load texts and labels,
2. split train/test,
3. vectorize with `TfidfVectorizer`,
4. train a library classifier such as logistic regression, linear SVM, or Naive
   Bayes,
5. report accuracy and confusion matrix,
6. print 5 misclassified examples.

Do not implement TF-IDF from scratch. If you want to check understanding, print
the vocabulary size and inspect the top weighted terms for one document.

Expected artifact:

- metric table,
- confusion matrix,
- 5 misclassified examples,
- short note: `covered: preprocessing + TF-IDF + text classification`.

Minimum acceptable result:

- one trained baseline,
- one metric,
- one inspected error.

## Task 2: Pretrained Text Encoder Workflow

Syllabus target:

- CEOAI related architectures: BERT/transformer text encoders.
- IOAI pre-trained text encoders such as BERT.
- IOAI transformers for text.

Time box: 45-75 minutes.

Use this exact setup:

- Dataset/input: 8 short texts from the existing 20 Newsgroups `X_test` split.
- Model: `distilbert-base-uncased`.
- Library: Hugging Face `AutoTokenizer` and `AutoModel`.
- Pooling target: mean pooling using `attention_mask`.

Run the real model. If download fails or takes too long, make a mock batch with
the same tensors and clearly mark the model step as blocked.

Do:

1. tokenize 8 real 20 Newsgroups sentences,
2. print `input_ids`, `attention_mask`, and decoded tokens for one example,
3. run `distilbert-base-uncased` or mock the encoder output,
4. extract one fixed-size sentence representation using attention-mask mean pooling,
5. train or evaluate a tiny classifier on top if time allows.

Expected artifact:

- tokenized batch shapes,
- one decoded example,
- encoder output or mock output shape,
- pooled embedding shape,
- note: `covered: pretrained text encoder / BERT workflow`.

Minimum acceptable result:

- correct token batch inspection and pooling contract.

Do not:

- derive attention equations,
- implement BERT from scratch,
- spend the session fighting a huge model.

## Task 3: Language Modeling / Encoder-Decoder Recognition

Syllabus target:

- CEOAI related architectures: Seq2Seq, T5, LLaMA-style models.
- IOAI language modeling.
- IOAI encoder-decoder models.
- IOAI pre-trained language models.

Time box: 35-50 minutes.

This is recognition plus practical use, not training a language model.

Use these exact models:

- Decoder-only model: `distilgpt2`.
- Encoder-decoder model: `google/flan-t5-small`.

Use these exact prompts:

```text
Write one sentence about space exploration.
Summarize why someone might ask about computer hardware.
Classify this topic: car engine repair and maintenance.
```

Do:

1. load tokenizer and model for `distilgpt2`,
2. run one short generation and label it decoder-only,
3. load tokenizer and model for `google/flan-t5-small`,
4. run one short generation and label it encoder-decoder,
5. make a short comparison table: BERT vs GPT/LLaMA vs T5,
6. for each family, write the input/output pattern and likely competition use.

If model access is blocked, do the comparison table anyway and mark generation
as blocked.

Expected artifact:

- generated example or architecture comparison table,
- one paragraph: `when would I use BERT, GPT/LLaMA, or T5 in a task?`,
- note: `covered: language modeling + encoder-decoder / pretrained LM`.

Minimum acceptable result:

- correct model-family recognition and task mapping.

Do not:

- train a transformer,
- read broad generative AI theory,
- spend more than 50 minutes here.

## Task 4: Competition-Style Mini Submission

Syllabus target:

- practical NLP workflow under time pressure.

Time box: 45-60 minutes.

Use this exact setup:

- Dataset: the existing 20 Newsgroups split from Task 1.
- Model: the existing `TfidfVectorizer` + `LogisticRegression` baseline.
- Hidden-style test: the first 20 examples from `X_test`.

Create a mini task with hidden-test-style behavior:

1. treat the first 20 `X_test` examples as unlabeled "test",
2. reuse the trained Task 1 baseline,
3. predict labels for those 20 examples,
4. generate a `submission` dataframe with `id` and `prediction`,
5. save or display it,
6. perform two sanity checks.

Sanity checks:

- no missing predictions,
- test row count equals submission row count,
- labels are in the allowed set,
- no train/test leakage in the final reporting.

Expected artifact:

- `submission` dataframe,
- sanity-check outputs,
- note: `covered: competition-style NLP pipeline`.

Minimum acceptable result:

- valid `id,prediction` output with sanity checks.

## Final 10-Minute Coverage Log

At the end of the notebook, write this table:

| Syllabus item | Evidence in notebook | Status |
| --- | --- | --- |
| NLP preprocessing/tokenization | cell/output reference | covered / weak / missing |
| TF-IDF / embeddings | cell/output reference | covered / weak / missing |
| text classification | cell/output reference | covered / weak / missing |
| pretrained text encoders | cell/output reference | covered / weak / missing |
| language modeling / encoder-decoder / LLMs | cell/output reference | covered / weak / missing |
| competition submission workflow | cell/output reference | covered / weak / missing |

If any row is `missing`, that is the next task. Do not start optional material.

## What Counts As Success

Strong result:

- You can point to 5-6 syllabus rows covered in the notebook.
- You produced metrics or submission-like outputs.
- You inspected at least one real error or misclassification.
- You used libraries correctly instead of rebuilding algorithms.

Weak result:

- You spent most of the time writing definitions.
- You implemented low-level internals that the syllabus only requires in
  practice.
- You cannot show which syllabus items were covered.
- You have no metric, no prediction output, or no sanity checks.

<!-- Archived prior generic drill list. Do not use during the three-day CEOAI
exam sprint; it is kept only for historical context.

## Source

Use the Caramanis resources in:

```text
../../../resources/caramanis_colab_notebooks/generative_ai_and_nlp/
```

Also use:

```text
../../../notes/caramanis_generative_ai_nlp_90pct.md
```

## Goal

This sprint is not for learning terminology. It is for turning the Caramanis
material into implementation reflexes.

By the end, you should be faster at:

- building a baseline,
- inspecting shapes,
- locating bad IDs,
- checking ranking metrics,
- debugging tokenization/vectorization,
- debugging PyTorch loss contracts.

No solution code should be generated here unless explicitly requested.

## Time Box

Target: 2 sessions of 75-90 minutes.

If you only have one session, do Drills 1-4. They are the highest yield for
competition speed.

## Required Notebook

Create one notebook named:

```text
solution2_caramanis_speed_drills.ipynb
```

Keep cells short. Every drill must include:

- one tiny hand-checkable example,
- printed shapes or IDs,
- one deliberate bug you introduce and then fix,
- a 2-4 line "contest reflex" note.

## Drill 1: Tokenizer And Vocabulary Contract

Use this fixed tiny corpus. Do not change it during Drills 1-4, because the
later retrieval and metric checks depend on stable document IDs.

| doc_id | text |
| --- | --- |
| `D1` | `Metformin lowers blood glucose and is often used for type 2 diabetes.` |
| `D2` | `Cimetidine can increase metformin levels by reducing renal tubular secretion.` |
| `D3` | `Insulin helps cells absorb glucose after meals.` |
| `D4` | `Patients taking metformin should ask a clinician about kidney function.` |
| `D5` | `A transformer model uses attention to build contextual token embeddings.` |
| `D6` | `Bag of words retrieval counts exact token overlap between a query and documents.` |
| `D7` | `TF-IDF gives more weight to rare terms than to common terms.` |
| `D8` | `BERT creates contextual embeddings and can be fine-tuned for text classification.` |
| `D9` | `A RAG system retrieves relevant documents before generating an answer.` |
| `D10` | `Cosine similarity compares the direction of two embedding vectors.` |

Use these fixed queries:

| query_id | text |
| --- | --- |
| `Q1` | `metformin cimetidine kidney interaction` |
| `Q2` | `contextual embeddings with attention` |
| `Q3` | `rare terms in tf idf retrieval` |

For later metric drills, use these relevance judgments:

| query_id | relevant docs |
| --- | --- |
| `Q1` | `D2` is highly relevant; `D4` is partially relevant; `D1` is partially relevant. |
| `Q2` | `D5` is highly relevant; `D8` is highly relevant; `D10` is partially relevant. |
| `Q3` | `D7` is highly relevant; `D6` is partially relevant. |

Task:

- write a simple tokenizer,
- print tokens for every document and query,
- build a vocabulary,
- create a document-term count matrix,
- verify one row manually.

Expected outputs:

- token list per document,
- vocabulary sorted or indexed consistently,
- matrix shape,
- one manually checked document vector.

Deliberate bugs to test:

- inconsistent lowercase handling,
- punctuation kept in one place and removed in another,
- vocabulary index order changes between train and query.

Stop when:

- you can explain exactly why each count appears in one chosen row.

## Drill 2: Bag-Of-Words Search From Scratch

Use the same corpus.

Task:

- vectorize queries with the same vocabulary,
- score each document by query-document dot product,
- return top-3 document IDs for each query,
- inspect one failure case.

Expected outputs:

- query vector shape,
- score vector shape,
- ranked document IDs,
- one explanation of a false positive or false negative.

Deliberate bugs to test:

- query uses a different vocabulary,
- scores are sorted ascending,
- document text is returned but document ID alignment is wrong.

Contest reflex:

- Retrieval bugs are often ID bugs, not model bugs.

## Drill 3: TF-IDF And BM25 Thinking

Do not start with Pyserini. First implement a simple TF-IDF-style score.

Task:

- compute document frequency for each token,
- compute a simple IDF,
- transform document counts into TF-IDF weights,
- compare top-3 results with Drill 2.

Expected outputs:

- document frequency table,
- top rare terms,
- BoW ranking vs TF-IDF ranking for each query,
- one case where TF-IDF improves or hurts.

Deliberate bugs to test:

- divide by zero for unseen query terms,
- IDF sign or smoothing mistake,
- common word dominates the ranking.

Stop when:

- you can explain why one rare term changes the ranking.

## Drill 4: Recall@K And NDCG@K By Hand

Create tiny qrels manually.

Example structure:

```text
query_id -> relevant_doc_ids with relevance scores
```

Task:

- compute Recall@1, Recall@3, and Recall@5,
- compute DCG and NDCG@3,
- compare your function against one hand calculation.

Expected outputs:

- retrieved IDs,
- relevant IDs,
- per-query metric values,
- macro average.

Deliberate bugs to test:

- using document rank starting at 0 in the log denominator,
- treating missing qrels as relevant,
- comparing integer IDs to string IDs.

Contest reflex:

- Before trusting metrics, make a 3-document example where you know the answer.

## Drill 5: Static Embedding Similarity

Use small fake embeddings first. Do not download a large Word2Vec model unless
you have time.

Task:

- create 6-10 words with 2D or 3D vectors,
- compute cosine similarity,
- show nearest neighbors,
- test one analogy-style vector operation.

Expected outputs:

- similarity matrix,
- top nearest neighbor for each word,
- one vector arithmetic example,
- one failure case.

Deliberate bugs to test:

- forget to normalize for cosine,
- use Euclidean distance but interpret it as cosine,
- compare row vectors with shape `(D,)` and `(D, 1)` accidentally.

Stop when:

- you can explain why static embeddings cannot handle context-dependent meaning.

## Drill 6: Dense Retrieval Mini-System

Use either fake embeddings or a lightweight sentence embedding model if already
available locally.

Task:

- assign one vector per document and query,
- compute `scores = query_embs @ doc_embs.T`,
- return top-k documents,
- compare with BoW/TF-IDF from earlier drills.

Expected outputs:

- query embedding shape `(num_queries, dim)`,
- document embedding shape `(num_docs, dim)`,
- score matrix shape `(num_queries, num_docs)`,
- top-k IDs.

Deliberate bugs to test:

- transpose missing in matrix multiplication,
- embeddings not normalized when cosine is expected,
- query/document order changes after scores are computed.

Contest reflex:

- Dense retrieval is still just shapes, scores, sorting, and IDs.

## Drill 7: BERT Token Batch Inspection

If transformers are installed and models are available, use a tiny BERT model or
`bert-base-uncased`. If not, write a mock batch with the same shapes.

Task:

- tokenize 4 sentences,
- print `input_ids`, `attention_mask`, and decoded tokens for one example,
- identify `[CLS]`, `[SEP]`, and padding,
- state the expected shape of `last_hidden_state`.

Expected outputs:

- `input_ids` shape,
- `attention_mask` shape,
- decoded first example,
- expected `last_hidden_state` shape.

Deliberate bugs to test:

- forget padding,
- forget truncation,
- attention mask has wrong dtype or shape,
- assume all sentences have the same real length.

Stop when:

- you can point to the exact token positions that should be ignored by pooling.

## Drill 8: Mean Pooling With Attention Mask

Use fake token embeddings first.

Task:

- create `token_embeddings` with shape `(batch, seq_len, hidden_dim)`,
- create `attention_mask` with shape `(batch, seq_len)`,
- implement mean pooling that ignores padding,
- verify one pooled vector by hand.

Expected outputs:

- pooled embedding shape `(batch, hidden_dim)`,
- hand-checked pooled vector,
- comparison against a wrong pooling version that averages padding.

Deliberate bugs to test:

- mask not expanded to hidden dimension,
- divide by total sequence length instead of real token count,
- denominator becomes zero.

Contest reflex:

- If sentence embeddings look bad, check pooling before blaming BERT.

## Drill 9: CLS Classification Contract

Use fake BERT outputs if needed.

Task:

- create a fake `last_hidden_state` with shape `(batch, seq_len, hidden_dim)`,
- extract `cls = last_hidden_state[:, 0, :]`,
- pass it through a linear classifier,
- compute cross entropy with integer labels.

Expected outputs:

- CLS shape,
- logits shape,
- labels shape and dtype,
- one loss value.

Deliberate bugs to test:

- labels are one-hot instead of integer class IDs,
- softmax is applied before cross entropy,
- classifier input dimension mismatches hidden dimension.

Stop when:

- you can state the exact contract of `CrossEntropyLoss` without checking notes.

## Drill 10: InfoNCE Similarity Matrix

Use fake query/document embeddings first.

Task:

- create `query_embs` and `doc_embs` with shape `(B, D)`,
- normalize both,
- compute `logits = query_embs @ doc_embs.T / temperature`,
- use labels `0..B-1`,
- compute cross entropy,
- inspect whether the diagonal is largest.

Expected outputs:

- similarity matrix shape `(B, B)`,
- labels,
- loss,
- diagonal vs off-diagonal comparison.

Deliberate bugs to test:

- batch size 1 gives no useful negatives,
- documents are shuffled but labels still assume diagonal positives,
- temperature too small makes logits extreme.

Contest reflex:

- InfoNCE is multiclass classification where the correct class is the diagonal.

## Drill 11: Full Mini-RAG Without An LLM

This is retrieval-only RAG scaffolding.

Task:

- build a small knowledge base,
- retrieve top-3 documents for a query,
- construct a prompt string with query plus retrieved context,
- do not call an LLM,
- inspect whether the context would be enough to answer.

Expected outputs:

- retrieved context IDs,
- assembled prompt,
- one "answerability" judgment.

Deliberate bugs to test:

- retrieved context does not include the answer,
- duplicate chunks dominate top-k,
- chunk IDs are lost after sorting.

Stop when:

- you can explain why generation quality depends on retrieval quality.

## Final Check

At the end, write a short self-review:

- Which bug took longest to find?
- Which shape contract do you still forget?
- Which retrieval metric can you compute by hand?
- Which part should be repeated tomorrow in under 20 minutes?

Strong signal:

- you debug by printing tokens, shapes, IDs, and scores,
- you use tiny examples before full data,
- you can explain the diagonal of InfoNCE,
- you can make a sparse and dense retrieval baseline.

Weak signal:

- you jump to a large pretrained model before a baseline works,
- you cannot tell whether a bad result is tokenization, ranking, or metric code,
- you trust average metrics without inspecting one query,
- you confuse logits, probabilities, labels, and loss.
-->
