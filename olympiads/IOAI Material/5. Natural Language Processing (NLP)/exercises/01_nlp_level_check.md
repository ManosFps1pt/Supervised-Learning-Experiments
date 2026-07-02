# Sprint 01: NLP Level Check Without Full Notebooks

## Source

Use:

- `../sources/hias_presentation_ion_short Androutsopoulos.pdf` for embeddings,
  contextual models, BERT fine-tuning examples, moderation, and current NLP
  applications.
- `../sources/diffusion.pdf` for what a language model is, next-token
  probabilities, prompting, and the idea of diffusion-style text generation.
- `../sources/deep_generative_mdels.pdf` for autoencoders, variational
  autoencoders, latent spaces, and the final PEFT slide.
- `../sources/fine_tuning.pdf` for the PEFT reminder.
- `../sources/L05f - Initialization strategies.docx` as a short concept note:
  random initialization breaks symmetry, and values that are too large create
  unstable training.

## What To Study First

Read only these ideas before coding:

1. what a language model predicts,
2. tokenization and text-to-vector thinking,
3. embeddings and cosine-style similarity intuition,
4. BERT fine-tuning as an application idea,
5. autoencoder or VAE intuition at a conceptual level,
6. PEFT as a lightweight adaptation idea.

Do **not** start by trying to train a transformer, a diffusion model, or a VAE.
That would test tooling and patience more than understanding.

## Why I Am Telling You To Do This

You said you have never programmed a generative model.

So the correct first step is not "build a big model." The correct first step is:

- understand how text becomes tokens or vectors,
- understand how a next-token distribution works,
- build one tiny classical NLP pipeline,
- build one tiny generation-style pipeline from counts,
- leave modern generative models at the intuition level for now.

That matches your current level better and respects your schedule.

## Time Box

Target: **65-80 minutes total**.

- **15-20 minutes** reading the source ideas above.
- **20-25 minutes** Exercise A.
- **20-25 minutes** Exercise B.
- **10 minutes** reflection.

If time is tight, do Exercise A first. It is the highest-value one.

## What Your Jupyter Notebook Should Contain

Create one notebook with these sections:

1. `Source notes`
   - short bullets on language models, embeddings, VAE intuition, and PEFT.
2. `Exercise A - tiny language model`
3. `Exercise B - text vectors and similarity`
4. `Reflection`
   - what felt easy, what felt slow, what should be repeated before harder NLP.

Keep the notebook compact. This is a level check, not a research notebook.

## Exercise A: Tiny Bigram Language Model

Build the smallest useful generative-model warm-up.

Task:

- write a tiny corpus of 10-20 short sentences,
- tokenize it consistently,
- count unigrams and bigrams,
- estimate `P(next_word | current_word)` from counts,
- show the top predicted next words for 3 prompts,
- generate 3 short sequences by repeatedly choosing the next word from your
  estimated distribution.

Produce:

- token counts,
- one printed conditional-probability table for a chosen word,
- three prompt examples with predicted next words,
- three short generated sequences,
- one paragraph explaining why this is already a generative model.

Tokenization note:

- For this exercise, "tokenize consistently" means use one simple rule for all
  sentences.
- Good default rule:
  - lowercase the sentence,
  - remove punctuation,
  - split by whitespace.
- Example:
  - `"I like pizza."` -> `["i", "like", "pizza"]`
  - `"You like pasta!"` -> `["you", "like", "pasta"]`
- Do not mix rules across sentences. For example, do not keep `"Pizza"` in one
  place and `"pizza"` in another, and do not sometimes keep punctuation as
  tokens and sometimes remove it.
- `word2vec` is not a tokenizer. It is a later representation method. For this
  warm-up, plain word-level splitting is enough.

Tiny corpus seed if you do not want to invent sentences:

- `i like pizza`
- `i like pasta`
- `you like pizza`
- `you cook pasta`
- `we cook dinner`
- `we like dinner`
- `they cook rice`
- `they like rice`

Reason:

This is the safest and cleanest first coding step into generative modeling for
your level. It teaches the core contract of language modeling without requiring
heavy libraries, GPUs, or a missing notebook.

## Exercise B: Text Vectors And Similarity

Build a tiny classical NLP pipeline on short texts.

Task:

- create 8-12 short texts from 2-3 themes,
- vectorize them with `CountVectorizer` or `TfidfVectorizer`,
- compute cosine similarities,
- inspect which texts are closest to each other,
- optionally train a tiny classifier if you add labels.

Produce:

- raw texts,
- vocabulary size,
- similarity matrix,
- top-2 nearest neighbors for at least 3 texts,
- one short interpretation of a failure case.

Reason:

This gives you the vector-space intuition behind the embeddings slides before
you touch pretrained models or fine-tuning.

## Optional Reflection Prompt

Write 4-6 lines answering:

- What is the difference between a vector representation and a next-token model?
- Why is a bigram model weak but still useful?
- Why am I not starting from diffusion or VAE code yet?
- What would PEFT help with if I later fine-tuned a larger model?

## Clear Goals

By the end of this sprint you should be able to:

- explain what a language model predicts,
- build a token-count-based next-word generator,
- explain what a text vector is doing,
- use cosine similarity on text features,
- describe autoencoders, VAEs, and PEFT at a high level without pretending you
  can already train them well.

## Stop Condition

Stop when:

- Exercise A is complete,
- Exercise B is complete or at least mostly complete,
- your reflection clearly states whether you are ready for harder NLP work or
  still need another classical-NLP pass.

Do not waste time trying to jump straight to a polished transformer notebook.

## Level Signal

Strong signal:

- you can explain your bigram probabilities in words,
- your generated text reflects the corpus structure,
- your similarity results mostly make sense,
- your reflection is honest about limits.

Weak signal:

- tokenization is inconsistent,
- probabilities do not sum to something sensible,
- you cannot explain why similar texts end up close,
- you try to skip directly to advanced generative code before understanding the
  basic pipeline.
