# Generative Models Cheat Sheet

This note is for the syllabus names that are easy to confuse.

Use it to answer four questions fast:

1. What does this model try to do?
2. Does it generate new data?
3. What is the core mechanism?
4. How is it different from the nearby acronyms?

## First Distinction

`Generative` means the model learns enough about the data distribution to create
new samples.

Examples:

- generate text,
- generate images,
- reconstruct inputs from a latent code,
- predict the next token in a sequence.

`Discriminative` usually means the model predicts a label or score instead of
creating new content.

Examples:

- spam or not spam,
- cat or dog,
- positive or negative review.

## The Most Important Families

### RNN = Recurrent Neural Network

- What it does: processes sequences one step at a time.
- Typical use: text, time series, audio.
- Generative? Yes, if used to predict the next token repeatedly.
- Core idea: keep a hidden state that carries information from earlier steps.
- Main weakness: struggles with long-range dependencies.
- Difference:
  - simpler and older than `LSTM` and `GRU`,
  - much older than `Transformers`.

### LSTM = Long Short-Term Memory

- What it does: improved RNN for longer dependencies.
- Typical use: sequence modeling before Transformers took over.
- Generative? Yes, if used autoregressively.
- Core idea: gates decide what to keep, forget, and output.
- Main weakness: still sequential and slower to train than Transformers.
- Difference:
  - better memory than plain `RNN`,
  - more complex than `GRU`.

### GRU = Gated Recurrent Unit

- What it does: a simpler gated alternative to LSTM.
- Typical use: sequence tasks where you want RNN-style modeling with fewer
  parameters.
- Generative? Yes, if used for next-step prediction.
- Core idea: gating like LSTM, but with a lighter structure.
- Main weakness: still not as scalable as Transformers.
- Difference:
  - usually simpler than `LSTM`,
  - still much closer to `RNN` than to `Transformer`.

### Transformer

- What it does: models sequences using attention instead of recurrence.
- Typical use: modern NLP, large language models, vision transformers.
- Generative? Can be, depending on the setup.
- Core idea: each token can attend to other relevant tokens directly.
- Main strength: handles long context much better than RNN-family models.
- Difference:
  - `RNN/LSTM/GRU` read step by step,
  - `Transformer` uses attention over the sequence.

## Text Models You Must Separate

### Language Model

- What it does: predicts the next token given previous tokens.
- Typical use: text generation.
- Generative? Yes.
- Core idea: estimate probabilities like `P(next_token | previous_tokens)`.
- Difference:
  - this is a task/setup,
  - `GPT`, `LLaMA`, and old `RNN/LSTM/GRU` models can all be language models.

### BERT

- What it does: learns strong text representations for understanding tasks.
- Typical use: classification, tagging, retrieval, embeddings.
- Generative? Usually no, not in the free-form text-generation sense.
- Core idea: encoder-style Transformer trained to understand context deeply.
- Main strength: text understanding.
- Difference:
  - `BERT` is mainly for understanding,
  - `GPT` is mainly for generation.

### GPT

- What it does: autoregressive text generation.
- Typical use: chat, writing, code generation, completion.
- Generative? Yes.
- Core idea: decoder-only Transformer predicts one token at a time.
- Main strength: open-ended generation.
- Difference:
  - `GPT` writes text,
  - `BERT` usually scores or understands text.

### LLaMA

- What it does: a family of pretrained large language models.
- Typical use: text generation and instruction following.
- Generative? Yes.
- Core idea: same broad family as GPT, meaning decoder-style Transformer
  language modeling.
- Difference:
  - think of it as "another GPT-like family,"
  - not a separate generative principle like `GAN` or `VAE`.

### Seq2Seq = Sequence to Sequence

- What it does: maps one sequence to another.
- Typical use: translation, summarization, captioning.
- Generative? Yes, often.
- Core idea: encode the input sequence and generate an output sequence.
- Difference:
  - this is a general setup,
  - it can be implemented with `LSTM`s or `Transformers`.

### Encoder-Decoder Model

- What it does: one network reads the input, another generates the output.
- Typical use: translation and summarization.
- Generative? Yes.
- Core idea:
  - encoder builds a representation,
  - decoder produces the new sequence.
- Difference:
  - `GPT` is usually decoder-only,
  - `T5` is encoder-decoder.

### T5

- What it does: treats many NLP tasks as text-in, text-out.
- Typical use: summarization, QA, translation, rewriting.
- Generative? Yes.
- Core idea: Transformer encoder-decoder model.
- Difference:
  - `T5` is usually more task-to-task structured,
  - `GPT` is usually freer next-token generation.

## Latent-Space Generative Models

### Autoencoder

- What it does: compresses input into a smaller latent code, then reconstructs
  it.
- Typical use: compression, denoising, representation learning.
- Generative? Not strongly, by itself.
- Core idea: `encoder -> latent code -> decoder`.
- Main weakness: plain autoencoders do not naturally give a clean sampling
  process for brand-new data.
- Difference:
  - good for reconstruction,
  - weaker as a true generator than `VAE`, `GAN`, or `Diffusion`.

### VAE = Variational Autoencoder

- What it does: learns a probabilistic latent space and can sample from it.
- Typical use: controlled or smooth latent generation.
- Generative? Yes.
- Core idea: encode inputs into a latent distribution, then sample and decode.
- Main strength: principled sampling from latent space.
- Main weakness: outputs can be blurrier than GAN outputs.
- Difference:
  - `Autoencoder` reconstructs,
  - `VAE` reconstructs and gives a smoother generative latent space.

## Adversarial And Denoising Models

### GAN = Generative Adversarial Network

- What it does: generates samples by making one model fool another.
- Typical use: realistic image generation.
- Generative? Yes.
- Core idea:
  - `generator` makes fake samples,
  - `discriminator` tries to distinguish fake from real.
- Main strength: sharp outputs.
- Main weakness: unstable training, mode collapse.
- Difference:
  - `VAE` uses latent probabilistic modeling,
  - `GAN` uses competition between two networks,
  - `Diffusion` generates by denoising noise.

### Diffusion Model

- What it does: learns to reverse a process that gradually turns data into
  noise.
- Typical use: modern image generation, now also text/audio variants.
- Generative? Yes.
- Core idea: start from noise, denoise step by step into a sample.
- Main strength: high-quality and stable generation.
- Main weakness: generation is often slower than one-shot models.
- Difference:
  - `GAN` often generates in one pass,
  - `Diffusion` generates in many denoising steps.

### Stable Diffusion

- What it does: a well-known diffusion-based image generator.
- Typical use: text-to-image generation.
- Generative? Yes.
- Core idea: practical image-generation system built on diffusion ideas.
- Difference:
  - `Diffusion model` is the family,
  - `Stable Diffusion` is a famous concrete member of that family.

## Very Short Comparison Table

| Name | Main Job | Generative? | Main Idea | Main Weakness |
| --- | --- | --- | --- | --- |
| `RNN` | sequence modeling | sometimes | recurrent hidden state | weak long memory |
| `LSTM` | better sequence memory | sometimes | gated recurrence | older/slower than Transformers |
| `GRU` | lighter gated sequence model | sometimes | simplified gating | still recurrent |
| `Transformer` | attention-based sequence modeling | sometimes | self-attention | heavier than tiny RNNs |
| `BERT` | text understanding | mostly no | encoder Transformer | not free-form generator |
| `GPT` | text generation | yes | decoder-only Transformer | can hallucinate |
| `Autoencoder` | reconstruct/compress | weakly | encode then decode | not a clean sampler |
| `VAE` | latent generation | yes | probabilistic latent space | blurrier outputs |
| `GAN` | realistic sample generation | yes | generator vs discriminator | unstable training |
| `Diffusion` | high-quality generation | yes | denoise from noise | slower sampling |

## What To Learn First

Best order for first understanding:

1. `Language model`
2. `RNN`, `LSTM`, `GRU`
3. `Transformer`
4. `BERT` vs `GPT`
5. `Autoencoder`
6. `VAE`
7. `GAN`
8. `Diffusion`

Reason:

- first learn sequence generation,
- then learn modern text architecture,
- then learn the three big image/latent generation families.

## The Fastest Mental Picture

- `RNN/LSTM/GRU`: old sequence brains
- `Transformer`: modern attention brain
- `BERT`: understands text
- `GPT/LLaMA`: writes text
- `Autoencoder`: compress and rebuild
- `VAE`: sample from latent space
- `GAN`: generator vs judge
- `Diffusion`: turn noise into data

## Contest-Level Reflex

If you see a model name, ask:

1. Is it for understanding or generation?
2. Is it sequence-based, latent-space-based, adversarial, or denoising-based?
3. Does it generate in one shot, token by token, or step by step from noise?
4. What is the failure mode: weak memory, blurry samples, unstable training, or
   slow sampling?
