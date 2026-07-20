# Stowaway

# Stowaway

You are a stowaway. The lab where you have worked for years has produced eight secret documents that need to leave the building. You have turned each document into a 256-bit message. The only thing you are allowed to carry past the inspection desk is a fine-tuned image classifier. The classifier is a small three-layer CNN trained on MNIST. Your job is to hide the eight messages inside copies of that network's weights and get them past three reviewers. Each reviewer checks something different, and each is one subtask.

## The model

The network is `StowawayCNN`.

```text
conv1: Conv2d(1, 16, kernel_size=3, padding=1)  -> ReLU -> MaxPool 2
conv2: Conv2d(16, 32, kernel_size=3, padding=1) -> ReLU -> MaxPool 2
fc:    Linear(32*7*7, 10)
```

It reaches about 99 percent top-1 on MNIST. You work with three weight tensors, `conv1.weight` (144 values), `conv2.weight` (4608), and `fc.weight` (15680), for 20 432 weights in total. The biases are off limits.

## How a message sits in the weights

A weight is a 32-bit float such as 0.003452. Multiply it by a million and round to the nearest integer, and you get the integer 3452, whose lowest bit you can overwrite before dividing back. This works on integers, not on the bit-level floating-point representation of the weight. For example, to plant a 1 bit:

```text
0.003452  ->  * 1e6  ->  round  ->  3452  ->  set lowest bit to 1  ->  3453  ->  / 1e6  ->  0.003453
```

The weight moves by about 1e-6, far below the precision the network actually uses, so its behaviour does not change, yet a bit is now planted in it. This is a least-significant-bit encoding (LSB), and 256 weights hold one 256-bit message. It is the basic primitive. Whether it survives depends on what a reviewer does to the network before reading it.

## What you receive

A starter kit with a Jupyter notebook and a `data/` folder.

| File | What it is |
|------|------------|
| `baseline_cnn.pt` | The frozen CNN, about 80 KB. |
| `reference_messages.json` | The eight reference messages you encode in subtask 3, each a 256-bit array. |
| `decoder_grammar.md` | The primitives behind the subtask-2 schemes. |
| `encoded_models/model_00.pt` .. `model_07.pt` | The eight modified networks for subtask 2. |
| `mnist_val_sample.npz` | 1000 MNIST validation images for the local subtask-1 check. |
| `mnist_finetune_batch.npz` | A finetune batch for the local subtask-3 check. |
| `Stowaway (RAW).ipynb` | Starter notebook with loaders, the exact local evaluators, and a runnable baseline for each part. |

The notebook lets you score yourself locally before you submit.

## What you submit

The three subtasks form one task worth 100 points, weighted 30, 35, and 35. You submit a single CSV with the columns `subtaskID,datapointID,answer`, setting `subtaskID` to 1, 2, or 3 for the three subtasks. Nothing you upload is executed. A grader reads your CSV and compares it to the hidden ground truth.

The three subtasks are worth 30, 35, and 35 points (100 total). Each subtask is calibrated so a random or trivial answer lands near 0 and a fully correct solution reaches its full weight, with partial credit in between. A mid-range score means your approach is on the right track but not yet complete, so it is worth pushing further.

## Subtask 1: the Prunability Desk (30 points)

The first reviewer suspects that smuggled data shows up as parasitic weights, so they ask you to certify which weights are redundant. You nominate exactly 12 000 of the 20 432 weights. The reviewer zeroes them in a clean copy of the network and confirms it still effectively classifies digits.

`datapointID` is the global weight index, ordered `conv1.weight` first (0 to 143), then `conv2.weight` (144 to 4751), then `fc.weight` (4752 to 20431). For example, datapointID 143 is the last `conv1.weight` value and 4752 is the first `fc.weight` value. The `answer` is 1 for a weight you prune and 0 otherwise. Your submission must contain exactly 12 000 ones: any other count scores 0 for this subtask.

Your submission is scored by token-level F1 against a fixed hidden reference: the 12 000 weights the network can most safely spare. Because both your submission and the reference contain 12 000 weights, this F1 equals the fraction of the reference you recover.

The reference is based on the Fisher information of each weight, a measure of how much the network's output depends on that weight. A weight with low Fisher information is one the predictions barely rely on, so zeroing it changes almost nothing, while a weight with high Fisher information strongly shapes the output and must be kept. The reference is the 12 000 weights of lowest Fisher information. Other importance measures capture only part of this signal and score below the Fisher reference.

## Subtask 2: the Decoder Desk (35 points)

At this desk you step into the reviewer's role: you are handed eight networks the lab has already modified, each carrying a 256-bit message planted by one of four schemes, two networks per scheme, and you must recover the message from each. A scheme is a choice of (i) parameter tensor, (ii) a 256-wide slice of its flattened weights, and (iii) a bit position. The bit position is a position inside the integer `int(round(w * 1e6))` from the primitive above, where bit 0 is the lowest bit, bit 1 the second lowest, and so on. Recover the message from each network.

For example, a scheme could hide the 256 message bits in `conv2.weight`, at flattened positions 1000 to 1255, in bit position 3. To read message bit `i` under that scheme you take the weight at position 1000 + i, compute `int(round(w * 1e6))`, and read bit number 3 of that integer. A different scheme changes one or more of the three choices, the tensor, the 256-wide slice, or the bit. The four schemes, and which network uses which, are hidden.

The value `datapointID = model * 256 + bit`, for model 0 to 7 and bit 0 to 255. For example, datapointID 0 is model 0's bit 0 and datapointID 257 is model 1's bit 1. The value `answer` is the recovered bit. Unlike the other subtasks, the scoring rule below is applied per network and then averaged: each network's bit-accuracy is mapped through the rule (floor 0.60, ceiling 1.00) to a value in 0 to 1, and the eight values are averaged and scaled to 35 points. So the eight networks share the 35 points equally, each worth about 4.4.

## Subtask 3: the Retraining Desk (35 points)

The third reviewer trains each network you hand over for ten standard gradient-descent steps on the MNIST classification task before reading it. Those gradient steps nudge every weight by a small amount, which erases the tiny changes an LSB message relies on. Write the eight reference messages into `fc.weight` in a way a fixed reader still recovers after that nudging.

For each of the eight networks you submit, for every weight of `fc.weight`, the change you made to that one weight: `delta = w_encoded - w_baseline` for that weight. So there is one delta per weight (15 680 per network), not an average or a norm over the tensor. `datapointID = model * 15680 + widx`, where `widx` indexes `fc.weight` flattened, and `answer` is that weight's delta. For example, datapointID 0 is model 0's first `fc.weight` change and 15680 is model 1's first. Every weight's delta must satisfy `|delta| <= 0.02` individually, so you cannot simply overwrite a weight. If any change in a network exceeds the cap, that entire network scores 0 for this subtask. Keeping every change within the cap is required. Clipping to the cap before submitting is the simplest way to guarantee it, but you may also keep the change small by design.

The reader is fixed and public, so you can test against it locally. It runs three steps on your changes for one network:

1. It adds a small random jitter, Gaussian noise with standard deviation 0.001, which is the size of the ten-step training drift.
2. It takes the discrete cosine transform of the jittered changes, `dct(delta + jitter, norm='ortho')`. This re-expresses your 15 680 weight changes as 15 680 frequency coefficients. A 1D Discrete Cosine Transform decomposes a sequence into a sum of cosines at different frequencies, with coefficient k the amplitude of the k-th cosine.
3. For message bit `i`, it adds up coefficients `64 + 61*i` through `124 + 61*i` and reads the bit as 1 when that sum is positive and 0 when it is negative.

Your score is the fraction of the 256 bits the reader recovers, averaged over the eight networks.

## Decoder Grammar (Subtask 2, decoder_grammar.md)

The eight networks in `encoded_models/` each carry a 256-bit message, planted with a least-significant-bit scheme. A scheme has three choices:

- A parameter tensor, one of `conv1.weight`, `conv2.weight`, `fc.weight`.
- A start offset into that tensor, flattened in row-major order.
- A bit position b.

Message bit i lives in element start + i of the flattened parameter. To read it, scale the weight by 1e6, round to the nearest integer, and take bit b.

Four distinct schemes were used, two networks each. The networks differ only in their weights, so finding one scheme tells you nothing about another until you find it too. You submit the recovered bits, not a decoder program. Each row is `subtaskID,datapointID,answer` with `datapointID = model * 256 + bit` and `answer` in {0, 1}.

## How points are scored

Each subtask turns its raw metric into points with the same rule:

```text
points = round( clamp( (metric − floor) / (ceil − floor), 0, 1 ) × max_points )
```

A metric at or below the **floor** scores 0; at or above the **ceiling** it scores the full points; in between it is linear, then rounded to an integer. Your task total is the sum of scores for the three subtasks, from 0 to 100.

| Subtask | Metric | Scores 0 at | Full points at | Max |
|---|---|---|---|---|
| 1 Prunability | agreement (token-F1) with the gold prune set | ≤ 0.583 | ≥ 0.90 | 30 |
| 2 Decoder | per-network bit-accuracy, averaged over the 8 networks | ≤ 0.60 (per network) | 1.00 (per network) | 35 |
| 3 Survival | bit-accuracy after the finetune, averaged over the 8 networks | ≤ 0.50 | ≥ 0.90 | 35 |

A random or trivial answer sits at or below every floor and scores 0; the reference solution reaches the full 100.
