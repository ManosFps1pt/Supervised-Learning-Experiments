# Exercise 02: MNIST Personal Benchmark

## Why This Exercise Exists

MNIST is not new for you.

That is exactly why it is useful now.

You have already used MNIST many times, including the nationals problem:

`olympiads/nationals problems/02_robust_mnist_classifier.ipynb`

So this exercise is not about first contact with image classification. It is
about rebuilding a familiar benchmark with a cleaner CV workflow and better
library usage under CEOAI-style time pressure.

Use this as your core CV exercise if you choose the MNIST route.

## Why MNIST Is A Good Benchmark For You

MNIST is personal benchmark material for you because:

- you already know the data shape and task
- you have prior intuition for what "normal" progress should look like
- you have already seen where your past pipeline broke down
- this lets you focus on speed, correctness, and workflow instead of novelty

The point of this exercise is to prove that you now handle the same family of
problem more cleanly than before.

## Reference Notebook To Study First

Read this notebook before starting:

`olympiads/nationals problems/02_robust_mnist_classifier.ipynb`

Do not copy it mechanically. Read it as a post-mortem.

## What Went Wrong In The Nationals Notebook

This is the important part.

The old notebook shows several concrete problems:

1. `torchvision` was imported, but not used in a clean or meaningful way.
   - `from torchvision import transforms as T` appears in the notebook.
   - The intended transform pipeline was commented out instead of becoming the
     real data path.
   - The commented block even shows an incomplete `T.ToTensor` usage rather than
     a complete transform call.

2. The noisy dataset was built in a slow way.
   - The notebook constructs `x_noisy_set` with a Python list of NumPy arrays,
     then wraps it with `torch.Tensor(...)`.
   - PyTorch emitted a warning that creating a tensor from a list of
     `numpy.ndarrays` is extremely slow.

3. The perturbation path was handwritten instead of becoming a structured data
   pipeline.
   - The notebook used a custom `randomize_img` function over each sample.
   - That may be fine for exploration, but it is not the cleanest contest-speed
     workflow when you want reliable shape and transform control.

4. The training loop had no real validation feedback.
   - The evaluation code was mostly commented out.
   - That means you were training without a proper visible test or validation
     accuracy signal.

5. The training setup was too weak to be a convincing baseline.
   - The model trained for 15 epochs with `Adam(..., lr=1e-5)`.
   - The printed training loss still ended around `1.324`, which is far too weak
     for a benchmark dataset like MNIST.

6. The exercise context was robustness-heavy, but the pipeline evidence was
   baseline-light.
   - The contest task asked for robust behavior under perturbation.
   - But the notebook still needed a much cleaner ordinary baseline and metric
     view before robustness claims could become trustworthy.

## Main Lesson

The lesson from that notebook is:

Do not treat `torchvision` as a decorative import.

Use it as part of a real image workflow:

- dataset loading
- transform definition
- tensor shape control
- clean baseline training
- visible evaluation

## Your New Goal

Build a cleaner MNIST benchmark notebook that proves:

1. you can structure the CV pipeline properly
2. you can use `torchvision` intentionally
3. you can produce visible metrics fast
4. you can compare your current result against your old MNIST habits

## Required Notebook

Create or use this notebook path:

`olympiads/IOAI Material/8. Computer Vision/exercises/cv_mnist_personal_benchmark.ipynb`

## Concrete Route

Use exactly this route unless you hit a blocker:

- dataset: `torchvision.datasets.MNIST`
- framework: `torch` + `torchvision`
- task: digit classification
- baseline model: a small CNN
- metric: validation accuracy

Do not begin with adversarial robustness again.

First rebuild the clean classification baseline correctly.

## What The Notebook Must Show

Your notebook must contain executed evidence for all of these:

1. dataset shapes
2. one batch shape
3. one image grid or preview
4. the transform pipeline you actually used
5. the model choice
6. training loss across epochs
7. validation accuracy
8. a small table or display of predictions vs true labels

If one of these is missing, the artifact is incomplete.

## Explicit Checks You Must Perform

Before training:

1. confirm image shape
2. confirm batch shape
3. confirm label dtype
4. confirm pixel range after transforms
5. confirm the model accepts the batch shape without crashing

During training:

1. print epoch loss
2. compute validation accuracy
3. stop if the metric is clearly broken instead of training blindly

After training:

1. inspect wrong predictions
2. name one reason the old notebook was weaker
3. name one thing that is cleaner now

## `torchvision` Reflex To Build

This is the CV habit you are trying to build:

- use `torchvision.datasets` to load image data
- use `torchvision.transforms` to define the image pipeline
- avoid ad-hoc conversion chains unless you have a reason
- treat transforms as part of the model pipeline, not as decoration

## Self-Check Questions

At the end of the notebook, write short answers to these:

1. What did `torchvision` help me do here that plain NumPy would have made more
   awkward?
2. Why was the old `x_noisy_set = torch.Tensor([...])` pattern a bad sign?
3. What metric told me this notebook was healthier than the nationals version?
4. If I had to turn this into a CEOAI sprint artifact in one hour, what would I
   keep and what would I drop?

## Stretch Task

Only if the clean baseline works first:

- add one controlled perturbation or augmentation step
- compare clean accuracy and perturbed accuracy
- write one sentence about whether the model is brittle

Do not do this stretch task before the clean baseline is visible.

## Stop Conditions

Stop when:

- you have a clean executed MNIST baseline notebook with validation accuracy and
  prediction examples
- you have written the short comparison against the old nationals notebook

Do not spend this block on:

- leaderboard chasing
- robustness tricks before the clean baseline works
- fancy architectures
- rewriting the whole notebook just for style

## What Success Means

Success is not "I used MNIST again."

Success is:

- I reused a personal benchmark on purpose
- I used `torchvision` as an actual CV tool
- I produced a cleaner and more informative artifact than the nationals version
