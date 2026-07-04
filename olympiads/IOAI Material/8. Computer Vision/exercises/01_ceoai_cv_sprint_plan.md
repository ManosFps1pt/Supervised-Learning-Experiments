# Sprint 01: CEOAI CV Study Plan

## Why This Exists

You are preparing for CEOAI first, not IOAI first.

This means the goal is not to study all of Computer Vision. The goal is to
cover the CEOAI CV rows with the smallest useful artifact that proves you can:

- recognize the task type
- choose a sensible baseline
- write the code fast
- inspect outputs without getting lost in theory

Use this sprint only after the current higher-priority Search/RL gap is either
closed or explicitly deferred for this block.

## CEOAI Target

This sprint maps to the local CEOAI CV syllabus:

- `5(a)` Processing: filtering, edge detection, HOG
- `5(b)` CNN architectures: AlexNet, VGG, ResNet, Inception, EfficientNet
- `5(c)` Related architectures: YOLO, Stable Diffusion, Vision Transformers

## What CV Is

Computer Vision is the part of AI where the input is an image and the output is
usually one of:

- one label for the whole image
- one or more objects with locations
- one label per pixel region

For this sprint, treat the three main task types as:

- `classification`: what is in the image?
- `detection`: where is it and what is it?
- `segmentation`: which pixels belong to which object?

## What You Already Have

You are not starting from zero.

Local repo evidence already suggests you have:

- NumPy and pandas basics
- scikit-learn basics and metrics
- PyTorch basics
- neural-network classification/regression practice
- some CNN/vision foundations from older material

So this sprint should not turn into a general introduction to Python or ML.

## Default Libraries

Default tool stack for this sprint:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `torch`
- `torchvision`
- `PIL`

Do not start with a custom image pipeline unless the default route fails.

## Required Artifact

Create exactly one notebook at:

`olympiads/IOAI Material/8. Computer Vision/exercises/cv_baseline_digits_or_cifar.ipynb`

The notebook must contain executed cells showing:

- dataset shape
- one small image preview
- train/validation split
- one model baseline
- one metric such as validation accuracy or loss
- a small table or display of predictions vs true labels

This is the minimum artifact that counts as real progress.

## Default Route

Choose one route only.

### Route A: Fastest Win

Use:

- `sklearn.datasets.load_digits()`

Why:

- tiny dataset
- fast training
- fast debugging
- good for proving the full pipeline quickly

### Route B: More Real CV

Use:

- `torchvision.datasets.CIFAR10`

Why:

- closer to real image-classification workflows
- good practice for transforms, channels, and CNNs

If you are short on time or energy, choose Route A first.

## Suggested Model Choice

Use the smallest baseline that closes the artifact:

- for `load_digits()`: a simple baseline classifier or a tiny MLP/CNN
- for `CIFAR10`: a tiny CNN first

Do not jump straight into YOLO, ViT, or diffusion for the first artifact.

## Concepts To Learn Briefly

### Processing

Know these at recognition level first:

- `filtering`: modifies local pixel neighborhoods
- `edge detection`: highlights strong intensity changes
- `HOG`: hand-crafted feature descriptor based on gradient directions

Your goal is to know what kind of problem each one helps with, not to derive
the math from scratch.

### CNN Architectures

Know the role of:

- `AlexNet`: early influential CNN
- `VGG`: deeper stacked convolutions
- `ResNet`: skip connections, strong practical baseline
- `Inception`: multi-scale feature extraction
- `EfficientNet`: scaled family balancing accuracy and efficiency

For competition speed, `ResNet` is the most important name to recognize.

### Related Architectures

Know these at quick-recognition level:

- `YOLO`: object detection
- `ViT`: transformer over image patches
- `Stable Diffusion`: image generation, lower priority for this sprint

## Fast Coding Reflexes

Before training anything, always check:

1. input shape
2. label shape
3. label dtype
4. number of classes
5. one batch or one image preview

During the sprint, prefer these habits:

- print shapes early
- test one batch before full training
- use one simple metric
- keep notebook cells short
- stop after the first valid baseline works

## What Completion Looks Like

This sprint is complete only if you can say:

1. I built one working image-classification baseline.
2. I can explain the difference between classification, detection, and
   segmentation.
3. I know when HOG, ResNet, YOLO, and ViT are the relevant names.
4. I have one saved notebook with visible metric and predictions.

## Small Recognition Challenge

At the end of the notebook, add five short written answers:

1. When would I use edge detection instead of a classifier?
2. What does HOG try to preserve from the image?
3. Why is ResNet a more practical baseline than memorizing AlexNet details?
4. What extra output does YOLO produce compared with classification?
5. What changes when a ViT sees an image as patches instead of local
   convolution windows?

## Stop Conditions

Stop the sprint when one of these is true:

- the notebook artifact is complete
- the notebook runs but a shape/device/metric bug blocks progress, and you have
  written down the exact blocker for debugging next

Do not spend the block on:

- broad theory reading
- collecting more slides or PDFs
- coding detection or segmentation from scratch
- polishing notebook style before the baseline works
- studying diffusion deeply before classification is covered

## Next Step After This Sprint

If this notebook works, the best follow-up is one of:

- a short HOG + edge-detection recognition drill
- a transfer-learning notebook using a pretrained vision encoder
- a YOLO-style detection note explaining inputs, outputs, and evaluation at a
  practical level
