# IOAI Preparation Repository Handoff

This file summarizes the work done in `D:\projects\Supervised-Learning-Experiments` so it can be copied into a new IOAI preparation repository and give a future coding agent enough context to continue from here.

## High-Level Purpose

This workspace was used as a preparation sandbox for IOAI / PDTN-style machine learning problems. The work covers:

- Classical supervised learning with NumPy and scikit-learn.
- PyTorch basics: tensors, datasets, dataloaders, CNNs, optimizers, schedulers, GPU training, and reusable training loops.
- Computer vision challenges: MNIST, CIFAR-10, CIFAR-100, emotion classification attacks, deepfake classification, and image-to-function neural approximation.
- NLP / transformer practice with Hugging Face on IMDB sentiment classification.
- Competition preparation documents: syllabus summaries, cheat sheets, study plans, and performance/debugging notes.
- Submission-style output generation for challenge platforms, usually as JSON files.

The repo is not yet organized like a polished package. It is a collection of notebooks, exported scripts, challenge assets, generated notes, and experiments. A new repo should keep the useful learning artifacts, but avoid copying heavy generated datasets, virtual environments, caches, and duplicated notebooks unless needed.

## Current Git State

At the time this handoff was generated, `git status --short` was clean. There were no uncommitted tracked changes.

## Main Concepts Practiced

### Classical Machine Learning

- Implemented entropy and information gain manually in `decision_trees.py`.
- Built a toy recursive decision tree from scratch using NumPy.
- Used tabular datasets such as Iris, Wine, and Diabetes in notebooks.
- Studied common supervised-learning concepts:
  - train/test split
  - preprocessing
  - model evaluation
  - decision trees
  - regression/classification distinction
  - overfitting and validation

### PyTorch Fundamentals

- Built reusable training loops with:
  - `model.train()` and `model.eval()`
  - `CrossEntropyLoss`
  - `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
  - validation accuracy tracking
  - best-model checkpointing with `copy.deepcopy(model.state_dict())`
  - `StepLR` and `CosineAnnealingLR`
  - CPU/GPU device handling
  - `pin_memory=True` and `non_blocking=True`
- Practiced CNN architecture design for MNIST, CIFAR-10, and CIFAR-100.
- Practiced transfer learning with `torchvision.models.resnet18`.
- Learned that `CrossEntropyLoss` expects raw logits, so the model output should not pass through softmax before the loss.

### Computer Vision

- MNIST digit classification with a small CNN.
- CAPTCHA-style equation solving by splitting an image into digit/operator windows.
- Noise augmentation and denoising for noisy MNIST/CAPTCHA data.
- CIFAR-10 CNN from scratch with data augmentation.
- CIFAR-100 transfer learning with ResNet-18, warm-up phase, full fine-tuning phase, label smoothing, and cosine LR schedules.
- Emotion-classification challenge using a FER+ ShuffleNet model and sparse L1 adversarial pixel attacks.
- Deepfake classification practice in `pdtn2025` / `challenges/deepfakes.ipynb`.
- Image approximation challenge where a small MLP learns a function `(x, y) -> (r, g, b)`.

### NLP / Transformers

- Created `challenges/nlp.py` for a Hugging Face IMDB sentiment classifier.
- Used:
  - `datasets.load_dataset("imdb")`
  - `AutoTokenizer.from_pretrained("bert-base-uncased")`
  - `AutoModelForSequenceClassification`
  - `Trainer`
  - `evaluate.load("accuracy")`
  - `pipeline("sentiment-analysis")`
- Saved model/tokenizer outputs to `./my_bert_sentiment`, which is ignored in `.gitignore`.

### Competition Strategy

- Created IOAI-focused study materials and cheat sheets.
- Emphasis was placed on rapid templates rather than memorization.
- The intended competition workflow is:
  1. Build a simple baseline quickly.
  2. Verify data shapes and target labels.
  3. Add augmentation/preprocessing.
  4. Tune learning rate, batch size, optimizer, regularization.
  5. Save submissions in the exact required format.
  6. Keep reusable templates for future tasks.

## Important Files and What They Contain

### Root-Level Python Scripts

- `train_function.py`
  - Reusable PyTorch training loop.
  - Handles train/test loss, accuracy, StepLR, best model weights, and loss/accuracy plots.
  - Uses `CrossEntropyLoss` internally.
  - Useful as a base template, but could be cleaned so the criterion is passed in and metrics are returned instead of just plotted.

- `mnist.py`
  - MNIST CNN training script.
  - Downloads MNIST through `torchvision.datasets.MNIST`.
  - Defines a simple 2-convolution CNN with fully connected layers.
  - Uses `train_function.train`.
  - Good starter example for image classification.

- `decision_trees.py`
  - NumPy-only entropy, information gain, best feature split, and recursive tree construction.
  - Good for understanding decision trees theoretically.
  - It is educational code, not production-quality. The feature iteration assumes a particular orientation of the data array.

- `emotions.py`
  - Exported notebook/script for the emotion adversarial challenge.
  - Downloads FER+ model and sample images with `gdown`.
  - Loads `model_ferplus.pth` into a grayscale ShuffleNet v2 model.
  - Implements:
    - `loadImage`
    - `tensorToImage`
    - `sparse_l1_attack`
    - `generate_pair`
    - `untargeted_sparse_attack`
    - `compare_images`
    - JSON export to `answers.json`
  - Important lesson: use `.clone().detach()`, `.requires_grad_()`, `.view(-1)`, `.argsort()`, `.clamp()`, and `.round()` carefully for pixel-level adversarial attacks.

- `emotions_draft.py`
  - Earlier/draft version of the emotions work.
  - Only copy if you want historical attempts; otherwise `emotions.py` is the cleaner artifact.

- `captcha.py`
  - Exported Colab notebook for the MNIST CAPTCHA challenge.
  - Downloads clean/noisy public/private CAPTCHA images and labels.
  - Splits each equation image into digit and operator windows.
  - Trains separate CNN models for clean and noisy digit recognition.
  - Includes salt-and-pepper noise augmentation, largest connected component denoising, and a simple operator heuristic.
  - Exports answers to `answers.json`.
  - It contains a lot of notebook/Colab residue and some mojibake from Greek text encoding, so keep it as a reference but consider rewriting a clean version in the new repo.

- `answers.json`
  - Generated challenge submission file.
  - Only copy if it is still relevant to a challenge. In a clean repo, generated submissions should usually go under `outputs/` or be ignored.

### Root-Level Notebooks

- `dicision_trees.ipynb`
  - Notebook for decision-tree experimentation. Filename has a typo (`dicision`).

- `emotions.ipynb`
  - Original notebook version of `emotions.py`.
  - Contains the emotion model loading, attack functions, image comparison, and answer export.

### Root-Level Assets

- `model_ferplus.pth`
  - Pretrained FER+ emotion classifier weights used by `emotions.py`.
  - Required if running the emotion challenge offline.
  - It is about 5 MB, so it is reasonable to copy if the new repo keeps the emotion challenge.

- `angry.png`, `happy.png`, `neutral.png`
  - Input images for the emotion challenge.
  - Required by `emotions.py` unless the script downloads them again.

- `public-clean.png`, `public-clean.txt`, `public-noisy.png`, `public-noisy.txt`, `private-clean.png`, `private-noisy.png`
  - CAPTCHA challenge files.
  - These are generated/downloaded challenge assets. Copy only if you want the new repo to run without re-downloading.

- `diabetes.csv`
  - Tabular dataset used for classical ML practice.
  - Small enough to copy if you want local tabular practice data.

- `output1.png`, `output2.png`
  - Generated image outputs. Likely not essential unless they document a result you care about.

### `challenges/`

This is the main challenge-practice directory.

- `challenges/ioai_material.md`
  - Extracted IOAI 2026 syllabus summary.
  - Covers foundational skills, classical ML, neural networks, deep learning, CV, NLP, audio, and data modalities.
  - Some text has mojibake encoding artifacts, but the content is still useful.

- `challenges/ioai_7_day_study_plan.md`
  - Original 7-day study plan.

- `challenges/optimized_7_day_study_plan.md`
  - Better fast-track plan for a 1-week IOAI sprint.
  - Focuses on building reusable code templates and spending most time coding/debugging.

- `challenges/resources_dump.md`
  - Markdown table of resources from the IOAI educational resources spreadsheet.
  - Contains resource type, title, author/institution, contents, level, and placeholder links.

- `challenges/IOAI - Educational Resources.xlsx`
  - Source spreadsheet for resource planning.

- `challenges/Syllabus.pdf`
  - Official or extracted IOAI syllabus PDF.

- `challenges/cfar1.py`
  - CIFAR-10 CNN training script with augmentation.
  - Uses random crop, horizontal flip, normalization, custom CNN, Adam, weight decay, and a validation loop.
  - Trains for 60 epochs with batch size 256 and DataLoader workers/pinned memory.

- `challenges/cfar10.ipynb`
  - Notebook version of CIFAR-10 training.

- `challenges/CFAR1001.py`
  - CIFAR-100 ResNet-18 transfer learning experiment.
  - Uses stronger augmentation: crop, flip, rotation, color jitter, normalization.
  - Freezes pretrained backbone, replaces `conv1`, removes maxpool, replaces `fc`.
  - Uses two training phases:
    - Phase 1 warm-up: train new `conv1` and `fc`.
    - Phase 2 fine-tune: unfreeze all layers with lower LR.
  - Uses label smoothing and fresh optimizers/schedulers per phase to avoid exhausted scheduler bugs.

- `challenges/cfar100.ipynb`
  - Notebook version of CIFAR-100 work.

- `challenges/mnist.ipynb` and `challenges/mnist1.py`
  - MNIST CNN training practice.
  - `mnist1.py` includes `torch.backends.cudnn.benchmark = True` and a cleaner training loop signature with external criterion.

- `challenges/captcha.ipynb`
  - Notebook version of CAPTCHA work.

- `challenges/squarepainting.py` and `challenges/squarepainting.ipynb`
  - Neural image approximation challenge.
  - Defines target functions:
    - `basic_f(x, y) = (x, y, 0)`
    - `easy_f`
    - `medium_f`
    - `advanced_f` based on `rubiks.jpg`
    - `hard_f` based on `cat.jpg`
  - Defines fixed MLP architecture with input size 2 and RGB output size 3.
  - Generates coordinate/RGB datasets with `TensorDataset`.
  - Trains one network per level and exports all weights to `answer.json`.

- `challenges/copy_of_squarepainting.py`
  - Duplicate/copy of square painting work. Prefer `squarepainting.py`.

- `challenges/nlp.py` and `challenges/nlp.ipynb`
  - Hugging Face BERT/IMDB sentiment classification practice.

- `challenges/nextmovie.ipynb`
  - Embedding/optimization-style challenge involving movie data in `movies.json`.
  - Headings indicate data loading, model initialization, level solving, optimization function, and subtasks.

- `challenges/deepfakes.ipynb`
  - Deepfake classification notebook.
  - Headings indicate data download, dataset creation, model definition, training, and answer export.

- `challenges/emotions_practice.ipynb`
  - Practice version of emotion adversarial challenge.

- `challenges/iris.ipynb`, `challenges/wine.ipynb`
  - Classical ML dataset practice notebooks.

- `challenges/embeddings.json`, `challenges/levels.json`, `challenges/movies.json`
  - Data files for challenge notebooks.

- `challenges/rubiks.jpg`, `challenges/cat.jpg`
  - Image inputs for square painting advanced/hard levels.

- `challenges/answer.json`
  - Generated submission output for square painting or another challenge.
  - Usually do not copy to a clean repo unless you want to preserve a solved submission.

- `challenges/extracted.py`
  - Extracted script; inspect before relying on it.

### `comp/`

This directory contains competition notes, study plans, and helper code.

- `comp/ioai_focused_syllabus.md`
  - Streamlined syllabus focused on scikit-learn models, XGBoost, neural networks, optimizers, and embeddings.

- `comp/ioai_competition_cheat_sheet.md`
  - High-value cheat sheet with:
    - pandas/NumPy EDA
    - missing value handling
    - one-hot encoding
    - scaling
    - model evaluation
    - Random Forest/XGBoost template
    - PyTorch training loops
    - CNN template
    - transfer learning snippet
    - embedding layer snippet
    - tensor debugging functions
    - competition debugging checklist
  - Some headings have mojibake characters, but the snippets are useful.

- `comp/pdtn_phase_b_notes_study_plan.md`
  - Study plan generated from PDTN Phase B notes.

- `comp/pdtn_phase_b_offline_notes.md`
  - Offline notes for PDTN Phase B.

- `comp/tutor_advice_classification_vs_regression.md`
  - Short conceptual note explaining classification vs regression.

- `comp/training_bug.md`
  - PyTorch performance/debugging note for CIFAR-10 training.
  - Explains GPU underutilization, pinned memory, `non_blocking=True`, DataLoader workers, first epoch slowness, RAM usage, and recommended settings.
  - The file currently contains pasted HTML/code-block markup, but the content is important. Consider cleaning this file in the new repo.

- `comp/train_func.py`
  - Cleaner reusable PyTorch training loop similar to the one in `mnist1.py`.

- `comp/PDTN_Phase_B_Notes_GR.pdf`, `comp/odhgies.docx`, generated `.html` and `.pdf` files
  - Reference/study materials and rendered outputs.
  - Copy only if the new repo will include local competition references.

### `Numpy Tutorial/`

- `Numpy Tutorial/numpy_tutorial.ipynb`
  - Tutorial notebook covering NumPy basics:
    - why NumPy
    - arrays
    - dtypes
    - multidimensional arrays
  - Useful if the new repo includes foundational notes.

### `tutorials/`

- `tutorials/Copy_of_GenAI_Coding_Lecture_3_CIFAR10_Embedding.ipynb`
  - CIFAR-10 embedding/linear separability lecture notebook.
  - Useful as reference material, but not core to the main IOAI templates.

### `pdtn2025/`

This directory is ignored by `.gitignore` and contains large PDTN challenge artifacts:

- `pdtn2025/deepfakes.zip`
- `pdtn2025/deepfakes/`
- `pdtn2025/deepfakes_final.ipynb`
- `pdtn2025/knit.zip`
- `pdtn2025/knit/`
- `pdtn2025/weights.png`
- Greek-named notebooks for embeddings and knit challenges.

Do not blindly copy the whole directory. It contains large datasets and many image files. If you want to preserve PDTN work, copy only the final notebooks and small reference assets, then add instructions for downloading datasets separately.

## Environment and Dependencies

The experiments use Python with the following libraries:

- `torch`
- `torchvision`
- `torchinfo`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `Pillow`
- `opencv-python` / `cv2`
- `gdown`
- `datasets`
- `transformers`
- `evaluate`
- `jupyter`
- `nbconvert` / `nbformat` if converting notebooks

Recommended new-repo setup:

```txt
torch
torchvision
torchinfo
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
pillow
opencv-python
gdown
datasets
transformers
evaluate
jupyter
```

For GPU work, install the PyTorch build matching the local CUDA version. This project was tested on a Windows machine with an NVIDIA RTX 4070 and Ryzen 7 5700X according to the training performance notes.

## Important Training Lessons Learned

- For vision tensors, always verify shapes. PyTorch CNNs usually expect `(batch, channels, height, width)`.
- `CrossEntropyLoss` expects class indices and raw logits.
- Use `model.train()` during training and `model.eval()` during validation/inference.
- Wrap validation/inference in `torch.no_grad()`.
- Always zero gradients before backpropagation.
- Use `.to(device, non_blocking=True)` together with `DataLoader(pin_memory=True)` for asynchronous GPU transfers.
- Too many DataLoader workers can increase RAM usage and context switching. For the local 8-core/16-thread CPU, `num_workers=6` to `8` was considered a stable range.
- First epochs can be slower due to CUDA initialization, cuDNN benchmarking, DataLoader worker startup, and OS cache warm-up.
- For transfer learning, freeze the backbone first, train the new head, then unfreeze and fine-tune with a lower learning rate.
- Use fresh optimizers/schedulers for distinct training phases.
- Label smoothing can help with many-class image classification such as CIFAR-100.
- For adversarial image attacks, avoid in-place operations on tensors that require gradients. Work with cloned/detached tensors.

## What To Copy Into the New IOAI Repo

### Strongly Recommended

Copy these because they are useful for continuing IOAI prep:

- `IOAI_REPO_HANDOFF.md`
- `.gitignore`
- `train_function.py`
- `comp/train_func.py`
- `mnist.py`
- `decision_trees.py`
- `challenges/cfar1.py`
- `challenges/CFAR1001.py`
- `challenges/mnist1.py`
- `challenges/nlp.py`
- `challenges/squarepainting.py`
- `challenges/ioai_material.md`
- `challenges/optimized_7_day_study_plan.md`
- `challenges/resources_dump.md`
- `challenges/Syllabus.pdf`
- `challenges/IOAI - Educational Resources.xlsx`
- `comp/ioai_focused_syllabus.md`
- `comp/ioai_competition_cheat_sheet.md`
- `comp/training_bug.md` after cleaning the pasted HTML markup
- `Numpy Tutorial/numpy_tutorial.ipynb` if you want foundational reference material

### Copy If You Want the Challenge To Run Offline

- Emotion challenge:
  - `emotions.py`
  - `emotions.ipynb`
  - `model_ferplus.pth`
  - `angry.png`
  - `happy.png`
  - `neutral.png`

- CAPTCHA challenge:
  - `captcha.py`
  - `challenges/captcha.ipynb`
  - `public-clean.png`
  - `public-clean.txt`
  - `public-noisy.png`
  - `public-noisy.txt`
  - `private-clean.png`
  - `private-noisy.png`

- Square painting challenge:
  - `challenges/squarepainting.py`
  - `challenges/squarepainting.ipynb`
  - `challenges/rubiks.jpg`
  - `challenges/cat.jpg`

- NLP / embedding challenges:
  - `challenges/nlp.py`
  - `challenges/nlp.ipynb`
  - `challenges/nextmovie.ipynb`
  - `challenges/movies.json`
  - `challenges/embeddings.json`
  - `challenges/levels.json`

- Classical data practice:
  - `diabetes.csv`
  - `challenges/iris.ipynb`
  - `challenges/wine.ipynb`

### Copy Only If You Want Historical/Reference Material

- `dicision_trees.ipynb` (typo in filename; consider renaming)
- `emotions_draft.py`
- `challenges/copy_of_squarepainting.py`
- `tutorials/Copy_of_GenAI_Coding_Lecture_3_CIFAR10_Embedding.ipynb`
- `comp/*.browser.html`
- `comp/*.browser.pdf`
- `comp/*.highlighted.html`
- `comp/*.highlighted.pdf`
- `output1.png`
- `output2.png`

### Do Not Copy Into a Clean Repo

Avoid copying these unless there is a specific reason:

- `.venv/`
- `.cache/`
- `.mypy_cache/`
- `__pycache__/`
- `.idea/`
- `.obsidian/`
- `data/`
- `challenges/data/`
- `graphify-out/`
- generated model output folders such as `results/`, `my_bert_sentiment/`, `challenges/my_bert_sentiment/`
- full `pdtn2025/deepfakes/` extracted dataset
- full `pdtn2025/deepfakes.zip` unless archive storage is intentional
- generated answer files unless preserving a specific submission:
  - `answers.json`
  - `challenges/answer.json`
  - `challenges/answers.json`

## Suggested New Repo Structure

Consider reorganizing the new repository like this:

```txt
ioai-prep/
  README.md
  IOAI_REPO_HANDOFF.md
  requirements.txt
  .gitignore
  notes/
    ioai_material.md
    optimized_7_day_study_plan.md
    ioai_competition_cheat_sheet.md
    training_performance_notes.md
  templates/
    train_function.py
    train_func.py
  classical_ml/
    decision_trees.py
    notebooks/
      iris.ipynb
      wine.ipynb
  vision/
    mnist.py
    cifar10.py
    cifar100_transfer.py
    squarepainting.py
    emotions.py
    captcha.py
  nlp/
    imdb_bert_sentiment.py
  data_small/
    diabetes.csv
  assets/
    emotions/
    squarepainting/
  outputs/
    .gitkeep
```

Add generated datasets/models/submissions to `.gitignore`, not to source control, unless they are small and intentionally part of the lesson.

## Cleanup Tasks for the New Repo

- Rename typo files:
  - `dicision_trees.ipynb` -> `decision_trees.ipynb`
  - `cfar1.py` -> `cifar10.py`
  - `CFAR1001.py` -> `cifar100_transfer.py`
- Clean mojibake/encoding artifacts in Greek markdown comments if preserving notebook explanations.
- Convert large notebook exports into clean Python modules where possible.
- Create `requirements.txt`.
- Add a root `README.md` explaining:
  - purpose
  - install instructions
  - recommended order of study
  - how to run each script
- Standardize generated outputs under `outputs/`.
- Make training functions return metrics instead of only plotting/printing.
- Add small smoke tests for core functions:
  - decision tree entropy/information gain
  - `train` loop on a tiny fake dataset
  - squarepainting dataset generation
  - CAPTCHA image segmentation shapes

## Suggested README Opening for the New Repo

```md
# IOAI Preparation

This repository contains my preparation work for the International Olympiad in Artificial Intelligence. It focuses on reusable competition templates, fast experimentation, and hands-on implementations across classical ML, PyTorch, computer vision, NLP, and challenge-style submissions.

Start with `notes/optimized_7_day_study_plan.md`, then use the scripts in `templates/`, `classical_ml/`, `vision/`, and `nlp/` as runnable examples.
```

## Most Important Files for a Future Agent

If an agent has limited time, start with these:

1. `comp/ioai_competition_cheat_sheet.md`
2. `challenges/optimized_7_day_study_plan.md`
3. `comp/training_bug.md`
4. `comp/train_func.py`
5. `challenges/CFAR1001.py`
6. `challenges/cfar1.py`
7. `challenges/nlp.py`
8. `emotions.py`
9. `captcha.py`
10. `challenges/squarepainting.py`

These files explain the current preparation strategy and contain the highest-value reusable code.
