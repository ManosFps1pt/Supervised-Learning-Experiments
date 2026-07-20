# IOAI Practice Environment

Date checked: 2026-07-19

## Current Target

Use IOAI 2026 as the primary environment target. CEOAI / EUROAI material remains
useful practice, but the local setup should now mimic the official IOAI
technical appendix as closely as possible.

The IOAI 2026 rules and technical appendix describe:

- Python 3.13 with exact package versions to be published before the contest.
- Yandex Contest for task statements, datasets, submissions, and scores.
- Web-based JupyterLab as the main development platform and GPU access route.
- VS Code as an offline editor on contestant machines, without direct GPU access.
- Ubuntu laptops without local GPUs.
- GPU-enabled training/evaluation machines. Current appendix details mention
  NVIDIA H200 141GB GPUs split into 7 MIG slices, with an 18GB VRAM limit.
- Core AI/ML packages: `torch`, `torchvision`, `torchaudio`, `transformers`,
  `accelerate`, `peft`, `trl`, `scikit-learn`, `xgboost`, `lightgbm`,
  `catboost`, `sentence-transformers`, `datasets`, `evaluate`, `spacy`, `nltk`,
  `gensim`, and `fasttext`.
- Data/CV/plotting/utilities: `numpy`, `pandas`, `scipy`, `polars`,
  `pyarrow`, `h5py`, `opencv-python`, `Pillow`, `scikit-image`,
  `albumentations`, `matplotlib`, `seaborn`, `plotly`, `tqdm`, `joblib`,
  `tensorboard`, `pytorch-lightning`, `pydantic`, and `pyyaml`.
- TensorFlow and Keras are not available. Installing additional packages during
  the contest is not permitted.
- Whitelist-only internet access. No unrestricted browsing and no external
  pretrained model downloads during the contest.
- Only organizer-approved pretrained models may be used.
- Expected evaluation limit: notebook runtime up to 20 minutes per task unless
  the task statement says otherwise.
- Expected submission limit: up to 60 submissions per task.

## Provided LLM

The Individual Contest provides an official LLM integrated into the contest
platform:

- Individual Contest: Gemma 3, at most 1000 output tokens per query.
- GAITE Contest: Gemma 4, at most 2000 output tokens per query.
- Exact model, context window, rate limits, and usage quotas will be announced
  before the contest.

External LLMs, coding agents, browser assistants, AI copilots, and external APIs
including LLM APIs are prohibited unless a task statement explicitly allows
them.

Local Gemma practice is only an approximation. Use `google/gemma-3-4b-it` if
hardware allows, otherwise `google/gemma-3-1b-it`. Always cap practice outputs
with `max_new_tokens=1000` and manually verify the response.

## Local Setup

This repository already has a `.venv` using Python 3.13.14.

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install or refresh the practice dependencies:

```powershell
python -m pip install -r requirements-ioai.txt
```

Register the Jupyter kernel:

```powershell
python -m ipykernel install --user --name ioai-practice --display-name "IOAI Practice (.venv)"
```

Launch JupyterLab from the repo root:

```powershell
jupyter lab
```

## Editor Recommendation

Use JupyterLab for most timed IOAI-style practice. It best matches the contest development environment, and its autocomplete is weaker than a full IDE. That explains why typing something like `tree.` may not show a rich `fit` suggestion panel.

Use VS Code for larger local `.py` files, templates, and repo navigation. Keep AI extensions disabled for contest-style practice. If the goal is strict contest simulation, also avoid relying on rich IntelliSense. In the actual IOAI environment, VS Code is an offline editor and does not directly access GPUs.

Anaconda is not the important target. It may look familiar because Anaconda often launches JupyterLab, but the practical thing to train is the JupyterLab workflow itself.

## Local Gemma Practice

Install support packages:

```powershell
python -m pip install -U torch "transformers>=4.50.0" accelerate pillow sentencepiece
```

Notebook pattern:

```python
import torch
from transformers import GenerationConfig, pipeline

MODEL_ID = "google/gemma-3-4b-it"  # fallback: "google/gemma-3-1b-it"

pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 1000
config.do_sample = False

messages = [
    {
        "role": "user",
        "content": "I am debugging a PyTorch training loop. What shape checks should I run first?",
    }
]

out = pipe(messages, return_full_text=False, generation_config=config)
print(out[0]["generated_text"])
```

Use this for prompt discipline, API-discovery rehearsal, and debugging
questions. Do not use it to replace your own baseline implementation during
strict mock contests.
