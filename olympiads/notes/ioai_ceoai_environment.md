# IOAI / CEOAI Practice Environment

Date checked: 2026-06-30

## Current Target

Use IOAI 2026 as the primary environment target. Public CEOAI environment details are harder to verify, while IOAI publishes current contest rules and a technical appendix.

The IOAI 2026 appendix describes:

- Python 3.13.
- Yandex Contest for task statements, answers, submissions, and evaluation.
- JupyterLab as the primary development platform.
- VS Code as an offline code editor without direct GPU access.
- Core Python ML libraries including NumPy, pandas, scikit-learn, PyTorch, torchvision, torchaudio, and common data/plotting/NLP/CV packages.
- No internet access during contest sessions.
- No AI-based code assistance during contest sessions.

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

Use VS Code for larger local `.py` files, templates, and repo navigation. Keep AI extensions disabled for contest-style practice. If the goal is strict contest simulation, also avoid relying on rich IntelliSense.

Anaconda is not the important target. It may look familiar because Anaconda often launches JupyterLab, but the practical thing to train is the JupyterLab workflow itself.

## Nitro / CEOAI Note

If CEOAI practice uses the Nitro development platform, treat it as a platform-specific contest wrapper unless official technical documentation says otherwise. The safest preparation path is:

- Train in JupyterLab first.
- Keep code portable and dependency-light.
- Practice without internet and without AI suggestions.
- Prefer explicit imports, shape checks, and small smoke tests because platform autocomplete may be limited.

Add any official Nitro documentation links here once available.
