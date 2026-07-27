# Portable IOAI notebooks

This directory contains canonical, tracked copies of six high-priority official
IOAI notebooks. A portable notebook is guaranteed to pass imports, verified
asset bootstrap, preprocessing, and a small CPU smoke run after one setup
command. Full training, paid API calls, contest proxies, and GPU-heavy sections
remain optional.

## Setup on Windows

From any PowerShell working directory:

```powershell
& "D:\path\to\repo\olympiads\portable_ioai\setup.ps1" -Smoke
```

From the repository root:

```powershell
.\olympiads\portable_ioai\setup.ps1 -Smoke
```

If Windows has PowerShell script execution disabled for the current process,
use the equivalent one-time invocation:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\olympiads\portable_ioai\setup.ps1 -Smoke
```

The script requires Python 3.12 and internet access. It creates
`.venv-ioai-portable` at the repository root, installs pinned CPU dependencies,
registers the `Python 3.12 (portable-ioai)` Jupyter kernel, downloads and
verifies task assets, and executes each notebook's smoke cells.

For the laptop, avoid transferring the repository's old 2+ GiB history:

```powershell
git clone --depth 1 https://github.com/ManosFps1pt/Supervised-Learning-Experiments.git
cd Supervised-Learning-Experiments
.\olympiads\portable_ioai\setup.ps1 -Smoke
```

Use a task ID to limit setup:

```powershell
.\olympiads\portable_ioai\setup.ps1 -Task home_task_1 -Smoke
.\olympiads\portable_ioai\setup.ps1 -Task home_task_2,help_bobai -Smoke
```

Run dataset-download-free checks with:

```powershell
.\olympiads\portable_ioai\setup.ps1 -Preflight
```

`-Preflight` still creates or updates the Python environment on its first run,
but it does not download task datasets or model assets. With neither
`-Preflight` nor `-Smoke`, setup installs the environment and fetches all
verified assets without executing notebook smoke cells.

## Task IDs

| Task ID | Official task |
| --- | --- |
| `home_task_1` | IOAI 2026 Home Task 1 |
| `home_task_2` | IOAI 2026 Home Task 2 |
| `home_task_3` | IOAI 2026 Home Task 3 |
| `chicken_counting` | IOAI 2025 Chicken Counting |
| `concepts_cpu` | IOAI 2025 Concepts CPU baseline |
| `help_bobai` | IOAI 2024 Help BOBAI |

Downloads and generated files stay under task-local `.data/`, `.cache/`, and
`outputs/` directories. These paths are ignored by Git; the notebooks and
small helper files remain tracked. Home Task 1 needs roughly 4 GiB for its
archive, extracted data, and model, and model caches can require additional
space.

The canonical sources are pinned to:

- IOAI 2026: `4d02a6dfc8b4fb9eebdc6cbe098e203ee5506482`
- IOAI 2025: `39558e9a639d170bd92de91958fcdc915f670463`
- IOAI 2024: `d04f1df7662252e9e1912fd473a3ea884147c2a5`

The notebooks retain their original full-workload cells, marked separately
from the portable smoke cells. Non-priority local notebook work is preserved
under `reference_only/` without a runnable guarantee.

See `PORTABILITY_REPORT.md` for the exact tested environment, per-task shapes,
download coverage, clean-checkout checks, and the boundary of the guarantee.
