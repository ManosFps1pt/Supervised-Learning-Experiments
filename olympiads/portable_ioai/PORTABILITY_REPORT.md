# Portable IOAI verification report

Verified on 2026-07-27 for the travel-laptop export.

## Environment

- Windows x64 (`10.0.26200`)
- Python `3.12.10`
- PyTorch `2.7.1+cpu`
- Torchvision `0.22.1+cpu`
- Transformers `4.53.2`
- Datasets `4.0.0`
- nbclient `0.10.2`
- CPU smoke mode; no CUDA device and no paid API were required

The complete entry point was run successfully:

```powershell
.\olympiads\portable_ioai\setup.ps1 -Smoke
```

PowerShell script execution is restricted in the Codex test process, so the
equivalent documented `-ExecutionPolicy Bypass` invocation was used there.
The script itself was also run from the Home Task 2 notebook directory with
`-Task home_task_2 -Smoke`; task selection and launch-directory independence
both passed end to end.

## Results

| Task | Verified CPU smoke contract | Result |
| --- | --- | --- |
| IOAI 2026 Home Task 1 | Verified six core asset hashes and 1,283 referenced WAVs; loaded one 80,000-sample clip at 16 kHz; AST preprocessing `(1, 1024, 128)`; local checkpoint forward `(1, 16)` | Passed |
| IOAI 2026 Home Task 2 | Fresh Google Drive folder download; all three pickle hashes and counts; 397-feature sample; MLP `(1, 6)` forward/backward/update; one mask-valid simulator step | Passed |
| IOAI 2026 Home Task 3 | Fresh Google Drive folder download; tracked helper import; pools `1472/559`; pinned Qwen 0.5B smoke model; one constrained CPU oracle interaction | Passed |
| IOAI 2025 Chicken Counting | Pinned Hugging Face train/validation/test splits, 100 rows each; official eight-tensor weight; full sample contracts `(3,720,1280)` and `(1,180,320)`; cropped model forward `(1,1,32,32)` | Passed |
| IOAI 2025 Concepts CPU | Pinned public splits `118/30/50/100`; 384-dimensional MiniLM embeddings for all hints and one 100-option example; valid clue; one-row JSONL and ZIP round-trip; no OpenRouter call | Passed |
| IOAI 2024 Help BOBAI | Fresh pinned HTTP downloads for all four assets; tensor shapes `(2473,1,769)`, `(200,1,768)`, `(700,1,768)`; base weight `(5,768)`; four valid integer predictions | Passed |

Each canonical notebook was then executed through `nbclient`, using only its
two `portable-smoke` cells. All six passed. The runner compared the notebook
SHA-256 before and after execution and confirmed that no notebook changed.

## Clean-checkout and Git checks

- Built a temporary clean-checkout simulation containing no ignored datasets,
  caches, outputs, smoke reports, or nested-repository state.
- All six notebooks parsed as valid notebook v4 JSON and all executable cells
  parsed as Python.
- Preflight passed from the repository root and path discovery passed from a
  notebook directory.
- Every `.data/`, `.cache/`, `.downloads/`, `outputs/`, and runtime-report path
  is ignored; every canonical notebook and tracked helper is not ignored.
- The portable tracked candidate set is 55 files / 27.41 MiB.
- The largest new file is `Home-Task-1.ipynb` at 6,933,319 bytes, well below
  GitHub's 100 MiB limit.
- All original outputs and execution counts were preserved in the six exported
  notebooks. No export exceeded the 10 MiB output-stripping threshold.
- Twenty-two non-canonical local notebooks were preserved as `reference_only`;
  they are valid notebook JSON but are not smoke-tested.

## Home Task 1 download note

The 1,508,471,337-byte official ZIP already present on this PC was verified
against SHA-256
`10742c5d31059f797bfa15e8c730404dec43e2dd00c14769fe5560d11e2a492a`.
Its 2,582 entries, 2,028,763,704 uncompressed bytes, required members, and path
safety were checked. The extracted data and AST checkpoint then passed the
real CPU smoke run.

The 1.5 GiB Google Drive transfer was not repeated merely to duplicate a known
identical archive. The bootstrap's mirror retry, exact archive verification,
safe extraction, and post-extraction consumer checks are active for a clean
laptop. Keep at least 4 GiB free for this task.

## Outside the guarantee

- Home Task 1 full fine-tuning and Chicken Counting full training remain
  GPU/long-runtime work.
- Home Task 3's official full workload uses the pinned Qwen 3B model; smoke
  mode deliberately uses the pinned 0.5B model on CPU.
- Concepts OpenRouter evaluation and the obsolete contest proxy are optional.
- `reference_only/` snapshots are preserved for retrieval, not presented as
  portable notebooks.

No history was rewritten, and no file was staged, committed, or pushed.
