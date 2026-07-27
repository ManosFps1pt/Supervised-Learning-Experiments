"""Export the audited local notebook work into tracked portable artifacts.

The ignored repositories under ``competition_samples/raw`` are build inputs.
This script is intentionally deterministic, but a clean laptop clone does not
need those repositories: all produced notebooks and helpers are committed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


PORTABLE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PORTABLE_ROOT.parents[1]
RAW_ROOT = REPOSITORY_ROOT / "olympiads" / "competition_samples" / "raw"
OUTPUT_STRIP_THRESHOLD = 10 * 1024 * 1024

CANONICAL = {
    "home_task_1": {
        "source": RAW_ROOT / "IOAI-2026-sparse" / "Home Task" / "problem1" / "Home-Task-1.ipynb",
        "destination": PORTABLE_ROOT / "tasks" / "home_task_1" / "Home-Task-1.ipynb",
        "transform": "home_task_1",
    },
    "home_task_2": {
        "source": RAW_ROOT / "IOAI-2026-sparse" / "Home Task" / "problem2" / "Home-Task-2.ipynb",
        "destination": PORTABLE_ROOT / "tasks" / "home_task_2" / "Home-Task-2.ipynb",
        "transform": "home_task_2",
    },
    "home_task_3": {
        "source": RAW_ROOT / "IOAI-2026-sparse" / "Home Task" / "problem3" / "Home-Task-3.ipynb",
        "destination": PORTABLE_ROOT / "tasks" / "home_task_3" / "Home-Task-3.ipynb",
        "transform": "home_task_3",
    },
    "chicken_counting": {
        "source": RAW_ROOT / "IOAI-2025-sparse" / "Individual-Contest" / "Chicken_Counting" / "Chicken_Counting.ipynb",
        "destination": PORTABLE_ROOT / "tasks" / "chicken_counting" / "Chicken_Counting.ipynb",
        "transform": "chicken_counting",
    },
    "concepts_cpu": {
        "source": RAW_ROOT / "IOAI-2025-sparse" / "Individual-Contest" / "Concepts" / "Concepts_baseline-Copy1.ipynb",
        "destination": PORTABLE_ROOT / "tasks" / "concepts_cpu" / "Concepts_CPU.ipynb",
        "transform": "concepts_cpu",
    },
    "help_bobai": {
        "source": RAW_ROOT / "IOAI-2024-sparse" / "On-Site-Round" / "Help_BOBAI" / "Help_BOBAI.ipynb",
        "destination": PORTABLE_ROOT / "tasks" / "help_bobai" / "Help_BOBAI.ipynb",
        "transform": "help_bobai",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def get_source(cell: dict[str, Any]) -> str:
    raw = cell.get("source", [])
    return raw if isinstance(raw, str) else "".join(raw)


def set_source(cell: dict[str, Any], text: str) -> None:
    cell["source"] = source_lines(text)


def find_code_cell(notebook: dict[str, Any], needle: str) -> dict[str, Any]:
    matches = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and needle in get_source(cell)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one code cell containing {needle!r}, found {len(matches)}")
    return matches[0]


def replace_code_cell(notebook: dict[str, Any], needle: str, replacement: str) -> None:
    set_source(find_code_cell(notebook, needle), replacement.strip() + "\n")


def add_tag(cell: dict[str, Any], tag: str) -> None:
    tags = cell.setdefault("metadata", {}).setdefault("tags", [])
    if tag not in tags:
        tags.append(tag)


def portable_cells(task_id: str) -> list[dict[str, Any]]:
    markdown = {
        "cell_type": "markdown",
        "id": f"portable-{task_id}-about"[:64],
        "metadata": {"tags": ["portable-info"]},
        "source": source_lines(
            "## Portable execution layer\n\n"
            "This tracked copy preserves the latest local work and its saved evidence. "
            "Downloads are verified against `manifest.json` and stored only in this "
            "task's ignored `.data/` and `.cache/` directories. Generated files go to "
            "`outputs/`. Set `PORTABLE_IOAI_SMOKE=1` to execute only the small CPU "
            "portability contract; the original workload cells are tagged `full-run`.\n"
        ),
    }
    preamble = {
        "cell_type": "code",
        "execution_count": None,
        "id": f"portable-{task_id}-context"[:64],
        "metadata": {"tags": ["portable-smoke"]},
        "outputs": [],
        "source": source_lines(
            "from pathlib import Path\n"
            "import json\n"
            "import os\n"
            "import sys\n\n"
            "def _find_portable_root():\n"
            "    start = Path.cwd().resolve()\n"
            "    for parent in (start, *start.parents):\n"
            "        for candidate in (parent, parent / 'olympiads' / 'portable_ioai'):\n"
            "            if (candidate / 'bootstrap.py').is_file() and (candidate / 'manifest.json').is_file():\n"
            "                return candidate\n"
            "    raise FileNotFoundError(\n"
            "        'Could not locate olympiads/portable_ioai. Start Jupyter from the '\n"
            "        'repository root or this notebook directory, after running setup.ps1.'\n"
            "    )\n\n"
            "PORTABLE_ROOT = _find_portable_root()\n"
            "if str(PORTABLE_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PORTABLE_ROOT))\n"
            "from bootstrap import NotebookContext, load_hf_datasets, smoke_task\n\n"
            f"PORTABLE = NotebookContext({task_id!r}).prepare_paths()\n"
            "DATA_DIR = PORTABLE.data_dir\n"
            "OUTPUT_DIR = PORTABLE.output_dir\n"
            "SMOKE_MODE = os.environ.get('PORTABLE_IOAI_SMOKE', '').lower() in {'1', 'true', 'yes'}\n"
            "os.environ[PORTABLE.spec['data_environment']] = str(DATA_DIR)\n"
            f"if {task_id!r} == 'home_task_3':\n"
            "    model_mode = 'smoke' if SMOKE_MODE else 'full'\n"
            "    model_spec = PORTABLE.spec['models'][model_mode]\n"
            "    os.environ['PORTABLE_IOAI_HT3_MODEL'] = model_spec['id']\n"
            "    os.environ['PORTABLE_IOAI_HT3_MODEL_REVISION'] = model_spec['revision']\n"
            "print(json.dumps({**PORTABLE.describe(), 'smoke_mode': SMOKE_MODE}, indent=2))\n"
        ),
    }
    smoke = {
        "cell_type": "code",
        "execution_count": None,
        "id": f"portable-{task_id}-smoke"[:64],
        "metadata": {"tags": ["portable-smoke"]},
        "outputs": [],
        "source": source_lines(
            "if SMOKE_MODE:\n"
            f"    SMOKE_RESULT = smoke_task({task_id!r}, ensure=True)\n"
            "    print(json.dumps(SMOKE_RESULT, indent=2, default=str))\n"
            "else:\n"
            "    print('Portable context ready; continuing with the preserved full-workload cells.')\n"
        ),
    }
    return [markdown, preamble, smoke]


def transform_home_task_1(notebook: dict[str, Any]) -> None:
    replace_code_cell(
        notebook,
        "# !pip -q install gdown",
        """
# The original commented Drive flow is replaced by the shared verified bootstrap.
LOCAL_DATA_DIR = PORTABLE.ensure_data()
print(f"Verified Home Task 1 data: {LOCAL_DATA_DIR}")
""",
    )
    replace_code_cell(
        notebook,
        "cwd = Path.cwd()# / \"problem1\"",
        """
from pathlib import Path

LOCAL_DATA_DIR = PORTABLE.ensure_data()
print("Dataset found:")
for path in sorted(LOCAL_DATA_DIR.iterdir()):
    print(path)
""",
    )
    imports = find_code_cell(notebook, "from transformers import ASTFeatureExtractor")
    text = get_source(imports).replace(
        'DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")',
        'DEVICE = "cpu" if SMOKE_MODE else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))',
    )
    set_source(imports, text)
    training = find_code_cell(notebook, "train(model,optimizer,criterion,loader,val_loader,10)")
    set_source(
        training,
        get_source(training).replace(
            "train(model,optimizer,criterion,loader,val_loader,10)",
            "train(model, optimizer, criterion, loader, val_loader, 1 if SMOKE_MODE else 10)",
        ),
    )


def transform_home_task_2(notebook: dict[str, Any]) -> None:
    replace_code_cell(
        notebook,
        "!pip install -q gdown",
        """
from pathlib import Path

DATA_DIR = PORTABLE.ensure_data()
print("Verified Robot Delivery data:", DATA_DIR)
sorted(path.name for path in DATA_DIR.iterdir())
""",
    )
    imports = find_code_cell(notebook, "ACTION_NAMES = {")
    set_source(
        imports,
        get_source(imports).replace(
            'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
            'DEVICE = torch.device("cpu" if SMOKE_MODE else ("cuda" if torch.cuda.is_available() else "cpu"))',
        ),
    )
    training = find_code_cell(notebook, "EPOCHS = 30")
    set_source(training, get_source(training).replace("EPOCHS = 30", "EPOCHS = 1 if SMOKE_MODE else 30"))
    evaluation = find_code_cell(notebook, "EVAL_LIMIT = None")
    set_source(
        evaluation,
        get_source(evaluation).replace(
            "EVAL_LIMIT = None  # set to None for all validation scenarios",
            "EVAL_LIMIT = 1 if SMOKE_MODE else None  # None evaluates all scenarios",
        ),
    )
    saver = find_code_cell(notebook, "def save_predictions_zip")
    saver_text = get_source(saver).replace(
        '    jsonl_path = Path("predictions.jsonl")',
        "    path = Path(path)\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        '    jsonl_path = path.with_name("predictions.jsonl")',
    )
    set_source(saver, saver_text)
    replace_code_cell(
        notebook,
        "test_predictions = generate_predictions(test_scenarios",
        """
# Generate a one-scenario artifact in smoke mode and the full submission otherwise.
submission_limit = 1 if SMOKE_MODE else None
test_predictions = generate_predictions(
    test_scenarios, mlp_action_model, limit=submission_limit
)
save_predictions_zip(test_predictions, OUTPUT_DIR / "predictions.zip")
""",
    )


def transform_home_task_3(notebook: dict[str, Any]) -> None:
    replace_code_cell(
        notebook,
        "!pip install -q gdown transformers accelerate",
        """
import os
import sys
from pathlib import Path

LOCAL_DIR = PORTABLE.ensure_data()
SUPPORT_DIR = PORTABLE.task_dir / "support"
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))
os.environ["PORTABLE_IOAI_HOME_TASK_3_DATA"] = str(LOCAL_DIR)
print("Data directory:", LOCAL_DIR)
print("Tracked helpers:", sorted(path.name for path in SUPPORT_DIR.iterdir()))
print("Data files:", sorted(path.name for path in LOCAL_DIR.iterdir()))
""",
    )
    replace_code_cell(notebook, "Path.cwd()", "LOCAL_DIR")
    replace_code_cell(
        notebook,
        "animals_pool, questions_pool = load_pools()",
        """
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from interactor import Interactor
from evaluate import evaluate, load_pools

DEVICE = "cpu" if SMOKE_MODE else ("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

animals_pool, questions_pool = load_pools(
    LOCAL_DIR / "animals_pool.txt",
    LOCAL_DIR / "questions_pool.txt",
)
print(f"animals_pool size:   {len(animals_pool):>6}  (e.g. {animals_pool[:5]})")
print(f"questions_pool size: {len(questions_pool):>6}  (e.g. {questions_pool[:3]})")

probe = Interactor(
    gold_animal="octopus",
    animals_pool=animals_pool,
    questions_pool=questions_pool,
)
probe_questions = questions_pool[:1] if SMOKE_MODE else [
    "is it a mammal?",
    "does it live in water?",
]
for question in probe_questions:
    print(f"ask({question!r}) ->", probe.ask(question))
print("Queries used:", probe.queries_used, "/", probe.budget)

if torch.cuda.is_available() and next(Interactor._model.parameters()).dtype != torch.float16:
    Interactor._model = Interactor._model.half()
    print("oracle model -> float16")
""",
    )
    replace_code_cell(
        notebook,
        "dev_results   = evaluate(solution, 'dev.csv')",
        """
import os

# solution = MySolution(animals_pool, questions_pool)
dev_results = evaluate(solution, LOCAL_DIR / "dev.csv")
test1_results = evaluate(solution, LOCAL_DIR / "test1.csv")

splits = [("dev", dev_results), ("test1", test1_results)]
test2_path = LOCAL_DIR / "test2.csv"
if test2_path.exists():
    splits.append(("test2", evaluate(solution, test2_path)))

rows = [{
    "split": name, "n": result["n"], "mean_score": result["mean_score"],
    "solved_rate": result["solved_rate"], "mean_queries": result["mean_queries"],
} for name, result in splits]

tests = [result for name, result in splits if name.startswith("test")]
n_test = sum(result["n"] for result in tests)
rows.append({
    "split": "FINAL",
    "n": n_test,
    "mean_score": sum(result["mean_score"] * result["n"] for result in tests) / n_test,
    "solved_rate": sum(result["solved_rate"] * result["n"] for result in tests) / n_test,
    "mean_queries": sum(result["mean_queries"] * result["n"] for result in tests) / n_test,
})
pd.DataFrame(rows)
""",
    )


def transform_chicken_counting(notebook: dict[str, Any]) -> None:
    replace_code_cell(
        notebook,
        "#Contestants should mount",
        """
import os
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

DATA_DIR = PORTABLE.ensure_data()
BASE_MODEL_PATH = DATA_DIR / "base.pth"
DTYPE = torch.float32
AMP_ENABLED = False if SMOKE_MODE else torch.cuda.is_available()
DEVICE = torch.device("cpu" if SMOKE_MODE else ("cuda:0" if torch.cuda.is_available() else "cpu"))
scale = 100.0
""",
    )
    model_cell = find_code_cell(notebook, "class FeatureExtraction")
    set_source(
        model_cell,
        get_source(model_cell).replace(
            "save_model = torch.load(weights_path)",
            'save_model = torch.load(weights_path, map_location="cpu", weights_only=True)',
        ),
    )
    replace_code_cell(
        notebook,
        "# 从 Hugging Face 加载数据集",
        """
from datasets import load_dataset
import matplotlib.pyplot as plt

train_dataset = load_dataset(
    "ioaihsc/Task2_Chicken_Counting_Train2",
    revision="377f01f034683afc5a49468001e75360af722393",
    data_dir="train",
    split="train",
    cache_dir=str(PORTABLE.cache_dir / "huggingface" / "datasets"),
)
if SMOKE_MODE:
    train_dataset = train_dataset.select(range(1))

image_transform = transforms.Compose([transforms.ToTensor()])

def collate_fn(batch, scale=scale):
    return {
        "image": torch.stack([image_transform(item["image"]) for item in batch]),
        "density": torch.stack([
            torch.tensor(item["density"], dtype=DTYPE).unsqueeze(0) * scale
            for item in batch
        ]),
    }

batch_size = 1 if SMOKE_MODE else 6
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
""",
    )
    settings = find_code_cell(notebook, "learning_rate = 1e-4")
    text = get_source(settings)
    text = text.replace('save_path = "model.pth"', 'save_path = OUTPUT_DIR / "model.pth"')
    text = text.replace("epochs = 20", "epochs = 1 if SMOKE_MODE else 20")
    set_source(settings, text)
    load_model = find_code_cell(notebook, "model.load_state_dict(torch.load(save_path")
    set_source(
        load_model,
        get_source(load_model).replace(
            "torch.load(save_path, map_location=DEVICE)",
            "torch.load(save_path, map_location=DEVICE, weights_only=True)",
        ),
    )
    test_cell = find_code_cell(notebook, 'test_dataset = load_dataset("ioaihsc/Task2_Chicken_Counting_Test"')
    text = get_source(test_cell)
    text = text.replace(
        'data_dir="valandtest",\n                            split="validation")',
        'revision="decfcd957868ace7595df19638b04b0fe9deafbe",\n'
        '                            data_dir="valandtest",\n'
        '                            split="validation",\n'
        '                            cache_dir=str(PORTABLE.cache_dir / "huggingface" / "datasets"))',
        1,
    )
    text = text.replace(
        'data_dir="valandtest",\n                            split="test")',
        'revision="decfcd957868ace7595df19638b04b0fe9deafbe",\n'
        '                            data_dir="valandtest",\n'
        '                            split="test",\n'
        '                            cache_dir=str(PORTABLE.cache_dir / "huggingface" / "datasets"))',
        1,
    )
    text = text.replace(
        "np.savez('submission.npz', pred_a=pred_a, pred_b=pred_b)",
        'np.savez(OUTPUT_DIR / "submission.npz", pred_a=pred_a, pred_b=pred_b)',
    )
    set_source(test_cell, text)
    replace_code_cell(
        notebook,
        "%run ./metrics.py",
        """
if SMOKE_MODE:
    print("Official 100-row metric skipped in smoke mode.")
else:
    os.environ["PORTABLE_IOAI_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ["HF_DATASETS_CACHE"] = str(PORTABLE.cache_dir / "huggingface" / "datasets")
    metrics_path = str(PORTABLE.task_dir / "support" / "metrics.py")
    get_ipython().run_line_magic("run", metrics_path)
""",
    )


def transform_concepts_cpu(notebook: dict[str, Any]) -> None:
    replace_code_cell(
        notebook,
        "# Windows/local baseline imports.",
        """
import json
import math
import os
import random
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import torch
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Each public split is loaded by repository ID and immutable revision. The
# shared helper verifies the underlying Parquet file hashes and row counts.
CONCEPTS_DATA = load_hf_datasets("concepts_cpu")
CONCEPTS_ARROW = {name: name for name in CONCEPTS_DATA}

def _read_arrow_rows(name):
    return [dict(row) for row in CONCEPTS_DATA[str(name)]]
""",
    )
    replace_code_cell(
        notebook,
        "# The original `judge_api.py`",
        """
# The official API-based guesser is optional and outside the CPU guarantee.
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.5-flash-lite-preview-06-17"

_HINT_DICT = {
    int(row["ID"]): row["Description"].replace("\\n", ", ")
    for row in _read_arrow_rows(CONCEPTS_ARROW["hint_descriptions"])
}
_ORDINALS = ["first", "second", "third", "fourth"]

def guess(clues: List[List[int]], options=None, N: int = 10) -> List[str]:
    if not API_KEY:
        return []
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    clue_str = ""
    for i, clue in enumerate(clues):
        clue_str += f"{_ORDINALS[i]} clue:\\n"
        for hint_idx in clue:
            clue_str += f" - {_HINT_DICT.get(int(hint_idx), f'[hint {hint_idx}]')}\\n"
        clue_str += "\\n"
    option_str = "\\n".join(options) if options else ""
    prompt = (
        "You are playing a Concepts game. A player has a secret keyword and "
        f"provided these clues:\\n{clue_str}\\nOptions:\\n{option_str}\\n"
        f"Return exactly {N} guesses, one per line."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content or ""
    return [
        line.strip(" -0123456789.\\t").lower()
        for line in text.splitlines() if line.strip()
    ][:N]
""",
    )
    game_client = find_code_cell(notebook, "class GameClient:")
    set_source(game_client, get_source(game_client).replace("# import httpx", "import httpx"))
    clue_giver = find_code_cell(notebook, "class ClueGiver:")
    text = get_source(clue_giver)
    text = text.replace(
        "self.model = SentenceTransformer('all-MiniLM-L6-v2')",
        "model_spec = PORTABLE.spec['models']['embedding']\n"
        "        self.model = SentenceTransformer(\n"
        "            model_spec['id'], revision=model_spec['revision'], device='cpu',\n"
        "            cache_folder=str(PORTABLE.cache_dir / 'huggingface' / 'sentence_transformers'),\n"
        "        )",
    )
    text = text.replace(
        'print({"embedder": "sklearn TfidfVectorizer",',
        'print({"embedder": "sentence-transformers/all-MiniLM-L6-v2",',
    )
    set_source(clue_giver, text)
    api_eval = find_code_cell(notebook, "with ThreadPoolExecutor() as executor:")
    original = get_source(api_eval)
    set_source(
        api_eval,
        "if not API_KEY:\n"
        '    print("Optional OpenRouter evaluation skipped: OPENROUTER_API_KEY is not set.")\n'
        "    predictions = []\n"
        "else:\n"
        + "".join(f"    {line}" if line.strip() else line for line in original.splitlines(keepends=True)),
    )
    sequential = find_code_cell(notebook, "for i, data in enumerate(dev):")
    original = get_source(sequential)
    set_source(
        sequential,
        "if API_KEY:\n"
        + "".join(f"    {line}" if line.strip() else line for line in original.splitlines(keepends=True))
        + "else:\n"
        '    print("Optional per-row API debug loop skipped.")\n',
    )
    duplicate = find_code_cell(notebook, "Test Case Prediction: Hit@10")
    set_source(
        duplicate,
        'print("Duplicate API debug loop removed from the portable full run; saved outputs remain in this notebook.")\n',
    )
    output = find_code_cell(notebook, 'OUTPUT_DIR = Path("out")')
    set_source(output, get_source(output).replace('OUTPUT_DIR = Path("out")', "OUTPUT_DIR = PORTABLE.output_dir"))


def transform_help_bobai(notebook: dict[str, Any]) -> None:
    replace_code_cell(
        notebook,
        "dataset = torch.load('./training_set/train-dev_dataset_with_labels.pt')",
        """
import torch

DATA_DIR = PORTABLE.ensure_data()
dataset = torch.load(
    DATA_DIR / "training_set" / "train-dev_dataset_with_labels.pt",
    map_location="cpu",
    weights_only=True,
)
inputs = dataset[:, :, :-1]
labels = dataset[:, :, -1]
""",
    )
    diagnostic = find_code_cell(notebook, "score = accuracy_score(preds, uncertain_mask")
    set_source(
        diagnostic,
        get_source(diagnostic).replace(
            "score = accuracy_score(preds, uncertain_mask.float().reshape(-1).numpy())",
            "score = accuracy_score(model_labels, preds)",
        ),
    )
    classifier = find_code_cell(notebook, "class SevenWayClassifier")
    text = get_source(classifier)
    text = text.replace(
        'base_clf.load_state_dict(torch.load("training_set/base_classifier.pth"))',
        'base_clf.load_state_dict(torch.load(\n'
        '        DATA_DIR / "training_set" / "base_classifier.pth",\n'
        '        map_location="cpu", weights_only=True,\n'
        "    ))",
    )
    text = text.replace("    return predicted_class\n", "    return int(predicted_class)\n")
    set_source(classifier, text)
    f1_cell = find_code_cell(notebook, "def compute_f1")
    set_source(
        f1_cell,
        get_source(f1_cell).replace(
            "return f1_score(labels, predictions, average='macro')",
            "return f1_score(np.asarray(labels).reshape(-1), np.asarray(predictions).reshape(-1), average='macro')",
        ),
    )
    validation = find_code_cell(notebook, "eval_inputs = torch.load")
    text = get_source(validation)
    text = text.replace(
        "df.to_csv(output_fpath, index=False)",
        'df["class"] = df["class"].astype(int)\n    df.to_csv(output_fpath, index=False)',
    )
    text = text.replace(
        "eval_inputs = torch.load('./Solution/validation_set/eval_dataset.pt')",
        'eval_inputs = torch.load(\n'
        '    DATA_DIR / "Solution" / "validation_set" / "eval_dataset.pt",\n'
        '    map_location="cpu", weights_only=True,\n'
        ")",
    )
    text = text.replace(
        "submission_to_csv(eval_predictions)",
        'submission_to_csv(eval_predictions, OUTPUT_DIR / "submission.csv")',
    )
    set_source(validation, text)
    replace_code_cell(
        notebook,
        "# this download link will not work until two hours",
        """
# The pinned official test tensor is downloaded by the shared bootstrap.
test_inputs = torch.load(
    DATA_DIR / "Solution" / "test_set" / "test_dataset.pt",
    map_location="cpu",
    weights_only=True,
)
test_predictions = inference(clf, test_inputs)
prediction_path = OUTPUT_DIR / "Team Name_predictions.txt"
prediction_path.write_text(
    "\\n".join(str(int(prediction)) for prediction in test_predictions),
    encoding="utf-8",
)
print("Saved", prediction_path)
""",
    )


TRANSFORMS: dict[str, Callable[[dict[str, Any]], None]] = {
    "home_task_1": transform_home_task_1,
    "home_task_2": transform_home_task_2,
    "home_task_3": transform_home_task_3,
    "chicken_counting": transform_chicken_counting,
    "concepts_cpu": transform_concepts_cpu,
    "help_bobai": transform_help_bobai,
}


def export_notebook(task_id: str, config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    destination = config["destination"]
    if not source.is_file():
        raise FileNotFoundError(source)
    notebook = json.loads(source.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            add_tag(cell, "full-run")
    TRANSFORMS[config["transform"]](notebook)
    insertion = 2 if len(notebook["cells"]) >= 2 else len(notebook["cells"])
    notebook["cells"][insertion:insertion] = portable_cells(task_id)
    notebook.setdefault("metadata", {})["kernelspec"] = {
        "display_name": "Python 3.12 (portable-ioai)",
        "language": "python",
        "name": "portable-ioai",
    }
    notebook["metadata"]["portable_ioai"] = {
        "task_id": task_id,
        "source_path": source.relative_to(REPOSITORY_ROOT).as_posix(),
        "source_sha256": sha256(source),
        "smoke_tag": "portable-smoke",
        "full_run_tag": "full-run",
    }
    stripped = False
    if source.stat().st_size > OUTPUT_STRIP_THRESHOLD:
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        stripped = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return {
        "task": task_id,
        "source": source.relative_to(REPOSITORY_ROOT).as_posix(),
        "source_sha256": sha256(source),
        "destination": destination.relative_to(REPOSITORY_ROOT).as_posix(),
        "destination_sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "outputs_stripped": stripped,
    }


def patch_interactor(source: str) -> str:
    source = source.replace("import re\n", "import os\nimport re\n", 1)
    source = source.replace(
        'DEFAULT_LLM  = "Qwen/Qwen2.5-3B-Instruct"',
        'DEFAULT_LLM  = os.environ.get("PORTABLE_IOAI_HT3_MODEL", "Qwen/Qwen2.5-3B-Instruct")',
    )
    start = source.index("    @classmethod\n    def _ensure_llm")
    end = source.index("    def __init__(", start)
    replacement = '''    @classmethod
    def _ensure_llm(cls, model_name: str = DEFAULT_LLM):
        if cls._model is not None and cls._model_name == model_name:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        force_cpu = os.environ.get("PORTABLE_IOAI_SMOKE", "").lower() in {"1", "true", "yes"}
        device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else "auto"
        revision = os.environ.get("PORTABLE_IOAI_HT3_MODEL_REVISION") or None
        cache_dir = os.environ.get("HF_HUB_CACHE") or None
        print(f"  [interactor] loading {model_name} on {device}...")
        cls._tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, cache_dir=cache_dir,
        )
        cls._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        cls._model = cls._model.to(device).eval()
        cls._model_name = model_name
        print("  [interactor] LLM ready.")

'''
    return source[:start] + replacement + source[end:]


def patch_evaluate(source: str) -> str:
    source = source.replace("import argparse\n", "import argparse\nimport os\n", 1)
    source = source.replace(
        '# All data files live in the same folder as this module (the "dataset" folder).\n'
        "HERE = Path(__file__).resolve().parent\n"
        "DEFAULT_ANIMALS_POOL   = HERE / \"animals_pool.txt\"\n"
        "DEFAULT_QUESTIONS_POOL = HERE / \"questions_pool.txt\"\n"
        "DEFAULT_DEV            = HERE / \"dev.csv\"\n",
        '# Portable data is task-local and can be overridden for an existing verified copy.\n'
        "HERE = Path(__file__).resolve().parent\n"
        "DATA_ROOT = Path(os.environ.get(\n"
        '    "PORTABLE_IOAI_HOME_TASK_3_DATA", HERE.parent / ".data"\n'
        "))\n"
        'DEFAULT_ANIMALS_POOL = DATA_ROOT / "animals_pool.txt"\n'
        'DEFAULT_QUESTIONS_POOL = DATA_ROOT / "questions_pool.txt"\n'
        'DEFAULT_DEV = DATA_ROOT / "dev.csv"\n',
    )
    return source


def patch_chicken_metrics(source: str) -> str:
    source = source.replace(
        '        test_dataset = load_dataset("ioaihsc/Task2_Chicken_Counting_LABEL", \n'
        '                            data_dir="valandtest",\n'
        "                            split=tag)",
        '        test_dataset = load_dataset(\n'
        '                            "ioaihsc/Task2_Chicken_Counting_LABEL",\n'
        '                            revision="c67fb0ac0cada3d502ef0f1f8779dc668897dcfb",\n'
        '                            data_dir="valandtest",\n'
        "                            split=tag,\n"
        '                            cache_dir=os.environ.get("HF_DATASETS_CACHE"),\n'
        "        )",
    )
    source = source.replace(
        '            preds = np.load("submission.npz", allow_pickle=False)',
        '            output_dir = os.environ.get("PORTABLE_IOAI_OUTPUT_DIR", ".")\n'
        '            preds = np.load(os.path.join(output_dir, "submission.npz"), allow_pickle=False)',
    )
    source = source.replace(
        "        with open('score.json', 'w') as f:",
        '        output_dir = os.environ.get("PORTABLE_IOAI_OUTPUT_DIR", ".")\n'
        "        os.makedirs(output_dir, exist_ok=True)\n"
        "        with open(os.path.join(output_dir, 'score.json'), 'w') as f:",
    )
    return source


def copy_helpers_and_figures() -> None:
    ht3_source = RAW_ROOT / "IOAI-2026-sparse" / "Home Task" / "problem3" / "problem3"
    ht3_destination = PORTABLE_ROOT / "tasks" / "home_task_3" / "support"
    ht3_destination.mkdir(parents=True, exist_ok=True)
    (ht3_destination / "interactor.py").write_text(
        patch_interactor((ht3_source / "interactor.py").read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    (ht3_destination / "evaluate.py").write_text(
        patch_evaluate((ht3_source / "evaluate.py").read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    chicken_source = RAW_ROOT / "IOAI-2025-sparse" / "Individual-Contest" / "Chicken_Counting"
    chicken_destination = PORTABLE_ROOT / "tasks" / "chicken_counting"
    support = chicken_destination / "support"
    support.mkdir(parents=True, exist_ok=True)
    (support / "metrics.py").write_text(
        patch_chicken_metrics((chicken_source / "metrics.py").read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    figure_sources = {
        chicken_source / "figs": chicken_destination / "figs",
        RAW_ROOT / "IOAI-2025-sparse" / "Individual-Contest" / "Concepts" / "figs":
            PORTABLE_ROOT / "tasks" / "concepts_cpu" / "figs",
        RAW_ROOT / "IOAI-2024-sparse" / "On-Site-Round" / "Help_BOBAI" / "figs":
            PORTABLE_ROOT / "tasks" / "help_bobai" / "figs",
    }
    for source_dir, destination_dir in figure_sources.items():
        if not source_dir.is_dir():
            continue
        for source in source_dir.rglob("*"):
            if source.is_file():
                destination = destination_dir / source.relative_to(source_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)


def git(repo: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.decode("utf-8", errors="replace").strip()


def changed_paths(repo: Path) -> list[tuple[str, str]]:
    output = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8", errors="surrogateescape")
    records = output.split("\0")
    changed: list[tuple[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status = record[:2]
        path = record[3:]
        if "R" in status or "C" in status:
            if index < len(records):
                path = records[index]
                index += 1
        changed.append((status, path))
    return changed


def export_reference_snapshots(repositories: list[Path]) -> list[dict[str, Any]]:
    canonical_sources = {config["source"].resolve() for config in CANONICAL.values()}
    entries: list[dict[str, Any]] = []
    for repo in repositories:
        origin = git(repo, "remote", "get-url", "origin", check=False)
        commit = git(repo, "rev-parse", "HEAD")
        for status, relative in changed_paths(repo):
            source = repo / relative
            if (
                source.suffix.lower() != ".ipynb"
                or ".ipynb_checkpoints" in source.parts
                or source.resolve() in canonical_sources
                or not source.is_file()
            ):
                continue
            # Validate JSON before preserving the exact bytes.
            notebook = json.loads(source.read_text(encoding="utf-8"))
            if notebook.get("nbformat") != 4:
                raise RuntimeError(f"Not a v4 notebook: {source}")
            destination = PORTABLE_ROOT / "reference_only" / repo.name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            entries.append({
                "id": f"{repo.name}:{relative}",
                "classification": "reference_only",
                "snapshot_path": destination.relative_to(PORTABLE_ROOT).as_posix(),
                "source_repo_url": origin,
                "source_commit": commit,
                "source_path": relative.replace("\\", "/"),
                "working_tree_status": status,
                "size_bytes": source.stat().st_size,
                "sha256": sha256(source),
                "smoke_tested": False,
            })
    entries.sort(key=lambda entry: entry["id"].lower())
    path = PORTABLE_ROOT / "reference_only" / "reference_catalog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "notebooks": entries}, indent=2) + "\n", encoding="utf-8")
    return entries


def export_source_catalog(repositories: list[Path]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for repo in repositories:
        changed = {path for _, path in changed_paths(repo)}
        tracked = [
            path
            for path in git(repo, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
            if path.lower().endswith(".ipynb") and ".ipynb_checkpoints" not in path
        ]
        sparse = git(repo, "sparse-checkout", "list", check=False).splitlines()
        catalog.append({
            "name": repo.name,
            "source_repo_url": git(repo, "remote", "get-url", "origin", check=False),
            "commit": git(repo, "rev-parse", "HEAD"),
            "sparse_paths": sparse,
            "tracked_notebook_count": len(tracked),
            "untouched_upstream_notebooks": [
                path for path in tracked if path not in changed
            ],
        })
    catalog.sort(key=lambda entry: entry["name"].lower())
    path = PORTABLE_ROOT / "source_catalog.json"
    path.write_text(json.dumps({"schema_version": 1, "repositories": catalog}, indent=2) + "\n", encoding="utf-8")
    return catalog


def main() -> None:
    exports = [
        export_notebook(task_id, config)
        for task_id, config in CANONICAL.items()
    ]
    copy_helpers_and_figures()
    repositories = sorted(
        [path for path in RAW_ROOT.iterdir() if path.is_dir() and (path / ".git").exists()],
        key=lambda path: path.name.lower(),
    )
    references = export_reference_snapshots(repositories)
    sources = export_source_catalog(repositories)
    export_manifest = {
        "schema_version": 1,
        "canonical_notebooks": exports,
        "reference_snapshot_count": len(references),
        "source_repository_count": len(sources),
    }
    (PORTABLE_ROOT / "export_manifest.json").write_text(
        json.dumps(export_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(export_manifest, indent=2))


if __name__ == "__main__":
    main()
