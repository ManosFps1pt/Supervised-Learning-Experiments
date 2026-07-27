"""Verified bootstrap and CPU smoke tests for the portable IOAI notebooks.

This module deliberately has only standard-library imports at module import time.
That keeps ``preflight`` useful even when the dedicated environment has not yet
been created. Task-specific dependencies are imported only by their fetch or
smoke routines.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PORTABLE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PORTABLE_ROOT.parents[1]
MANIFEST_PATH = PORTABLE_ROOT / "manifest.json"


class PortableIOAIError(RuntimeError):
    """An actionable portability/bootstrap failure."""


def _load_manifest() -> dict[str, Any]:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PortableIOAIError(f"Missing portable manifest: {MANIFEST_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise PortableIOAIError(f"Invalid portable manifest JSON: {exc}") from exc


MANIFEST = _load_manifest()
TASKS: dict[str, dict[str, Any]] = MANIFEST["tasks"]


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def _ensure_free_space(path: Path, required: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < required:
        raise PortableIOAIError(
            f"{path} has {_human_bytes(free)} free, but this task needs at least "
            f"{_human_bytes(required)}. Free disk space or set the task's "
            "PORTABLE_IOAI_*_DATA environment variable to another drive."
        )


def _validate_file(path: Path, spec: dict[str, Any], *, label: str = "asset") -> None:
    if not path.is_file():
        raise PortableIOAIError(
            f"Missing {label}: {path}. Run setup.ps1 -Task {spec.get('task', '<task>')} "
            "-Smoke while internet access is available."
        )
    expected_bytes = spec.get("bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise PortableIOAIError(
            f"{label.capitalize()} has the wrong size: {path} "
            f"({path.stat().st_size} bytes, expected {expected_bytes}). "
            "The download is incomplete or from a different revision."
        )
    expected_lf_hash = spec.get("sha256_lf")
    if expected_lf_hash:
        normalized = (
            path.read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .encode("utf-8")
        )
        actual = hashlib.sha256(normalized).hexdigest()
        if actual.lower() != expected_lf_hash.lower():
            raise PortableIOAIError(
                f"Normalized SHA-256 mismatch for {path}: got {actual}, "
                f"expected {expected_lf_hash}."
            )
    expected_hash = spec.get("sha256")
    if expected_hash and not expected_lf_hash:
        actual = _sha256(path)
        if actual.lower() != expected_hash.lower():
            raise PortableIOAIError(
                f"SHA-256 mismatch for {path}: got {actual}, expected {expected_hash}. "
                "Remove only this corrupt file and rerun setup."
            )


def _asset_errors(data_dir: Path, task: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for asset in task.get("assets", []):
        path = data_dir / asset["path"]
        if not path.is_file():
            errors.append(f"missing {asset['path']}")
            continue
        if asset.get("bytes") is not None and path.stat().st_size != asset["bytes"]:
            errors.append(
                f"wrong size for {asset['path']}: {path.stat().st_size} != {asset['bytes']}"
            )
    return errors


def _download_http(url: str, destination: Path, spec: dict[str, Any]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            _validate_file(destination, spec)
            return destination
        except PortableIOAIError:
            pass

    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "portable-ioai-bootstrap/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
    except (OSError, urllib.error.URLError) as exc:
        raise PortableIOAIError(
            f"Could not download {url}. Check internet access, proxy/firewall settings, "
            f"and available disk space, then rerun setup. Original error: {exc}"
        ) from exc

    _validate_file(partial, spec, label="download")
    partial.replace(destination)
    return destination


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(archive) as zipped:
            for info in zipped.infolist():
                target = (destination / info.filename).resolve()
                try:
                    target.relative_to(root)
                except ValueError as exc:
                    raise PortableIOAIError(
                        f"Unsafe path in downloaded archive: {info.filename!r}"
                    ) from exc
            zipped.extractall(destination)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PortableIOAIError(
            f"Downloaded archive is not a valid ZIP: {archive}. "
            "Delete that archive and rerun setup."
        ) from exc


def _configure_hugging_face(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "HF_HOME": cache_dir,
        "HF_HUB_CACHE": cache_dir / "hub",
        "HF_DATASETS_CACHE": cache_dir / "datasets",
        "SENTENCE_TRANSFORMERS_HOME": cache_dir / "sentence_transformers",
    }
    os.environ.pop("TRANSFORMERS_CACHE", None)
    for key, value in values.items():
        os.environ[key] = str(value)
        Path(value).mkdir(parents=True, exist_ok=True)


@dataclass
class NotebookContext:
    """Stable task paths, independent of the Jupyter launch directory."""

    task_id: str

    def __post_init__(self) -> None:
        if self.task_id not in TASKS:
            raise PortableIOAIError(
                f"Unknown task {self.task_id!r}. Available: {', '.join(TASKS)}"
            )
        self.spec = TASKS[self.task_id]
        self.notebook = PORTABLE_ROOT / self.spec["notebook"]
        self.task_dir = self.notebook.parent
        env_name = self.spec["data_environment"]
        override = os.environ.get(env_name)
        self.data_overridden = bool(override)
        self.data_dir = (
            Path(override).expanduser().resolve()
            if override
            else self.task_dir / ".data"
        )
        self.cache_dir = self.task_dir / ".cache"
        self.output_dir = self.task_dir / "outputs"
        self.download_dir = self.task_dir / ".downloads"

    def prepare_paths(self) -> "NotebookContext":
        self.task_dir.mkdir(parents=True, exist_ok=True)
        if not os.environ.get(self.spec["data_environment"]):
            self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        _configure_hugging_face(self.cache_dir / "huggingface")
        return self

    def ensure_data(self) -> Path:
        self.prepare_paths()
        ensure_task_data(self.task_id, self)
        return self.data_dir

    def describe(self) -> dict[str, str]:
        return {
            "task": self.task_id,
            "notebook": str(self.notebook),
            "data": str(self.data_dir),
            "cache": str(self.cache_dir),
            "outputs": str(self.output_dir),
        }


def _verify_assets(context: NotebookContext) -> None:
    for raw_spec in context.spec.get("assets", []):
        spec = dict(raw_spec)
        spec["task"] = context.task_id
        _validate_file(context.data_dir / spec["path"], spec)


def _validate_home_task_1(context: NotebookContext) -> None:
    _verify_assets(context)
    expected_rows = {
        "train.csv": context.spec["contracts"]["train_rows"],
        "fine_tune.csv": context.spec["contracts"]["fine_tune_rows"],
    }
    referenced: set[str] = set()
    for filename, row_count in expected_rows.items():
        with (context.data_dir / filename).open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != row_count:
            raise PortableIOAIError(
                f"{filename} has {len(rows)} rows; expected {row_count}."
            )
        required = {"path", "split", "target", "category"}
        if not rows or not required.issubset(rows[0]):
            raise PortableIOAIError(f"{filename} does not have columns {sorted(required)}.")
        referenced.update(row["path"] for row in rows)
    missing = [relative for relative in referenced if not (context.data_dir / relative).is_file()]
    if missing:
        raise PortableIOAIError(
            f"Home Task 1 is missing {len(missing)} CSV-referenced audio files; "
            f"first missing path: {missing[0]}"
        )
    expected_audio = context.spec["contracts"]["referenced_wav_files"]
    if len(referenced) != expected_audio:
        raise PortableIOAIError(
            f"Home Task 1 references {len(referenced)} distinct WAV files; "
            f"expected {expected_audio}."
        )


def _validate_home_task_2(context: NotebookContext) -> None:
    _verify_assets(context)
    with (context.data_dir / "train_demos.pkl").open("rb") as stream:
        train = pickle.load(stream)
    with (context.data_dir / "valid_scenarios.pkl").open("rb") as stream:
        valid = pickle.load(stream)
    with (context.data_dir / "test_scenarios.pkl").open("rb") as stream:
        test = pickle.load(stream)
    trajectories = train.get("trajectories", [])
    contracts = context.spec["contracts"]
    if len(trajectories) != contracts["trajectories"]:
        raise PortableIOAIError("train_demos.pkl trajectory count does not match the manifest.")
    if sum(int(t["num_steps"]) for t in trajectories) != contracts["state_action_samples"]:
        raise PortableIOAIError("train_demos.pkl step count does not match the manifest.")
    if len(valid) != contracts["valid_scenarios"] or len(test) != contracts["test_scenarios"]:
        raise PortableIOAIError("Home Task 2 scenario counts do not match the manifest.")


def _validate_home_task_3(context: NotebookContext) -> None:
    _verify_assets(context)
    animals = [
        line for line in (context.data_dir / "animals_pool.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    questions = [
        line for line in (context.data_dir / "questions_pool.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contracts = context.spec["contracts"]
    if len(animals) != contracts["animals"] or len(questions) != contracts["questions"]:
        raise PortableIOAIError("Home Task 3 pool sizes do not match the manifest.")
    for filename, expected in (("dev.csv", contracts["dev_rows"]), ("test1.csv", contracts["test1_rows"])):
        with (context.data_dir / filename).open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected or not rows or "animal" not in rows[0]:
            raise PortableIOAIError(f"{filename} does not satisfy the expected animal-row contract.")


def validate_task_data(task_id: str, context: NotebookContext | None = None) -> None:
    context = context or NotebookContext(task_id)
    if task_id == "home_task_1":
        _validate_home_task_1(context)
    elif task_id == "home_task_2":
        _validate_home_task_2(context)
    elif task_id == "home_task_3":
        _validate_home_task_3(context)
    else:
        _verify_assets(context)


def _fetch_google_drive_zip(context: NotebookContext) -> None:
    try:
        _validate_home_task_1(context)
        return
    except (PortableIOAIError, OSError, pickle.UnpicklingError):
        pass

    _ensure_free_space(context.task_dir, context.spec["minimum_free_bytes"])
    archive_spec = context.spec["bootstrap"]["archive"]
    archive = context.download_dir / archive_spec["name"]
    if archive.is_file():
        try:
            _validate_file(archive, archive_spec, label="archive")
        except PortableIOAIError:
            archive.unlink()

    if not archive.is_file():
        try:
            import gdown
        except ImportError as exc:
            raise PortableIOAIError(
                "gdown is unavailable. Run setup.ps1 to install the portable dependencies."
            ) from exc
        failures: list[str] = []
        for file_id in context.spec["bootstrap"]["file_ids"]:
            partial = archive.with_name(archive.name + ".part")
            if partial.exists():
                partial.unlink()
            try:
                result = gdown.download(
                    id=file_id,
                    output=str(partial),
                    quiet=False,
                    use_cookies=False,
                )
                if not result:
                    raise RuntimeError("Google Drive returned no file")
                _validate_file(partial, archive_spec, label="archive download")
                partial.replace(archive)
                break
            except Exception as exc:  # gdown exposes several backend exception types
                failures.append(f"{file_id}: {exc}")
        else:
            raise PortableIOAIError(
                "All Home Task 1 Google Drive mirrors failed or returned a file with "
                "the wrong checksum. Check internet access and Drive quota, then retry. "
                f"Failures: {'; '.join(failures)}"
            )

    _safe_extract_zip(archive, context.data_dir)
    _validate_home_task_1(context)


def _fetch_google_drive_folder(context: NotebookContext) -> None:
    try:
        validate_task_data(context.task_id, context)
        return
    except (PortableIOAIError, OSError, pickle.UnpicklingError):
        pass

    _ensure_free_space(context.task_dir, context.spec["minimum_free_bytes"])
    try:
        import gdown
    except ImportError as exc:
        raise PortableIOAIError(
            "gdown is unavailable. Run setup.ps1 to install the portable dependencies."
        ) from exc

    staging = context.download_dir / "google_drive_folder"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download_folder(
            id=context.spec["bootstrap"]["folder_id"],
            output=str(staging),
            quiet=False,
            use_cookies=False,
        )
    except Exception as exc:
        raise PortableIOAIError(
            f"Google Drive folder download failed for {context.task_id}. "
            "Check internet access and Drive quota, then retry. "
            f"Original error: {exc}"
        ) from exc

    for asset in context.spec["assets"]:
        candidates = [
            path for path in staging.rglob(Path(asset["path"]).name)
            if path.is_file()
        ]
        matching: Path | None = None
        for candidate in candidates:
            try:
                _validate_file(candidate, asset, label="staged download")
                matching = candidate
                break
            except PortableIOAIError:
                continue
        if matching is None:
            raise PortableIOAIError(
                f"Google Drive folder did not contain a valid {asset['path']}."
            )
        destination = context.data_dir / asset["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matching, destination)
    validate_task_data(context.task_id, context)


def _fetch_http_assets(context: NotebookContext) -> None:
    _ensure_free_space(context.task_dir, context.spec["minimum_free_bytes"])
    for asset in context.spec.get("assets", []):
        url = asset.get("url")
        if not url:
            continue
        _download_http(url, context.data_dir / asset["path"], asset)
    _verify_assets(context)


def _hf_download_dataset(context: NotebookContext, dataset_spec: dict[str, Any]):
    _configure_hugging_face(context.cache_dir / "huggingface")
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise PortableIOAIError(
            "Hugging Face dependencies are unavailable. Run setup.ps1 first."
        ) from exc

    remote = dataset_spec["remote_file"]
    try:
        raw_path = Path(
            hf_hub_download(
                repo_id=dataset_spec["id"],
                repo_type="dataset",
                revision=dataset_spec["revision"],
                filename=remote["path"],
                cache_dir=str(context.cache_dir / "huggingface" / "hub"),
            )
        )
        _validate_file(raw_path, remote, label="Hugging Face dataset file")
        kwargs: dict[str, Any] = {
            "path": dataset_spec["id"],
            "revision": dataset_spec["revision"],
            "split": dataset_spec["split"],
            "cache_dir": str(context.cache_dir / "huggingface" / "datasets"),
        }
        if dataset_spec.get("data_dir"):
            kwargs["data_dir"] = dataset_spec["data_dir"]
        dataset = load_dataset(**kwargs)
    except Exception as exc:
        if isinstance(exc, PortableIOAIError):
            raise
        raise PortableIOAIError(
            f"Could not load pinned Hugging Face dataset {dataset_spec['id']} "
            f"at {dataset_spec['revision']}. Check internet access and, if Hugging "
            f"Face requests authentication, run `hf auth login`. Original error: {exc}"
        ) from exc
    if len(dataset) != dataset_spec["rows"]:
        raise PortableIOAIError(
            f"{dataset_spec['id']}:{dataset_spec['split']} has {len(dataset)} rows; "
            f"expected {dataset_spec['rows']} at the pinned revision."
        )
    return dataset


def load_hf_datasets(task_id: str) -> dict[str, Any]:
    context = NotebookContext(task_id).prepare_paths()
    return {
        spec["name"]: _hf_download_dataset(context, spec)
        for spec in context.spec.get("datasets", [])
    }


def _fetch_embedding_model(context: NotebookContext):
    model_spec = context.spec["models"]["embedding"]
    _configure_hugging_face(context.cache_dir / "huggingface")
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(
            model_spec["id"],
            revision=model_spec["revision"],
            device="cpu",
            cache_folder=str(context.cache_dir / "huggingface" / "sentence_transformers"),
        )
    except Exception as exc:
        raise PortableIOAIError(
            f"Could not load pinned embedding model {model_spec['id']} at "
            f"{model_spec['revision']}. Check internet access and disk space. "
            f"Original error: {exc}"
        ) from exc


def _fetch_home_task_3_model(context: NotebookContext) -> None:
    model = context.spec["models"]["smoke"]
    _configure_hugging_face(context.cache_dir / "huggingface")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model["id"],
            revision=model["revision"],
            cache_dir=str(context.cache_dir / "huggingface" / "hub"),
        )
    except Exception as exc:
        raise PortableIOAIError(
            f"Could not download the pinned Home Task 3 smoke model {model['id']}. "
            f"Check internet access and leave at least {_human_bytes(context.spec['minimum_free_bytes'])} "
            f"free. Original error: {exc}"
        ) from exc


def ensure_task_data(task_id: str, context: NotebookContext | None = None) -> None:
    context = (context or NotebookContext(task_id)).prepare_paths()
    bootstrap_kind = context.spec["bootstrap"]["kind"]
    if context.data_overridden and bootstrap_kind != "hugging_face":
        try:
            validate_task_data(task_id, context)
            return
        except Exception as exc:
            raise PortableIOAIError(
                f"{context.spec['data_environment']} points to {context.data_dir}, "
                "but that directory does not satisfy the pinned asset contract. "
                "Correct the path or unset the variable to download into the task-local "
                f".data directory. Validation error: {exc}"
            ) from exc
    if bootstrap_kind == "google_drive_zip":
        _fetch_google_drive_zip(context)
    elif bootstrap_kind == "google_drive_folder":
        _fetch_google_drive_folder(context)
    elif bootstrap_kind == "http_assets":
        _fetch_http_assets(context)
    elif bootstrap_kind == "hugging_face":
        pass
    else:
        raise PortableIOAIError(f"Unsupported bootstrap kind: {bootstrap_kind}")


def fetch_task(task_id: str) -> dict[str, Any]:
    context = NotebookContext(task_id).prepare_paths()
    ensure_task_data(task_id, context)
    result: dict[str, Any] = context.describe()
    if task_id == "home_task_3":
        _fetch_home_task_3_model(context)
        result["model"] = context.spec["models"]["smoke"]
    elif task_id == "chicken_counting":
        datasets = load_hf_datasets(task_id)
        result["datasets"] = {name: len(value) for name, value in datasets.items()}
    elif task_id == "concepts_cpu":
        datasets = load_hf_datasets(task_id)
        _fetch_embedding_model(context)
        result["datasets"] = {name: len(value) for name, value in datasets.items()}
        result["model"] = context.spec["models"]["embedding"]
    return result


def _smoke_home_task_1(context: NotebookContext) -> dict[str, Any]:
    _validate_home_task_1(context)
    try:
        import librosa
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
    except ImportError as exc:
        raise PortableIOAIError(f"Home Task 1 import failed: {exc}. Run setup.ps1.") from exc

    with (context.data_dir / "train.csv").open("r", encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    audio_path = context.data_dir / row["path"]
    waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    extractor = ASTFeatureExtractor.from_pretrained(context.data_dir / "model")
    inputs = extractor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    expected_features = tuple(context.spec["contracts"]["feature_shape"])
    if tuple(inputs["input_values"].shape) != expected_features:
        raise PortableIOAIError(
            f"AST preprocessing returned {tuple(inputs['input_values'].shape)}; "
            f"expected {expected_features}."
        )
    model = ASTForAudioClassification.from_pretrained(context.data_dir / "model")
    model.to("cpu").eval()
    with torch.no_grad():
        logits = model(**{key: value.to("cpu") for key, value in inputs.items()}).logits
    expected_logits = tuple(context.spec["contracts"]["logits_shape"])
    if tuple(logits.shape) != expected_logits:
        raise PortableIOAIError(
            f"AST forward pass returned {tuple(logits.shape)}; expected {expected_logits}."
        )
    return {
        "audio": row["path"],
        "samples": int(len(waveform)),
        "sample_rate": int(sample_rate),
        "features": list(inputs["input_values"].shape),
        "logits": list(logits.shape),
        "device": "cpu",
    }


class _DeliverySimulator:
    grid_size = 8
    n_depots = 6
    max_steps = 80
    deltas = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}

    def reset(self, scenario: dict[str, Any]) -> tuple[int, int, int, int]:
        self.step_count = 0
        self.carrying = False
        self.walls = {tuple(cell) for cell in scenario["walls"]}
        self.depots = [tuple(cell) for cell in scenario["depots"]]
        self.agent_pos = tuple(scenario["agent_pos"])
        self.package_location = int(scenario["package_location"])
        self.destination = int(scenario["destination"])
        return self.state()

    def state(self) -> tuple[int, int, int, int]:
        package = self.n_depots if self.carrying else self.package_location
        return (
            int(self.agent_pos[0]),
            int(self.agent_pos[1]),
            int(package),
            int(self.destination),
        )

    def can_enter(self, row: int, col: int) -> bool:
        return (
            0 <= row < self.grid_size
            and 0 <= col < self.grid_size
            and (row, col) not in self.walls
        )

    def valid_action_mask(self):
        import numpy as np
        row, col, _, destination = self.state()
        mask = np.zeros(6, dtype=bool)
        for action, (dr, dc) in self.deltas.items():
            mask[action] = self.can_enter(row + dr, col + dc)
        mask[4] = (not self.carrying) and self.agent_pos == self.depots[self.package_location]
        mask[5] = self.carrying and self.agent_pos == self.depots[destination]
        return mask

    def step(self, action: int):
        action = int(action)
        done = False
        if action in self.deltas:
            dr, dc = self.deltas[action]
            target = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if self.can_enter(*target):
                self.agent_pos = target
        elif action == 4 and not self.carrying and self.agent_pos == self.depots[self.package_location]:
            self.carrying = True
        elif action == 5 and self.carrying and self.agent_pos == self.depots[self.destination]:
            done = True
            self.carrying = False
            self.package_location = self.destination
        elif action not in (4, 5):
            raise ValueError(f"unknown action: {action}")
        self.step_count += 1
        return self.state(), done, self.step_count >= self.max_steps and not done


def _smoke_home_task_2(context: NotebookContext) -> dict[str, Any]:
    _validate_home_task_2(context)
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise PortableIOAIError(f"Home Task 2 import failed: {exc}. Run setup.ps1.") from exc

    with (context.data_dir / "train_demos.pkl").open("rb") as stream:
        train = pickle.load(stream)
    with (context.data_dir / "valid_scenarios.pkl").open("rb") as stream:
        valid = pickle.load(stream)
    with (context.data_dir / "test_scenarios.pkl").open("rb") as stream:
        test = pickle.load(stream)
    trajectory = train["trajectories"][0]
    observation = trajectory["observations"][0]
    action = int(trajectory["actions"][0])
    grid = np.asarray(observation["grid"], dtype=np.float32)
    vector = np.asarray(observation["vector"], dtype=np.float32)
    mask = np.asarray(observation["action_mask"], dtype=bool)
    contracts = context.spec["contracts"]
    if (
        tuple(grid.shape) != tuple(contracts["grid_shape"])
        or tuple(vector.shape) != tuple(contracts["vector_shape"])
        or tuple(mask.shape) != tuple(contracts["action_mask_shape"])
    ):
        raise PortableIOAIError("Home Task 2 observation shapes do not match the manifest.")
    features = np.concatenate([grid.reshape(-1), vector])
    if features.size != contracts["flattened_features"]:
        raise PortableIOAIError("Home Task 2 flattened feature count is not 397.")
    model = torch.nn.Sequential(
        torch.nn.Linear(features.size, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, contracts["actions"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.tensor(features).unsqueeze(0)
    y = torch.tensor([action], dtype=torch.long)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    simulator = _DeliverySimulator()
    simulator.reset(valid[0])
    valid_actions = np.flatnonzero(simulator.valid_action_mask())
    if not len(valid_actions):
        raise PortableIOAIError("Home Task 2 validation scenario has no valid action.")
    next_state, done, truncated = simulator.step(int(valid_actions[0]))
    return {
        "contracts_loaded": {
            "trajectories": len(train["trajectories"]),
            "validation": len(valid),
            "test": len(test),
        },
        "features": int(features.size),
        "logits": list(logits.shape),
        "loss": float(loss.detach()),
        "rollout_state": list(next_state),
        "rollout_done": bool(done),
        "rollout_truncated": bool(truncated),
        "device": "cpu",
    }


def _smoke_home_task_3(context: NotebookContext) -> dict[str, Any]:
    _validate_home_task_3(context)
    _fetch_home_task_3_model(context)
    support = context.task_dir / "support"
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    smoke_model = context.spec["models"]["smoke"]
    os.environ["PORTABLE_IOAI_SMOKE"] = "1"
    os.environ["PORTABLE_IOAI_HT3_MODEL"] = smoke_model["id"]
    os.environ["PORTABLE_IOAI_HT3_MODEL_REVISION"] = smoke_model["revision"]
    os.environ["PORTABLE_IOAI_HOME_TASK_3_DATA"] = str(context.data_dir)
    _configure_hugging_face(context.cache_dir / "huggingface")
    try:
        from evaluate import load_pools
        from interactor import Interactor
    except Exception as exc:
        raise PortableIOAIError(f"Home Task 3 helper import failed: {exc}") from exc

    animals, questions = load_pools(
        context.data_dir / "animals_pool.txt",
        context.data_dir / "questions_pool.txt",
    )
    probe = Interactor(
        gold_animal="octopus",
        animals_pool=animals,
        questions_pool=questions,
        budget=1,
        model_name=smoke_model["id"],
    )
    response = probe.ask(questions[0])
    if response not in {"yes", "no"}:
        raise PortableIOAIError(f"Home Task 3 oracle returned invalid response {response!r}.")
    return {
        "animals": len(animals),
        "questions": len(questions),
        "model": smoke_model,
        "question": questions[0],
        "response": response,
        "queries_used": probe.queries_used,
        "device": str(probe._model.device),
    }


def _chicken_model():
    import torch
    import torch.nn.functional as functional

    class FeatureExtraction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=2, dilation=2)
            self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
            self.pool2 = torch.nn.MaxPool2d(2, 2)
            self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)
            self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
            self.pool4 = torch.nn.MaxPool2d(2, 2)

        def forward(self, values):
            values = functional.relu(self.conv1(values))
            values = functional.relu(self.conv2(values))
            values = self.pool2(values)
            values = functional.relu(self.conv3(values))
            values = functional.relu(self.conv4(values))
            return self.pool4(values)

    class ChickenCounting(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extraction = FeatureExtraction()
            self.feature_decoder = torch.nn.Sequential(
                torch.nn.Conv2d(128, 32, kernel_size=3, padding=2, dilation=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 1, kernel_size=3, padding=2, dilation=2),
                torch.nn.ReLU(),
            )

        def forward(self, values):
            return self.feature_decoder(self.feature_extraction(values))

    return ChickenCounting()


def _smoke_chicken_counting(context: NotebookContext) -> dict[str, Any]:
    _fetch_http_assets(context)
    datasets = load_hf_datasets("chicken_counting")
    try:
        import numpy as np
        import torch
        from torchvision.transforms.functional import pil_to_tensor
    except ImportError as exc:
        raise PortableIOAIError(f"Chicken Counting import failed: {exc}. Run setup.ps1.") from exc

    row = datasets["train"][0]
    image = pil_to_tensor(row["image"].convert("RGB")).float() / 255.0
    density = torch.as_tensor(np.asarray(row["density"]), dtype=torch.float32).unsqueeze(0)
    if tuple(image.shape) != tuple(context.spec["contracts"]["image_shape"]):
        raise PortableIOAIError(f"Chicken image shape is {tuple(image.shape)}, not the expected shape.")
    if tuple(density.shape) != tuple(context.spec["contracts"]["density_shape"]):
        raise PortableIOAIError(
            f"Chicken density shape is {tuple(density.shape)}, not the expected shape."
        )

    state = torch.load(
        context.data_dir / "base.pth",
        map_location="cpu",
        weights_only=True,
    )
    if len(state) != context.spec["contracts"]["weight_tensors"]:
        raise PortableIOAIError("Chicken Counting baseline weight does not contain eight tensors.")
    model = _chicken_model().cpu()
    feature_state = {
        key.split(".", 1)[1]: value
        for key, value in state.items()
        if key.startswith("feature_extraction.")
    }
    model.feature_extraction.load_state_dict(feature_state, strict=True)
    # A divisible-by-four crop exercises the exact model while keeping a CPU smoke
    # run small. The full image/density contracts were validated immediately above.
    crop = image[:, :128, :128].unsqueeze(0)
    with torch.no_grad():
        output = model(crop)
    if tuple(output.shape) != (1, 1, 32, 32):
        raise PortableIOAIError(f"Chicken smoke forward returned {tuple(output.shape)}.")
    return {
        "train_rows": len(datasets["train"]),
        "validation_rows": len(datasets["validation"]),
        "test_rows": len(datasets["test"]),
        "image": list(image.shape),
        "density": list(density.shape),
        "weight_tensors": len(state),
        "smoke_crop": list(crop.shape),
        "smoke_output": list(output.shape),
        "device": "cpu",
    }


def _smoke_concepts(context: NotebookContext) -> dict[str, Any]:
    datasets = load_hf_datasets("concepts_cpu")
    model = _fetch_embedding_model(context)
    try:
        import numpy as np
    except ImportError as exc:
        raise PortableIOAIError(f"Concepts import failed: {exc}. Run setup.ps1.") from exc

    hints = datasets["hint_descriptions"]
    train = datasets["train"]
    hint_ids = np.asarray([int(row["ID"]) for row in hints], dtype=np.int64)
    hint_texts = [str(row["Description"]).replace("\n", " ") for row in hints]
    hint_embeddings = model.encode(
        hint_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    sample = train[0]
    labels = list(sample["options"])
    if sample["label"] not in labels:
        labels.insert(0, sample["label"])
    option_embeddings = model.encode(
        labels,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    answer_index = labels.index(sample["label"])
    similarity = hint_embeddings @ option_embeddings[answer_index]
    clue = [[int(hint_ids[int(np.argmax(similarity))])]]
    low, high = context.spec["contracts"]["hint_ids"]
    if not (len(clue) == 1 and len(clue[0]) == 1 and low <= clue[0][0] <= high):
        raise PortableIOAIError(f"Concepts generated invalid smoke clue: {clue}")

    # Also force one row from each held-out public split to decode now.
    _ = datasets["validation"][0]
    _ = datasets["test"][0]
    output = context.output_dir / "smoke_clues.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clue) + "\n", encoding="utf-8")
    reloaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    if reloaded != [clue]:
        raise PortableIOAIError("Concepts JSONL output did not round-trip.")
    archive = context.output_dir / "smoke_submission.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(output, arcname="clues_a.jsonl")
    with zipfile.ZipFile(archive) as zipped:
        if zipped.namelist() != ["clues_a.jsonl"]:
            raise PortableIOAIError("Concepts smoke ZIP has the wrong members.")
    return {
        "dataset_rows": {name: len(dataset) for name, dataset in datasets.items()},
        "hint_embeddings": list(hint_embeddings.shape),
        "option_embeddings": list(option_embeddings.shape),
        "clue": clue,
        "jsonl_rows": 1,
        "output": str(output),
        "device": "cpu",
        "paid_api_used": False,
    }


def _smoke_help_bobai(context: NotebookContext) -> dict[str, Any]:
    _fetch_http_assets(context)
    try:
        import numpy as np
        import torch
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise PortableIOAIError(f"Help BOBAI import failed: {exc}. Run setup.ps1.") from exc

    training = torch.load(
        context.data_dir / "training_set/train-dev_dataset_with_labels.pt",
        map_location="cpu",
        weights_only=True,
    )
    validation = torch.load(
        context.data_dir / "Solution/validation_set/eval_dataset.pt",
        map_location="cpu",
        weights_only=True,
    )
    test = torch.load(
        context.data_dir / "Solution/test_set/test_dataset.pt",
        map_location="cpu",
        weights_only=True,
    )
    contracts = context.spec["contracts"]
    if tuple(training.shape) != tuple(contracts["training_shape"]):
        raise PortableIOAIError("Help BOBAI training tensor shape does not match.")
    if tuple(validation.shape) != tuple(contracts["validation_shape"]):
        raise PortableIOAIError("Help BOBAI validation tensor shape does not match.")
    if tuple(test.shape) != tuple(contracts["test_shape"]):
        raise PortableIOAIError("Help BOBAI test tensor shape does not match.")

    inputs = training[:, :, :-1].reshape(-1, 768).numpy()
    labels = training[:, :, -1].reshape(-1)
    gate_labels = torch.where(labels < 4, 0, labels - 4).numpy().astype(np.int64)
    gate = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    gate.fit(inputs, gate_labels)
    state = torch.load(
        context.data_dir / "training_set/base_classifier.pth",
        map_location="cpu",
        weights_only=True,
    )
    base = torch.nn.Linear(768, 5)
    base.load_state_dict(state)
    batch = validation[:4].reshape(-1, 768).float()
    gate_predictions = gate.predict(batch.numpy())
    with torch.no_grad():
        base_predictions = base(batch).argmax(dim=1).numpy()
    predictions = np.where(gate_predictions > 0, gate_predictions + 4, base_predictions).astype(int)
    low, high = contracts["prediction_classes"]
    if predictions.shape != (4,) or not np.all((predictions >= low) & (predictions <= high)):
        raise PortableIOAIError(f"Help BOBAI produced invalid predictions: {predictions!r}")
    return {
        "training": list(training.shape),
        "validation": list(validation.shape),
        "test": list(test.shape),
        "base_weight": list(state["weight"].shape),
        "predictions": predictions.tolist(),
        "prediction_shape": list(predictions.shape),
        "device": "cpu",
    }


SMOKE_FUNCTIONS = {
    "home_task_1": _smoke_home_task_1,
    "home_task_2": _smoke_home_task_2,
    "home_task_3": _smoke_home_task_3,
    "chicken_counting": _smoke_chicken_counting,
    "concepts_cpu": _smoke_concepts,
    "help_bobai": _smoke_help_bobai,
}


def smoke_task(task_id: str, *, ensure: bool = True) -> dict[str, Any]:
    context = NotebookContext(task_id).prepare_paths()
    if ensure:
        ensure_task_data(task_id, context)
    result = SMOKE_FUNCTIONS[task_id](context)
    return {"status": "passed", "task": task_id, **result}


def _dependency_modules(task_ids: Iterable[str]) -> list[str]:
    mapping = {
        "pillow": "PIL",
        "scikit-learn": "sklearn",
        "sentence-transformers": "sentence_transformers",
    }
    modules: set[str] = {"nbclient", "nbformat"}
    for task_id in task_ids:
        for dependency in TASKS[task_id].get("dependencies", []):
            modules.add(mapping.get(dependency, dependency.replace("-", "_")))
    return sorted(modules)


def preflight(task_ids: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    largest_file = {"path": "", "bytes": 0}
    if sys.version_info[:2] != (3, 12):
        errors.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is active; "
            "the portable environment requires Python 3.12."
        )
    for module in _dependency_modules(task_ids):
        if importlib.util.find_spec(module) is None:
            errors.append(f"missing Python dependency: {module}")
    for task_id in task_ids:
        context = NotebookContext(task_id)
        if not context.notebook.is_file():
            errors.append(f"missing canonical notebook: {context.notebook}")
        else:
            try:
                notebook = json.loads(context.notebook.read_text(encoding="utf-8"))
                if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
                    errors.append(f"invalid notebook contract: {context.notebook}")
                smoke_cells = [
                    cell for cell in notebook.get("cells", [])
                    if cell.get("cell_type") == "code"
                    and "portable-smoke" in cell.get("metadata", {}).get("tags", [])
                ]
                if len(smoke_cells) != 2:
                    errors.append(
                        f"{context.notebook} has {len(smoke_cells)} portable-smoke cells; expected 2"
                    )
                for index, cell in enumerate(notebook.get("cells", [])):
                    if cell.get("cell_type") != "code":
                        continue
                    source = cell.get("source", "")
                    if isinstance(source, list):
                        source = "".join(source)
                    if source.lstrip().startswith(("%", "!")):
                        continue
                    try:
                        ast.parse(source)
                    except SyntaxError as exc:
                        errors.append(
                            f"{context.notebook} code cell {index} is not valid Python: {exc}"
                        )
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"cannot parse {context.notebook}: {exc}")
        try:
            free = shutil.disk_usage(context.task_dir).free
            if free < context.spec["minimum_free_bytes"]:
                warnings.append(
                    f"{task_id}: only {_human_bytes(free)} free; "
                    f"recommended {_human_bytes(context.spec['minimum_free_bytes'])}"
                )
        except OSError as exc:
            errors.append(f"{task_id}: cannot inspect disk space: {exc}")
        for support in context.spec.get("support", []):
            path = context.task_dir / support["path"]
            if not path.is_file():
                errors.append(f"{task_id}: missing tracked helper {path}")
            elif path.suffix == ".py":
                try:
                    _validate_file(path, support, label="tracked helper")
                    ast.parse(path.read_text(encoding="utf-8"))
                except (OSError, SyntaxError, PortableIOAIError) as exc:
                    errors.append(f"{task_id}: invalid tracked helper {path}: {exc}")

        ignored_probe = context.task_dir / ".data" / "__portable_ignore_probe__"
        try:
            ignored = subprocess.run(
                ["git", "-C", str(REPOSITORY_ROOT), "check-ignore", "-q", str(ignored_probe)],
                check=False,
            ).returncode == 0
            notebook_ignored = subprocess.run(
                ["git", "-C", str(REPOSITORY_ROOT), "check-ignore", "-q", str(context.notebook)],
                check=False,
            ).returncode == 0
            if not ignored:
                errors.append(f"{task_id}: task-local .data is not ignored by Git")
            if notebook_ignored:
                errors.append(f"{task_id}: canonical notebook is unexpectedly ignored by Git")
        except OSError as exc:
            warnings.append(f"{task_id}: could not run Git ignore checks: {exc}")

    excluded_parts = {".data", ".cache", ".downloads", "outputs", "smoke-results", "__pycache__"}
    for path in PORTABLE_ROOT.rglob("*"):
        if not path.is_file() or any(part in excluded_parts for part in path.parts):
            continue
        size = path.stat().st_size
        if size > largest_file["bytes"]:
            largest_file = {
                "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "bytes": size,
            }
        if size >= 100 * 1024 * 1024:
            errors.append(
                f"portable tracked candidate exceeds GitHub's 100 MiB limit: {path} ({size} bytes)"
            )
    return {
        "status": "passed" if not errors else "failed",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "tasks": task_ids,
        "largest_portable_file": largest_file,
        "errors": errors,
        "warnings": warnings,
    }


def notebook_smoke(task_id: str) -> dict[str, Any]:
    context = NotebookContext(task_id).prepare_paths()
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    runtime_dir = context.cache_dir / "jupyter-runtime"
    temp_dir = context.cache_dir / "tmp"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)
    os.environ["IPYTHONDIR"] = str(context.cache_dir / "ipython")
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise PortableIOAIError(f"Notebook smoke dependencies are unavailable: {exc}") from exc

    notebook = nbformat.read(context.notebook, as_version=4)
    smoke_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "portable-smoke" in cell.get("metadata", {}).get("tags", [])
    ]
    if len(smoke_cells) < 2:
        raise PortableIOAIError(
            f"{context.notebook} has {len(smoke_cells)} portable-smoke cells; expected at least two."
        )
    ephemeral = nbformat.v4.new_notebook(
        cells=smoke_cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3.12 (portable-ioai)",
                "language": "python",
                "name": MANIFEST["kernel"],
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    before = _sha256(context.notebook)
    old_smoke = os.environ.get("PORTABLE_IOAI_SMOKE")
    os.environ["PORTABLE_IOAI_SMOKE"] = "1"
    try:
        NotebookClient(
            ephemeral,
            timeout=1800,
            kernel_name=MANIFEST["kernel"],
            resources={"metadata": {"path": str(context.task_dir)}},
            allow_errors=False,
        ).execute()
    except Exception as exc:
        raise PortableIOAIError(
            f"Notebook smoke execution failed for {task_id}: {exc}"
        ) from exc
    finally:
        if old_smoke is None:
            os.environ.pop("PORTABLE_IOAI_SMOKE", None)
        else:
            os.environ["PORTABLE_IOAI_SMOKE"] = old_smoke
    after = _sha256(context.notebook)
    if before != after:
        raise PortableIOAIError(f"Smoke execution changed tracked notebook {context.notebook}.")
    return {
        "status": "passed",
        "task": task_id,
        "notebook": str(context.notebook),
        "smoke_cells": len(smoke_cells),
        "notebook_sha256": after,
    }


def _select_tasks(raw: list[str]) -> list[str]:
    selected: list[str] = []
    for value in raw:
        for task_id in value.split(","):
            task_id = task_id.strip()
            if not task_id:
                continue
            if task_id == "all":
                return list(TASKS)
            if task_id not in TASKS:
                raise PortableIOAIError(
                    f"Unknown task {task_id!r}. Available tasks: {', '.join(TASKS)}"
                )
            if task_id not in selected:
                selected.append(task_id)
    return selected or list(TASKS)


def _save_runtime_report(action: str, results: list[dict[str, Any]]) -> Path:
    report_dir = PORTABLE_ROOT / "smoke-results"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "action": action,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "results": results,
    }
    path = report_dir / f"{action}-latest.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("list", "preflight", "fetch", "smoke", "notebook-smoke"),
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task ID, comma-separated task IDs, or all. May be repeated.",
    )
    args = parser.parse_args(argv)
    try:
        task_ids = _select_tasks(args.task)
        if args.action == "list":
            print(json.dumps({task_id: TASKS[task_id]["title"] for task_id in task_ids}, indent=2))
            return 0
        if args.action == "preflight":
            result = preflight(task_ids)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "passed" else 1

        results: list[dict[str, Any]] = []
        for task_id in task_ids:
            print(f"\n[{args.action}] {task_id}", flush=True)
            if args.action == "fetch":
                result = {"status": "passed", "task": task_id, **fetch_task(task_id)}
            elif args.action == "smoke":
                result = smoke_task(task_id, ensure=True)
            else:
                result = notebook_smoke(task_id)
            results.append(result)
            print(json.dumps(result, indent=2, default=str), flush=True)
        report = _save_runtime_report(args.action, results)
        print(f"\nRuntime report: {report}")
        return 0
    except PortableIOAIError as exc:
        print(f"\nPORTABILITY ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Partial downloads remain under task-local ignored directories.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
