from __future__ import annotations

import csv
import math
import shutil
import textwrap
import zipfile
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from sklearn.datasets import load_digits


ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260708)


TASKS = {
    "round-1": {
        "stochastic_rift": "stochastic_rift.md",
        "project_kraken": "project_kraken.md",
        "star_observatory": "star_observatory.md",
    },
    "round-2": {
        "panda_mnist": "panda_mnist.md",
        "trace_twins": "trace_twins.md",
    },
}


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def copy_prompt(round_name: str, task_name: str, source_md: str) -> Path:
    task_dir = ROOT / round_name / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / round_name / source_md
    target = task_dir / "prompt.md"
    if source.exists():
        prompt_text = source.read_text(encoding="utf-8")
    elif target.exists():
        prompt_text = target.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(f"missing prompt source: {source}")
    target.write_text(prompt_text, encoding="utf-8")
    return task_dir


def write_dataset_note(task_dir: Path, official_url: str) -> None:
    write_text(
        task_dir / "DATASET_NOTE.md",
        f"""
        # Dataset Note

        The files in `data/` are deterministic local fixture datasets generated for
        notebook practice. They match the task's file names, shapes, interfaces, and
        submission contracts, but they are not the hidden official Nitro judge data.

        Official attachments are available through Nitro after login:

        {official_url}

        Replace `data/` with the official downloaded files before submitting to the
        judge. Keep the notebook output format unchanged.
        """,
    )


def make_notebook(path: Path, title: str, cells: list[tuple[str, str]]) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    nb.cells = [nbf.v4.new_markdown_cell(f"# {title}")]
    for cell_type, source in cells:
        source = textwrap.dedent(source).strip("\n")
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        elif cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(source))
        else:
            raise ValueError(f"unknown cell type: {cell_type}")
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in source_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(source_dir))


def value_iteration(transitions: dict[tuple[int, int], list[tuple[float, float, int]]], n_states: int, gamma: float) -> np.ndarray:
    values = np.zeros(n_states, dtype=np.float64)
    for _ in range(500):
        old = values.copy()
        for state in range(n_states):
            q_values = []
            for action in range(4):
                outcomes = transitions[(state, action)]
                q_values.append(sum(prob * (reward + gamma * old[next_state]) for prob, reward, next_state in outcomes))
            values[state] = max(q_values)
        if np.max(np.abs(values - old)) < 1e-8:
            break
    return values


def build_stochastic_rift(task_dir: Path) -> None:
    data_dir = task_dir / "data"
    clean_dir(data_dir)
    n_states = 24
    n_actions = 4
    gamma = 0.99
    goal_state = n_states - 1
    bad_state = n_states - 2

    transitions: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
    for state in range(n_states):
        for action in range(n_actions):
            if state == goal_state:
                outcomes = [(1.0, 0.0, goal_state)]
            else:
                forward = min(goal_state, state + action + 1)
                slip = max(0, state - 1)
                jump = bad_state if (state + action) % 7 == 0 else forward
                base_reward = -1.0 + 0.15 * action
                if forward == goal_state:
                    base_reward += 35.0
                if jump == bad_state:
                    base_reward -= 12.0
                outcomes = [
                    (0.70, base_reward, forward),
                    (0.20, base_reward - 2.0, slip),
                    (0.10, base_reward - 4.0, jump),
                ]
            transitions[(state, action)] = outcomes

    true_values = value_iteration(transitions, n_states, gamma)

    rows = []
    for state in range(n_states):
        for action in range(n_actions):
            for _ in range(60):
                probs = [outcome[0] for outcome in transitions[(state, action)]]
                idx = int(RNG.choice(len(probs), p=np.array(probs) / np.sum(probs)))
                _, reward, next_state = transitions[(state, action)][idx]
                noisy_reward = reward + float(RNG.normal(0, 0.05))
                rows.append((state, action, round(noisy_reward, 4), next_state))
    for _ in range(240):
        state = int(RNG.integers(0, n_states - 1))
        action = int(RNG.integers(0, n_actions))
        probs = [outcome[0] for outcome in transitions[(state, action)]]
        idx = int(RNG.choice(len(probs), p=np.array(probs) / np.sum(probs)))
        _, reward, next_state = transitions[(state, action)][idx]
        noisy_reward = reward + float(RNG.normal(0, 0.05))
        rows.append((state, action, round(noisy_reward, 4), next_state))
    pd.DataFrame(rows, columns=["current_state", "action", "reward", "next_state"]).sample(frac=1, random_state=7).to_csv(
        data_dir / "sector_logs.csv", index=False
    )
    query_states = pd.DataFrame({"id": range(12), "state_id": np.linspace(0, n_states - 1, 12, dtype=int)})
    query_states.to_csv(data_dir / "query_states.csv", index=False)
    pd.DataFrame({"state_id": query_states["state_id"], "true_value": true_values[query_states["state_id"]]}).to_csv(
        data_dir / "ground_truth_values.csv", index=False
    )

    env_text = f'''
    import numpy as np


    class Sector7Env:
        """Small fixture environment matching the Stochastic Rift starter API."""

        n_states = {n_states}
        n_actions = {n_actions}
        gamma = {gamma}

        def __init__(self, seed=0):
            self.rng = np.random.default_rng(seed)
            self.state = 0

        def reset(self, state=0):
            self.state = int(state)
            return self.state

        def step(self, action):
            action = int(action)
            state = self.state
            if state == {goal_state}:
                return state, 0.0, True, {{}}
            forward = min({goal_state}, state + action + 1)
            slip = max(0, state - 1)
            jump = {bad_state} if (state + action) % 7 == 0 else forward
            base_reward = -1.0 + 0.15 * action
            if forward == {goal_state}:
                base_reward += 35.0
            if jump == {bad_state}:
                base_reward -= 12.0
            outcomes = [
                (0.70, base_reward, forward),
                (0.20, base_reward - 2.0, slip),
                (0.10, base_reward - 4.0, jump),
            ]
            probs = np.array([x[0] for x in outcomes], dtype=float)
            idx = int(self.rng.choice(len(outcomes), p=probs / probs.sum()))
            _, reward, next_state = outcomes[idx]
            self.state = int(next_state)
            return int(next_state), float(reward), next_state == {goal_state}, {{}}
    '''
    write_text(data_dir / "env.py", env_text)

    make_notebook(
        task_dir / "solution.ipynb",
        "CEOAI Practice 1 - Stochastic Rift Minimum Solution",
        [
            (
                "markdown",
                """
                Objective: build the smallest end-to-end value-estimation pipeline:

                1. Load transition logs and query states.
                2. Estimate an empirical MDP.
                3. Run value iteration.
                4. Export `predictions.csv`.

                This notebook uses the local fixture data in `data/`. Replace that folder with the official Nitro files before judging.
                """,
            ),
            (
                "code",
                """
                from pathlib import Path
                import importlib.util
                import numpy as np
                import pandas as pd

                ROOT = Path.cwd()
                DATA = ROOT / "data"
                OUT = ROOT / "outputs"
                OUT.mkdir(exist_ok=True)
                GAMMA = 0.99
                """,
            ),
            (
                "code",
                """
                logs = pd.read_csv(DATA / "sector_logs.csv")
                queries = pd.read_csv(DATA / "query_states.csv")

                spec = importlib.util.spec_from_file_location("env", DATA / "env.py")
                env_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(env_module)
                env = env_module.Sector7Env(seed=0)

                n_states = int(env.n_states)
                n_actions = int(env.n_actions)
                print(logs.head())
                print({"rows": len(logs), "states": n_states, "actions": n_actions, "queries": len(queries)})
                """,
            ),
            (
                "code",
                """
                grouped = (
                    logs.groupby(["current_state", "action", "next_state"], as_index=False)
                    .agg(count=("reward", "size"), reward=("reward", "mean"))
                )

                model = {}
                global_reward = float(logs["reward"].mean())
                for state in range(n_states):
                    for action in range(n_actions):
                        block = grouped[(grouped.current_state == state) & (grouped.action == action)]
                        if len(block) == 0:
                            # Conservative fallback for unseen pairs.
                            model[(state, action)] = [(1.0, global_reward - 1.0, state)]
                            continue
                        total = block["count"].sum()
                        model[(state, action)] = [
                            (row["count"] / total, row["reward"], int(row["next_state"]))
                            for _, row in block.iterrows()
                        ]

                known_pairs = logs[["current_state", "action"]].drop_duplicates().shape[0]
                print({"known_state_action_pairs": known_pairs, "total_pairs": n_states * n_actions})
                """,
            ),
            (
                "code",
                """
                values = np.zeros(n_states, dtype=float)
                deltas = []
                for iteration in range(2500):
                    old = values.copy()
                    for state in range(n_states):
                        q_values = []
                        for action in range(n_actions):
                            q = sum(prob * (reward + GAMMA * old[next_state]) for prob, reward, next_state in model[(state, action)])
                            q_values.append(q)
                        values[state] = max(q_values)
                    delta = float(np.max(np.abs(values - old)))
                    deltas.append(delta)
                    if delta < 1e-6:
                        break

                print({"iterations": len(deltas), "last_delta": deltas[-1]})
                """,
            ),
            (
                "code",
                """
                pred = pd.DataFrame({
                    "subtaskID": 1,
                    "datapointID": queries["id"],
                    "answer": queries["state_id"].map(lambda s: float(values[int(s)])),
                })
                pred.to_csv(OUT / "predictions.csv", index=False)
                pred.head()
                """,
            ),
            (
                "code",
                """
                truth_path = DATA / "ground_truth_values.csv"
                if truth_path.exists():
                    truth = pd.read_csv(truth_path)
                    mse = np.mean((pred["answer"].to_numpy() - truth["true_value"].to_numpy()) ** 2)
                    print({"fixture_mse": float(mse)})

                assert len(pred) == len(queries)
                assert list(pred.columns) == ["subtaskID", "datapointID", "answer"]
                assert pred["answer"].notna().all()
                print("wrote", OUT / "predictions.csv")
                """,
            ),
        ],
    )


def build_project_kraken(task_dir: Path) -> None:
    data_dir = task_dir / "data"
    clean_dir(data_dir)
    n_train, n_test = 48, 16
    ids_train = [f"train_{i:05d}" for i in range(n_train)]
    ids_test = [f"test_{i:05d}" for i in range(n_test)]

    train_slices = RNG.normal(0, 0.7, size=(n_train, 3, 128, 128)).astype("float32")
    test_slices = RNG.normal(0, 0.7, size=(n_test, 3, 128, 128)).astype("float32")
    train_echoes = RNG.normal(0, 1, size=(n_train, 1024, 2)).astype("float32")
    test_echoes = RNG.normal(0, 1, size=(n_test, 1024, 2)).astype("float32")

    for i in range(n_train):
        label = i % 8
        train_slices[i, :, 20 + label : 35 + label, 25:45] += 1.5 + label * 0.1
        train_echoes[i, :, 0] += np.sin(np.linspace(0, (label + 1) * math.pi, 1024))
    for i in range(n_test):
        label = i % 8
        test_slices[i, :, 20 + label : 35 + label, 25:45] += 1.5 + label * 0.1
        test_echoes[i, :, 0] += np.sin(np.linspace(0, (label + 1) * math.pi, 1024))

    np.save(data_dir / "train_slices.npy", train_slices)
    np.save(data_dir / "test_slices.npy", test_slices)
    np.save(data_dir / "train_echoes.npy", train_echoes)
    np.save(data_dir / "test_echoes.npy", test_echoes)

    glyph_vocab = ["NAB", "VEX", "ION", "KAI", "MOR", "ZED", "LUX", "ORB"]

    def glyph_row(i: int) -> str:
        label = i % 8
        tokens = [glyph_vocab[(label + j) % len(glyph_vocab)] for j in range(8)]
        if i % 5 == 0:
            tokens.insert(2, "PHI")
        return " ".join(tokens)

    pd.DataFrame({"datapointID": ids_train, "glyphs": [glyph_row(i) for i in range(n_train)]}).to_csv(
        data_dir / "train_glyphs.csv", index=False
    )
    pd.DataFrame({"datapointID": ids_test, "glyphs": [glyph_row(i) for i in range(n_test)]}).to_csv(
        data_dir / "test_glyphs.csv", index=False
    )

    coeff_base = np.stack(
        [
            np.linspace(0.0, 1.0, 10) * (0.1 + (i % 8) * 0.02)
            + RNG.normal(0, 0.005, size=10)
            for i in range(n_train)
        ]
    )
    stability = np.clip(0.2 + 0.07 * (np.arange(n_train) % 8) + RNG.normal(0, 0.02, n_train), 0, 1)
    targets = pd.DataFrame({"datapointID": ids_train})
    for j in range(10):
        targets[f"coef_{j}"] = coeff_base[:, j]
    targets["topology_class"] = np.arange(n_train) % 8
    targets["stability"] = stability
    targets.to_csv(data_dir / "train_targets.csv", index=False)

    make_notebook(
        task_dir / "solution.ipynb",
        "CEOAI Practice 1 - Project KRAKEN Minimum Solution",
        [
            (
                "markdown",
                """
                Objective: create a first valid multimodal baseline:

                1. Load slices, echoes, glyphs, and targets.
                2. Convert each modality into cheap statistical features.
                3. Train one simple model per subtask.
                4. Export one submission CSV with all three subtask outputs.
                """,
            ),
            (
                "code",
                """
                from pathlib import Path
                import numpy as np
                import pandas as pd
                from sklearn.linear_model import Ridge, LogisticRegression
                from sklearn.metrics import mean_squared_error, f1_score
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import make_pipeline

                ROOT = Path.cwd()
                DATA = ROOT / "data"
                OUT = ROOT / "outputs"
                OUT.mkdir(exist_ok=True)
                """,
            ),
            (
                "code",
                """
                train_slices = np.load(DATA / "train_slices.npy")
                test_slices = np.load(DATA / "test_slices.npy")
                train_echoes = np.load(DATA / "train_echoes.npy")
                test_echoes = np.load(DATA / "test_echoes.npy")
                train_glyphs = pd.read_csv(DATA / "train_glyphs.csv")
                test_glyphs = pd.read_csv(DATA / "test_glyphs.csv")
                targets = pd.read_csv(DATA / "train_targets.csv")

                print({
                    "train_slices": train_slices.shape,
                    "test_slices": test_slices.shape,
                    "train_echoes": train_echoes.shape,
                    "test_echoes": test_echoes.shape,
                })
                """,
            ),
            (
                "code",
                """
                def make_features(slices, echoes, glyph_df):
                    image_features = np.concatenate([
                        slices.mean(axis=(2, 3)),
                        slices.std(axis=(2, 3)),
                        slices.max(axis=(2, 3)),
                    ], axis=1)
                    echo_features = np.concatenate([
                        echoes.mean(axis=1),
                        echoes.std(axis=1),
                        np.mean(np.isclose(echoes, 0), axis=1),
                    ], axis=1)
                    glyph_features = []
                    for text in glyph_df["glyphs"]:
                        tokens = text.split()
                        glyph_features.append([
                            len(tokens),
                            len(set(tokens)),
                            tokens.count("PHI"),
                            sum(len(tok) for tok in tokens) / max(len(tokens), 1),
                        ])
                    return np.concatenate([image_features, echo_features, np.array(glyph_features, dtype=float)], axis=1)

                X = make_features(train_slices, train_echoes, train_glyphs)
                X_test = make_features(test_slices, test_echoes, test_glyphs)
                coef_cols = [f"coef_{i}" for i in range(10)]
                print(X.shape, X_test.shape)
                """,
            ),
            (
                "code",
                """
                idx_train, idx_val = train_test_split(np.arange(len(X)), test_size=0.25, random_state=0, stratify=targets["topology_class"])
                X_train, X_val = X[idx_train], X[idx_val]

                coef_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                class_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                stability_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

                coef_model.fit(X_train, targets.loc[idx_train, coef_cols])
                class_model.fit(X_train, targets.loc[idx_train, "topology_class"])
                stability_model.fit(X_train, targets.loc[idx_train, "stability"])

                coef_pred = coef_model.predict(X_val)
                class_pred = class_model.predict(X_val)
                stability_pred = np.clip(stability_model.predict(X_val), 0, 1)

                print({
                    "coef_mse": float(mean_squared_error(targets.loc[idx_val, coef_cols], coef_pred)),
                    "class_macro_f1": float(f1_score(targets.loc[idx_val, "topology_class"], class_pred, average="macro")),
                    "stability_rmse": float(np.sqrt(mean_squared_error(targets.loc[idx_val, "stability"], stability_pred))),
                })
                """,
            ),
            (
                "code",
                """
                test_coef = coef_model.predict(X_test)
                test_class = class_model.predict(X_test)
                test_stability = np.clip(stability_model.predict(X_test), 0, 1)

                rows = []
                for i, datapoint_id in enumerate(test_glyphs["datapointID"]):
                    rows.append({
                        "subtaskID": 1,
                        "datapointID": datapoint_id,
                        "answer": ";".join(f"{x:.6f}" for x in test_coef[i]),
                    })
                    rows.append({"subtaskID": 2, "datapointID": datapoint_id, "answer": int(test_class[i])})
                    rows.append({"subtaskID": 3, "datapointID": datapoint_id, "answer": float(test_stability[i])})

                submission = pd.DataFrame(rows)
                submission.to_csv(OUT / "submission.csv", index=False)
                assert len(submission) == 3 * len(test_glyphs)
                print("wrote", OUT / "submission.csv")
                submission.head(9)
                """,
            ),
        ],
    )


def render_star_image(center_x: float, center_y: float, flux: float, fried: float, airmass: float) -> np.ndarray:
    yy, xx = np.mgrid[0:128, 0:128]
    sigma = 1.8 + (1.0 / fried) * 2.5 + (airmass - 1.0) * 0.8
    amplitude = flux / 90.0
    image = amplitude * np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2))
    image += RNG.normal(3, 1.5, size=image.shape)
    image = np.clip(image, 0, 255)
    return image.astype("uint8")


def build_star_observatory(task_dir: Path) -> None:
    data_dir = task_dir / "data"
    clean_dir(data_dir)
    train_img_dir = data_dir / "train_images"
    test_img_dir = data_dir / "test_images"
    train_img_dir.mkdir(exist_ok=True)
    test_img_dir.mkdir(exist_ok=True)

    train_rows = []
    for i in range(80):
        image_id = f"{i:05d}.png"
        fried = float(RNG.uniform(0.6, 2.0))
        airmass = float(RNG.uniform(1.0, 2.2))
        center_x = float(RNG.uniform(25, 103))
        center_y = float(RNG.uniform(25, 103))
        flux = float(700 + 2500 * fried / airmass + RNG.normal(0, 80))
        arr = render_star_image(center_x, center_y, flux, fried, airmass)
        Image.fromarray(arr).save(train_img_dir / image_id)
        train_rows.append({"image_id": image_id, "fried_parameter": fried, "airmass": airmass, "target_flux": flux})
    pd.DataFrame(train_rows).to_csv(data_dir / "train.csv", index=False)

    test_rows = []
    hidden_rows = []
    for i in range(24):
        image_id = f"{i:05d}.png"
        fried = float(RNG.uniform(0.6, 2.0))
        airmass = float(RNG.uniform(1.0, 2.2))
        center_x = float(RNG.uniform(25, 103))
        center_y = float(RNG.uniform(25, 103))
        flux = float(700 + 2500 * fried / airmass + RNG.normal(0, 80))
        arr = render_star_image(center_x, center_y, flux, fried, airmass)
        Image.fromarray(arr).save(test_img_dir / image_id)
        test_rows.append({"image_id": image_id})
        hidden_rows.append({"image_id": image_id, "center_x": center_x, "center_y": center_y, "target_flux": flux})
    pd.DataFrame(test_rows).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame(hidden_rows).to_csv(data_dir / "test_hidden.csv", index=False)

    make_notebook(
        task_dir / "solution.ipynb",
        "CEOAI Practice 1 - Star Observatory Minimum Solution",
        [
            (
                "markdown",
                """
                Objective: produce the first valid 600-row-style submission:

                1. Estimate star centers with an intensity-weighted centroid.
                2. Train a simple flux regressor from image summary features.
                3. Export center and flux rows in the required CSV format.
                """,
            ),
            (
                "code",
                """
                from pathlib import Path
                import numpy as np
                import pandas as pd
                from PIL import Image
                from sklearn.linear_model import Ridge
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                from sklearn.model_selection import train_test_split
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler

                ROOT = Path.cwd()
                DATA = ROOT / "data"
                OUT = ROOT / "outputs"
                OUT.mkdir(exist_ok=True)
                """,
            ),
            (
                "code",
                """
                train = pd.read_csv(DATA / "train.csv")
                test = pd.read_csv(DATA / "test.csv")
                print(train.head())
                print({"train": len(train), "test": len(test)})
                """,
            ),
            (
                "code",
                """
                def load_gray(path):
                    return np.asarray(Image.open(path).convert("L"), dtype=float)

                def image_features(image):
                    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
                    weights = np.maximum(image - np.percentile(image, 98), 0)
                    total = weights.sum() + 1e-9
                    cx = float((weights * xx).sum() / total)
                    cy = float((weights * yy).sum() / total)
                    return {
                        "sum": float(image.sum()),
                        "mean": float(image.mean()),
                        "std": float(image.std()),
                        "max": float(image.max()),
                        "centroid_x": cx,
                        "centroid_y": cy,
                    }

                def build_feature_frame(df, folder):
                    rows = []
                    for image_id in df["image_id"]:
                        image = load_gray(DATA / folder / image_id)
                        rows.append({"image_id": image_id, **image_features(image)})
                    return pd.DataFrame(rows)

                train_features = build_feature_frame(train, "train_images").merge(train, on="image_id")
                test_features = build_feature_frame(test, "test_images")
                train_features.head()
                """,
            ),
            (
                "code",
                """
                feature_cols = ["sum", "mean", "std", "max", "centroid_x", "centroid_y"]
                X_train, X_val, y_train, y_val = train_test_split(
                    train_features[feature_cols],
                    np.log1p(train_features["target_flux"]),
                    test_size=0.25,
                    random_state=0,
                )
                flux_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                flux_model.fit(X_train, y_train)
                val_pred = np.expm1(flux_model.predict(X_val))
                print({"fixture_flux_rmse": float(np.sqrt(mean_squared_error(np.expm1(y_val), val_pred)))})
                """,
            ),
            (
                "code",
                """
                test_flux = np.expm1(flux_model.predict(test_features[feature_cols]))
                rows = []
                for i, row in test_features.iterrows():
                    image_id = row["image_id"]
                    rows.append({
                        "subtaskID": 1,
                        "datapointID": image_id,
                        "answer": f"({row['centroid_x']:.2f}, {row['centroid_y']:.2f})",
                    })
                    rows.append({
                        "subtaskID": 2,
                        "datapointID": image_id,
                        "answer": float(test_flux[i]),
                    })

                submission = pd.DataFrame(rows)
                submission.to_csv(OUT / "submission.csv", index=False)
                assert len(submission) == 2 * len(test)
                submission.head()
                """,
            ),
            (
                "code",
                """
                hidden_path = DATA / "test_hidden.csv"
                if hidden_path.exists():
                    hidden = pd.read_csv(hidden_path).merge(test_features, on="image_id")
                    center_mae = mean_absolute_error(hidden[["center_x", "center_y"]], hidden[["centroid_x", "centroid_y"]])
                    flux_rmse = np.sqrt(mean_squared_error(hidden["target_flux"], test_flux))
                    print({"fixture_center_coord_mae": float(center_mae), "fixture_flux_rmse": float(flux_rmse)})

                print("wrote", OUT / "submission.csv")
                """,
            ),
        ],
    )


def make_digit_image(label: int, scanner: int, channels: int) -> np.ndarray:
    digits = load_digits()
    candidates = np.where(digits.target == label)[0]
    base = digits.images[candidates].mean(axis=0)
    img = np.kron(base, np.ones((3, 3)))
    img = np.pad(img, ((2, 2), (2, 2)), mode="constant")
    img = img[:28, :28]
    img = img / max(img.max(), 1) * 255.0
    img = np.roll(img, shift=(scanner % 3) - 1, axis=1)
    img = np.roll(img, shift=((scanner + 1) % 3) - 1, axis=0)
    img = img * (0.85 + 0.02 * scanner) + scanner * 2
    img += RNG.normal(0, 4 + scanner * 0.5, size=img.shape)
    img = np.clip(img, 0, 255).astype("uint8")
    if channels == 1:
        return img[None, :, :]
    return np.stack([
        img,
        np.clip(np.roll(img, shift=1, axis=0) * 0.95, 0, 255).astype("uint8"),
        np.clip(np.roll(img, shift=-1, axis=1) * 1.05, 0, 255).astype("uint8"),
    ])


def build_panda_mnist(task_dir: Path) -> None:
    data_dir = task_dir / "data"
    clean_dir(data_dir)
    train_root = data_dir / "train_data"
    test_root = data_dir / "test_data"
    (train_root / "subtask1").mkdir(parents=True, exist_ok=True)
    (train_root / "subtask2").mkdir(parents=True, exist_ok=True)
    (test_root / "subtask1").mkdir(parents=True, exist_ok=True)
    (test_root / "subtask2").mkdir(parents=True, exist_ok=True)
    label_root = data_dir / "fixture_test_labels"
    label_root.mkdir(exist_ok=True)

    for scanner in range(1, 4):
        labels = np.tile(np.arange(10), 8)
        RNG.shuffle(labels)
        X = np.stack([make_digit_image(int(y), scanner, 1) for y in labels])
        np.save(train_root / "subtask1" / f"scanner{scanner}_X.npy", X)
        np.save(train_root / "subtask1" / f"scanner{scanner}_y.npy", labels.astype("int64"))

        test_labels = np.tile(np.arange(10), 2)
        X_test = np.stack([make_digit_image(int(y), scanner, 1) for y in test_labels])
        np.save(test_root / "subtask1" / f"scanner{scanner}_X.npy", X_test)
        np.save(label_root / f"subtask1_scanner{scanner}_y.npy", test_labels.astype("int64"))

    for scanner in range(1, 9):
        labels = np.tile(np.arange(10), 5)
        RNG.shuffle(labels)
        X = np.stack([make_digit_image(int(y), scanner, 3) for y in labels])
        np.save(train_root / "subtask2" / f"scanner{scanner}_X.npy", X)
        np.save(train_root / "subtask2" / f"scanner{scanner}_y.npy", labels.astype("int64"))

        test_labels = np.tile(np.arange(10), 2)
        X_test = np.stack([make_digit_image(int(y), scanner, 3) for y in test_labels])
        np.save(test_root / "subtask2" / f"scanner{scanner}_X.npy", X_test)
        np.save(label_root / f"subtask2_scanner{scanner}_y.npy", test_labels.astype("int64"))

    zip_dir(train_root, data_dir / "train_data.zip")
    zip_dir(test_root, data_dir / "test_data.zip")

    make_notebook(
        task_dir / "solution.ipynb",
        "CEOAI Practice 2 - Panda MNIST Minimum Solution",
        [
            (
                "markdown",
                """
                Objective: train two tiny Torch models and package a valid submission:

                1. Load scanner arrays from `train_data.zip`.
                2. Train `model_sub1.pt` for `(N, 1, 28, 28)` inputs.
                3. Train `model_sub2.pt` for `(N, 3, 28, 28)` inputs.
                4. Zip both TorchScript models into `submission.zip`.
                """,
            ),
            (
                "code",
                """
                from pathlib import Path
                import zipfile
                import numpy as np
                import torch
                import torch.nn as nn
                import torch.nn.functional as F
                from torch.utils.data import DataLoader, TensorDataset, random_split

                ROOT = Path.cwd()
                DATA = ROOT / "data"
                OUT = ROOT / "outputs"
                OUT.mkdir(exist_ok=True)
                torch.manual_seed(0)
                """,
            ),
            (
                "code",
                """
                def ensure_unzipped(zip_path, target_dir):
                    if target_dir.exists():
                        return
                    target_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(target_dir)

                ensure_unzipped(DATA / "train_data.zip", DATA / "train_data")
                ensure_unzipped(DATA / "test_data.zip", DATA / "test_data")
                """,
            ),
            (
                "code",
                """
                def load_scanners(subtask, scanners):
                    X_parts, y_parts, scanner_parts = [], [], []
                    folder = DATA / "train_data" / subtask
                    for scanner in scanners:
                        X = np.load(folder / f"scanner{scanner}_X.npy")
                        y = np.load(folder / f"scanner{scanner}_y.npy")
                        X_parts.append(X)
                        y_parts.append(y)
                        scanner_parts.extend([scanner] * len(y))
                    X = torch.tensor(np.concatenate(X_parts), dtype=torch.float32) / 255.0
                    y = torch.tensor(np.concatenate(y_parts), dtype=torch.long)
                    return X, y, np.array(scanner_parts)

                X1, y1, scanners1 = load_scanners("subtask1", range(1, 4))
                X2, y2, scanners2 = load_scanners("subtask2", range(1, 9))
                print(X1.shape, y1.shape, X2.shape, y2.shape)
                """,
            ),
            (
                "code",
                """
                class TinyDigitNet(nn.Module):
                    def __init__(self, in_channels):
                        super().__init__()
                        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
                        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
                        self.pool = nn.AdaptiveAvgPool2d((7, 7))
                        self.fc = nn.Linear(16 * 7 * 7, 10)

                    def forward(self, x):
                        x = F.relu(self.conv1(x))
                        x = F.relu(self.conv2(x))
                        x = self.pool(x)
                        x = torch.flatten(x, 1)
                        return self.fc(x)

                def train_model(X, y, in_channels, epochs=18):
                    dataset = TensorDataset(X, y)
                    n_val = max(20, len(dataset) // 5)
                    n_train = len(dataset) - n_val
                    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
                    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=128)
                    model = TinyDigitNet(in_channels)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    for epoch in range(epochs):
                        model.train()
                        for xb, yb in train_loader:
                            optimizer.zero_grad()
                            loss = F.cross_entropy(model(xb), yb)
                            loss.backward()
                            optimizer.step()
                    model.eval()
                    correct = total = 0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            pred = model(xb).argmax(dim=1)
                            correct += int((pred == yb).sum())
                            total += len(yb)
                    return model, correct / total

                model1, acc1 = train_model(X1, y1, in_channels=1)
                model2, acc2 = train_model(X2, y2, in_channels=3)
                print({"fixture_val_acc_sub1": acc1, "fixture_val_acc_sub2": acc2})
                print({"params_sub1": sum(p.numel() for p in model1.parameters()), "params_sub2": sum(p.numel() for p in model2.parameters())})
                """,
            ),
            (
                "code",
                """
                model1.eval()
                model2.eval()
                scripted1 = torch.jit.trace(model1, torch.zeros(1, 1, 28, 28))
                scripted2 = torch.jit.trace(model2, torch.zeros(1, 3, 28, 28))
                scripted1.save(OUT / "model_sub1.pt")
                scripted2.save(OUT / "model_sub2.pt")

                with zipfile.ZipFile(OUT / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.write(OUT / "model_sub1.pt", "model_sub1.pt")
                    zf.write(OUT / "model_sub2.pt", "model_sub2.pt")

                with zipfile.ZipFile(OUT / "submission.zip") as zf:
                    assert sorted(zf.namelist()) == ["model_sub1.pt", "model_sub2.pt"]
                print("wrote", OUT / "submission.zip")
                """,
            ),
        ],
    )


def build_trace_twins(task_dir: Path) -> None:
    data_dir = task_dir / "data"
    clean_dir(data_dir)
    categories = ["adware", "trojan", "ransom", "worm"]
    base_tokens = {
        "adware": ["open_url", "read_cookie", "spawn_popup", "write_cache", "dns_query"],
        "trojan": ["open_file", "read_file", "inject_proc", "reg_read", "net_send"],
        "ransom": ["scan_dir", "read_file", "encrypt_file", "write_note", "delete_shadow"],
        "worm": ["scan_net", "open_socket", "copy_self", "exec_remote", "dns_query"],
    }
    rows = []
    for program_idx in range(28):
        category = categories[program_idx % len(categories)]
        private = [f"p{program_idx}_a", f"p{program_idx}_b"]
        tokens = []
        period = 5 + (program_idx % 7)
        for pos in range(620):
            if pos % period == 0:
                tokens.append(private[0])
            elif pos % period == 1 and program_idx % 2 == 0:
                tokens.append(private[0])
            elif RNG.random() < 0.12:
                tokens.append(str(RNG.choice(private)))
            else:
                tokens.append(str(RNG.choice(base_tokens[category])))
        rows.append({"program_id": f"program_{program_idx:03d}", "category": category, "tokens": " ".join(tokens)})
    trace_csv = data_dir / "public_traces.csv"
    pd.DataFrame(rows).to_csv(trace_csv, index=False)
    with zipfile.ZipFile(data_dir / "public_traces.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(trace_csv, "public_traces.csv")

    make_notebook(
        task_dir / "solution.ipynb",
        "CEOAI Practice 2 - Trace Twins Minimum Solution",
        [
            (
                "markdown",
                """
                Objective: serialize a valid `Submission` object:

                1. Build local validation windows and pairs from `public_traces.csv`.
                2. Use token overlap for Part A.
                3. Use frequency-shape similarity for Part B, where token names may be scrambled.
                4. Save `submission.pkl` with `score_A` and `score_B`.
                """,
            ),
            (
                "code",
                """
                from pathlib import Path
                from collections import Counter
                import itertools
                import zipfile
                import cloudpickle
                import numpy as np
                import pandas as pd
                from sklearn.metrics import roc_auc_score

                ROOT = Path.cwd()
                DATA = ROOT / "data"
                OUT = ROOT / "outputs"
                OUT.mkdir(exist_ok=True)
                """,
            ),
            (
                "code",
                """
                zip_path = DATA / "public_traces.zip"
                if zip_path.exists() and not (DATA / "public_traces.csv").exists():
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(DATA)

                traces = pd.read_csv(DATA / "public_traces.csv")
                traces.head()
                """,
            ),
            (
                "code",
                """
                def make_windows(tokens, size=200, stride=200):
                    return [tokens[i:i + size] for i in range(0, len(tokens) - size + 1, stride)]

                windows = []
                program_for_window = []
                category_for_window = []
                for _, row in traces.iterrows():
                    toks = row["tokens"].split()
                    for window in make_windows(toks):
                        windows.append(window)
                        program_for_window.append(row["program_id"])
                        category_for_window.append(row["category"])

                by_program = {}
                by_category = {}
                for idx, program in enumerate(program_for_window):
                    by_program.setdefault(program, []).append(idx)
                    by_category.setdefault(category_for_window[idx], []).append(idx)

                pairs = []
                labels = []
                for indexes in by_program.values():
                    for i, j in zip(indexes, indexes[1:]):
                        pairs.append((i, j))
                        labels.append(1)

                target_negatives = len(labels)
                candidate_negatives = []
                for i, j in itertools.combinations(range(len(windows)), 2):
                    if program_for_window[i] == program_for_window[j]:
                        continue
                    if category_for_window[i] == category_for_window[j] or len(candidate_negatives) % 3 == 0:
                        candidate_negatives.append((i, j))
                rng = np.random.default_rng(0)
                chosen = rng.choice(len(candidate_negatives), size=target_negatives, replace=False)
                for idx in chosen:
                    pairs.append(candidate_negatives[int(idx)])
                    labels.append(0)
                print({"windows": len(windows), "pairs": len(pairs), "positives": sum(labels)})
                """,
            ),
            (
                "code",
                """
                def jaccard_score(a, b):
                    sa, sb = set(a), set(b)
                    return len(sa & sb) / max(len(sa | sb), 1)

                def frequency_shape(window, top_k=20):
                    counts = sorted(Counter(window).values(), reverse=True)
                    counts = counts[:top_k] + [0] * max(0, top_k - len(counts))
                    arr = np.array(counts, dtype=float)
                    return arr / max(arr.sum(), 1.0)

                def repeat_signature(window):
                    adjacent_repeats = sum(1 for a, b in zip(window, window[1:]) if a == b) / max(len(window) - 1, 1)
                    counts = Counter(window)
                    top_counts = sorted(counts.values(), reverse=True)[:8]
                    top_counts = top_counts + [0] * (8 - len(top_counts))
                    top_counts = np.array(top_counts, dtype=float) / len(window)
                    return np.concatenate([[adjacent_repeats], top_counts])

                def cosine(a, b):
                    denom = np.linalg.norm(a) * np.linalg.norm(b)
                    return float(np.dot(a, b) / denom) if denom else 0.0

                class Submission:
                    def score_A(self, windows, pairs):
                        return [float(jaccard_score(windows[i], windows[j])) for i, j in pairs]

                    def score_B(self, windows, pairs):
                        shapes = [np.concatenate([frequency_shape(w), repeat_signature(w)]) for w in windows]
                        return [float(cosine(shapes[i], shapes[j])) for i, j in pairs]

                sub = Submission()
                scores_a = sub.score_A(windows, pairs)
                scores_b = sub.score_B(windows, pairs)
                print({
                    "fixture_auc_A": float(roc_auc_score(labels, scores_a)),
                    "fixture_auc_B": float(roc_auc_score(labels, scores_b)),
                })
                """,
            ),
            (
                "code",
                """
                with open(OUT / "submission.pkl", "wb") as f:
                    cloudpickle.dump(sub, f)

                with open(OUT / "submission.pkl", "rb") as f:
                    loaded = cloudpickle.load(f)
                assert len(loaded.score_A(windows[:4], [(0, 1), (2, 3)])) == 2
                assert len(loaded.score_B(windows[:4], [(0, 1), (2, 3)])) == 2
                print("wrote", OUT / "submission.pkl")
                """,
            ),
        ],
    )


def write_root_readme() -> None:
    write_text(
        ROOT / "README.md",
        """
        # CEOAI/EUROAI 2026 Practice Rounds

        Source: Nitro Judge

        - Round 1 URL: https://judge.nitro-ai.org/competitions/ceoai/ceoai-2026-practice-1
        - Round 2 URL: https://judge.nitro-ai.org/competitions/ceoai/ceoai-2026-practice-2

        This folder is organized as one folder per task. Each task folder contains:

        - `prompt.md`: local copy of the task statement summary.
        - `solution.ipynb`: a runnable minimum-solution notebook with starter code.
        - `data/`: deterministic fixture data matching the task contract.
        - `DATASET_NOTE.md`: reminder that fixture data is not official judge data.
        - `outputs/`: created when the notebook is executed.

        The official Nitro attachments require a logged-in Nitro session. The local
        fixture data is for practice, debugging, and output-contract fluency. Before
        judge submission, replace the fixture `data/` folder with the official task
        files from Nitro and rerun the notebook.

        ## Tasks

        Round 1:

        - `round-1/stochastic_rift/`
        - `round-1/project_kraken/`
        - `round-1/star_observatory/`

        Round 2:

        - `round-2/panda_mnist/`
        - `round-2/trace_twins/`
        """,
    )


def main() -> None:
    for round_name, tasks in TASKS.items():
        for task_name, source_md in tasks.items():
            copy_prompt(round_name, task_name, source_md)

    build_stochastic_rift(ROOT / "round-1" / "stochastic_rift")
    write_dataset_note(
        ROOT / "round-1" / "stochastic_rift",
        "https://judge.nitro-ai.org/competitions/ceoai/ceoai-2026-practice-1/1/view",
    )

    build_project_kraken(ROOT / "round-1" / "project_kraken")
    write_dataset_note(
        ROOT / "round-1" / "project_kraken",
        "https://judge.nitro-ai.org/competitions/ceoai/ceoai-2026-practice-1/2/view",
    )

    build_star_observatory(ROOT / "round-1" / "star_observatory")
    write_dataset_note(
        ROOT / "round-1" / "star_observatory",
        "https://judge.nitro-ai.org/competitions/ceoai/ceoai-2026-practice-1/3/view",
    )

    build_panda_mnist(ROOT / "round-2" / "panda_mnist")
    write_dataset_note(
        ROOT / "round-2" / "panda_mnist",
        "https://judge.nitro-ai.org/competitions/ceoai/ceoai-2026-practice-2/1/view",
    )

    build_trace_twins(ROOT / "round-2" / "trace_twins")
    write_dataset_note(
        ROOT / "round-2" / "trace_twins",
        "https://judge.nitro-ai.org/competitions/ceoai/ceoai-2026-practice-2/2/view",
    )

    write_root_readme()


if __name__ == "__main__":
    main()
