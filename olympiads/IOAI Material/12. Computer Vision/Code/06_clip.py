"""
06_clip.py
==========
VISION-LANGUAGE MODELS with CLIP — now a whole MODEL ZOO, with METRICS.

CLIP-style models are trained to put images and their text descriptions into
the SAME embedding space. That lets us compare a picture directly with a
sentence and ask "how well do they match?".

The magic result: ZERO-SHOT CLASSIFICATION. We classify an image into ANY
categories we invent on the spot — just by writing them as text — even
categories the model was never explicitly trained to output.

Like the detection (04) and segmentation (05) scripts, we don't stop at one
model. We load EVERY available vision-language model and compare them two ways:
    PART A  a live zero-shot demo on one image with labels WE choose
    PART B  REAL zero-shot accuracy (top-1 / top-5) on a labelled dataset
            (CIFAR-100) — the standard way CLIP-style models are benchmarked
Any model whose weights are missing — or the dataset if it can't download —
is skipped with a note, so the script still runs with whatever you have.

Run it:
    python 06_clip.py

------------------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------------------
* IMAGE-TEXT EMBEDDINGS:
    Each model has TWO encoders (image + text) whose vectors live in one
    shared space, so a "dog photo" vector lands near the "a photo of a dog"
    text vector. We L2-normalise both, then a dot product = similarity.

* ZERO-SHOT LEARNING:
    Classifying into categories with ZERO training examples for them.
    Describe the classes in words; change the words -> change the
    classifier, instantly, with no retraining.

* HOW WE SCORE IT (Part B):
    For each image, embed it once, embed all class prompts once, take the
    dot products, and the highest-scoring class is the prediction. Compare
    to the true label over many images -> top-1 / top-5 accuracy.
"""

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")          # cached labelled datasets
RESULTS_DIR = os.path.join(HERE, "results", "clip")   # saved figures
EVAL_N = 128               # how many labelled images to score for accuracy
DEMO_SCALE = 100.0         # sharpen the demo softmax (CLIP's own logit scale ~100)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_sample_image() -> str:
    images_dir = os.path.join(HERE, "images")
    for name in ("cats.jpg", "dog.jpg", "street.jpg"):
        path = os.path.join(images_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No image found. Please run: python download.py")


# ===========================================================================
# THE VISION-LANGUAGE MODEL REGISTRY
# ---------------------------------------------------------------------------
# Every model exposes the SAME tiny interface after loading:
#     embed_image(image)   -> (1, D) L2-normalised image vector
#     embed_texts(prompts) -> (C, D) L2-normalised text vectors
#     n_params             -> parameter count
# Zero-shot scoring is then just  image_vec @ text_vecs.T  for every model,
# so the rest of the script never cares whether it's CLIP, OpenCLIP, or SigLIP.
# ===========================================================================
@dataclass
class VLModel:
    name: str
    family: str
    build: Callable                   # build(device) -> (embed_image, embed_texts, n_params)
    embed_image: Optional[Callable] = None
    embed_texts: Optional[Callable] = None
    n_params: int = 0

    def load(self, device: str) -> None:
        self.embed_image, self.embed_texts, self.n_params = self.build(device)


def _feat(out) -> torch.Tensor:
    """Pull the embedding tensor out of get_*_features().

    transformers <5 returned a plain tensor; transformers >=5 returns a
    ModelOutput whose projected embedding sits in `.pooler_output`.
    """
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    raise TypeError(f"Unexpected feature output: {type(out).__name__}")


def _normalize(out) -> torch.Tensor:
    return torch.nn.functional.normalize(_feat(out), dim=-1)


# ---- CLIP-architecture models (OpenAI CLIP, OpenCLIP/LAION, MetaCLIP) ------
# All three load through CLIPModel + CLIPProcessor and share the exact same
# get_image_features / get_text_features interface.
def _build_clip(model_name: str):
    def builder(device: str):
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        proc = CLIPProcessor.from_pretrained(model_name)

        def embed_image(image):
            ins = proc(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                return _normalize(model.get_image_features(**ins))

        def embed_texts(prompts):
            ins = proc(text=prompts, return_tensors="pt", padding=True,
                       truncation=True).to(device)
            with torch.no_grad():
                return _normalize(model.get_text_features(**ins))

        n = sum(p.numel() for p in model.parameters())
        return embed_image, embed_texts, n

    return builder


# ---- SigLIP (Google) — same idea, sigmoid-trained, uses AutoProcessor ------
# SigLIP's text tower needs fixed-length padding ("max_length"); otherwise the
# interface is identical.
def _build_siglip(model_name: str):
    def builder(device: str):
        from transformers import SiglipModel, AutoProcessor
        model = SiglipModel.from_pretrained(model_name).to(device).eval()
        proc = AutoProcessor.from_pretrained(model_name)

        def embed_image(image):
            ins = proc(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                return _normalize(model.get_image_features(**ins))

        def embed_texts(prompts):
            ins = proc(text=prompts, return_tensors="pt", padding="max_length",
                       truncation=True).to(device)
            with torch.no_grad():
                return _normalize(model.get_text_features(**ins))

        n = sum(p.numel() for p in model.parameters())
        return embed_image, embed_texts, n

    return builder


# The models we TRY to load, roughly smallest -> largest download. Anything
# whose weights/library are missing is skipped (see load_available).
REGISTRY = [
    VLModel("CLIP-B/32",    "OpenAI CLIP",
            _build_clip("openai/clip-vit-base-patch32")),
    VLModel("CLIP-B/16",    "OpenAI CLIP",
            _build_clip("openai/clip-vit-base-patch16")),
    VLModel("MetaCLIP-B/32","MetaCLIP",
            _build_clip("facebook/metaclip-b32-400m")),
    VLModel("LAION-B/32",   "OpenCLIP / LAION-2B",
            _build_clip("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")),
    VLModel("SigLIP-B/16",  "Google SigLIP",
            _build_siglip("google/siglip-base-patch16-224")),
    VLModel("CLIP-L/14",    "OpenAI CLIP",
            _build_clip("openai/clip-vit-large-patch14")),
]


def load_available(device: str) -> list:
    """Build every registered model, skipping any that fail to load."""
    ready = []
    for vl in REGISTRY:
        try:
            vl.load(device)
            ready.append(vl)
            print(f"    [ok]   {vl.name:<14} ({vl.family}, "
                  f"{vl.n_params/1e6:.0f}M params)")
        except Exception as e:
            short = str(e).splitlines()[0][:70]
            print(f"    [skip] {vl.name:<14} — {short}")
    return ready


def zero_shot_probs(vl: VLModel, image, labels: list) -> np.ndarray:
    """Softmax probabilities over a small set of custom labels (the demo)."""
    prompts = [f"a photo of {label}" for label in labels]
    img = vl.embed_image(image)                    # (1, D)
    txt = vl.embed_texts(prompts)                  # (C, D)
    sims = (img @ txt.T)[0] * DEMO_SCALE           # (C,)
    return sims.softmax(dim=0).cpu().numpy()


# ===========================================================================
# PART B — REAL zero-shot accuracy (top-1 / top-5) on a labelled dataset
# ===========================================================================
def load_cifar100_slice(n: int):
    """`n` CIFAR-100 test images as (RGB image, label int) + the class names."""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"    [skip] `datasets` not installed ({e}). Accuracy needs labels.")
        return None, None
    # HF now requires a namespaced repo id ("cifar100" alone is rejected); the
    # canonical mirror is uoft-cs/cifar100. Try a couple of ids to be safe.
    ds = None
    for repo in ("uoft-cs/cifar100", "cifar100"):
        try:
            ds = load_dataset(repo, split=f"test[:{n}]", cache_dir=DATA_DIR)
            break
        except Exception as e:
            last = str(e).splitlines()[0]
    if ds is None:
        print(f"    [skip] couldn't load CIFAR-100 ({last}).")
        return None, None

    class_names = ds.features["fine_label"].names          # 100 human names
    samples = [(ex["img"].convert("RGB"), int(ex["fine_label"])) for ex in ds]
    return samples, class_names


def evaluate_zeroshot(models: list, samples: list, class_names: list) -> list:
    """Score each model's top-1 / top-5 zero-shot accuracy on the slice."""
    print(f"\n[B] Zero-shot accuracy on {len(samples)} CIFAR-100 test images "
          f"({len(class_names)} classes)")
    print("    (no training on CIFAR — we just describe the 100 classes in")
    print("     words; top-1 = right answer is #1, top-5 = in the top five)\n")

    # A prompt per class, built once. "a photo of a ___" is the classic CLIP
    # template and reliably beats bare class names.
    # Underscores in dataset names ("aquarium_fish") read poorly to CLIP; the
    # "a photo of a ___" template is the classic zero-shot prompt.
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
    k = min(5, len(class_names))                           # top-5, or fewer classes

    rows = []
    for vl in models:
        text_feats = vl.embed_texts(prompts)               # (C, D), computed once
        top1 = top5 = 0
        t0 = time.perf_counter()
        for image, label in samples:
            sims = (vl.embed_image(image) @ text_feats.T)[0]
            top = sims.topk(k).indices.cpu().tolist()
            top1 += int(top[0] == label)
            top5 += int(label in top)
        ms = 1000.0 * (time.perf_counter() - t0) / len(samples)

        acc1, acc5 = top1 / len(samples), top5 / len(samples)
        rows.append({"name": vl.name, "family": vl.family,
                     "params_m": vl.n_params / 1e6,
                     "top1": acc1, "top5": acc5, "ms": ms})
        print(f"    {vl.name:<14} top-1={acc1*100:5.1f}   top-5={acc5*100:5.1f}"
              f"   ({ms:.0f} ms/img)")

    rows.sort(key=lambda r: r["top1"], reverse=True)
    return rows


def plot_metrics(rows: list, out_path: str) -> None:
    """Bar charts: top-1 accuracy and speed for every scored model."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"    [warn] plotting unavailable ({e}); skipping metrics plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)

    # Left: accuracy (higher is better).
    ax = axes[0][0]
    m = sorted(rows, key=lambda r: r["top1"])
    ax.barh([r["name"] for r in m], [r["top1"] * 100 for r in m], color="#55A868")
    ax.set_xlabel("CIFAR-100 top-1 accuracy (%)  — higher is better")
    ax.set_title(f"Zero-shot accuracy on {EVAL_N} images")
    for i, r in enumerate(m):
        ax.text(r["top1"] * 100, i, f"  {r['top1']*100:.1f}", va="center", fontsize=8)

    # Right: speed (lower is better).
    ax = axes[0][1]
    s = sorted(rows, key=lambda r: r["ms"], reverse=True)
    ax.barh([r["name"] for r in s], [r["ms"] for r in s], color="#4C72B0")
    ax.set_xlabel("inference time (ms / image)  — lower is faster")
    ax.set_title("Speed")
    for i, r in enumerate(s):
        ax.text(r["ms"], i, f"  {r['ms']:.0f}ms", va="center", fontsize=8)

    fig.suptitle("CLIP-family comparison: zero-shot accuracy / speed trade-off",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved metrics plot -> {out_path}")


def main() -> None:
    print("=" * 60)
    print(" 06 - CLIP: Vision-Language Zero-Shot — MODEL ZOO + METRICS")
    print("=" * 60)

    device = get_device()
    print(f"[info] Using device: {device}")

    # -------------------------------------------------------------------
    # STEP 1: Load every available vision-language model.
    # First run downloads weights (CLIP-B/32 ~600MB; CLIP-L/14 ~1.7GB).
    # -------------------------------------------------------------------
    print("[1] Loading the vision-language model zoo...")
    models = load_available(device)
    if not models:
        raise RuntimeError("No vision-language models could be loaded.")
    print(f"    Loaded {len(models)} model(s).")

    # -------------------------------------------------------------------
    # STEP 2 (PART A): live zero-shot demo — OUR labels, every model.
    # Edit `labels` and re-run to experiment!
    # -------------------------------------------------------------------
    path = find_sample_image()
    print(f"[2] Loading image: {path}")
    image = Image.open(path).convert("RGB")

    labels = ["a dog", "a cat", "a car"]
    print(f"[3] Candidate labels: {labels}")
    print("\n[A] Each model's zero-shot probabilities on this image:\n")
    header = "    " + "model".ljust(14) + "".join(f"{l:>10}" for l in labels) + "   verdict"
    print(header)
    print("    " + "-" * (14 + 10 * len(labels) + 12))
    for vl in models:
        probs = zero_shot_probs(vl, image, labels)
        best = labels[int(np.argmax(probs))]
        cells = "".join(f"{p*100:9.1f}%" for p in probs)
        print(f"    {vl.name:<14}{cells}   {best}")

    # -------------------------------------------------------------------
    # STEP 4 (PART B): REAL zero-shot accuracy on a labelled dataset.
    # -------------------------------------------------------------------
    print(f"\n[4] Measuring REAL zero-shot accuracy on CIFAR-100 "
          f"(first run downloads ~170MB)...")
    samples, class_names = load_cifar100_slice(EVAL_N)
    metric_rows = []
    if samples:
        metric_rows = evaluate_zeroshot(models, samples, class_names)
        print(f"\n    {'model':<14}{'family':<22}{'params':>8}"
              f"{'top-1':>8}{'top-5':>8}{'ms/img':>9}")
        print("    " + "-" * 69)
        for r in metric_rows:
            print(f"    {r['name']:<14}{r['family']:<22}{r['params_m']:>6.0f}M"
                  f"{r['top1']*100:>7.1f}{r['top5']*100:>8.1f}{r['ms']:>9.0f}")
        print("    (top-1/top-5 are % on unseen CIFAR-100 — zero training, "
              "just text prompts.)")
    else:
        print("    (accuracy skipped — install `datasets` to enable it; the "
              "zero-shot demo above still stands.)")

    # -------------------------------------------------------------------
    # STEP 5: Save the accuracy / speed comparison chart.
    # -------------------------------------------------------------------
    if metric_rows:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print("\n[5] Saving the accuracy / speed comparison chart...")
        plot_metrics(metric_rows, os.path.join(RESULTS_DIR, "clip_metrics.png"))

    print("\nDone. We classified WITHOUT training on these labels — that's")
    print("zero-shot learning. Edit `labels` above and run again to experiment!")


if __name__ == "__main__":
    main()
