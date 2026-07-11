"""
02_image_classification.py
==========================
IMAGE CLASSIFICATION with a pretrained deep-learning model.

"Classification" = "Look at this whole image and tell me what it is."
The model outputs ONE label for the ENTIRE picture (e.g. "Egyptian cat").

We use a pretrained Vision Transformer (ViT) from Hugging Face. It was
already trained on ImageNet (1000 everyday categories), so we don't train
anything - we just USE it. This works fine on a CPU.

This script has THREE parts:
    A) Classify a single sample image (the classic demo).
    B) EVALUATE the model on a real dataset and MEASURE how good it is
       (accuracy, top-5 accuracy, per-class accuracy, confusion matrix).
    C) A reality check: run the SAME model on an out-of-domain dataset
       (bean-leaf diseases) to SEE why a pretrained ImageNet model can't
       classify things it never learned - motivating fine-tuning (08) & CLIP (06).

All plots are saved as PNGs under  results/classifications/.

Run it:
    python 02_image_classification.py

------------------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------------------
* CLASSIFICATION vs DETECTION:
    - Classification: one label for the whole image ("this is a cat").
    - Detection (script 04): WHERE objects are, with boxes.

* METRICS (part B) - a single demo image tells you nothing about how GOOD
  a model is. To know that, you run it on MANY labelled images and count:
    - Top-1 accuracy : how often the #1 guess is exactly right.
    - Top-5 accuracy : how often the right answer is in the top 5 guesses.
    - Per-class accuracy + confusion matrix : WHERE it gets confused.

* IN-DOMAIN vs OUT-OF-DOMAIN (part C):
    A model can only predict classes it was trained on. Imagenette's classes
    ARE ImageNet classes, so the model scores well. Bean diseases are NOT,
    so the model is helpless - that gap is exactly what fine-tuning fixes.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification
from model_summary import print_summary   # prints layers + params of each model

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results", "classifications")

# How many images to evaluate / show. Small keeps it fast on a CPU.
EVAL_N = 150      # images used to measure accuracy in part B
GALLERY_N = 8     # images shown in the out-of-domain gallery in part C
BATCH_SIZE = 16   # images fed to the model at once


# ---------------------------------------------------------------------------
# Dataset registry. Each entry describes ONE dataset in data/.
#   aligned=True  -> its labels ARE ImageNet classes, so we can score accuracy.
#                    'imagenet_index' maps dataset-label i -> ImageNet class id.
#   aligned=False -> classes the ImageNet model never saw (out-of-domain demo).
# ---------------------------------------------------------------------------
DATASETS = {
    "imagenette": {
        "repo": "johnowhitaker/imagenette2-320",
        "split": "train",
        "label_key": "label",
        "aligned": True,
        # Imagenette's 10 classes, and the ImageNet class id each one maps to.
        "imagenet_index": [0, 217, 482, 491, 497, 566, 569, 571, 574, 701],
        "pretty": ["tench", "English springer", "cassette", "chain saw",
                   "church", "French horn", "garbage truck", "gas pump",
                   "golf ball", "parachute"],
    },
    "beans": {
        "repo": "AI-Lab-Makerere/beans",
        "split": "train",
        "label_key": "labels",
        "aligned": False,   # bean-leaf diseases are NOT ImageNet classes
        "pretty": ["angular_leaf_spot", "bean_rust", "healthy"],
    },
    "flowers": {
        "repo": "nelorth/oxford-flowers",
        "split": "train",
        "label_key": "label",
        "aligned": False,   # 102 flower species, not ImageNet classes
        "pretty": None,     # filled in from the dataset's own class names
    },
}


def class_names(spec: dict, ds) -> list:
    """Human-readable class names: use `pretty` if given, else the dataset's."""
    if spec["pretty"] is not None:
        return spec["pretty"]
    names = ds.features[spec["label_key"]].names
    return [f"flower-{n}" for n in names]  # flowers repo names are bare numbers


def get_device() -> str:
    """Use a GPU if available, otherwise fall back to CPU (always works)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset_shuffled(spec: dict, n: int):
    """
    Load `n` images from a dataset in data/, SHUFFLED so we get a mix of all
    classes (the raw files are grouped by class, which would bias metrics).
    Returns a small in-memory list of examples.
    """
    from datasets import load_dataset
    ds = load_dataset(spec["repo"], split=spec["split"], cache_dir=DATA_DIR)
    ds = ds.shuffle(seed=0).select(range(min(n, len(ds))))
    return ds


@torch.no_grad()
def classify(model, processor, pil_images, device) -> np.ndarray:
    """
    Run the model on a list of PIL images and return a (N, 1000) array of
    probabilities (one row per image, one column per ImageNet class).
    We process in BATCHES so we don't run out of memory on big inputs.
    """
    all_probs = []
    for start in range(0, len(pil_images), BATCH_SIZE):
        batch = pil_images[start:start + BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        logits = model(**inputs).logits            # (B, 1000) raw scores
        probs = torch.softmax(logits, dim=-1)      # -> probabilities
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ===========================================================================
# PART A — classify a single image (the classic demo)
# ===========================================================================
def demo_single_image(model, processor, device) -> None:
    from PIL import Image
    print("\n[A] Single-image demo")
    sample = None
    for name in ("cats.jpg", "dog.jpg", "street.jpg"):
        p = os.path.join(HERE, "images", name)
        if os.path.exists(p):
            sample = p
            break
    if sample is None:
        print("    (no sample image found - run download.py; skipping)")
        return

    image = Image.open(sample).convert("RGB")
    probs = classify(model, processor, [image], device)[0]  # (1000,)
    top5 = np.argsort(-probs)[:5]
    print(f"    Image: {os.path.basename(sample)}   Top-5 predictions:")
    for rank, idx in enumerate(top5, start=1):
        label = model.config.id2label[idx]
        bar = "#" * int(probs[idx] * 40)
        print(f"      {rank}. {label:<28} {probs[idx]*100:5.1f}%  {bar}")


# ===========================================================================
# PART B — evaluate on an ImageNet-aligned dataset and MEASURE quality
# ===========================================================================
def evaluate_aligned(model, processor, device, spec: dict) -> dict:
    """Run the model over EVAL_N labelled images and compute metrics."""
    print(f"\n[B] Evaluating on '{spec['repo']}' ({EVAL_N} images)")
    ds = load_dataset_shuffled(spec, EVAL_N)
    images = [ex["image"].convert("RGB") for ex in ds]
    y_true = np.array(ds[spec["label_key"]])            # 0..9 (dataset labels)
    imagenet_index = np.array(spec["imagenet_index"])   # 0..9 -> ImageNet id
    true_imagenet = imagenet_index[y_true]              # the "correct" ImageNet id

    probs = classify(model, processor, images, device)  # (N, 1000)

    # --- Full 1000-way metrics: how hard the real ImageNet task is ----------
    pred_top1 = probs.argmax(axis=1)
    top1_acc = float((pred_top1 == true_imagenet).mean())
    top5_idx = np.argsort(-probs, axis=1)[:, :5]
    top5_acc = float(np.mean([t in row for t, row in zip(true_imagenet, top5_idx)]))

    # --- 10-way metrics: restrict scores to Imagenette's 10 columns ---------
    # This is the standard Imagenette setup and gives a clean 10x10 confusion.
    restricted = probs[:, imagenet_index]               # (N, 10)
    pred10 = restricted.argmax(axis=1)                  # 0..9
    acc10 = float((pred10 == y_true).mean())

    n_classes = len(spec["pretty"])
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, pred10):
        confusion[t, p] += 1
    # --- Per-class precision / recall / F1 (all read off the confusion matrix)
    #   recall[c]    = of all TRUE c images, how many did we get right?
    #                = diagonal / row-sum   (this equals per-class accuracy)
    #   precision[c] = of all images we PREDICTED as c, how many were right?
    #                = diagonal / column-sum
    #   F1[c]        = harmonic mean of precision & recall (one balanced score)
    diag = np.diag(confusion).astype(float)
    row_totals = confusion.sum(axis=1)      # how many were truly each class
    col_totals = confusion.sum(axis=0)      # how many we predicted each class
    recall = np.divide(diag, row_totals, out=np.zeros(n_classes), where=row_totals > 0)
    precision = np.divide(diag, col_totals, out=np.zeros(n_classes), where=col_totals > 0)
    pr_sum = precision + recall
    f1 = np.divide(2 * precision * recall, pr_sum,
                   out=np.zeros(n_classes), where=pr_sum > 0)
    per_class = recall                      # recall IS per-class accuracy
    macro_acc = float(recall.mean())

    print(f"    Top-1 accuracy (1000-way ImageNet) : {top1_acc*100:5.1f}%")
    print(f"    Top-5 accuracy (1000-way ImageNet) : {top5_acc*100:5.1f}%")
    print(f"    Accuracy (10-way, restricted)      : {acc10*100:5.1f}%")
    print(f"    Macro-averaged  P/R/F1 (10-way)    : "
          f"{precision.mean()*100:.1f}% / {recall.mean()*100:.1f}% / {f1.mean()*100:.1f}%")

    # A compact per-class table (sorted worst-F1 first, so problems surface).
    pretty = class_names(spec, ds)
    print(f"\n    {'class':<18}{'precision':>10}{'recall':>9}{'f1':>7}{'n':>6}")
    print(f"    {'-'*49}")
    for c in np.argsort(f1):
        print(f"    {pretty[c]:<18}{precision[c]*100:9.1f}%{recall[c]*100:8.1f}%"
              f"{f1[c]*100:6.1f}%{int(row_totals[c]):6d}")

    return {
        "confusion": confusion, "per_class": per_class, "pretty": pretty,
        "precision": precision, "recall": recall, "f1": f1,
        "top1": top1_acc, "top5": top5_acc, "acc10": acc10, "macro": macro_acc,
    }


def plot_metrics(m: dict, out_path: str) -> None:
    """Save a confusion matrix + per-class accuracy figure as a PNG."""
    pretty, cm, per_class = m["pretty"], m["confusion"], m["per_class"]
    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(
        f"Pretrained ViT on Imagenette   |   "
        f"top-1 {m['top1']*100:.0f}%   top-5 {m['top5']*100:.0f}%   "
        f"10-way {m['acc10']*100:.0f}%",
        fontsize=13,
    )

    # Left: confusion matrix. Rows = true class, columns = predicted class.
    # A perfect model has all counts on the diagonal (top-left to bottom-right).
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_title("Confusion matrix (rows=true, cols=predicted)")
    ax_cm.set_xticks(range(len(pretty)))
    ax_cm.set_yticks(range(len(pretty)))
    ax_cm.set_xticklabels(pretty, rotation=45, ha="right", fontsize=8)
    ax_cm.set_yticklabels(pretty, fontsize=8)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax_cm.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                           color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # Right: per-class accuracy bars (which classes the model nails vs fumbles).
    order = np.argsort(per_class)
    ax_bar.barh([pretty[i] for i in order], [per_class[i] * 100 for i in order],
                color="#4C72B0")
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("accuracy (%)")
    ax_bar.set_title("Per-class accuracy")
    for i, idx in enumerate(order):
        ax_bar.text(per_class[idx] * 100 + 1, i, f"{per_class[idx]*100:.0f}%",
                    va="center", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved metrics plot -> {out_path}")


# ===========================================================================
# PART C — out-of-domain reality check (why we ever need to fine-tune)
# ===========================================================================
def reality_check(model, processor, device, specs: list, out_path: str) -> None:
    """Run the ImageNet model on SEVERAL out-of-domain datasets at once and
    build a single gallery, so the 'it can't classify these' point is obvious."""
    repos = ", ".join(s["repo"].split("/")[-1] for s in specs)
    print(f"\n[C] Out-of-domain check on: {repos} (none are ImageNet classes)")

    per_dataset = max(1, GALLERY_N // len(specs))
    samples = []   # each item: (PIL image, true class name, dataset short name)
    for spec in specs:
        ds = load_dataset_shuffled(spec, per_dataset)
        names = class_names(spec, ds)
        short = spec["repo"].split("/")[-1]
        for ex in ds:
            samples.append((ex["image"].convert("RGB"),
                            names[ex[spec["label_key"]]], short))

    images = [s[0] for s in samples]
    probs = classify(model, processor, images, device)      # one pass for all
    pred_top1 = probs.argmax(axis=1)

    print("    The ImageNet model has no matching classes, so it must guess the")
    print("    visually-closest ImageNet label - usually unrelated:")
    for i, (_, truth, short) in enumerate(samples):
        guess = model.config.id2label[pred_top1[i]]
        conf = probs[i, pred_top1[i]] * 100
        print(f"      [{short:<11}] true={truth:<18} -> '{guess}' ({conf:.0f}%)")

    # Save one combined gallery so the mismatch is visible at a glance.
    cols = 4
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.4 * rows))
    fig.suptitle("ImageNet model on out-of-domain images "
                 "(predictions are wrong by design)", fontsize=13)
    for ax, (img, truth, short), i in zip(np.ravel(axes), samples, range(len(samples))):
        ax.imshow(img)
        ax.axis("off")
        guess = model.config.id2label[pred_top1[i]].split(",")[0]
        ax.set_title(f"[{short}]\npred: {guess}\ntrue: {truth}", fontsize=8)
    for ax in np.ravel(axes)[len(images):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.0)  # h_pad: keep 3-line titles clear
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved gallery -> {out_path}")
    print("    Lesson: to classify THESE, you must fine-tune (08) or use CLIP (06).")


def main() -> None:
    print("=" * 60)
    print(" 02 - Image Classification (pretrained ViT + metrics)")
    print("=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = get_device()
    print(f"[info] Using device: {device}")

    # Load the pretrained model + its matching preprocessor once, reuse everywhere.
    model_name = "google/vit-base-patch16-224"
    print(f"[info] Loading pretrained model: {model_name}")
    print("       (first run downloads ~350MB - be patient)")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    model.eval()
    print_summary(model, "ViT image classifier")

    # A) quick single-image demo
    demo_single_image(model, processor, device)

    # B) measure real accuracy on an ImageNet-aligned dataset
    try:
        metrics = evaluate_aligned(model, processor, device, DATASETS["imagenette"])
        plot_metrics(metrics, os.path.join(RESULTS_DIR, "imagenette_metrics.png"))
    except Exception as e:
        print(f"    [warn] evaluation skipped ({e}). Run download.py first.")

    # C) out-of-domain reality check (beans AND flowers in one gallery)
    try:
        reality_check(model, processor, device,
                      [DATASETS["beans"], DATASETS["flowers"]],
                      os.path.join(RESULTS_DIR, "out_of_domain.png"))
    except Exception as e:
        print(f"    [warn] out-of-domain check skipped ({e}).")

    print("\nDone. See results/classifications/ for the saved plots.")
    print("Next: 04_object_detection.py finds *where* multiple objects are.")


if __name__ == "__main__":
    main()
