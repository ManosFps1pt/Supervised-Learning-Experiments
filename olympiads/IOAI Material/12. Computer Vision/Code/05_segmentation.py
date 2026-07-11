"""
05_segmentation.py
==================
SEMANTIC SEGMENTATION — now with a whole MODEL ZOO.

Segmentation is the most detailed vision task: it classifies EVERY SINGLE
PIXEL. Instead of one label for the image (classification) or boxes around
objects (detection), we colour each pixel by what it belongs to
("this pixel is road, that pixel is person, those pixels are sky").

Just like the detection script (04), we don't stop at one model — we load
EVERY available segmentation model, run them all on the same image, and
compare them two ways:
    PART A  speed, size, and the classes each finds (always runs)
    PART B  REAL accuracy — mIoU + pixel accuracy on labelled validation
            images (ADE20K for the transformers, Pascal VOC for the CNNs)
Any model whose library/weights are missing — or any dataset we can't
download — is simply skipped with a note, so the script still runs with
whatever subset you have installed.

Run it:
    python 05_segmentation.py

------------------------------------------------------------------------
THE THREE TASKS, FROM COARSE TO FINE
------------------------------------------------------------------------
    CLASSIFICATION : one label for the whole image        -> "a street"
    DETECTION      : boxes around objects                 -> "car here"
    SEGMENTATION   : a label for EVERY pixel              -> pixel-perfect
                                                             object shapes

* PIXEL-LEVEL PREDICTION:
    Each model outputs a grid the same size as the image, where each cell
    holds the predicted class of that pixel. This grid is the "mask".

* Semantic vs Instance segmentation (good to know):
    - Semantic (this script): all cars share one "car" colour.
    - Instance: each car gets its OWN colour (car #1, car #2, ...).

------------------------------------------------------------------------
THE MODEL ZOO (two families, two datasets)
------------------------------------------------------------------------
    Transformer family, trained on ADE20K (150 scene classes):
        SegFormer-b0 / -b2 . NVIDIA's efficient transformer (b0 is tiny!)
        BEiT ............... masked-image-pretrained ViT backbone
        DPT ................ dense prediction transformer (large)
        UperNet-ConvNeXt ... ConvNeXt backbone + UperNet head
        Mask2Former ........ mask-classification transformer (SOTA-style)

    CNN family, trained on Pascal VOC (21 everyday classes):
        DeepLabV3-R50 ...... atrous convolutions, classic strong baseline
        FCN-R50 ............ the original fully-convolutional net
        LR-ASPP-MNv3 ....... lightweight MobileNetV3 (fast on CPU)

    NOTE: ADE20K and VOC use DIFFERENT label sets, so each model paints
    with its OWN palette and reports its OWN class names. That's expected —
    it's part of seeing how model + training data shape the output.
"""

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")          # cached validation datasets
RESULTS_DIR = os.path.join(HERE, "results", "segmentation")   # saved figures
MASK_ALPHA = 0.45          # overlay blend: how strongly the mask tints the photo
TIMING_PASSES = 3          # average inference time over this many passes
EVAL_N = 8                 # how many labelled val images to score for accuracy


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_sample_image() -> str:
    images_dir = os.path.join(HERE, "images")
    for name in ("street.jpg", "cats.jpg", "dog.jpg"):
        path = os.path.join(images_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No image found. Please run: python download.py")


# ===========================================================================
# THE SEGMENTER REGISTRY
# ---------------------------------------------------------------------------
# Every model exposes the SAME tiny interface after loading:
#     predict(image) -> mask   (H, W) ndarray of class ids at full resolution
#     id2label       -> {class_id: "name"} for colouring + reporting
#     n_params       -> parameter count (for the size column)
# A "build(device)" function does the model-specific loading and returns those
# three things, so the rest of the script never has to care which family a
# model belongs to.
# ===========================================================================
@dataclass
class Segmenter:
    name: str
    family: str                       # e.g. "Transformer / ADE20K"
    build: Callable                   # build(device) -> (predict, n_params, id2label)
    predict: Optional[Callable] = None
    id2label: Optional[dict] = None
    n_params: int = 0

    def load(self, device: str) -> None:
        self.predict, self.n_params, self.id2label = self.build(device)


# ---- Transformer family: HF models that output class LOGITS ---------------
# SegFormer, BEiT, DPT and UperNet all share the "logits (b, C, h, w) then
# upsample + argmax" recipe, so one factory covers all of them.
def _build_hf_logits(model_name: str):
    def builder(device: str):
        from transformers import (
            AutoImageProcessor,
            AutoModelForSemanticSegmentation,
        )
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = (
            AutoModelForSemanticSegmentation.from_pretrained(model_name)
            .to(device)
            .eval()
        )
        id2label = dict(model.config.id2label)

        def predict(image):
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits          # (1, C, h/4, w/4)
            upsampled = torch.nn.functional.interpolate(
                logits, size=image.size[::-1], mode="bilinear", align_corners=False
            )
            return upsampled.argmax(dim=1)[0].cpu().numpy()   # (H, W)

        n = sum(p.numel() for p in model.parameters())
        return predict, n, id2label

    return builder


# ---- Transformer family: Mask2Former (mask-classification, not logits) ----
# Mask2Former predicts a SET of masks + labels, so it needs the processor's
# dedicated post-processor to collapse them into one per-pixel map.
def _build_mask2former(model_name: str):
    def builder(device: str):
        from transformers import (
            AutoImageProcessor,
            Mask2FormerForUniversalSegmentation,
        )
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
            .to(device)
            .eval()
        )
        id2label = dict(model.config.id2label)

        def predict(image):
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            seg = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]
            return seg.cpu().numpy()                          # (H, W)

        n = sum(p.numel() for p in model.parameters())
        return predict, n, id2label

    return builder


# ---- CNN family: torchvision segmentation models (VOC, 21 classes) --------
def _build_torchvision_seg(kind: str):
    def builder(device: str):
        from torchvision.models.segmentation import (
            deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
            fcn_resnet50, FCN_ResNet50_Weights,
            lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights,
        )
        table = {
            "deeplabv3": (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights.DEFAULT),
            "fcn":       (fcn_resnet50, FCN_ResNet50_Weights.DEFAULT),
            "lraspp":    (lraspp_mobilenet_v3_large,
                          LRASPP_MobileNet_V3_Large_Weights.DEFAULT),
        }
        ctor, weights = table[kind]
        model = ctor(weights=weights).to(device).eval()
        transform = weights.transforms()                     # resize + normalise
        categories = weights.meta["categories"]              # 21 VOC names
        id2label = {i: c for i, c in enumerate(categories)}

        def predict(image):
            x = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x)["out"]                         # (1, 21, h, w)
            upsampled = torch.nn.functional.interpolate(
                out, size=image.size[::-1], mode="bilinear", align_corners=False
            )
            return upsampled.argmax(dim=1)[0].cpu().numpy()   # (H, W)

        n = sum(p.numel() for p in model.parameters())
        return predict, n, id2label

    return builder


# The models we TRY to load, roughly smallest -> largest download. Anything
# whose library/weights are missing is skipped (see load_available).
REGISTRY = [
    Segmenter("SegFormer-b0",     "Transformer / ADE20K",
              _build_hf_logits("nvidia/segformer-b0-finetuned-ade-512-512")),
    Segmenter("SegFormer-b2",     "Transformer / ADE20K",
              _build_hf_logits("nvidia/segformer-b2-finetuned-ade-512-512")),
    Segmenter("UperNet-ConvNeXt", "Transformer / ADE20K",
              _build_hf_logits("openmmlab/upernet-convnext-tiny")),
    Segmenter("BEiT",             "Transformer / ADE20K",
              _build_hf_logits("microsoft/beit-base-finetuned-ade-640-640")),
    Segmenter("Mask2Former",      "Transformer / ADE20K",
              _build_mask2former("facebook/mask2former-swin-tiny-ade-semantic")),
    Segmenter("DPT",              "Transformer / ADE20K",
              _build_hf_logits("Intel/dpt-large-ade")),
    Segmenter("DeepLabV3-R50",    "CNN / VOC-21",
              _build_torchvision_seg("deeplabv3")),
    Segmenter("FCN-R50",          "CNN / VOC-21",
              _build_torchvision_seg("fcn")),
    Segmenter("LR-ASPP-MNv3",     "CNN / VOC-21",
              _build_torchvision_seg("lraspp")),
]


def load_available(device: str) -> list:
    """Build every registered segmenter, skipping any that fail to load."""
    ready = []
    for seg in REGISTRY:
        try:
            seg.load(device)
            ready.append(seg)
            print(f"    [ok]   {seg.name:<17} ({seg.family}, "
                  f"{seg.n_params/1e6:.1f}M params)")
        except Exception as e:
            short = str(e).splitlines()[0][:70]
            print(f"    [skip] {seg.name:<17} — {short}")
    return ready


# ===========================================================================
# Colour a mask using a model's own label set.
# ===========================================================================
def colour_mask(mask: np.ndarray, id2label: dict) -> np.ndarray:
    """Map each class id to a fixed random colour (deterministic per run)."""
    n_classes = max(max(id2label) + 1, int(mask.max()) + 1)
    rng = np.random.default_rng(seed=42)
    palette = rng.integers(0, 255, size=(n_classes, 3), dtype=np.uint8)
    return palette[mask]                                     # (H, W, 3)


# ===========================================================================
# PART B — REAL accuracy (mIoU + pixel accuracy) on labelled validation data
# ---------------------------------------------------------------------------
# Detection reports mAP; segmentation's standard metric is mIoU (mean
# Intersection-over-Union): for each class, IoU = correctly-labelled pixels /
# (predicted-or-true pixels of that class), then averaged over the classes
# that actually appear. Pixel accuracy = fraction of all pixels labelled
# right. Both need GROUND-TRUTH masks, so we score each family on the dataset
# it was trained on: ADE20K for the transformers, Pascal VOC for the CNNs.
# ===========================================================================
def _confusion(pred: np.ndarray, gt: np.ndarray, k: int, ignore: int) -> np.ndarray:
    """Fast k x k confusion matrix (rows = true class, cols = predicted)."""
    valid = (gt != ignore) & (gt >= 0) & (gt < k) & (pred >= 0) & (pred < k)
    idx = gt[valid].astype(np.int64) * k + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=k * k).reshape(k, k)


def load_ade20k_slice(n: int):
    """`n` ADE20K val images as (RGB image, gt mask). gt ids match the models.

    The HF annotation stores class 1..150 (0 = unlabeled). The transformer
    models number the SAME classes 0..149, so we shift by one and mark the
    unlabeled pixels as 255 (ignored during scoring).
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"    [skip] `datasets` not installed ({e}). mIoU needs labelled data.")
        return None
    try:
        ds = load_dataset("scene_parse_150", split=f"validation[:{n}]",
                          cache_dir=DATA_DIR, trust_remote_code=True)
    except Exception as e:
        print(f"    [skip] couldn't load ADE20K val slice ({str(e).splitlines()[0]}).")
        return None

    samples = []
    for ex in ds:
        ann = np.array(ex["annotation"], dtype=np.int64)
        if ann.ndim == 3:                       # some rows come back RGB
            ann = ann[..., 0]
        gt = ann - 1                            # ADE class 1..150 -> id 0..149
        gt[ann == 0] = 255                      # unlabeled -> ignore
        samples.append((ex["image"].convert("RGB"), gt))
    return samples or None


def load_voc_slice(n: int):
    """`n` Pascal VOC-2012 val images as (RGB image, gt mask).

    VOC masks already use ids 0..20 (0 = background) matching the torchvision
    CNN models, with 255 marking the 'void' border pixels we ignore.
    """
    try:
        from torchvision.datasets import VOCSegmentation
    except Exception as e:
        print(f"    [skip] torchvision VOC dataset unavailable ({e}).")
        return None
    try:
        voc = VOCSegmentation(root=DATA_DIR, year="2012", image_set="val",
                              download=True)
    except Exception as e:
        print(f"    [skip] couldn't load Pascal VOC ({str(e).splitlines()[0]}).")
        return None

    samples = []
    for i in range(min(n, len(voc))):
        image, target = voc[i]
        samples.append((image.convert("RGB"), np.array(target, dtype=np.int64)))
    return samples or None


def evaluate_seg(models: list, samples: list, k: int, ignore: int,
                 dataset: str) -> list:
    """Score each model's mIoU + pixel accuracy on a labelled slice."""
    print(f"\n[B] Measuring accuracy on {len(samples)} labelled {dataset} images")
    print("    (mIoU = mean per-class Intersection-over-Union — the number")
    print("     segmentation papers report; higher is better)\n")

    rows = []
    for seg in models:
        conf = np.zeros((k, k), dtype=np.int64)
        for image, gt in samples:
            conf += _confusion(seg.predict(image), gt, k, ignore)
        inter = np.diag(conf).astype(np.float64)
        union = conf.sum(1) + conf.sum(0) - inter
        present = conf.sum(1) > 0               # classes that appear in the GT
        iou = inter / np.maximum(union, 1.0)
        miou = float(iou[present].mean()) if present.any() else 0.0
        pixacc = float(inter.sum() / max(conf.sum(), 1))
        rows.append({"name": seg.name, "family": seg.family, "dataset": dataset,
                     "miou": miou, "pixacc": pixacc})
        print(f"    {seg.name:<18} mIoU={miou*100:5.1f}   pixAcc={pixacc*100:5.1f}")

    rows.sort(key=lambda r: r["miou"], reverse=True)
    return rows


def plot_metrics(results: list, metric_rows: list, out_path: str) -> None:
    """Bar charts: speed (all models) and, if measured, mIoU accuracy."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"    [warn] plotting unavailable ({e}); skipping metrics plot.")
        return

    has_acc = bool(metric_rows)
    fig, axes = plt.subplots(1, 2 if has_acc else 1,
                             figsize=(13 if has_acc else 7, 5), squeeze=False)

    # Left: speed (lower is better).
    ax = axes[0][0]
    s = sorted(results, key=lambda r: r["ms"], reverse=True)
    ax.barh([r["name"] for r in s], [r["ms"] for r in s], color="#4C72B0")
    ax.set_xlabel("inference time (ms / image)  — lower is faster")
    ax.set_title("Speed")
    for i, r in enumerate(s):
        ax.text(r["ms"], i, f"  {r['ms']:.0f}ms", va="center", fontsize=8)

    # Right: accuracy (higher is better) — only if we scored it.
    if has_acc:
        ax = axes[0][1]
        m = sorted(metric_rows, key=lambda r: r["miou"])
        ax.barh([f"{r['name']} ({r['dataset']})" for r in m],
                [r["miou"] * 100 for r in m], color="#55A868")
        ax.set_xlabel("mIoU (%)  — higher is better")
        ax.set_title(f"Accuracy on {EVAL_N} labelled images")
        for i, r in enumerate(m):
            ax.text(r["miou"] * 100, i, f"  {r['miou']*100:.1f}",
                    va="center", fontsize=8)

    fig.suptitle("Segmentation model comparison: accuracy / speed trade-off",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved metrics plot -> {out_path}")


def main() -> None:
    print("=" * 60)
    print(" 05 - Semantic Segmentation — MODEL ZOO")
    print("=" * 60)

    device = get_device()
    print(f"[info] Using device: {device}")

    # -------------------------------------------------------------------
    # STEP 1: Load every available segmentation model.
    # First run downloads weights (SegFormer-b0 ~15MB; heavier ones like
    # DPT-large are ~1GB). Missing libs/weights are skipped automatically.
    # -------------------------------------------------------------------
    print("[1] Loading the segmentation model zoo...")
    segmenters = load_available(device)
    if not segmenters:
        raise RuntimeError("No segmentation models could be loaded.")
    print(f"    Loaded {len(segmenters)} model(s).")

    # -------------------------------------------------------------------
    # STEP 2: Load the image every model will run on.
    # -------------------------------------------------------------------
    path = find_sample_image()
    print(f"[2] Loading image: {path}")
    image = Image.open(path).convert("RGB")
    original = np.array(image)

    # -------------------------------------------------------------------
    # STEP 3: Run every model, time it, and record the classes it found.
    # -------------------------------------------------------------------
    print(f"[3] Predicting a class for every pixel with {len(segmenters)} models "
          f"(timing over {TIMING_PASSES} passes)...")
    results = []   # one dict per model: name, family, mask, overlay, ms, classes
    for seg in segmenters:
        mask = seg.predict(image)                            # warm-up pass

        t0 = time.perf_counter()
        for _ in range(TIMING_PASSES):
            mask = seg.predict(image)
        ms = 1000.0 * (time.perf_counter() - t0) / TIMING_PASSES

        class_names = [seg.id2label.get(int(c), str(c)) for c in np.unique(mask)]
        cmask = colour_mask(mask, seg.id2label)
        overlay = ((1 - MASK_ALPHA) * original + MASK_ALPHA * cmask).astype(np.uint8)
        results.append({
            "name": seg.name, "family": seg.family,
            "params_m": seg.n_params / 1e6, "ms": ms,
            "n_classes": len(class_names), "classes": class_names,
            "overlay": overlay, "cmask": cmask,
        })

    # -------------------------------------------------------------------
    # STEP 4: Print a comparison table (fastest first) + classes found.
    # -------------------------------------------------------------------
    results.sort(key=lambda r: r["ms"])
    print(f"\n    {'model':<18}{'family':<22}{'params':>8}{'ms/img':>9}{'#classes':>10}")
    print("    " + "-" * 67)
    for r in results:
        print(f"    {r['name']:<18}{r['family']:<22}{r['params_m']:>6.1f}M"
              f"{r['ms']:>9.1f}{r['n_classes']:>10}")
    print(f"    (inference time = wall-clock per image, averaged over "
          f"{TIMING_PASSES} passes on {device})")

    print("\n    Classes each model found:")
    for r in results:
        shown = ", ".join(r["classes"][:12])
        more = "" if len(r["classes"]) <= 12 else f", (+{len(r['classes']) - 12} more)"
        print(f"       {r['name']:<18}: {shown}{more}")

    # -------------------------------------------------------------------
    # STEP 4b (PART B): REAL accuracy on labelled validation images.
    # The two families use different label sets, so each is scored on the
    # dataset it was trained on. Any slice we can't download is skipped.
    # -------------------------------------------------------------------
    print("\n[4b] Measuring REAL accuracy (mIoU) on labelled validation data...")
    ade_models = [s for s in segmenters if s.family.endswith("ADE20K")]
    voc_models = [s for s in segmenters if s.family.endswith("VOC-21")]
    metric_rows = []
    if ade_models:
        ade_samples = load_ade20k_slice(EVAL_N)
        if ade_samples:
            metric_rows += evaluate_seg(ade_models, ade_samples, 150, 255, "ADE20K")
    if voc_models:
        voc_samples = load_voc_slice(EVAL_N)
        if voc_samples:
            metric_rows += evaluate_seg(voc_models, voc_samples, 21, 255, "VOC")

    if metric_rows:
        by_miou = sorted(metric_rows, key=lambda r: r["miou"], reverse=True)
        print(f"\n    {'model':<18}{'dataset':<10}{'mIoU':>8}{'pixAcc':>9}")
        print("    " + "-" * 45)
        for r in by_miou:
            print(f"    {r['name']:<18}{r['dataset']:<10}"
                  f"{r['miou']*100:>7.1f}{r['pixacc']*100:>9.1f}")
        print("    (mIoU/pixAcc are % — compare within a dataset; ADE20K's 150 "
              "classes make its mIoU look lower than VOC's 21.)")
    else:
        print("    (accuracy skipped — install `datasets` (ADE20K) / allow the "
              "VOC download to enable mIoU; the speed comparison still stands.)")

    # -------------------------------------------------------------------
    # STEP 5: Save the speed-vs-accuracy comparison chart.
    # -------------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\n[5] Saving the speed / accuracy comparison chart...")
    plot_metrics(results, metric_rows, os.path.join(RESULTS_DIR, "segmentation_metrics.png"))

    # -------------------------------------------------------------------
    # STEP 6: Show the original + every model's overlay side by side.
    # -------------------------------------------------------------------
    print("\n[6] Displaying results (close the window to finish)")
    try:
        import matplotlib.pyplot as plt
        panels = [("Original", original)] + [(r["name"], r["overlay"]) for r in results]
        cols = 3
        rows = (len(panels) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
        for ax in axes[len(panels):]:          # hide any empty cells
            ax.axis("off")
        plt.tight_layout()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "segmentation_result.png")
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        print(f"    Saved figure to {out_path}")
        plt.show()
    except Exception as e:
        print(f"    (display unavailable: {e})")

    print("\nDone. Segmentation gives the finest, pixel-level understanding — "
          "and different models 'see' it differently.")


if __name__ == "__main__":
    main()
