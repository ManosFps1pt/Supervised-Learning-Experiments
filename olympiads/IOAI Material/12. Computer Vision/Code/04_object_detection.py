"""
04_object_detection.py
======================
OBJECT DETECTION — and a fair COMPARISON of the main detector families.

Detection answers TWO questions at once for EVERY object in the image:
    1. WHAT is it?      (a label, e.g. "cat")
    2. WHERE is it?     (a BOUNDING BOX: a rectangle around the object)

A single demo image looks impressive but tells you almost nothing about how
GOOD a detector is, or which one to pick. So this script does what a real
practitioner does: it puts several detectors side by side and MEASURES them.

We compare two big families on the SAME images:

    CNN-based detectors (convolutions):
        * SSD           - one-stage, older, very fast, less accurate
        * RetinaNet     - one-stage, focal-loss, strong accuracy/speed balance
        * Faster R-CNN  - two-stage (propose boxes, then classify), accurate
        * YOLOv8        - one-stage, the popular real-time detector

    Transformer-based detector (attention, no hand-made anchors/NMS):
        * DETR          - "DEtection TRansformer", predicts a fixed set of boxes

Every model here is PRETRAINED on COCO (80 everyday object categories), so we
just run them — no training needed.

Run it:
    python 04_object_detection.py

The sample images (street.jpg, cats.jpg, dog.jpg) are real COCO val2017
photos, so if the labelled COCO slice is available we can compute the SAME
metrics the research papers report (mAP). Everything degrades gracefully:
missing libraries or datasets just skip a section with a warning.

------------------------------------------------------------------------
KEY CONCEPTS — the metrics
------------------------------------------------------------------------
* BOUNDING BOX:
    A rectangle (x_min, y_min, x_max, y_max) tightly around an object. Each
    detection also carries a CONFIDENCE score in [0, 1].

* IoU (Intersection over Union):
    IoU = (area of overlap) / (area of union) between two boxes.
    IoU = 1.0 is a perfect match, 0.0 is no overlap. A prediction counts as
    "correct" only if its IoU with a true box is above a threshold (e.g. 0.5).

* PRECISION / RECALL:
    - Precision = of the boxes we PREDICTED, how many were right?
    - Recall    = of the TRUE boxes, how many did we find?
    Sweeping the confidence threshold traces a precision-recall curve.

* AP and mAP (the headline detection metric):
    - AP  (Average Precision) = area under the precision-recall curve for ONE
          class at one IoU threshold.
    - mAP (mean AP) averages AP over all classes. "COCO mAP" (a.k.a mAP@[.5:.95])
          averages again over IoU thresholds 0.50, 0.55, ... 0.95 — a strict,
          all-round score. mAP@0.50 is the looser "PASCAL VOC" style score.
    - AR  (Average Recall) = how many true objects are found, averaged the
          same way. Higher is better for all of these.

* SPEED matters too:
    We time each model (milliseconds per image, and frames-per-second). The
    best detector is a TRADE-OFF between mAP and speed for YOUR use case.

* ONE-STAGE vs TWO-STAGE vs TRANSFORMER:
    - One-stage (SSD, RetinaNet, YOLO): predict boxes directly in one pass.
      Fast; historically a bit less accurate on small objects.
    - Two-stage (Faster R-CNN): first PROPOSE regions, then classify them.
      Slower but accurate.
    - Transformer (DETR): treats detection as set prediction with attention;
      no anchor boxes and no Non-Maximum-Suppression post-step.

* NMS (Non-Maximum Suppression):
    CNN detectors fire many overlapping boxes for one object, then use IoU to
    keep only the best and drop the duplicates. DETR avoids this by design.
"""

import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

warnings.filterwarnings("ignore")  # keep the workshop output readable

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(HERE, "images")
DATA_DIR = os.path.join(HERE, "data")
MODELS_DIR = os.path.join(HERE, "models")   # local weights live here (e.g. YOLO)
RESULTS_DIR = os.path.join(HERE, "results", "detections")

# Confidence threshold used for the visual comparison + the count/speed table.
# (mAP ignores this: it sweeps ALL thresholds internally.)
CONF_THRESHOLD = 0.5
EVAL_N = 50          # labelled COCO images used to measure mAP (small = fast)
TIMING_PASSES = 3    # how many times we re-run each model to average its speed


# ===========================================================================
# COCO label space — so we can compare models that number their classes
# differently. Everything is mapped to these 80 canonical names.
# ===========================================================================
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
# Different toolkits spell a few classes differently (VOC vs COCO conventions).
_ALIASES = {
    "aeroplane": "airplane", "motorbike": "motorcycle", "sofa": "couch",
    "pottedplant": "potted plant", "diningtable": "dining table",
    "tvmonitor": "tv", "cellphone": "cell phone",
}
_NAME_TO_ID = {name: i for i, name in enumerate(COCO80)}


def name_to_id(name: str) -> int:
    """Map any detector's class name to a canonical 0..79 COCO id (-1 if unknown)."""
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    return _NAME_TO_ID.get(key, -1)


@dataclass
class Detections:
    """One model's output on one image, in a shared format."""
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # xyxy, pixels
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    names: list = field(default_factory=list)                            # class names


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_sample_images() -> list:
    """All local demo images, in a nice order (street first: it's the busiest)."""
    paths = []
    for name in ("street.jpg", "cats.jpg", "dog.jpg"):
        p = os.path.join(IMAGES_DIR, name)
        if os.path.exists(p):
            paths.append(p)
    if not paths:
        raise FileNotFoundError("No image found. Please run: python download.py")
    return paths


# ===========================================================================
# DETECTOR REGISTRY
# ---------------------------------------------------------------------------
# Each detector exposes the same tiny interface via a `Detector`:
#     .load()                       -> download + build the model once
#     .predict(pil_image, thresh)   -> a `Detections` in shared xyxy format
# The loaders below return  (predict_fn, n_params)  so every model looks the
# same to the rest of the script, no matter which library it comes from.
# ===========================================================================
class Detector:
    def __init__(self, name: str, family: str, builder):
        self.name = name
        self.family = family
        self._builder = builder
        self.n_params = 0
        self._predict = None

    def load(self, device: str) -> None:
        self._predict, self.n_params = self._builder(device)

    def predict(self, image, threshold: float) -> Detections:
        return self._predict(image, threshold)


# ---- Transformer family: DETR (Hugging Face) ------------------------------
def _build_detr(device: str):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    name = "facebook/detr-resnet-50"
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForObjectDetection.from_pretrained(name).to(device).eval()

    @torch.no_grad()
    def predict(image, threshold):
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        target = torch.tensor([image.size[::-1]])  # (h, w)
        r = processor.post_process_object_detection(
            outputs, target_sizes=target, threshold=threshold)[0]
        names = [model.config.id2label[i.item()] for i in r["labels"]]
        return Detections(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), names)

    return predict, sum(p.numel() for p in model.parameters())


# ---- CNN family: torchvision detectors (SSD / RetinaNet / Faster R-CNN) ----
def _build_torchvision(kind: str):
    def builder(device: str):
        import torchvision
        m = torchvision.models.detection
        table = {  # (constructor, weights enum)
            "ssd": (m.ssd300_vgg16, m.SSD300_VGG16_Weights.COCO_V1),
            "retinanet": (m.retinanet_resnet50_fpn_v2,
                          m.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1),
            "fasterrcnn": (m.fasterrcnn_resnet50_fpn_v2,
                           m.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
        }
        ctor, weights = table[kind]
        # A low internal score threshold keeps enough boxes for an honest mAP
        # sweep. SSD's constructor doesn't accept that knob, so special-case it.
        extra = {} if kind == "ssd" else {"box_score_thresh": 0.01}
        model = ctor(weights=weights, **extra).to(device).eval()
        preprocess = weights.transforms()
        categories = weights.meta["categories"]  # index -> COCO name (91-way)

        @torch.no_grad()
        def predict(image, threshold):
            batch = [preprocess(image).to(device)]
            out = model(batch)[0]
            keep = out["scores"] >= threshold
            boxes = out["boxes"][keep].cpu().numpy()
            scores = out["scores"][keep].cpu().numpy()
            names = [categories[i] for i in out["labels"][keep].cpu().tolist()]
            return Detections(boxes, scores, names)

        return predict, sum(p.numel() for p in model.parameters())
    return builder


# ---- CNN family: YOLOv8 (ultralytics) -------------------------------------
def _build_yolo(device: str):
    from ultralytics import YOLO
    # Keep the weights inside the project's models/ folder (like the other
    # cached weights) instead of dumping yolov8n.pt in the working directory.
    # If the file isn't there yet, ultralytics downloads it straight to this
    # path (~6MB, 'n' = nano: tiny + fast).
    os.makedirs(MODELS_DIR, exist_ok=True)
    weights = os.path.join(MODELS_DIR, "yolov8n.pt")
    model = YOLO(weights)

    def predict(image, threshold):
        r = model.predict(image, conf=threshold, device=device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return Detections()
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        names = [r.names[int(c)] for c in r.boxes.cls.cpu().tolist()]
        return Detections(boxes, scores, names)

    return predict, sum(p.numel() for p in model.model.parameters())


# The models we TRY to compare. Any whose library is missing is skipped with a
# note, so the script still runs with whatever subset you have installed.
REGISTRY = [
    Detector("DETR",         "Transformer",      _build_detr),
    Detector("SSD300",       "CNN (one-stage)",  _build_torchvision("ssd")),
    Detector("RetinaNet",    "CNN (one-stage)",  _build_torchvision("retinanet")),
    Detector("Faster R-CNN", "CNN (two-stage)",  _build_torchvision("fasterrcnn")),
    Detector("YOLOv8n",      "CNN (one-stage)",  _build_yolo),
]


def load_available(device: str) -> list:
    """Build every registered detector, skipping any whose library is missing."""
    ready = []
    for det in REGISTRY:
        try:
            det.load(device)
            ready.append(det)
            print(f"    [ok]   {det.name:<13} ({det.family}, "
                  f"{det.n_params/1e6:.1f}M params)")
        except Exception as e:
            short = str(e).splitlines()[0][:70]
            print(f"    [skip] {det.name:<13} — {short}")
            if det.name == "YOLOv8n":
                print("           (install with:  pip install ultralytics)")
    return ready


# ===========================================================================
# PART A — run every model on the SAME local images and measure speed + counts
# ===========================================================================
def compare_on_images(detectors: list, image_paths: list) -> list:
    """Time each detector and tally what it finds. No labels needed here."""
    from PIL import Image
    print(f"\n[A] Speed + output comparison on {len(image_paths)} local image(s) "
          f"(conf > {CONF_THRESHOLD})")
    images = [Image.open(p).convert("RGB") for p in image_paths]
    rows = []

    for det in detectors:
        # Warm-up pass (first call is slow: lazy init, memory allocation).
        det.predict(images[0], CONF_THRESHOLD)

        # Timed passes: average milliseconds PER IMAGE over several passes.
        n_det, conf_sum = 0, 0.0
        t0 = time.perf_counter()
        for _ in range(TIMING_PASSES):
            for img in images:
                d = det.predict(img, CONF_THRESHOLD)
                n_det += len(d.scores)
                conf_sum += float(d.scores.sum())
        elapsed = time.perf_counter() - t0

        total_imgs = TIMING_PASSES * len(images)
        ms = 1000.0 * elapsed / total_imgs
        rows.append({
            "name": det.name, "family": det.family,
            "params_m": det.n_params / 1e6,
            "ms": ms, "fps": 1000.0 / ms,
            "avg_det": n_det / total_imgs,
            "avg_conf": (conf_sum / n_det) if n_det else 0.0,
        })

    # Pretty table, fastest first.
    rows.sort(key=lambda r: r["ms"])
    print(f"\n    {'model':<14}{'family':<18}{'params':>8}"
          f"{'ms/img':>9}{'FPS':>7}{'avg #det':>10}{'avg conf':>10}")
    print("    " + "-" * 76)
    for r in rows:
        print(f"    {r['name']:<14}{r['family']:<18}{r['params_m']:>6.1f}M"
              f"{r['ms']:>9.1f}{r['fps']:>7.1f}{r['avg_det']:>10.1f}"
              f"{r['avg_conf']*100:>9.1f}%")
    print("    (inference time = wall-clock per image, averaged over "
          f"{TIMING_PASSES} passes on {get_device()})")
    return rows


# ===========================================================================
# PART B — REAL metrics (mAP / mAP@50 / AR) on a labelled COCO slice
# ===========================================================================
def load_coco_slice(n: int):
    """Load `n` labelled COCO val images (image + true boxes + true classes).

    Returns a list of (PIL image, gt_boxes_xyxy ndarray, gt_names list) or None
    if the dataset / `datasets` library isn't available. Parsing is defensive:
    HF COCO repos differ slightly in how they name the annotation columns.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"    [skip] `datasets` not installed ({e}). "
              "mAP needs labelled data.")
        return None

    try:
        ds = load_dataset("rafaelpadilla/coco2017", split=f"val[:{n}]",
                          cache_dir=DATA_DIR)
    except Exception as e:
        print(f"    [skip] couldn't load COCO val slice ({str(e).splitlines()[0]}).")
        return None

    # Find the column holding per-object annotations (a dict-of-lists).
    obj_col = next((c for c in ("objects", "annotations") if c in ds.column_names),
                   None)
    if obj_col is None:
        print("    [skip] COCO slice has no recognizable annotation column.")
        return None

    # Map category ids -> COCO names using the dataset's own label feature.
    label_names = None
    feat = ds.features[obj_col]
    inner = getattr(feat, "feature", feat)
    for key in ("label", "category", "category_id"):
        sub = inner.get(key) if hasattr(inner, "get") else None
        if sub is not None and getattr(sub, "names", None):
            label_names = sub.names
            break

    samples = []
    for ex in ds:
        obj = ex[obj_col]
        bboxes = obj.get("bbox") or obj.get("boxes")
        labels = obj.get("label")
        if labels is None:
            labels = obj.get("category") or obj.get("category_id")
        if not bboxes:
            continue
        boxes_xyxy, names = [], []
        for bb, lab in zip(bboxes, labels):
            x, y, w, h = bb            # COCO stores boxes as [x, y, width, height]
            boxes_xyxy.append([x, y, x + w, y + h])
            names.append(label_names[lab] if label_names else str(lab))
        samples.append((ex["image"].convert("RGB"),
                        np.array(boxes_xyxy, dtype=np.float32), names))
    return samples if samples else None


def _to_ids(names):
    """Vectorised name -> canonical id (keeps -1 for unknowns; caller filters)."""
    return np.array([name_to_id(n) for n in names], dtype=np.int64)


def evaluate_map(detectors: list, samples: list) -> list:
    """Compute COCO mAP / mAP@50 / AR for each detector on the labelled slice."""
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except Exception as e:
        print(f"    [skip] torchmetrics not installed ({e}). "
              "Install with:  pip install torchmetrics pycocotools")
        return []

    print(f"\n[B] Measuring accuracy on {len(samples)} labelled COCO images")
    print("    (mAP sweeps every confidence + IoU threshold — this is the")
    print("     number detection papers report; higher is better)\n")

    rows = []
    for det in detectors:
        metric = MeanAveragePrecision(box_format="xyxy")
        for image, gt_boxes, gt_names in samples:
            # Predict with a LOW threshold: mAP wants the full ranked list.
            d = det.predict(image, 0.05)
            p_ids = _to_ids(d.names)
            g_ids = _to_ids(gt_names)
            pk, gk = p_ids >= 0, g_ids >= 0   # drop classes outside COCO-80
            metric.update(
                [{"boxes": torch.tensor(d.boxes[pk], dtype=torch.float32).reshape(-1, 4),
                  "scores": torch.tensor(d.scores[pk], dtype=torch.float32),
                  "labels": torch.tensor(p_ids[pk])}],
                [{"boxes": torch.tensor(gt_boxes[gk], dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor(g_ids[gk])}],
            )
        res = metric.compute()
        rows.append({
            "name": det.name, "family": det.family,
            "map": float(res["map"]), "map50": float(res["map_50"]),
            "map75": float(res["map_75"]), "mar": float(res["mar_100"]),
        })
        print(f"    {det.name:<14} mAP={rows[-1]['map']*100:5.1f}  "
              f"mAP@50={rows[-1]['map50']*100:5.1f}  "
              f"mAP@75={rows[-1]['map75']*100:5.1f}  "
              f"AR={rows[-1]['mar']*100:5.1f}")

    rows.sort(key=lambda r: r["map"], reverse=True)
    print(f"\n    {'model':<14}{'family':<18}{'mAP':>7}{'mAP@50':>9}"
          f"{'mAP@75':>9}{'AR@100':>9}")
    print("    " + "-" * 66)
    for r in rows:
        print(f"    {r['name']:<14}{r['family']:<18}{r['map']*100:>6.1f}"
              f"{r['map50']*100:>9.1f}{r['map75']*100:>9.1f}{r['mar']*100:>9.1f}")
    return rows


# ===========================================================================
# Visualisation — one figure with every model's boxes on the same image
# ===========================================================================
_FAMILY_COLOR = {
    "Transformer": "#C44E52",       # red
    "CNN (one-stage)": "#4C72B0",   # blue
    "CNN (two-stage)": "#55A868",   # green
}


def plot_detection_grid(detectors: list, image_path: str, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image
    except Exception as e:
        print(f"    [warn] plotting unavailable ({e}); skipping figure.")
        return

    image = Image.open(image_path).convert("RGB")
    n = len(detectors)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows))
    fig.suptitle(f"Same image, {n} detectors  (conf > {CONF_THRESHOLD})  —  "
                 f"{os.path.basename(image_path)}", fontsize=14)

    for ax, det in zip(np.ravel(axes), detectors):
        d = det.predict(image, CONF_THRESHOLD)
        color = _FAMILY_COLOR.get(det.family, "#8172B3")
        ax.imshow(image)
        for (x0, y0, x1, y1), name, score in zip(d.boxes, d.names, d.scores):
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=color, linewidth=2))
            ax.text(x0, y0 - 2, f"{name} {score:.2f}", fontsize=6, color="white",
                    bbox=dict(facecolor=color, edgecolor="none", pad=0.5))
        ax.set_title(f"{det.name}  ·  {det.family}\n{len(d.scores)} objects",
                     fontsize=10)
        ax.axis("off")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved comparison image -> {out_path}")


def plot_metrics(speed_rows: list, map_rows: list, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"    [warn] plotting unavailable ({e}); skipping figure.")
        return

    has_map = bool(map_rows)
    fig, axes = plt.subplots(1, 2 if has_map else 1,
                             figsize=(13 if has_map else 7, 5), squeeze=False)

    # Left: speed (lower is better).
    ax = axes[0][0]
    s = sorted(speed_rows, key=lambda r: r["ms"])
    colors = [_FAMILY_COLOR.get(r["family"], "#8172B3") for r in s]
    ax.barh([r["name"] for r in s], [r["ms"] for r in s], color=colors)
    ax.set_xlabel("inference time (ms / image)  — lower is faster")
    ax.set_title("Speed")
    for i, r in enumerate(s):
        ax.text(r["ms"], i, f"  {r['ms']:.0f}ms · {r['fps']:.0f}fps",
                va="center", fontsize=8)

    # Right: accuracy (higher is better) — only if we measured it.
    if has_map:
        ax = axes[0][1]
        m = sorted(map_rows, key=lambda r: r["map"])
        colors = [_FAMILY_COLOR.get(r["family"], "#8172B3") for r in m]
        ax.barh([r["name"] for r in m], [r["map"] * 100 for r in m], color=colors)
        ax.set_xlabel("COCO mAP@[.5:.95] (%)  — higher is better")
        ax.set_title(f"Accuracy on {EVAL_N} labelled images")
        for i, r in enumerate(m):
            ax.text(r["map"] * 100, i, f"  {r['map']*100:.1f}",
                    va="center", fontsize=8)

    fig.suptitle("Detector comparison: the accuracy / speed trade-off", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved metrics plot -> {out_path}")


def main() -> None:
    print("=" * 64)
    print(" 04 - Object Detection: comparing CNN vs Transformer detectors")
    print("=" * 64)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = get_device()
    print(f"[info] Using device: {device}")
    print("[info] Loading detectors (first run downloads model weights)...")
    detectors = load_available(device)
    if not detectors:
        print("\nNo detectors could be loaded. Install torch + transformers "
              "(and optionally torchvision / ultralytics) and retry.")
        return

    image_paths = find_sample_images()

    # A) speed + output comparison on the local sample images (always runs)
    speed_rows = compare_on_images(detectors, image_paths)

    # B) proper accuracy (mAP) on a labelled COCO slice (optional)
    print("\n[B] Loading a small labelled COCO slice for mAP...")
    samples = load_coco_slice(EVAL_N)
    map_rows = evaluate_map(detectors, samples) if samples else []
    if not samples:
        print("    (mAP skipped — the count/speed comparison above still stands. "
              "It needs `datasets` + internet for the COCO val slice.)")

    # Figures
    print("\n[C] Saving figures...")
    plot_detection_grid(detectors, image_paths[0],
                        os.path.join(RESULTS_DIR, "detectors_side_by_side.png"))
    plot_metrics(speed_rows, map_rows,
                 os.path.join(RESULTS_DIR, "detector_metrics.png"))

    print("\nDone. Takeaways:")
    print("  * There is no single 'best' detector — it's a mAP-vs-speed trade-off.")
    print("  * CNN one-stage (YOLO/SSD) win on speed; two-stage & DETR often win")
    print("    on accuracy. DETR (Transformer) needs no anchors or NMS.")
    print(f"  * See {RESULTS_DIR}/ for the side-by-side + metrics plots.")


if __name__ == "__main__":
    main()
