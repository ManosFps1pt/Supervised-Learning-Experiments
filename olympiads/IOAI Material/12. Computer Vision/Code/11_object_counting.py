"""
12_object_counting.py   (OLYMPIAD SPECIAL)
==========================================
COUNTING OBJECTS in an image.

Counting is a deceptively popular olympiad task: IOAI 2025 asked to count
chickens, and Poland's national olympiad asked to build a coin counter.
"How many X are there?" sounds easy but hides real challenges: overlapping
objects, tiny objects, and objects the detector wasn't trained on.

This script shows the TWO main ways to count, so you can pick the right
tool in a contest:

    METHOD 1 (deep learning): run an object detector (DETR) and count how
             many boxes of each class it found. Great for real-world objects.

    METHOD 2 (classical CV): for simple, well-separated blobs (like coins on
             a plain background) you don't even need a neural network — you
             can threshold + find contours with OpenCV. Fast and label-free.

Run it:
    python 12_object_counting.py

------------------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------------------
* COUNTING vs DETECTION:
    Detection finds objects; counting just needs the NUMBER. Often you count
    by detecting then tallying — but watch out for double-counting and
    missed overlapping objects.

* WHEN TO USE CLASSICAL CV:
    If objects are similar blobs on a clean background, contour-counting is
    faster and needs no model. Contests reward picking the SIMPLEST method
    that works.

* THRESHOLD & CONTOUR:
    Threshold = turn a grayscale image into black/white (object vs background).
    Contour  = the outline of a connected white blob. Count the blobs = count
    the objects.
"""

import os
import csv
import cv2
import numpy as np
import torch
from collections import Counter
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from model_summary import print_summary   # prints layers + params of each model

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "counting")


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_sample_image() -> str:
    images_dir = os.path.join(HERE, "images")
    for name in ("street.jpg", "cats.jpg", "dog.jpg"):
        path = os.path.join(images_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No image found. Please run: python download.py")


# =====================================================================
# METHOD 1 — count with a deep-learning detector (DETR)
# =====================================================================
def count_with_detector(path: str, device: str) -> Counter:
    print("\n--- METHOD 1: Counting with an object detector (DETR) ------")
    model_name = "facebook/detr-resnet-50"
    print(f"[1] Loading detector: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    print_summary(model, "DETR object detector")

    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.7
    )[0]

    # Tally how many boxes we found per class label.
    names = [model.config.id2label[i.item()] for i in results["labels"]]
    counts = Counter(names)

    print(f"[2] Detected {len(names)} objects total. Counts per class:")
    if not counts:
        print("    (none above the confidence threshold — try lowering it)")
    for name, n in counts.most_common():
        print(f"       {name:<15} x {n}")
    print("    TIP: to count a SPECIFIC class (e.g. chickens/coins), just")
    print("    filter the labels — e.g. counts['bird'].")

    # Export an annotated image showing every counted box.
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(image)
        for (x0, y0, x1, y1), name, score in zip(
                results["boxes"].tolist(), names, results["scores"].tolist()):
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="#C44E52", linewidth=2))
            ax.text(x0, y0 - 2, f"{name} {score:.2f}", fontsize=7, color="white",
                    bbox=dict(facecolor="#C44E52", edgecolor="none", pad=0.5))
        total = sum(counts.values())
        summary = ", ".join(f"{n}x {name}" for name, n in counts.most_common())
        ax.set_title(f"DETR counted {total} objects  ({summary})", fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "detector_counts.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved annotated image -> {out}")
    except Exception as e:
        print(f"    (plot unavailable: {e})")

    return counts


# =====================================================================
# METHOD 2 — count simple blobs with classical OpenCV (no model!)
# =====================================================================
def count_with_classical_cv() -> tuple:
    print("\n--- METHOD 2: Counting blobs with classical CV (no model) --")
    # We MAKE a clean synthetic "coins on a table" image so the demo always
    # works and clearly shows the idea. In a contest you'd load a real photo.
    print("[1] Creating a synthetic 'coins on a table' image...")
    canvas = np.full((300, 400, 3), 230, dtype=np.uint8)  # light gray table
    coin_centers = [(70, 80), (160, 90), (250, 70), (330, 120),
                    (110, 200), (220, 210), (310, 220)]
    for (x, y) in coin_centers:
        cv2.circle(canvas, (x, y), 28, (90, 90, 90), -1)   # draw 7 "coins"
    true_count = len(coin_centers)

    print("[2] Threshold -> find contours -> count blobs")
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Objects are darker than the table, so threshold picks them out.
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Keep only reasonably large blobs (ignore tiny noise specks).
    blobs = [c for c in contours if cv2.contourArea(c) > 200]

    print(f"    True number of coins  = {true_count}")
    print(f"    Counted by contours   = {len(blobs)}")
    print(f"    {'CORRECT!' if len(blobs)==true_count else 'off — tune the threshold/area'}")

    # Draw the detected blobs for the figure.
    annotated = canvas.copy()
    cv2.drawContours(annotated, blobs, -1, (0, 0, 255), 3)
    for i, c in enumerate(blobs, start=1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(annotated, str(i), (x + 5, y + h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Input (synthetic coins)")
        axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Counted {len(blobs)} objects")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "classical_counting.png")
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved figure to {out}")
    except Exception as e:
        print(f"    (display unavailable: {e})")

    return true_count, len(blobs)


def main() -> None:
    print("=" * 60)
    print(" 12 - Object Counting  [OLYMPIAD SPECIAL]")
    print("=" * 60)
    device = get_device()
    print(f"[info] Using device: {device}")

    path = find_sample_image()
    print(f"[info] Real image for Method 1: {path}")

    detector_counts = count_with_detector(path, device)
    true_count, classical_count = count_with_classical_cv()

    # Export both methods' counts to one CSV.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "counts.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "item", "count"])
        for name, n in detector_counts.most_common():
            w.writerow(["detector (DETR)", name, n])
        w.writerow(["detector (DETR)", "TOTAL", sum(detector_counts.values())])
        w.writerow(["classical CV", "coins (true)", true_count])
        w.writerow(["classical CV", "coins (counted)", classical_count])
    print(f"\n[export] Saved counts CSV -> {csv_path}")

    print("\nDone. Lesson: match the METHOD to the PROBLEM. Fancy detector for")
    print("messy real scenes; simple contours for clean, separated objects.")


if __name__ == "__main__":
    main()
