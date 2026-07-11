"""
11_quantization_efficiency.py   (OLYMPIAD SPECIAL)
==================================================
MODEL EFFICIENCY: making a model SMALLER and FASTER with quantization.

Efficiency is a recurring olympiad theme (IOAI 2024 had a quantization task;
IOAI 2025 had a "pixel efficiency" challenge). Contests give you limited
compute and time, so a solution that is 4x smaller and faster — while keeping
almost the same accuracy — can be the difference between passing and failing.

We take the pretrained ViT classifier and apply DYNAMIC QUANTIZATION, then
compare size, speed, and predictions. This runs on CPU (quantization is
actually a CPU-inference technique).

Run it:
    python 11_quantization_efficiency.py

------------------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------------------
* PRECISION (float32 vs int8):
    By default weights are 32-bit floats (float32) — very precise, but big.
    Quantization stores them as 8-bit integers (int8): 4x smaller, and
    integer math is faster on CPUs. We trade a tiny bit of accuracy for a
    lot of efficiency.

* DYNAMIC QUANTIZATION:
    The simplest kind: PyTorch quantizes the weights of Linear layers to
    int8 and converts on-the-fly during inference. One line of code, no
    retraining. Perfect for a time-limited contest.

* WHAT TO MEASURE:
    1. Model SIZE on disk (MB)
    2. Inference SPEED (seconds per image)
    3. AGREEMENT: does the small model still predict the same thing?
"""

import os
import csv
import time
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from model_summary import print_summary   # prints layers + params of each model

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "quantization")


def find_sample_image() -> str:
    images_dir = os.path.join(HERE, "images")
    for name in ("cats.jpg", "dog.jpg", "street.jpg"):
        path = os.path.join(images_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No image found. Please run: python download.py")


def model_size_mb(model) -> float:
    """Save the model's weights to a temp file and measure the size in MB."""
    tmp = os.path.join(HERE, "models", "_tmp_size.pt")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    torch.save(model.state_dict(), tmp)
    mb = os.path.getsize(tmp) / (1024 * 1024)
    os.remove(tmp)
    return mb


def time_inference(model, pixel_values, runs=5) -> float:
    """Average seconds per forward pass (after a warm-up run)."""
    with torch.no_grad():
        model(pixel_values=pixel_values)  # warm-up (not timed)
        start = time.perf_counter()
        for _ in range(runs):
            model(pixel_values=pixel_values)
        return (time.perf_counter() - start) / runs


def top1(model, pixel_values):
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits[0]
    probs = torch.softmax(logits, dim=-1)
    idx = int(probs.argmax())
    return idx, model.config.id2label[idx], float(probs[idx])


def export_results(metrics: dict) -> None:
    """Write the fp32-vs-int8 metrics to a CSV and a comparison chart."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) CSV — the raw numbers, easy to paste into a report or spreadsheet.
    csv_path = os.path.join(RESULTS_DIR, "quantization_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "float32", "int8"])
        w.writerow(["model size (MB)", f"{metrics['size_fp32']:.1f}",
                    f"{metrics['size_int8']:.1f}"])
        w.writerow(["time per image (ms)", f"{metrics['speed_fp32']*1000:.1f}",
                    f"{metrics['speed_int8']*1000:.1f}"])
        w.writerow(["top-1 prediction", metrics['label32'], metrics['label8']])
        w.writerow(["confidence (%)", f"{metrics['conf32']*100:.1f}",
                    f"{metrics['conf8']*100:.1f}"])
    print(f"    Saved metrics CSV -> {csv_path}")

    # 2) Figure — size + speed bars, annotated with the reduction / speed-up.
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"    (chart unavailable: {e}; CSV still written)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = ["#4C72B0", "#55A868"]
    for ax, key_f, key_i, title, unit, values in [
        (axes[0], "size_fp32", "size_int8", "Model size", "MB",
         [metrics["size_fp32"], metrics["size_int8"]]),
        (axes[1], "speed_fp32", "speed_int8", "Inference time / image", "ms",
         [metrics["speed_fp32"] * 1000, metrics["speed_int8"] * 1000]),
    ]:
        ax.bar(["float32", "int8"], values, color=colors)
        ax.set_ylabel(unit)
        ax.set_title(title)
        for i, v in enumerate(values):
            ax.text(i, v, f" {v:.1f}", ha="center", va="bottom", fontsize=9)

    size_drop = (1 - metrics["size_int8"] / metrics["size_fp32"]) * 100
    speedup = metrics["speed_fp32"] / metrics["speed_int8"]
    agree = "same prediction" if metrics["idx32"] == metrics["idx8"] else "prediction changed"
    fig.suptitle(f"Dynamic quantization: -{size_drop:.0f}% size, "
                 f"{speedup:.2f}x faster, {agree}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png_path = os.path.join(RESULTS_DIR, "quantization_comparison.png")
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved comparison chart -> {png_path}")


def main() -> None:
    print("=" * 60)
    print(" 11 - Quantization & Efficiency  [OLYMPIAD SPECIAL]")
    print("=" * 60)
    print("[info] Quantization targets CPU inference, so we use CPU here.")

    # -------------------------------------------------------------------
    # STEP 1: Load the full-precision (float32) model.
    # -------------------------------------------------------------------
    model_name = "google/vit-base-patch16-224"
    print(f"[1] Loading full-precision model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model_fp32 = AutoModelForImageClassification.from_pretrained(model_name)
    model_fp32.eval()
    print_summary(model_fp32, "ViT (float32)")

    path = find_sample_image()
    print(f"[2] Using image: {path}")
    image = Image.open(path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]

    # -------------------------------------------------------------------
    # STEP 2: Apply dynamic quantization (float32 -> int8) in ONE line.
    # We quantize the Linear layers, which hold most of a transformer's weights.
    # -------------------------------------------------------------------
    print("[3] Quantizing the model (float32 -> int8)...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    model_int8.eval()
    print_summary(model_int8, "ViT (int8 quantized)")

    # -------------------------------------------------------------------
    # STEP 3: Measure SIZE, SPEED, and AGREEMENT.
    # -------------------------------------------------------------------
    print("[4] Measuring size, speed, and predictions...\n")

    size_fp32 = model_size_mb(model_fp32)
    size_int8 = model_size_mb(model_int8)

    speed_fp32 = time_inference(model_fp32, pixel_values)
    speed_int8 = time_inference(model_int8, pixel_values)

    idx32, label32, conf32 = top1(model_fp32, pixel_values)
    idx8, label8, conf8 = top1(model_int8, pixel_values)

    print(f"    {'metric':<24}{'float32':>14}{'int8':>14}")
    print(f"    {'-'*24}{'-'*14}{'-'*14}")
    print(f"    {'model size (MB)':<24}{size_fp32:>14.1f}{size_int8:>14.1f}")
    print(f"    {'time / image (ms)':<24}{speed_fp32*1000:>14.1f}{speed_int8*1000:>14.1f}")
    print(f"    {'top-1 prediction':<24}{label32[:14]:>14}{label8[:14]:>14}")
    print(f"    {'confidence (%)':<24}{conf32*100:>14.1f}{conf8*100:>14.1f}")

    print("\n[5] Summary:")
    print(f"    * Size reduced by  {(1 - size_int8/size_fp32)*100:.0f}%  "
          f"({size_fp32:.0f} MB -> {size_int8:.0f} MB)")
    if speed_int8 < speed_fp32:
        print(f"    * Inference {speed_fp32/speed_int8:.2f}x faster on this machine")
    else:
        print("    * Speed varies by CPU; on many machines int8 is faster.")
    agree = "SAME" if idx32 == idx8 else "different"
    print(f"    * Prediction is {agree} after quantization "
          f"({'accuracy preserved' if idx32==idx8 else 'small accuracy cost'})")

    print("\n[6] Exporting results...")
    export_results({
        "size_fp32": size_fp32, "size_int8": size_int8,
        "speed_fp32": speed_fp32, "speed_int8": speed_int8,
        "idx32": idx32, "label32": label32, "conf32": conf32,
        "idx8": idx8, "label8": label8, "conf8": conf8,
    })

    print("\nDone. Same model, ~4x smaller, and the answer barely changed.")
    print("In a contest with tight limits, that trade-off can win you points.")


if __name__ == "__main__":
    main()
