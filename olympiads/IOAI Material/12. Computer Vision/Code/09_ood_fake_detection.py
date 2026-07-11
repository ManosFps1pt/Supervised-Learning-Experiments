"""
10_ood_fake_detection.py   (OLYMPIAD SPECIAL)
=============================================
OUT-OF-DISTRIBUTION (OOD) & FAKE-IMAGE DETECTION.

Two closely-related olympiad favourites (IOAI 2025 had an OOD task; China's
national olympiad had a "real or fake AI image" task):

    * OOD detection: "Is this input UNLIKE anything the model knows?"
    * Fake detection: "Was this image made by an AI, not a camera?"

The unifying trick: turn every image into an EMBEDDING (a vector), then
reason about DISTANCES and SIMILARITIES in that vector space. No training
needed — this is why embeddings are an olympiad Swiss-army knife.

Run it (optionally run 07_stable_diffusion.py first to create a 'fake' image):
    python 10_ood_fake_detection.py

------------------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------------------
* IN-DISTRIBUTION vs OUT-OF-DISTRIBUTION:
    "In-distribution" = similar to your reference/training data.
    "Out-of-distribution" = something new/weird the model shouldn't trust.
    A confident classifier can be confidently WRONG on OOD inputs, so we
    add a SEPARATE check: how far is this image from what we know?

* EMBEDDING-DISTANCE OOD SCORE:
    Build a "memory" of known images (their embeddings). For a new image,
    measure its similarity to the closest known image. Low similarity =
    likely OOD.

* ZERO-SHOT FAKE DETECTION with CLIP:
    Ask CLIP to compare the image against the texts "a real photograph" vs
    "an AI-generated image". Whichever wins is the guess. (A simple, clever
    baseline — real contest solutions go further, but this shows the idea.)
"""

import os
import glob
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from model_summary import print_summary   # prints layers + params of each model


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "ood")


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def list_images() -> list[str]:
    images_dir = os.path.join(HERE, "images")
    paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) +
                   glob.glob(os.path.join(images_dir, "*.png")))
    if not paths:
        raise FileNotFoundError("No images found. Please run: python download.py")
    return paths


def _as_tensor(feats):
    """transformers >=5 returns a ModelOutput (embedding in .pooler_output);
    older versions returned a plain tensor. Normalize both to a tensor."""
    if isinstance(feats, torch.Tensor):
        return feats
    out = getattr(feats, "pooler_output", None)
    if out is None:
        raise TypeError("Unexpected get_*_features output")
    return out


def clip_image_embedding(path, model, processor, device):
    """Embed one image into CLIP's shared image/text space (normalized)."""
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = _as_tensor(model.get_image_features(**inputs))
    return F.normalize(feats, dim=-1)[0]   # unit-length vector


def save_results_figure(paths, report):
    """One panel per image: the picture titled with its OOD score + verdict."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"    (plot unavailable: {e}; results printed above)")
        return
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.8), squeeze=False)
    for ax, p in zip(axes[0], paths):
        r = report[p]
        ax.imshow(Image.open(p).convert("RGB"))
        star = "  <-- most OOD" if r.get("most_ood") else ""
        title = (f"{os.path.basename(p)}{star}\n"
                 f"OOD score = {r.get('ood', 0):.3f}\n"
                 f"{r.get('verdict', '?')}  "
                 f"(real {r.get('real', 0)*100:.0f}% / fake {r.get('fake', 0)*100:.0f}%)")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("OOD score + real/fake verdict per image", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "ood_fake_detection.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[5] Saved results figure -> {out}")


def main() -> None:
    print("=" * 60)
    print(" 10 - OOD & Fake-Image Detection  [OLYMPIAD SPECIAL]")
    print("=" * 60)

    device = get_device()
    print(f"[info] Using device: {device}")

    # -------------------------------------------------------------------
    # STEP 1: Load CLIP — it gives us both image and text embeddings.
    # -------------------------------------------------------------------
    model_name = "openai/clip-vit-base-patch32"
    print(f"[1] Loading CLIP: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    print_summary(model, "CLIP (vision + text encoders)")

    paths = list_images()
    print(f"[2] Found {len(paths)} images in images/")

    # ===================================================================
    # PART A — OOD detection by embedding distance
    # ===================================================================
    print("\n--- PART A: Out-of-distribution detection ------------------")
    # We treat all-but-one image as our "known world" (the reference set),
    # then score each image by its BEST similarity to a *different* image.
    # The odd-one-out will have the lowest best-similarity = most OOD.
    embeddings = {p: clip_image_embedding(p, model, processor, device) for p in paths}

    print("[3] OOD score = 1 - (highest similarity to any OTHER image)")
    print("    (higher score = more unusual / more out-of-distribution)\n")
    report = {p: {} for p in paths}   # per-image results, collected for the figure
    scored = []
    for p in paths:
        sims = [F.cosine_similarity(embeddings[p].unsqueeze(0),
                                    embeddings[q].unsqueeze(0)).item()
                for q in paths if q != p]
        best = max(sims) if sims else 0.0
        ood_score = 1.0 - best
        scored.append((p, ood_score))
        report[p]["ood"] = ood_score
        print(f"    {os.path.basename(p):<18} OOD score = {ood_score:.3f}")
    scored.sort(key=lambda x: x[1], reverse=True)
    report[scored[0][0]]["most_ood"] = True
    print(f"\n    => Most out-of-distribution image: "
          f"{os.path.basename(scored[0][0])}")
    print("       (If you generated one with script 07, it often stands out!)")

    # ===================================================================
    # PART B — Real-vs-AI zero-shot fake detection with CLIP
    # ===================================================================
    print("\n--- PART B: Real-vs-AI (fake) detection --------------------")
    real_prompt = "a real photograph taken by a camera"
    fake_prompt = "an AI-generated computer image"
    text_inputs = processor(text=[real_prompt, fake_prompt],
                            return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feats = F.normalize(_as_tensor(model.get_text_features(**text_inputs)), dim=-1)

    print(f"[4] Comparing each image to:")
    print(f"      REAL prompt: \"{real_prompt}\"")
    print(f"      FAKE prompt: \"{fake_prompt}\"\n")
    for p in paths:
        img_feat = embeddings[p]                      # already normalized
        sims = img_feat @ text_feats.T                # 2 similarities
        probs = torch.softmax(sims * 100, dim=-1)     # CLIP uses a temperature
        verdict = "REAL" if probs[0] > probs[1] else "FAKE (AI)"
        report[p].update(real=probs[0].item(), fake=probs[1].item(), verdict=verdict)
        print(f"    {os.path.basename(p):<18} "
              f"real={probs[0]*100:5.1f}%  fake={probs[1]*100:5.1f}%  -> {verdict}")

    # ===================================================================
    # Save a figure: each image with its OOD score + real/fake verdict.
    # ===================================================================
    save_results_figure(paths, report)

    print("\nDone. Notice this is just 'embed, then compare'. That single")
    print("pattern solves matching, retrieval, OOD, and fake-detection tasks.")
    print("Tip: real contest solutions combine this with a trained detector.")


if __name__ == "__main__":
    main()
