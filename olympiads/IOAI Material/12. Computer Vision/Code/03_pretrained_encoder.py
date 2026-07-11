"""
03_pretrained_encoder.py
========================
IMAGE EMBEDDINGS: turning a picture into a vector of numbers.

A pretrained "encoder" reads an image and outputs a fixed-length list of
numbers (a VECTOR) that captures its meaning. This vector is called an
EMBEDDING. Similar images produce similar vectors - which lets us do
search, clustering, and comparison WITHOUT any labels.

This script has TWO parts:
    A) Embed a few images and measure their COSINE SIMILARITY.
    B) COMPARE several pretrained encoders: embed hundreds of images with
       each one, squash the vectors to 2-D with t-SNE and UMAP, and plot
       them so you can SEE which encoders separate the classes best.

The encoders we compare span the main families:
    * ViT-B/16   - a Vision Transformer trained with LABELS (ImageNet).
    * ResNet-18/50, ConvNeXt-T - CONVOLUTIONAL networks (CNNs), the classic
       and modern workhorses of computer vision.
    * DINOv2 / DINOv3 - transformers trained SELF-SUPERVISED (NO labels).
       These famously produce very clean, well-separated embeddings.

The 2-D plots are saved under  results/embeddings/.

Run it:
    python 03_pretrained_encoder.py

------------------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------------------
* EMBEDDING:
    A point in a high-dimensional space that represents the image. Different
    encoders output different sizes (512, 384, 768, ...).

* POOLING an image into ONE vector:
    - A CNN outputs a grid of features (C x H x W). We GLOBAL-AVERAGE-POOL
      over the grid -> one vector of length C.
    - A transformer outputs one vector per patch token plus a special [CLS]
      token that summarises the whole image. We take that [CLS] vector.

* DIMENSIONALITY REDUCTION (part B):
    t-SNE and UMAP place high-D points on a 2-D map so nearby points stay
    nearby. Neither uses the labels; we only COLOUR by label to check how
    well each encoder grouped the classes ON ITS OWN.
      - SELF-SUPERVISED encoders (DINO) usually give the cleanest clusters.
"""

import gc
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
from model_summary import print_summary   # prints layers + params of each model

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results", "embeddings")

EMBED_N = 300     # how many images to embed for the 2-D maps
BATCH_SIZE = 16   # images encoded at once

# The dataset we project. Imagenette has 10 clearly-different classes, so the
# clusters are easy to see. (Same 10 classes used in 02_image_classification.)
IMAGENETTE = {
    "repo": "johnowhitaker/imagenette2-320",
    "split": "train",
    "label_key": "label",
    "pretty": ["tench", "English springer", "cassette", "chain saw", "church",
               "French horn", "garbage truck", "gas pump", "golf ball", "parachute"],
}

# The encoders we compare in part B. 'optional' models are skipped (with a
# note) if they fail to load - e.g. DINOv3 is GATED and needs you to accept
# its licence on Hugging Face and set an HF token.
ENCODERS = [
    {"name": "ViT-B/16 (supervised)", "repo": "google/vit-base-patch16-224"},
    {"name": "ResNet-18 (CNN)",       "repo": "microsoft/resnet-18"},
    {"name": "ResNet-50 (CNN)",       "repo": "microsoft/resnet-50"},
    {"name": "ConvNeXt-T (CNN)",      "repo": "facebook/convnext-tiny-224"},
    {"name": "DINOv2-S (self-sup)",   "repo": "facebook/dinov2-small"},
    {"name": "DINOv3-S (self-sup)",   "repo": "facebook/dinov3-vits16-pretrain-lvd1689m",
     "optional": True},
]


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Turn an encoder's raw output into ONE vector per image.

    - CNNs give a feature MAP (B, C, H, W) -> GLOBAL-AVERAGE-POOL over H,W.
    - Transformers give TOKENS (B, tokens, D) -> take the [CLS] token (the
      first token), which is the model's summary of the whole image. This is
      the standard image embedding for ViT and DINO.
    We deliberately don't use `pooler_output`: for some models (e.g. plain
    ViT) that extra head is randomly initialised and would give garbage.
    """
    if last_hidden_state.dim() == 4:              # CNN feature map
        return last_hidden_state.mean(dim=(2, 3))
    return last_hidden_state[:, 0]                # transformer [CLS] token


@torch.no_grad()
def embed_images(pil_images, processor, model, device) -> np.ndarray:
    """Embed a list of PIL images -> (N, D) array, in batches for speed."""
    out = []
    for start in range(0, len(pil_images), BATCH_SIZE):
        batch = pil_images[start:start + BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        hidden = model(**inputs).last_hidden_state
        out.append(pool(hidden).cpu().numpy())
    return np.concatenate(out, axis=0)


def find_images() -> list:
    images_dir = os.path.join(HERE, "images")
    found = [os.path.join(images_dir, n) for n in ("cats.jpg", "dog.jpg", "street.jpg")
             if os.path.exists(os.path.join(images_dir, n))]
    if not found:
        raise FileNotFoundError("No images found. Please run: python download.py")
    return found


# ===========================================================================
# PART A — a few images and their cosine similarity (using one ViT encoder)
# ===========================================================================
def similarity_demo(device) -> None:
    from PIL import Image
    print("\n[A] Embedding + cosine similarity on the sample images")
    name = "google/vit-base-patch16-224"
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device)
    model.eval()
    print_summary(model, "ViT encoder")

    paths = find_images()
    images = [Image.open(p).convert("RGB") for p in paths]
    vecs = embed_images(images, processor, model, device)
    print(f"    Each image -> a vector of {vecs.shape[1]} numbers.")
    if len(paths) > 1:
        v0 = torch.tensor(vecs[0]).unsqueeze(0)
        print("    Cosine similarity to the first image (1.0 = identical):")
        for path, v in zip(paths, vecs):
            sim = F.cosine_similarity(v0, torch.tensor(v).unsqueeze(0)).item()
            print(f"      {os.path.basename(path):<12} = {sim:.3f}")
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ===========================================================================
# PART B — embed the same images with MANY encoders, project each to 2-D
# ===========================================================================
def load_imagenette():
    from datasets import load_dataset
    ds = load_dataset(IMAGENETTE["repo"], split=IMAGENETTE["split"], cache_dir=DATA_DIR)
    ds = ds.shuffle(seed=0).select(range(min(EMBED_N, len(ds))))
    images = [ex["image"].convert("RGB") for ex in ds]
    labels = np.array(ds[IMAGENETTE["label_key"]])
    return images, labels


def project(embeddings: np.ndarray):
    """L2-normalise, then reduce to 2-D with both t-SNE and UMAP."""
    from sklearn.manifold import TSNE
    import umap
    x = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    tsne_xy = TSNE(n_components=2, perplexity=30, init="pca",
                   learning_rate="auto", random_state=0).fit_transform(x)
    umap_xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        random_state=0).fit_transform(x)
    return tsne_xy, umap_xy


def compare_encoders(device) -> None:
    print(f"\n[B] Comparing encoders on {EMBED_N} Imagenette images")
    images, labels = load_imagenette()
    pretty = IMAGENETTE["pretty"]

    results = []   # list of (display_name, dim, tsne_xy, umap_xy)
    for enc in ENCODERS:
        try:
            processor = AutoImageProcessor.from_pretrained(enc["repo"])
            model = AutoModel.from_pretrained(enc["repo"]).to(device)
            model.eval()
        except Exception as e:
            tag = "gated/optional" if enc.get("optional") else "failed"
            print(f"    [skip] {enc['name']} ({tag}): {str(e)[:80]}")
            if enc.get("optional"):
                print("           To enable DINOv3: accept its licence on the model's")
                print("           HF page, then `huggingface-cli login` (or set HF_TOKEN).")
            continue

        print(f"\n    -> {enc['name']}  [{enc['repo']}]")
        print_summary(model, enc["name"])
        emb = embed_images(images, processor, model, device)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        tsne_xy, umap_xy = project(emb)
        results.append((enc["name"], emb.shape[1], tsne_xy, umap_xy))

    if not results:
        print("    [warn] no encoders loaded; nothing to plot.")
        return

    # One row per encoder, columns = [t-SNE, UMAP]. Colour = class.
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5.0 * n), squeeze=False)
    fig.suptitle(f"Encoder comparison: 2-D projections of {len(images)} "
                 "Imagenette images (colour = class)", fontsize=14, y=0.997)
    cmap = plt.get_cmap("tab10")
    handles = None
    for row, (name, dim, tsne_xy, umap_xy) in enumerate(results):
        for col, (xy, method) in enumerate([(tsne_xy, "t-SNE"), (umap_xy, "UMAP")]):
            ax = axes[row][col]
            for c in range(len(pretty)):
                m = labels == c
                ax.scatter(xy[m, 0], xy[m, 1], s=14, color=cmap(c),
                           label=pretty[c], alpha=0.8, edgecolors="none")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(method, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{name}\n({dim}-D)", fontsize=10)
        handles = axes[row][0].collections
    # Single shared legend on the right.
    fig.legend([plt.Line2D([0], [0], marker="o", linestyle="", color=cmap(c))
                for c in range(len(pretty))], pretty,
               loc="center right", fontsize=9, title="class",
               bbox_to_anchor=(1.06, 0.5))

    fig.tight_layout(rect=(0, 0, 0.98, 0.99))
    out_path = os.path.join(RESULTS_DIR, "encoder_comparison_tsne_umap.png")
    fig.savefig(out_path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    print(f"\n    Saved comparison -> {out_path}")
    print("    Look for TIGHT, well-separated blobs: that encoder captured the")
    print("    class structure best - usually the self-supervised DINO models.")


def main() -> None:
    print("=" * 60)
    print(" 03 - Pretrained Encoders (compare embeddings + t-SNE/UMAP)")
    print("=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = get_device()
    print(f"[info] Using device: {device}")

    similarity_demo(device)

    try:
        compare_encoders(device)
    except Exception as e:
        print(f"    [warn] encoder comparison skipped ({e}). Run download.py first.")

    print("\nDone. See results/embeddings/ for the encoder comparison map.")
    print("Next: 06_clip.py matches these image embeddings to TEXT.")


if __name__ == "__main__":
    main()
