"""
download.py
===========
Run this ONCE before the workshop to pre-fetch EVERYTHING the numbered scripts
need — sample images, datasets, and every model weight — so the lessons then
run offline with no waiting.

    python download.py                 # grab everything (several GB)
    python download.py --skip-heavy    # skip the big models/datasets (fast)
    python download.py --only images   # images / datasets / models / torchvision

For GATED models (e.g. DINOv3) log in with your Hugging Face token first:
    export HF_TOKEN=hf_xxx  &&  python download.py
    python download.py --hf-token hf_xxx

What it fetches
---------------
1. Sample photos                 -> images/
2. Small real datasets (HF)      -> data/       (classification, CLIP, LoRA...)
3. Hugging Face model weights    -> HF cache    (classifiers, CLIP, segmentation,
                                                 detection, Stable Diffusion...)
4. torchvision + YOLO weights    -> torch cache / models/

Everything is DEFENSIVE: a failed or gated download just prints a warning and
the rest continues. Anything already cached is skipped instantly, so re-running
is cheap.
"""

import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(HERE, "images")
DATA_DIR = os.path.join(HERE, "data")
MODELS_DIR = os.path.join(HERE, "models")


# ===========================================================================
# 1) Sample images (public COCO photos), saved with simple names.
# ===========================================================================
IMAGES = {
    "cats.jpg":   "http://images.cocodataset.org/val2017/000000039769.jpg",
    "street.jpg": "http://images.cocodataset.org/val2017/000000000139.jpg",
    "dog.jpg":    "http://images.cocodataset.org/val2017/000000000285.jpg",
}


def download_images() -> None:
    print("\n[images] sample photos -> images/")
    os.makedirs(IMAGES_DIR, exist_ok=True)          # make images/ if missing
    for name, url in IMAGES.items():
        dest = os.path.join(IMAGES_DIR, name)       # where this photo will live
        if os.path.exists(dest):                    # already have it -> don't re-download
            print(f"  [skip] {name}")
            continue
        try:
            # A browser-like User-Agent header stops some servers rejecting us.
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            # Open the URL and stream the bytes straight into the local file.
            with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
                f.write(r.read())
            print(f"  [ok  ] {name}")
        except Exception as e:                      # network error -> warn, keep going
            print(f"  [warn] {name}: {e}")


# ===========================================================================
# 2) Hugging Face datasets (small slices). label_key is optional.
# ===========================================================================
DATASETS = [
    # (repo, config, split, label_key, heavy, note)
    ("johnowhitaker/imagenette2-320", None, "train[:150]", "label", False,
     "10-class natural images (~320px)"),
    ("nelorth/oxford-flowers", None, "train[:150]", "label", False,
     "102 fine-grained flower classes"),
    ("AI-Lab-Makerere/beans", None, "train[:120]", "labels", False,
     "3-class leaf disease (LoRA fine-tuning)"),
    ("AI-Lab-Makerere/beans", None, "validation[:30]", "labels", False,
     "beans validation slice"),
    ("uoft-cs/cifar100", None, "test[:200]", "fine_label", False,
     "CIFAR-100 (CLIP zero-shot accuracy)"),
    ("scene_parse_150", None, "validation[:16]", None, True,
     "ADE20K (segmentation mIoU) — larger download"),
    ("rafaelpadilla/coco2017", None, "val[:16]", None, True,
     "COCO val slice (detection mAP) — larger download"),
]


def download_datasets(skip_heavy: bool) -> None:
    print("\n[datasets] real-image datasets -> data/")
    try:
        from datasets import load_dataset          # Hugging Face `datasets` library
    except Exception as e:                          # not installed? datasets are optional
        print(f"  [warn] `datasets` not installed, skipping: {e}")
        return
    for repo, config, split, label_key, heavy, note in DATASETS:
        if heavy and skip_heavy:                    # --skip-heavy -> jump the big ones
            print(f"  [skip-heavy] {repo} ({split})")
            continue
        try:
            kw = {"cache_dir": DATA_DIR}            # cache into data/ (not the home dir)
            if repo == "scene_parse_150":           # ADE20K ships a loader script...
                kw["trust_remote_code"] = True      # ...which needs explicit permission
            # Download just the requested slice (e.g. "train[:150]") — fast + small.
            ds = load_dataset(repo, config, split=split, **kw)
            # If the label column exposes class names, show how many there are.
            extra = ""
            if label_key and label_key in ds.features and \
                    getattr(ds.features[label_key], "names", None):
                extra = f", {len(ds.features[label_key].names)} classes"
            print(f"  [ok  ] {repo} ({split}) -> {len(ds)} items{extra} — {note}")
        except Exception as e:                      # dataset moved/offline -> warn, continue
            print(f"  [warn] {repo}: {str(e).splitlines()[0][:80]}")


# ===========================================================================
# 3) Hugging Face model weights (cached in the shared HF cache).
# ===========================================================================
MODELS = [
    # (repo_id, heavy, used-by note)
    ("google/vit-base-patch16-224", False, "02 classify / 08 LoRA / 10 quant"),
    ("microsoft/resnet-18", False, "03 encoders"),
    ("microsoft/resnet-50", False, "03 encoders"),
    ("facebook/convnext-tiny-224", False, "03 encoders"),
    ("facebook/dinov2-small", False, "03 encoders / 08 LoRA"),
    ("facebook/detr-resnet-50", False, "04 detection / 11 counting"),
    ("openai/clip-vit-base-patch32", False, "06 CLIP / 09 OOD"),
    ("openai/clip-vit-base-patch16", False, "06 CLIP"),
    ("facebook/metaclip-b32-400m", False, "06 CLIP"),
    ("google/siglip-base-patch16-224", False, "06 CLIP"),
    ("nvidia/segformer-b0-finetuned-ade-512-512", False, "05 segmentation"),
    ("openmmlab/upernet-convnext-tiny", False, "05 segmentation"),
    ("openai/clip-vit-large-patch14", True, "06 CLIP (large)"),
    ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", True, "06 CLIP (OpenCLIP)"),
    ("nvidia/segformer-b2-finetuned-ade-512-512", True, "05 segmentation"),
    ("microsoft/beit-base-finetuned-ade-640-640", True, "05 segmentation"),
    ("facebook/mask2former-swin-tiny-ade-semantic", True, "05 segmentation"),
    ("Intel/dpt-large-ade", True, "05 segmentation (large)"),
    ("runwayml/stable-diffusion-v1-5", True, "07 diffusion / 08 SD LoRA (~4GB)"),
    ("lllyasviel/sd-controlnet-canny", True, "07 ControlNet"),
    # Gated (needs `huggingface-cli login` + access) — optional in 03.
    ("facebook/dinov3-vits16-pretrain-lvd1689m", True, "03 encoders (GATED)"),
]


def download_models(skip_heavy: bool) -> None:
    print("\n[models] Hugging Face weights -> HF cache")
    try:
        # snapshot_download grabs ALL files of a model repo into the shared HF
        # cache (~/.cache/huggingface). Every script's from_pretrained() then
        # loads instantly from that cache instead of hitting the network.
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"  [warn] huggingface_hub not installed, skipping: {e}")
        return
    for repo, heavy, note in MODELS:
        if heavy and skip_heavy:                    # --skip-heavy -> skip the multi-GB ones
            print(f"  [skip-heavy] {repo}")
            continue
        try:
            # ignore_patterns skips redundant formats (ONNX / TF / msgpack) we
            # never use — keeps the download smaller. (.bin/.safetensors kept.)
            snapshot_download(repo, ignore_patterns=["*.onnx", "*.msgpack", "*.h5"])
            print(f"  [ok  ] {repo:<48} — {note}")
        except Exception as e:
            first = str(e).splitlines()[0][:70]
            # A 403 / "gated" error means the repo needs an access request + login.
            gated = "  (gated: run `huggingface-cli login`)" if "gated" in str(e).lower() \
                or "403" in first else ""
            print(f"  [warn] {repo}: {first}{gated}")


# ===========================================================================
# 4) torchvision + YOLO weights (used by detection 04 and segmentation 05).
# ===========================================================================
def download_torchvision(skip_heavy: bool) -> None:
    print("\n[torchvision] detector + segmentation weights -> torch cache")
    try:
        import torchvision.models.detection as det
        import torchvision.models.segmentation as seg
    except Exception as e:
        print(f"  [warn] torchvision not installed, skipping: {e}")
        return
    builders = [
        ("ssd300_vgg16", det.ssd300_vgg16, det.SSD300_VGG16_Weights.DEFAULT),
        ("retinanet_resnet50_fpn_v2", det.retinanet_resnet50_fpn_v2,
         det.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT),
        ("fasterrcnn_resnet50_fpn_v2", det.fasterrcnn_resnet50_fpn_v2,
         det.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
        ("deeplabv3_resnet50", seg.deeplabv3_resnet50,
         seg.DeepLabV3_ResNet50_Weights.DEFAULT),
        ("fcn_resnet50", seg.fcn_resnet50, seg.FCN_ResNet50_Weights.DEFAULT),
        ("lraspp_mobilenet_v3_large", seg.lraspp_mobilenet_v3_large,
         seg.LRASPP_MobileNet_V3_Large_Weights.DEFAULT),
    ]
    for name, ctor, weights in builders:
        try:
            # Building the model WITH weights=DEFAULT downloads + caches those
            # weights as a side effect. We throw the model away; the cache stays.
            ctor(weights=weights)
            print(f"  [ok  ] {name}")
        except Exception as e:
            print(f"  [warn] {name}: {str(e).splitlines()[0][:70]}")

    # YOLOv8n (ultralytics): constructing YOLO(path) downloads the .pt weights
    # to that path if missing — we point it at models/yolov8n.pt, where 04 looks.
    try:
        from ultralytics import YOLO
        os.makedirs(MODELS_DIR, exist_ok=True)
        YOLO(os.path.join(MODELS_DIR, "yolov8n.pt"))
        print("  [ok  ] yolov8n.pt")
    except Exception as e:
        print(f"  [warn] yolov8n (install `ultralytics`): {str(e).splitlines()[0][:60]}")


# ===========================================================================
# 0) Optional Hugging Face login — needed for GATED models (e.g. DINOv3).
#    Get a token at https://huggingface.co/settings/tokens, then either:
#       export HF_TOKEN=hf_xxx        (recommended)
#       python download.py --hf-token hf_xxx
# ===========================================================================
def hf_login(token: str) -> None:
    # Prefer an explicit --hf-token, otherwise look in the usual env variables.
    token = (token or os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_TOKEN"))
    if not token:                                   # no token -> that's fine
        print("[hf] No token found — public models work fine; GATED ones "
              "(e.g. DINOv3) will be skipped.")
        print("     To enable them: set HF_TOKEN=hf_xxx  or  pass --hf-token hf_xxx")
        return
    try:
        from huggingface_hub import login, whoami
        # Store the token for this session so every download is authenticated.
        # We never print the token; add_to_git_credential=False keeps it out of git.
        login(token=token, add_to_git_credential=False)
        who = whoami(token=token).get("name", "?")  # confirm WHO we logged in as
        print(f"[hf] Logged in to Hugging Face as '{who}'.")
    except Exception as e:                          # bad token -> warn, keep going
        print(f"[hf] Login failed ({str(e).splitlines()[0][:70]}); "
              "continuing without auth.")


def parse_args(argv):
    """Tiny hand-rolled CLI parser (no argparse) -> a config dict."""
    cfg = {"skip_heavy": False, "only": None, "hf_token": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--skip-heavy":                     # skip the multi-GB downloads
            cfg["skip_heavy"] = True
        elif a == "--only" and i + 1 < len(argv):   # run just one group
            cfg["only"] = argv[i + 1]; i += 1       # consume the next token too
        elif a == "--hf-token" and i + 1 < len(argv):
            cfg["hf_token"] = argv[i + 1]; i += 1
        elif a.startswith("--hf-token="):           # also accept --hf-token=hf_xxx
            cfg["hf_token"] = a.split("=", 1)[1]
        i += 1
    return cfg


def main() -> None:
    print("=" * 68)
    print(" Computer Vision Workshop — downloader (images + datasets + models)")
    print("=" * 68)
    cfg = parse_args(sys.argv[1:])
    only, skip_heavy = cfg["only"], cfg["skip_heavy"]
    if skip_heavy:
        print("[info] --skip-heavy: skipping the largest models/datasets.")

    hf_login(cfg["hf_token"])                       # authenticate first (for gated repos)

    # `only` filters which groups run; when it's None, ALL of them run.
    if only in (None, "images"):
        download_images()
    if only in (None, "datasets"):
        download_datasets(skip_heavy)
    if only in (None, "models"):
        download_models(skip_heavy)
    if only in (None, "torchvision"):
        download_torchvision(skip_heavy)

    print("\nDone! Everything is cached. Now run the numbered scripts, e.g.:")
    print("    python 01_classical_cv.py")


if __name__ == "__main__":
    main()
