"""
07_stable_diffusion.py
=====================
IMAGE GENERATION with Stable Diffusion — and the DIFFERENT WAYS to CONDITION it.

So far every script ANALYSED an existing image. Now we GENERATE brand new
images. Text-to-image is the famous one, but a diffusion model can be STEERED
("conditioned") by much more than words. This script shows FOUR conditioning
signals side by side so students can see what each one controls:

    TEXT              a prompt only                  -> "imagine this"
    IMAGE (img2img)   a starting picture + prompt    -> "repaint this"
    IMAGE + MASK      a picture, a hole, + prompt    -> "fill this region"
    EDGES (ControlNet) an edge map + prompt          -> "follow these lines"

The first three REUSE the Stable Diffusion 1.5 weights we already download
(no extra cost). The ControlNet edge demo pulls a small extra model (~1.4GB)
and is skipped with a note if it can't be fetched — so the script still runs.

It runs on CPU, but be warned: on CPU each image takes MINUTES. On a GPU it's
seconds. We use few inference steps to keep the demo bearable on a laptop.

Run it:
    python 07_stable_diffusion.py
    python 07_stable_diffusion.py "a red bicycle on the moon, digital art"
    python 07_stable_diffusion.py --modes text,canny         # only some modes

------------------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------------------
* DIFFUSION MODELS:
    Add noise to an image step by step until it's pure static, then train a
    network to REVERSE that. To GENERATE, start from random noise and denoise
    repeatedly until a clean image appears.

* CONDITIONING = the guidance that shapes the denoising:
    - TEXT: a CLIP text vector nudges every step toward the words.
    - IMAGE (img2img): start from a NOISED version of a real photo instead of
      pure noise, so the result keeps its layout/colours (strength = how much
      to change).
    - MASK (inpainting): only regenerate pixels inside a mask; keep the rest.
    - CONTROLNET: a second network injects a STRUCTURE map (edges, depth,
      pose...) so the output follows that structure while the text sets style.
"""

import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
GEN_DIR = os.path.join(RESULTS_DIR, "generation")   # generated images + figure live here
IMAGES_DIR = os.path.join(HERE, "images")           # source photos (conditioning inputs)
SIZE = 512                                   # generate at 512x512 (SD 1.5's native size)
INPAINT_FILL = "a vase of bright colourful flowers, photorealistic"
ALL_MODES = ["text", "img2img", "inpaint", "canny"]


def get_device_and_dtype():
    """GPU -> float16 (fast, low memory). CPU -> float32 (float16 is slow there)."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def find_sample_image() -> str:
    for name in ("dog.jpg", "cats.jpg", "street.jpg"):
        path = os.path.join(IMAGES_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No image found. Please run: python download.py")


# ---------------------------------------------------------------------------
# Helpers that BUILD conditioning signals from a photo.
# ---------------------------------------------------------------------------
def box_mask(size: int, frac: float = 0.45) -> Image.Image:
    """A black canvas with a white box in the middle = 'repaint here'."""
    mask = Image.new("L", (size, size), 0)
    bw = int(size * frac)
    x0 = (size - bw) // 2
    ImageDraw.Draw(mask).rectangle([x0, x0, x0 + bw, x0 + bw], fill=255)
    return mask


def mask_preview(init: Image.Image, mask: Image.Image) -> Image.Image:
    """Show WHERE we'll inpaint: the photo with the masked box tinted red."""
    preview = init.convert("RGB").copy()
    red = Image.new("RGB", preview.size, (255, 0, 0))
    preview.paste(Image.blend(preview, red, 0.5), (0, 0), mask)
    return preview


def canny_edges(init: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """A Canny edge map (via OpenCV) = the STRUCTURE ControlNet must follow."""
    import cv2
    arr = np.array(init.convert("RGB"))
    edges = cv2.Canny(arr, low, high)               # (H, W) 0/255
    return Image.fromarray(np.stack([edges] * 3, axis=-1))   # -> 3-channel RGB


# ---------------------------------------------------------------------------
# One function per conditioning MODE. Each returns:
#   (title, conditioning_preview_or_None, output_image)
# and reuses the already-loaded base pipeline's weights via `.components`.
# ---------------------------------------------------------------------------
def run_text(base, ctx):
    out = base(ctx["prompt"], num_inference_steps=ctx["steps"],
               guidance_scale=7.5, generator=ctx["gen"]()).images[0]
    return "TEXT only", None, out


def run_img2img(base, ctx):
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline(**base.components)
    pipe.set_progress_bar_config(disable=True)
    out = pipe(prompt=ctx["prompt"], image=ctx["init"], strength=0.65,
               guidance_scale=7.5, num_inference_steps=ctx["steps"],
               generator=ctx["gen"]()).images[0]
    return "IMAGE (img2img)", ctx["init"], out


def run_inpaint(base, ctx):
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline(**base.components)
    pipe.set_progress_bar_config(disable=True)
    out = pipe(prompt=INPAINT_FILL, image=ctx["init"], mask_image=ctx["mask"],
               num_inference_steps=ctx["steps"], guidance_scale=7.5,
               generator=ctx["gen"]()).images[0]
    return f"IMAGE + MASK (inpaint)\n-> '{INPAINT_FILL.split(',')[0]}'", \
        mask_preview(ctx["init"], ctx["mask"]), out


def run_canny(base, ctx):
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=ctx["dtype"])
    pipe = StableDiffusionControlNetPipeline(controlnet=controlnet, **base.components)
    pipe.set_progress_bar_config(disable=True)
    edges = canny_edges(ctx["init"])
    out = pipe(prompt=ctx["prompt"], image=edges,
               num_inference_steps=ctx["steps"], guidance_scale=7.5,
               generator=ctx["gen"]()).images[0]
    return "EDGES (ControlNet-Canny)", edges, out


MODE_FUNCS = {
    "text": run_text,
    "img2img": run_img2img,
    "inpaint": run_inpaint,
    "canny": run_canny,
}


def parse_args(argv):
    """prompt = first non-flag arg; --modes a,b,c selects modes (default all)."""
    prompt = "a cute robot reading a book in a cozy library, digital art"
    modes = list(ALL_MODES)
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--modes" and i + 1 < len(argv):
            modes = argv[i + 1].split(","); i += 2; continue
        if a.startswith("--modes="):
            modes = a.split("=", 1)[1].split(","); i += 1; continue
        if not a.startswith("-"):
            prompt = a
        i += 1
    modes = [m for m in modes if m in MODE_FUNCS] or list(ALL_MODES)
    return prompt, modes


def main() -> None:
    print("=" * 64)
    print(" 07 - Stable Diffusion: conditioning BEYOND text")
    print("=" * 64)

    prompt, modes = parse_args(sys.argv[1:])
    print(f"[info] Prompt : \"{prompt}\"")
    print(f"[info] Modes  : {', '.join(modes)}")

    device, dtype = get_device_and_dtype()
    print(f"[info] Device : {device} (dtype={dtype})")
    if device == "cpu":
        print("[warn] Generating on CPU takes MINUTES per image. "
              f"You picked {len(modes)} mode(s) — be patient (or use --modes).")

    # -------------------------------------------------------------------
    # STEP 1: Load ONE Stable Diffusion 1.5 pipeline. Every text-based mode
    # reuses its weights via `.components`, so only ControlNet adds a download.
    # -------------------------------------------------------------------
    from diffusers import StableDiffusionPipeline
    model_name = "runwayml/stable-diffusion-v1-5"
    print(f"[1] Loading pipeline: {model_name}")
    print("    (first run downloads ~4GB - this is the big one!)")
    base = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype).to(device)
    base.safety_checker = None          # avoid blank images on harmless prompts (edu)
    base.set_progress_bar_config(disable=True)

    # -------------------------------------------------------------------
    # STEP 2: Build the conditioning inputs (a photo, a mask) shared by the
    # image-based modes.
    # -------------------------------------------------------------------
    init = Image.open(find_sample_image()).convert("RGB").resize((SIZE, SIZE))
    mask = box_mask(SIZE)
    steps = 20 if device == "cuda" else 12
    ctx = {
        "prompt": prompt, "init": init, "mask": mask, "steps": steps,
        "dtype": dtype,
        # A FRESH generator per run (same seed) keeps every mode reproducible
        # and comparable.
        "gen": lambda: torch.Generator(device=device).manual_seed(1234),
    }

    # -------------------------------------------------------------------
    # STEP 3: Run each selected conditioning mode. Any that fails (e.g. the
    # ControlNet download offline) is skipped with a note.
    # -------------------------------------------------------------------
    print(f"[2] Generating with {len(modes)} conditioning mode(s) "
          f"({steps} steps each)...")
    os.makedirs(GEN_DIR, exist_ok=True)
    panels = []   # one (title, conditioning_preview, output) per successful mode
    for m in modes:
        print(f"    -> {m} ...", flush=True)
        t0 = time.perf_counter()
        try:
            title, cond, out = MODE_FUNCS[m](base, ctx)
            dt = time.perf_counter() - t0
            out_path = os.path.join(GEN_DIR, f"generated_{m}.png")
            out.save(out_path)
            print(f"       done in {dt:.0f}s  ->  {out_path}")
            panels.append((title, cond, out))
        except Exception as e:
            short = str(e).splitlines()[0][:80]
            hint = "  (needs internet for the ControlNet weights)" if m == "canny" else ""
            print(f"       [skip] {m} — {short}{hint}")

    if not panels:
        print("\nNo modes succeeded. Try:  python 07_stable_diffusion.py --modes text")
        return

    # -------------------------------------------------------------------
    # STEP 4: Show conditioning input (top) vs generated output (bottom)
    # for every mode, side by side.
    # -------------------------------------------------------------------
    print("[3] Building the comparison figure...")
    try:
        import matplotlib.pyplot as plt
        n = len(panels)
        fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.6), squeeze=False)
        fig.suptitle(f'Conditioning a diffusion model  —  "{prompt}"', fontsize=13)
        for col, (title, cond, out) in enumerate(panels):
            top = axes[0][col]
            if cond is None:
                top.text(0.5, 0.5, "(no image input)\ntext only", ha="center",
                         va="center", fontsize=11)
            else:
                top.imshow(cond)
            top.set_title(title, fontsize=10)
            top.axis("off")
            axes[1][col].imshow(out)
            axes[1][col].axis("off")
        axes[0][0].set_ylabel("conditioning")
        axes[1][0].set_title("generated output", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        os.makedirs(GEN_DIR, exist_ok=True)
        out_path = os.path.join(GEN_DIR, "diffusion_conditioning.png")
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        print(f"    Saved comparison figure -> {out_path}")
        plt.show()
    except Exception as e:
        print(f"    (display unavailable: {e}; individual PNGs are in {GEN_DIR})")

    print("\nDone. Same model, same prompt — but TEXT, an IMAGE, a MASK, and an")
    print("EDGE map each steer the result differently. That's conditioning.")


if __name__ == "__main__":
    main()
