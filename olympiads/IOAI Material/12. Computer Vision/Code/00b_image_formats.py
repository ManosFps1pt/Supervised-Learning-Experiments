"""
00b_image_representations.py   (read AFTER 00_python_and_tensors_primer.py)
==========================================================================
The primer taught you that an image is a grid of numbers stored in an
array/tensor. This script zooms in on ONE thing that trips up every
beginner and half the experts:

    "The SAME image can be laid out in memory several different ways,
     and every library expects a DIFFERENT layout."

If you get the layout wrong you don't get an error — you get garbage
colours, a rotated picture, or a shape mismatch deep inside a model.
So we make the layouts explicit here, and show the exact functions that
translate between them.

Everything runs instantly (no downloads, no GPU needed).

Run it:
    python 00b_image_representations.py

------------------------------------------------------------------------
THE FOUR REPRESENTATIONS YOU WILL MEET
------------------------------------------------------------------------
* MATRIX          (H, W)              a grayscale image: 1 number/pixel
* HWC array       (H, W, C)           "channels-last": NumPy / OpenCV /
                                       matplotlib / skimage use this
* CHW tensor      (C, H, W)           "channels-first": PyTorch models
                                       want this
* NCHW batch      (N, C, H, W)        a *batch* of N images: what you
                                       actually feed a network

Two more axes of confusion, independent of the shape:
* DTYPE & RANGE   uint8 in [0,255]   (how images are stored on disk)
                  float32 in [0,1]   (what networks want)
* CHANNEL ORDER   RGB (most things)  vs  BGR (OpenCV's historical quirk)
------------------------------------------------------------------------
"""

import numpy as np
import torch
import torch.nn.functional as F   # F holds stateless ops like interpolate


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(" " + title)
    print("=" * 64)


def show(name: str, t) -> None:
    """Print a tensor/array's shape and dtype — your #1 debugging habit."""
    dtype = t.dtype
    kind = "tensor" if isinstance(t, torch.Tensor) else "array"
    print(f"  {name:<22} shape={tuple(t.shape)!s:<20} dtype={dtype}  ({kind})")


def main() -> None:
    # ===================================================================
    section("1 — MATRIX: a grayscale image is a 2D grid (H, W)")
    # ===================================================================
    # One number per pixel = one brightness value. Rows are the height,
    # columns are the width. This is a "matrix" in the maths sense.
    gray = np.array([
        [  0,  64, 128],
        [ 64, 128, 192],
        [128, 192, 255],
    ], dtype=np.uint8)
    show("grayscale matrix", gray)
    print("  Read it as rows=height, cols=width. gray[row, col] = one pixel.")
    print(f"  Pixel at row 2, col 0 = {gray[2, 0]}  (0=black .. 255=white)")

    # ===================================================================
    section("2 — HWC: a colour image stacks 3 matrices (H, W, C)")
    # ===================================================================
    # A colour image = 3 grayscale grids (Red, Green, Blue) stacked along
    # a new "channel" axis that comes LAST. This "channels-last" / HWC
    # layout is what NumPy, OpenCV, matplotlib and skimage all use.
    # We'll make a small synthetic 4x6 RGB image so nothing downloads.
    rng = np.random.default_rng(0)
    hwc = rng.integers(0, 256, size=(4, 6, 3), dtype=np.uint8)  # H=4, W=6, C=3
    show("HWC colour image", hwc)
    print("  Last axis is Channels: hwc[:, :, 0]=Red plane, 1=Green, 2=Blue.")
    show("  the Red plane only", hwc[:, :, 0])   # dropping the channel -> a matrix

    # ===================================================================
    section("3 — DTYPE & RANGE: uint8 [0,255]  vs  float32 [0,1]")
    # ===================================================================
    # Images are STORED as uint8 (whole numbers 0..255). Neural networks
    # want float32 usually scaled to 0..1 (or standardized, see step 7).
    # The conversion is: cast to float, then divide by 255.
    hwc_float = hwc.astype(np.float32) / 255.0
    show("uint8 original", hwc)
    show("float32 normalized", hwc_float)
    print(f"  min={hwc_float.min():.3f}  max={hwc_float.max():.3f}  (now in [0,1])")
    print("  WHY: floats let the network do smooth math & gradients; keeping")
    print("  0..255 would make activations huge and training unstable.")

    # ===================================================================
    section("4 — NumPy array  ->  PyTorch tensor")
    # ===================================================================
    # torch.from_numpy SHARES memory (fast, zero-copy) — edits to one
    # affect the other. torch.tensor(...) COPIES. Know which you want.
    t_hwc = torch.from_numpy(hwc_float)          # still HWC, still float32
    show("torch.from_numpy(...)", t_hwc)
    print("  Same numbers, now a torch.Tensor. Layout is UNCHANGED (still HWC).")
    print("  Note: from_numpy shares memory; torch.tensor(x) makes a copy.")

    # ===================================================================
    section("5 — HWC  ->  CHW  with .permute()  (the big one)")
    # ===================================================================
    # PyTorch layers (Conv2d, pretrained models...) expect CHANNELS-FIRST:
    # (C, H, W). .permute() reorders the axes by their INDEX. HWC is axes
    # (0=H, 1=W, 2=C); we want (C, H, W) = old axes (2, 0, 1).
    t_chw = t_hwc.permute(2, 0, 1)
    show("before .permute (HWC)", t_hwc)
    show("after  .permute (CHW)", t_chw)
    print("  .permute(2,0,1) means: new axis0=old axis2, new1=old0, new2=old1.")
    print("  NB: permute returns a VIEW with weird memory order. Some ops need")
    print("  contiguous memory -> call .contiguous() after permuting if you hit")
    print("  a 'not contiguous' error.")
    t_chw = t_chw.contiguous()

    # .transpose() swaps exactly TWO axes; .permute() reorders ALL of them.
    same = t_hwc.transpose(0, 2).transpose(1, 2)   # a clumsier way to reach CHW
    print(f"  (transpose can do it too, but permute is clearer: match={torch.equal(t_chw, same.contiguous())})")

    # ===================================================================
    section("6 — CHW  ->  NCHW  with .unsqueeze()  (add a batch axis)")
    # ===================================================================
    # Models process a BATCH of images at once, so they want a 4D tensor
    # (N, C, H, W). A single image is a batch of size 1. .unsqueeze(0)
    # inserts a new axis of length 1 at position 0.
    batch = t_chw.unsqueeze(0)                    # (C,H,W) -> (1,C,H,W)
    show("single image (CHW)", t_chw)
    show("as a batch (NCHW)", batch)
    print("  .unsqueeze(0) adds the batch axis. .squeeze(0) removes it again.")
    show("  after .squeeze(0)", batch.squeeze(0))

    # Build a real multi-image batch by STACKING several images.
    # torch.stack joins along a NEW axis; torch.cat joins along an EXISTING one.
    img_a, img_b = t_chw, t_chw * 0.5
    stacked = torch.stack([img_a, img_b], dim=0)  # (2, C, H, W) — a batch of 2
    show("torch.stack of 2 imgs", stacked)
    print("  stack -> new axis (makes the batch). cat -> glue along an axis that")
    print("  already exists (e.g. concatenate feature vectors).")

    # ===================================================================
    section("7 — NORMALIZE like ImageNet models expect (mean/std)")
    # ===================================================================
    # Pretrained CV models (steps 03+) were trained on images that were
    # STANDARDIZED per channel: (pixel - mean) / std. You must apply the
    # SAME numbers or the model sees the wrong distribution.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # shape (C,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = (t_chw - mean) / std
    show("normalized CHW", normalized)
    print("  .view(3,1,1) reshapes mean/std so they BROADCAST over H and W:")
    print("  one number per channel is applied to every pixel in that channel.")
    print("  (torchvision.transforms.Normalize does exactly this for you.)")

    # ===================================================================
    section("8 — RESIZE images with F.interpolate")
    # ===================================================================
    # Networks want a fixed size (e.g. 224x224). F.interpolate resizes the
    # SPATIAL dims of an NCHW batch. It needs 4D input (hence the batch axis).
    resized = F.interpolate(batch, size=(8, 8), mode="bilinear", align_corners=False)
    show("original batch", batch)
    show("resized to 8x8", resized)
    print("  mode='bilinear' smoothly blends pixels; 'nearest' just copies the")
    print("  closest one (use nearest for label masks so you don't invent classes).")

    # ===================================================================
    section("9 — FLATTEN / view / reshape (grid -> vector)")
    # ===================================================================
    # Classifiers eventually turn a feature grid into a flat vector.
    # .flatten() collapses dims; .reshape()/.view() re-interpret the shape.
    features = torch.arange(2 * 3 * 4).float().reshape(2, 3, 4)  # (C,H,W)
    show("feature grid", features)
    show(".flatten()", features.flatten())            # -> 1D of length 24
    show(".flatten(1) keep dim0", features.flatten(1))  # -> (2, 12)
    print("  .view() needs contiguous memory & you give EVERY dim (use -1 for")
    print("  'figure it out'): features.view(2, -1) -> (2, 12). .reshape() is the")
    print("  forgiving version that copies if it must. When unsure, use reshape.")

    # ===================================================================
    section("10 — TENSOR back to a displayable image (the round trip)")
    # ===================================================================
    # To SHOW a tensor with matplotlib you must undo everything: move to CPU,
    # drop the batch axis, CHW->HWC, clamp to a valid range, scale to 0..255,
    # cast to uint8, and hand off a NumPy array.
    display = (
        batch                       # (1, C, H, W), float, on some device
        .squeeze(0)                 # -> (C, H, W)      drop batch
        .permute(1, 2, 0)           # -> (H, W, C)      channels last for plotting
        .clamp(0, 1)                # keep values valid (normalize can overshoot)
        .mul(255).round()           # 0..1 -> 0..255
        .to(torch.uint8)            # back to storage dtype
        .cpu().numpy()              # tensor -> NumPy array matplotlib understands
    )
    show("display-ready image", display)
    print("  This exact chain (.detach().cpu().permute(1,2,0).numpy()) is how you")
    print("  visualise ANY model output later in the workshop.")
    print("  torch.clamp matters: after un-normalizing, values can drift outside")
    print("  [0,1]; clamping prevents wrap-around garbage when casting to uint8.")

    # ===================================================================
    section("CHEAT-SHEET — the functions worth memorizing")
    # ===================================================================
    lines = [
        ("torch.from_numpy(a)", "NumPy array -> tensor (shares memory)"),
        (".numpy()",            "tensor -> NumPy array (CPU only)"),
        (".float() / .to(...)", "change dtype / move to CPU or GPU ('device')"),
        (".permute(2,0,1)",     "reorder ALL axes: HWC <-> CHW"),
        (".transpose(a,b)",     "swap exactly two axes"),
        (".contiguous()",       "fix memory layout after permute/transpose"),
        (".unsqueeze(0)",       "add a length-1 axis (single image -> batch)"),
        (".squeeze(0)",         "remove a length-1 axis (batch -> single image)"),
        ("torch.stack(list)",   "join tensors along a NEW axis (build a batch)"),
        ("torch.cat(list,dim)", "join along an EXISTING axis"),
        (".view(-1) / .reshape","re-interpret shape (view=fast, reshape=safe)"),
        (".flatten(start)",     "collapse dims into one (grid -> vector)"),
        (".clamp(0,1)",         "keep pixel values in range before display"),
        ("F.interpolate(x,...)","resize the H,W of an NCHW batch"),
        (".detach()",           "drop gradient tracking before .numpy()/plotting"),
        ("with torch.no_grad()","run inference without building a gradient graph"),
    ]
    for fn, what in lines:
        print(f"  {fn:<22} {what}")

    print("\nRule of thumb for this workshop:")
    print("  load (HWC uint8) -> float/255 -> permute to CHW -> unsqueeze batch")
    print("  -> normalize -> model -> (reverse it all) to view the result.")
    print("\nNext step: run  python 01_classical_cv.py")


if __name__ == "__main__":
    main()
