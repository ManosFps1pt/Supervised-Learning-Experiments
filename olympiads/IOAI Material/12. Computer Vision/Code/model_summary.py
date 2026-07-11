"""
model_summary.py
================
A tiny shared helper used by the numbered scripts: right after they LOAD a
model, they call print_summary(model, "name") so you can SEE how big the
model is and what it's made of (layers + parameter counts).

It uses `torchinfo` (the maintained successor to the old `torchsummary`).
We DON'T pass an input tensor, so no forward pass runs - that keeps it safe
for every model here (ViT, DETR, CLIP, SegFormer, diffusion U-Nets, ...),
because each of those expects a different input format.

If torchinfo isn't installed, we fall back to a plain parameter count so the
lessons still run.
"""

from __future__ import annotations


def print_summary(model, name: str = "model", depth: int = 1) -> None:
    """Print a compact layer + parameter summary of a loaded model.

    `model` must be a torch.nn.Module. For pipelines (e.g. Stable Diffusion)
    pass the main sub-module you care about, like `pipe.unet`.
    `depth` controls how many nested levels of layers are shown.
    """
    print(f"\n[summary] {name} ------------------------------------------")
    try:
        from torchinfo import summary
    except ImportError:
        # Fallback: no torchinfo -> just count parameters by hand.
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"          {total/1e6:.1f}M parameters "
              f"({trainable/1e6:.1f}M trainable)")
        print("          (install torchinfo for a full layer breakdown)")
        return

    summary(
        model,
        depth=depth,                               # nested layer levels to show
        col_names=("num_params", "trainable"),     # no output_size => no forward
        row_settings=("var_names",),
        verbose=1,
    )
