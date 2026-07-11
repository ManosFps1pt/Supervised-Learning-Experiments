"""
08_finetune_lora.py   (BONUS)
=============================
FINE-TUNING pretrained models with LoRA (Low-Rank Adaptation) — on BOTH
kinds of vision model: image CLASSIFIERS and an image GENERATOR.

Everything before this script only *used* pretrained models. Here we adapt
them to NEW tasks with our own data, cheaply, using LoRA.

    PART A  Fine-tune small CLASSIFIERS on the 'beans' plant-disease dataset
            (3 classes) — a supervised ViT AND a self-supervised DINOv2.
            Same LoRA recipe, two very different backbones, compared.

    PART B  Fine-tune the STABLE DIFFUSION generator (script 07's model) on a
            single image so it learns a new concept ("sks dog"), DreamBooth-
            style. We generate the trigger prompt BEFORE and AFTER training to
            see the concept appear.

All outputs (loss curves, accuracy chart, before/after generations) are saved
under results/lora/.

Run download.py first, then:
    python 08_finetune_lora.py                      # everything
    python 08_finetune_lora.py --skip-sd            # classifiers only (fast)
    python 08_finetune_lora.py --skip-cls           # Stable Diffusion only

------------------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------------------
* FINE-TUNING / TRANSFER LEARNING:
    Start from a pretrained model and keep training on YOUR data so it
    specialises — far cheaper than training from scratch.

* WHY LoRA (PEFT = Parameter-Efficient Fine-Tuning):
    FREEZE the millions of original weights and insert tiny trainable
    "adapter" matrices. Train only those (often ~1%), save just a few MB.

* target_modules="all-linear":
    Tells peft to adapt EVERY linear layer, so the SAME code works on any
    architecture out of the box — no need to know each model's layer names.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODELS_DIR = os.path.join(HERE, "models")
RESULTS_DIR = os.path.join(HERE, "results", "lora")
SD_MODEL = "runwayml/stable-diffusion-v1-5"

# Small classifier backbones to LoRA-finetune and compare.
CLASSIFIERS = [
    ("google/vit-base-patch16-224", "ViT-base"),      # supervised pretraining
    ("facebook/dinov2-small",       "DINOv2-small"),  # self-supervised, tiny
]


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def lora_config(modules_to_save=None):
    """One LoRA recipe reused everywhere (see 'all-linear' note in the header)."""
    from peft import LoraConfig
    return LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
        target_modules="all-linear",
        modules_to_save=modules_to_save,
    )


# ===========================================================================
# PART A — LoRA-finetune small image classifiers on 'beans'
# ===========================================================================
def finetune_classifier(model_name, disp, labels, train_ds, val_ds,
                        device, epochs, lr):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from peft import get_peft_model

    print(f"\n  [{disp}] loading + wrapping with LoRA...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name, num_labels=len(labels), ignore_mismatched_sizes=True)
    model = get_peft_model(model, lora_config(modules_to_save=["classifier"])).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"       trainable {trainable/1e6:.2f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.2f}%)")

    def collate(batch):
        images = [ex["image"].convert("RGB") for ex in batch]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        targets = torch.tensor([ex["labels"] for ex in batch])
        return pixel_values, targets

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        run = 0.0
        for pixel_values, targets in train_loader:
            pixel_values, targets = pixel_values.to(device), targets.to(device)
            loss = model(pixel_values=pixel_values, labels=targets).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())
            run += loss.item()
        print(f"       epoch {epoch}/{epochs}  avg loss {run/len(train_loader):.4f}")

    model.eval()
    correct = total_n = 0
    with torch.no_grad():
        for pixel_values, targets in val_loader:
            pixel_values, targets = pixel_values.to(device), targets.to(device)
            preds = model(pixel_values=pixel_values).logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total_n += targets.size(0)
    acc = correct / max(total_n, 1)
    print(f"       validation accuracy: {correct}/{total_n} = {100*acc:.1f}%")

    save_dir = os.path.join(MODELS_DIR, f"beans_lora_{disp}")
    model.save_pretrained(save_dir)
    print(f"       saved adapter -> {save_dir}")
    return {"name": disp, "history": history, "val_acc": acc,
            "trainable": trainable, "total": total}


def part_a_classifiers(device, epochs, lr):
    print("=" * 64)
    print(" PART A — LoRA-finetune small classifiers on 'beans'")
    print("=" * 64)
    from datasets import load_dataset

    print("[A1] Loading a small slice of the 'beans' dataset...")
    train_ds = load_dataset("AI-Lab-Makerere/beans", split="train[:120]", cache_dir=DATA_DIR)
    val_ds = load_dataset("AI-Lab-Makerere/beans", split="validation[:30]", cache_dir=DATA_DIR)
    labels = train_ds.features["labels"].names
    print(f"     Classes: {labels}")

    results = []
    for model_name, disp in CLASSIFIERS:
        try:
            results.append(finetune_classifier(model_name, disp, labels,
                                                train_ds, val_ds, device, epochs, lr))
        except Exception as e:
            print(f"  [skip] {disp} — {str(e).splitlines()[0][:70]}")

    if not results:
        return

    print("\n     " + f"{'model':<16}{'trainable%':>12}{'val acc':>10}")
    print("     " + "-" * 38)
    for r in results:
        print(f"     {r['name']:<16}{100*r['trainable']/r['total']:>11.2f}%"
              f"{100*r['val_acc']:>9.1f}%")

    # Figure: training loss curves + validation-accuracy bars.
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for r in results:
            axes[0].plot(r["history"], label=r["name"])
        axes[0].set_xlabel("training step")
        axes[0].set_ylabel("loss")
        axes[0].set_title("LoRA fine-tuning loss")
        axes[0].legend()
        names = [r["name"] for r in results]
        axes[1].bar(names, [100 * r["val_acc"] for r in results], color="#55A868")
        axes[1].set_ylabel("validation accuracy (%)")
        axes[1].set_title("Beans accuracy after LoRA")
        axes[1].set_ylim(0, 100)
        for i, r in enumerate(results):
            axes[1].text(i, 100 * r["val_acc"],
                         f"{100*r['trainable']/r['total']:.1f}% trained",
                         ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "classifier_lora.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"     Saved chart -> {out}")
    except Exception as e:
        print(f"     (plot unavailable: {e})")


# ===========================================================================
# PART B — LoRA-finetune Stable Diffusion on ONE image (DreamBooth-lite)
# ===========================================================================
def find_concept_image() -> str:
    for name in ("dog.jpg", "cats.jpg", "street.jpg"):
        path = os.path.join(HERE, "images", name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No image found. Please run: python download.py")


def part_b_stable_diffusion(device, steps):
    print("\n" + "=" * 64)
    print(" PART B — LoRA-finetune Stable Diffusion on one image")
    print("=" * 64)
    try:
        from diffusers import StableDiffusionPipeline
    except Exception as e:
        print(f"[skip] diffusers not available ({e}).")
        return
    from PIL import Image

    dtype = torch.float32
    prompt = "a photo of sks dog"
    print(f"[B1] Loading {SD_MODEL} (first run downloads ~4GB)...")
    pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL, torch_dtype=dtype).to(device)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    def generate(tag):
        g = torch.Generator(device=device).manual_seed(0)
        return pipe(prompt, num_inference_steps=20, guidance_scale=7.5, generator=g).images[0]

    print(f"[B2] BEFORE training: generating '{prompt}' (sks means nothing yet)...")
    before = generate("before")

    # Insert LoRA into the U-Net's attention; freeze everything else.
    print(f"[B3] Adding LoRA to the U-Net and training {steps} steps on the "
          "concept image...")
    from peft import LoraConfig
    unet = pipe.unet
    # The U-Net's attention layers are named to_q/to_k/to_v/to_out.0 (diffusers
    # convention), so we target those explicitly here.
    unet.add_adapter(LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0,
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    unet.train()
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"     U-Net LoRA trainable: {sum(p.numel() for p in params)/1e6:.2f}M params")
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    concept = Image.open(find_concept_image()).convert("RGB").resize((512, 512))
    x = torch.from_numpy(np.array(concept)).float().permute(2, 0, 1)[None].to(device)
    x = x / 127.5 - 1.0
    with torch.no_grad():
        latents = pipe.vae.encode(x).latent_dist.sample() * pipe.vae.config.scaling_factor
        tok = pipe.tokenizer([prompt], padding="max_length",
                             max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").to(device)
        text_emb = pipe.text_encoder(tok.input_ids)[0]

    n_ts = pipe.scheduler.config.num_train_timesteps
    for step in range(1, steps + 1):
        noise = torch.randn_like(latents)
        ts = torch.randint(0, n_ts, (1,), device=device).long()
        noisy = pipe.scheduler.add_noise(latents, noise, ts)
        pred = unet(noisy, ts, encoder_hidden_states=text_emb).sample
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == 1:
            print(f"     step {step:>3}/{steps}  loss {loss.item():.4f}")

    print(f"[B4] AFTER training: generating '{prompt}' again (same seed)...")
    unet.eval()
    after = generate("after")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
        for ax, img, title in [(axes[0], concept, "concept image\n(the one we trained on)"),
                               (axes[1], before, f"BEFORE LoRA\n'{prompt}'"),
                               (axes[2], after, f"AFTER LoRA\n'{prompt}'")]:
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle("Stable Diffusion learns a new concept via LoRA", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = os.path.join(RESULTS_DIR, "sd_lora.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"     Saved before/after figure -> {out}")
    except Exception as e:
        print(f"     (plot unavailable: {e})")

    # Persist ONLY the tiny LoRA weights (a few MB) — NOT the whole 3GB U-Net.
    try:
        save_dir = os.path.join(MODELS_DIR, "sd_dog_lora")
        unet.save_lora_adapter(save_dir)
        print(f"     Saved U-Net LoRA adapter -> {save_dir}")
    except Exception as e:
        print(f"     (could not save SD adapter: {str(e).splitlines()[0][:60]})")


# ===========================================================================
def parse_args(argv):
    cfg = {"skip_cls": False, "skip_sd": False, "epochs": 2, "lr": 1e-3,
           "sd_steps": 100}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--skip-cls":
            cfg["skip_cls"] = True
        elif a == "--skip-sd":
            cfg["skip_sd"] = True
        elif a == "--epochs" and i + 1 < len(argv):
            cfg["epochs"] = int(argv[i + 1]); i += 1
        elif a == "--sd-steps" and i + 1 < len(argv):
            cfg["sd_steps"] = int(argv[i + 1]); i += 1
        i += 1
    return cfg


def main() -> None:
    print("=" * 64)
    print(" 08 - Fine-tuning with LoRA: classifiers + Stable Diffusion (BONUS)")
    print("=" * 64)
    cfg = parse_args(sys.argv[1:])
    device = get_device()
    print(f"[info] Using device: {device}")

    if not cfg["skip_cls"]:
        part_a_classifiers(device, cfg["epochs"], cfg["lr"])
    if not cfg["skip_sd"]:
        part_b_stable_diffusion(device, cfg["sd_steps"])

    print("\nDone. Same LoRA idea, three models: two classifiers and a diffusion")
    print("generator — each adapted by training a tiny fraction of its weights.")
    print("Results saved under results/lora/.")


if __name__ == "__main__":
    main()
