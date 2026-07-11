"""
00c_dataloader_and_a_simple_network.py   (read AFTER 00b_image_representations.py)
=================================================================================
The primer showed you a SINGLE image -> a model -> some logits. Real life is
never one image. You have thousands, they live on disk, and you feed them to
the network in small groups ("batches") over and over until it LEARNS.

This script closes the gap between "I have a folder of pictures" and "I trained
a network". Two ideas do all the work:

    1. DATASET   - an object that knows how to fetch ONE example: (image, label).
    2. DATALOADER- wraps a Dataset and hands you BATCHES, shuffled, in parallel.

Then we build the smallest useful NETWORK, write the 5-line TRAINING LOOP that
every PyTorch program shares, and watch a loss number go down. That loop is the
same whether you train this toy net or a giant Vision Transformer.

Everything runs instantly on a CPU: we GENERATE a tiny synthetic dataset in
memory (coloured shapes), so nothing downloads.

Run it (use the workshop's environment):
    python 00c_dataloader_and_a_simple_network.py

------------------------------------------------------------------------
THE MENTAL MODEL
------------------------------------------------------------------------
    Dataset[i]          -> (image_tensor, label)      # one example
    DataLoader(dataset) -> yields (images, labels)     # a BATCH of them
                           shapes: (N,C,H,W), (N,)

    for each epoch (a full pass over the data):
        for each batch from the DataLoader:
            predictions = model(images)      # forward
            loss        = criterion(preds, labels)
            loss.backward()                  # gradients
            optimizer.step()                 # nudge the weights
            optimizer.zero_grad()            # reset for next batch

Learn those two boxes and that loop once; you reuse them forever.
------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Make the run reproducible: same "random" shapes and weights every time, so
# your loss numbers match this file's comments. Real training does this too.
torch.manual_seed(0)


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(" " + title)
    print("=" * 64)


def show(name: str, t) -> None:
    """Print a tensor's shape and dtype — your #1 debugging habit (from 00b)."""
    print(f"  {name:<26} shape={tuple(t.shape)!s:<18} dtype={t.dtype}")


# =====================================================================
#  A tiny image generator so we DON'T need any downloads.
# ---------------------------------------------------------------------
#  We paint one of three shapes onto a small RGB image. The SHAPE is the
#  thing we want the network to recognise, so the shape index IS the label:
#     0 = horizontal stripe, 1 = vertical stripe, 2 = filled square.
# =====================================================================
IMG_SIZE = 16                       # 16x16 pixels — small = fast
CLASSES = ["h-stripe", "v-stripe", "square"]
NUM_CLASSES = len(CLASSES)


def make_one_image(label: int, generator: torch.Generator) -> torch.Tensor:
    """Return a (C, H, W) float image in [0,1] whose pattern matches `label`."""
    # Start from light random noise so no two images are identical (like real
    # photos: same class, never pixel-perfect copies).
    img = 0.15 * torch.rand(3, IMG_SIZE, IMG_SIZE, generator=generator)
    mid = IMG_SIZE // 2
    band = slice(mid - 2, mid + 2)          # a 4-pixel-thick region

    if label == 0:                          # horizontal stripe -> bright rows
        img[:, band, :] = 1.0
    elif label == 1:                        # vertical stripe -> bright columns
        img[:, :, band] = 1.0
    else:                                   # filled square in the middle
        img[:, band, band] = 1.0
    return img


# =====================================================================
#  1 — A DATASET: teach PyTorch how to fetch ONE example.
# ---------------------------------------------------------------------
#  A map-style Dataset is any object with __len__ and __getitem__. That's
#  the WHOLE contract. Here we generate images on the fly; a real dataset
#  would open a file from disk inside __getitem__ instead.
# =====================================================================
class ShapesDataset(Dataset):
    def __init__(self, n_per_class: int = 200, seed: int = 0):
        # Pre-decide the label of every example: [0,1,2, 0,1,2, ...].
        self.labels = [i % NUM_CLASSES for i in range(n_per_class * NUM_CLASSES)]
        # One generator per dataset -> deterministic but varied images.
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        # How many examples exist. The DataLoader uses this to know when an
        # epoch is over.
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Fetch example number `idx`: build (image, label). In a real project
        # this is where you'd do: img = Image.open(path); img = transform(img).
        label = self.labels[idx]
        image = make_one_image(label, self.gen)
        return image, label            # a tuple: (tensor (C,H,W), int)


def main() -> None:
    # ===================================================================
    section("1 — DATASET: an object that returns ONE (image, label)")
    # ===================================================================
    train_ds = ShapesDataset(n_per_class=200)   # 600 training images
    test_ds  = ShapesDataset(n_per_class=40, seed=999)  # 120 unseen test images
    print(f"  len(train_ds) = {len(train_ds)}   (that's what __len__ returns)")

    # Index it like a list. __getitem__ runs and hands back one example.
    image0, label0 = train_ds[0]
    show("train_ds[0] image", image0)
    print(f"  train_ds[0] label = {label0}  -> '{CLASSES[label0]}'")
    print("  The Dataset's ONLY job: given an index, return one (image, label).")
    print("  For real data you'd open a file here; the interface stays identical.")

    # ===================================================================
    section("2 — DATALOADER: turn single examples into shuffled BATCHES")
    # ===================================================================
    # A network trains on BATCHES, not one image at a time (faster + steadier
    # gradients). The DataLoader calls Dataset[i] for you, stacks the results
    # into (N,C,H,W) + (N,) tensors, and shuffles the order each epoch.
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)
    print("  DataLoader(train_ds, batch_size=32, shuffle=True)")
    print("  * batch_size=32 : how many images per step.")
    print("  * shuffle=True  : reorder every epoch so the net can't memorise order")
    print("                    (test/eval loaders use shuffle=False).")
    print("  * num_workers=k : (optional) load batches with k parallel processes.")

    # Grab ONE batch to see the shapes the model will actually receive.
    images, labels = next(iter(train_loader))
    show("one batch of images", images)     # (32, 3, 16, 16) — the NCHW from 00b!
    show("one batch of labels", labels)     # (32,)
    print(f"  So each training step sees {images.shape[0]} images at once.")
    print(f"  Batches per epoch = ceil(600/32) = {len(train_loader)}.")

    # ===================================================================
    section("3 — A SIMPLE NETWORK: numbers in -> one logit per class out")
    # ===================================================================
    # nn.Module is the base class for every network. You define the layers in
    # __init__ and the data flow in forward(). This one is deliberately tiny:
    # flatten the image to a vector, then two Linear layers with a ReLU between.
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            in_features = 3 * IMG_SIZE * IMG_SIZE      # C*H*W flattened = 768
            self.fc1 = nn.Linear(in_features, 64)      # learnable weights + bias
            self.fc2 = nn.Linear(64, NUM_CLASSES)      # -> one logit per class

        def forward(self, x):                          # x: (N, C, H, W)
            x = x.flatten(1)                           # -> (N, 768)  keep batch dim
            x = F.relu(self.fc1(x))                    # non-linearity: lets it bend
            x = self.fc2(x)                            # -> (N, 3) raw logits
            return x                                   # NO softmax here (see below)

    model = SimpleNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"  Trainable parameters: {n_params:,}")
    print("  forward() defines the math; calling model(x) runs it (never call")
    print("  model.forward(x) directly — model(x) does extra bookkeeping).")

    # Sanity check BEFORE training: feed the batch through untrained weights.
    with torch.no_grad():                    # no_grad = 'just predict, don't learn'
        logits = model(images)
    show("model(images) logits", logits)     # (32, 3): one score per class
    print("  Logits are raw scores (any real number). softmax(logits) -> the")
    print("  probabilities you saw in the primer. We keep logits raw because the")
    print("  loss function below wants logits, not probabilities.")

    # ===================================================================
    section("4 — LOSS + OPTIMIZER: how to measure error and fix it")
    # ===================================================================
    # criterion = HOW WRONG is a prediction? CrossEntropyLoss compares the 3
    # logits against the true class index. It applies softmax internally, which
    # is exactly why forward() returns raw logits.
    criterion = nn.CrossEntropyLoss()
    # optimizer = the RULE for nudging every weight to reduce that loss. Adam is
    # a solid default. lr (learning rate) = how big each nudge is.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("  criterion = nn.CrossEntropyLoss()   -> a single 'how wrong' number")
    print("  optimizer = Adam(model.parameters(), lr=1e-3) -> adjusts the weights")
    print("  You hand the optimizer model.parameters() so it knows what to change.")

    # ===================================================================
    section("5 — THE TRAINING LOOP: the 5 lines every PyTorch program shares")
    # ===================================================================
    # An EPOCH = one full pass over the whole dataset. We do a few, and within
    # each we iterate the DataLoader batch by batch. Watch the loss fall and the
    # accuracy rise: that IS learning.
    print("  epoch |  train loss | train acc")
    print("  ------+-------------+----------")
    EPOCHS = 6
    model.train()                            # 'training mode' (matters once you
    #                                          add Dropout/BatchNorm later).
    for epoch in range(1, EPOCHS + 1):
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:  # DataLoader yields batches for us
            # ---- the 5 lines you will type for the rest of your life ----
            logits = model(images)               # 1. forward pass -> predictions
            loss = criterion(logits, labels)     # 2. how wrong were we?
            optimizer.zero_grad()                # 3. clear last batch's gradients
            loss.backward()                      # 4. backprop: dLoss/dweight
            optimizer.step()                     # 5. nudge every weight
            # -------------------------------------------------------------

            running_loss += loss.item() * images.size(0)   # .item(): tensor->float
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"  {epoch:^5} | {running_loss/total:^11.4f} | {correct/total:^8.1%}")

    print("  WHY zero_grad? PyTorch ADDS up gradients by default; without the")
    print("  reset, batch 2 would train on batch 1's leftovers too. Forget it and")
    print("  training silently breaks — this is the #1 beginner bug.")

    # ===================================================================
    section("6 — EVALUATE on data the model NEVER trained on")
    # ===================================================================
    # Accuracy on the TRAINING set can lie (the net may just memorise). The
    # honest question is how it does on FRESH images -> the test set.
    model.eval()                             # 'eval mode' (turns off Dropout etc.)
    correct, total = 0, 0
    with torch.no_grad():                    # no gradients needed for evaluation
        for images, labels in test_loader:
            preds = model(images).argmax(1)  # take the highest-logit class
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"  Test accuracy on {total} unseen images: {correct/total:.1%}")
    print("  Train mode vs eval mode + torch.no_grad() are habits worth building")
    print("  now; every model you evaluate later (02, 03, ...) uses this pattern.")

    # ===================================================================
    section("CHEAT-SHEET — the pieces you just assembled")
    # ===================================================================
    lines = [
        ("class MyDataset(Dataset)", "define __len__ and __getitem__ -> (x, y)"),
        ("__getitem__(i)",           "fetch ONE example (open file, transform)"),
        ("DataLoader(ds, batch_size)","batches + shuffling + parallel loading"),
        ("next(iter(loader))",       "peek at one batch to check shapes"),
        ("class Net(nn.Module)",     "layers in __init__, data flow in forward()"),
        ("model(x)",                 "run the forward pass (NOT model.forward)"),
        ("nn.CrossEntropyLoss()",    "classification loss (wants raw logits)"),
        ("torch.optim.Adam(params)", "the weight-update rule; lr sets step size"),
        ("optimizer.zero_grad()",    "clear old gradients (do this EVERY batch)"),
        ("loss.backward()",          "backprop: compute every gradient"),
        ("optimizer.step()",         "apply the update to the weights"),
        ("model.train()/.eval()",    "switch modes (Dropout/BatchNorm behaviour)"),
        ("with torch.no_grad()",     "evaluate/predict without building gradients"),
        (".item()",                  "1-element tensor -> plain Python number"),
        (".argmax(1)",               "logits -> predicted class index per image"),
    ]
    for fn, what in lines:
        print(f"  {fn:<28} {what}")

    print("\nThe big picture:")
    print("  Dataset (one example) -> DataLoader (batches) -> model -> loss")
    print("  -> backward -> optimizer.step().  Every training script is this loop.")
    print("\nNext step: run  python 01_classical_cv.py")


if __name__ == "__main__":
    main()
