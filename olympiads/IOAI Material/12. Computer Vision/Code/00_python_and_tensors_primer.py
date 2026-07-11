"""
00_python_and_tensors_primer.py   (START HERE!)
===============================================
This is your VERY FIRST script. It assumes you know almost nothing about
machine learning. By the end you will understand the 5 ideas that EVERY
other script in this workshop is built on:

    1. An image is just a grid of numbers.
    2. Those numbers live in an "array" (NumPy) or a "tensor" (PyTorch).
    3. Tensors can live on a CPU or a GPU ("device").
    4. A neural network is a function: numbers in -> numbers out.
    5. "Inference" = giving a trained model an input and reading its output.

Nothing here downloads a big model. It runs instantly.

Run it:
    python 00_python_and_tensors_primer.py

Read every printed line and every comment. Take your time. 🙂
"""

import numpy as np          # NumPy: the standard library for number arrays
import torch                # PyTorch: like NumPy, but for deep learning


def section(title: str) -> None:
    """Just prints a nice header so the output is easy to follow."""
    print("\n" + "=" * 60)
    print(" " + title)
    print("=" * 60)


def main() -> None:
    # ===================================================================
    section("IDEA 1 — An image is a grid of numbers")
    # ===================================================================
    # Imagine a tiny 3x3 grayscale image. Each number is a brightness:
    # 0 = black, 255 = white. That's literally all an image is.
    tiny_image = np.array([
        [  0, 128, 255],
        [128, 255, 128],
        [255, 128,   0],
    ], dtype=np.uint8)
    print("A tiny 3x3 grayscale image (0=black, 255=white):")
    print(tiny_image)
    print(f"Its shape (rows, columns) = {tiny_image.shape}")
    print("A COLOR image simply has 3 of these grids stacked: Red, Green, Blue.")
    print("So a color image has shape (height, width, 3).")

    # ===================================================================
    section("IDEA 2 — Arrays (NumPy) and Tensors (PyTorch)")
    # ===================================================================
    # A NumPy 'array' and a PyTorch 'tensor' are both just boxes of numbers.
    # Deep-learning models want TENSORS, so we convert.
    np_array = np.array([1.0, 2.0, 3.0, 4.0])
    torch_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print(f"NumPy array : {np_array}   type={type(np_array).__name__}")
    print(f"Torch tensor: {torch_tensor}   type={type(torch_tensor).__name__}")

    # Converting between them is easy — you'll do this a lot.
    back_and_forth = torch.from_numpy(np_array)          # NumPy  -> Torch
    again = back_and_forth.numpy()                        # Torch  -> NumPy
    print(f"NumPy -> Torch -> NumPy round trip: {again}")

    # Math on whole arrays happens at once (no slow Python loops!). This is
    # called 'vectorization' and it is why deep learning is fast.
    print(f"Multiply every element by 10: {torch_tensor * 10}")
    print(f"Sum of all elements        : {torch_tensor.sum().item()}")

    # ===================================================================
    section("IDEA 3 — Shapes and dimensions (the #1 beginner confusion)")
    # ===================================================================
    # 90% of deep-learning bugs are SHAPE bugs. Learn to read shapes now.
    #   scalar : a single number          shape ()
    #   vector : a list of numbers        shape (n,)
    #   matrix : a grid                   shape (rows, cols)
    #   tensor : a stack of grids         shape (a, b, c, ...)
    scalar = torch.tensor(7.0)
    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.zeros(2, 3)                # 2 rows, 3 columns of zeros
    image_batch = torch.zeros(8, 3, 224, 224) # 8 color images, 224x224 each
    for name, t in [("scalar", scalar), ("vector", vector),
                    ("matrix", matrix), ("image_batch", image_batch)]:
        print(f"{name:<12} shape = {tuple(t.shape)}")
    print("\nModels take a BATCH of images at once. The shape (8, 3, 224, 224)")
    print("means: 8 images, 3 color channels, height 224, width 224.")

    # ===================================================================
    section("IDEA 4 — 'Device': CPU vs GPU")
    # ===================================================================
    # Tensors live somewhere: on the CPU (always available) or a GPU (fast,
    # optional). Every script in this workshop picks the best one for you.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"This computer will use device: {device}")
    if device == "cpu":
        print("No GPU found — that's totally fine! Everything here runs on CPU.")
    x = torch.tensor([1.0, 2.0, 3.0]).to(device)   # move the tensor to device
    print(f"Tensor now lives on: {x.device}")

    # ===================================================================
    section("IDEA 5 — A neural network is just a function")
    # ===================================================================
    # A network takes numbers in and gives numbers out. The simplest layer
    # is a 'linear' layer: output = input @ weights + bias.
    # 'Training' is the process of finding good weights. Here we use random
    # weights just to SEE the shapes flow through.
    print("Building a tiny fake 'network': 4 inputs -> 2 outputs")
    layer = torch.nn.Linear(in_features=4, out_features=2)  # random weights
    fake_input = torch.tensor([[0.5, -1.0, 2.0, 0.3]])       # shape (1, 4)
    output = layer(fake_input)                                # shape (1, 2)
    print(f"Input  shape {tuple(fake_input.shape)} -> Output shape {tuple(output.shape)}")
    print(f"Raw output numbers (called 'logits'): {output.detach().numpy().round(3)}")

    # Real classifiers output one 'logit' per class. We turn logits into
    # probabilities with 'softmax' (all values become 0..1 and sum to 1).
    probabilities = torch.softmax(output, dim=1)
    print(f"After softmax (probabilities): {probabilities.detach().numpy().round(3)}")
    print("The class with the highest probability is the model's answer.")

    # ===================================================================
    section("YOU'RE READY!")
    # ===================================================================
    print("Recap:")
    print("  * images = grids of numbers")
    print("  * we store them in tensors")
    print("  * tensors have a shape and a device")
    print("  * a model maps input numbers -> output numbers (logits)")
    print("  * softmax turns logits into probabilities")
    print("\nNext step: run  python 01_classical_cv.py")


if __name__ == "__main__":
    main()
