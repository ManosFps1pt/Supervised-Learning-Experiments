"""
01_classical_cv.py
===================
CLASSICAL (pre-deep-learning) COMPUTER VISION.

Before neural networks took over, computer vision was done with
hand-designed math operations on the pixel grid. This script shows the
classic pipeline that engineers used for decades:

    image -> grayscale -> blur -> edges -> handcrafted features (HOG)

Run it:
    python 01_classical_cv.py

------------------------------------------------------------------------
KEY CONCEPTS (read these!)
------------------------------------------------------------------------
* PIXELS:
    A digital image is just a grid of numbers. A colour image has 3
    numbers per pixel (Red, Green, Blue), each from 0 (dark) to 255 (bright).
    So an image is really a 3D array of shape (height, width, 3).

* FILTERS (a.k.a. kernels):
    A small matrix (e.g. 3x3) that slides over the image. At each position
    we multiply neighbouring pixels by the filter and sum them up. Different
    filters do different jobs: blurring, sharpening, edge detection...
    This "slide-and-multiply" operation is called CONVOLUTION - the same
    idea that powers Convolutional Neural Networks (CNNs) later!

* EDGES:
    Edges are places where brightness changes suddenly (object boundaries).
    Detecting them is a classic first step to "understanding" an image.

* HANDCRAFTED FEATURES:
    Instead of feeding raw pixels to a classifier, people designed clever
    summaries of the image (like HOG). "Handcrafted" = a human invented the
    formula. Deep learning's big idea was to LEARN these features instead.
"""

import os
import cv2                      # OpenCV: the classic computer-vision library
import numpy as np
import matplotlib.pyplot as plt
# from skimage.feature import hog # HOG = Histogram of Oriented Gradients


def find_sample_image() -> str:
    """Return a path to an example image, or make one if none exist."""
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    for name in ("cats.jpg", "dog.jpg", "street.jpg"):
        path = os.path.join(images_dir, name)
        if os.path.exists(path):
            return path
    # Fallback: create a simple synthetic image so the script always runs.
    print("[info] No sample image found. Run download.py to get real ones.")
    os.makedirs(images_dir, exist_ok=True)
    synthetic = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(synthetic, (60, 60), (200, 200), (200, 120, 40), -1)
    cv2.circle(synthetic, (128, 128), 40, (40, 220, 220), -1)
    path = os.path.join(images_dir, "synthetic.jpg")
    cv2.imwrite(path, synthetic)
    return path


def main() -> None:
    print("=" * 60)
    print(" 01 - Classical Computer Vision")
    print("=" * 60)

    # -------------------------------------------------------------------
    # STEP 1: Load an image.
    # OpenCV loads images in BGR order (Blue, Green, Red) - a historical
    # quirk. matplotlib expects RGB, so we convert for correct colours.
    # -------------------------------------------------------------------
    path = find_sample_image()
    print(f"[1] Loading image: {path}")
    bgr = cv2.imread(path)                       # shape: (H, W, 3), values 0-255
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"    Image shape (Height, Width, Channels) = {rgb.shape}")
    print(f"    A single pixel (top-left) = {rgb[0, 0]}  <- three numbers: R, G, B")

    # -------------------------------------------------------------------
    # STEP 2: RGB -> Grayscale.
    # Many classical algorithms only care about brightness, not colour.
    # Grayscale = 1 number per pixel instead of 3.
    # -------------------------------------------------------------------
    print("[2] Converting to grayscale (colour -> brightness only)")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # shape: (H, W)

    # -------------------------------------------------------------------
    # STEP 3: Gaussian blur (a smoothing FILTER).
    # This averages each pixel with its neighbours, weighted by a bell
    # curve. It removes noise. The (7,7) is the filter size.
    # -------------------------------------------------------------------
    print("[3] Applying Gaussian blur (smoothing filter)")
    blurred = cv2.GaussianBlur(gray, (7, 7), sigmaX=1.5)

    # -------------------------------------------------------------------
    # STEP 4: Canny edge detection.
    # Canny finds strong brightness changes = edges. The two numbers are
    # thresholds: weak edges below 50 are dropped, strong edges above 150
    # are kept, and in-between edges are kept only if connected to strong ones.
    # -------------------------------------------------------------------
    print("[4] Detecting edges with Canny")
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # -------------------------------------------------------------------
    # STEP 5: HOG features (a handcrafted feature descriptor).
    # HOG chops the image into small cells and, in each cell, records the
    # dominant EDGE DIRECTIONS. The result is a vector of numbers that
    # summarises the shapes in the image. Historically great for detecting
    # people. `hog(...)` also returns a picture so we can SEE the features.
    # -------------------------------------------------------------------
    # print("[5] Computing HOG (Histogram of Oriented Gradients) features")
    # hog_vector, hog_image = hog(
    #     gray,
    #     orientations=9,              # how many edge-direction "bins"
    #     pixels_per_cell=(16, 16),    # size of each cell
    #     cells_per_block=(2, 2),      # cells are grouped and normalised
    #     visualize=True,              # also return a picture of the features
    # )
    # print(f"    HOG produced a feature vector of length {hog_vector.shape[0]}")
    # print("    (This single vector 'describes' the image for a classifier.)")

    # -------------------------------------------------------------------
    # STEP 6: Show everything side by side.
    # -------------------------------------------------------------------
    print("[6] Displaying results (close the window to finish)")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("Classical Computer Vision Pipeline", fontsize=15)

    panels = [
        (rgb,       "1. Original (RGB)",        None),
        (gray,      "2. Grayscale",             "gray"),
        (blurred,   "3. Gaussian blur",         "gray"),
        (edges,     "4. Canny edges",           "gray"),
        # (hog_image, "5. HOG features",          "gray"),
    ]
    for ax, (img, title, cmap) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    axes.flat[5].axis("off")  # the 6th cell is empty

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(path), "..", "classical_cv_result.png")
    plt.savefig(os.path.abspath(out_path), dpi=110, bbox_inches="tight")
    print(f"    Saved figure to {os.path.abspath(out_path)}")
    plt.show()
    print("\nDone. Notice: NO neural network was used here - just math on pixels!")


if __name__ == "__main__":
    main()
