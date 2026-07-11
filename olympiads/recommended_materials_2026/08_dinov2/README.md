# DINOv2: Learning Robust Visual Features without Supervision

- Source: https://arxiv.org/abs/2304.07193
- Local source: `paper.pdf`
- Extracted text: `paper_extracted.md`

## What the paper actually says

DINOv2 trains Vision Transformers to produce general-purpose image features without text labels. Its central claim is that self-supervised methods can provide strong reusable image and patch representations when trained at sufficient scale on carefully curated data.

The system builds the LVD-142M training set by deduplicating a large web-image pool and retrieving images visually close to several curated datasets. Training combines a DINO image-level student/teacher objective with an iBOT masked-patch objective. The teacher is updated as an exponential moving average of the student. Additional components stabilize and spread the representation, and a short high-resolution phase improves dense tasks without paying the full cost of high-resolution training throughout.

The resulting frozen features work well for k-nearest-neighbor classification, linear probing, retrieval, semantic segmentation and depth prediction. Patch tokens retain local information; a class token or pooled representation summarizes an image. The paper stresses that fine-tuning is optional because frozen features are already strong, which is especially relevant under competition compute limits.

## CEOAI syllabus mapping

- `3(c) Architectures`: Transformers and Vision Transformers.
- `5(c) Related architectures`: ViT and foundation-model feature extraction.
- `5(a) Processing`: resizing, patch grids, local image representations.
- `2(b) Clustering` and `2(d) Dimensionality Reduction`: useful downstream analysis of embeddings.
- `2(a) Classification`: kNN or linear heads over frozen features.

## What to retain for competition

Expect to receive embeddings rather than train DINOv2. Determine whether a tensor represents one vector per image or a grid of patch vectors. Check shape, normalize embeddings before cosine similarity, map patch indices back to image coordinates carefully, and start with kNN, prototypes, clustering, or a linear head. Do not spend these three days studying how to reproduce DINOv2 pretraining.
