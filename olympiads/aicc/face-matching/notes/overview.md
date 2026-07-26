# Face Matching Overview

## Source

- AICC/problem URL: https://aicc-official.org/solutions/round-2/face-matching
- Kaggle URL: https://www.kaggle.com/competitions/face-matching-aicc-round-2
- Platform: Kaggle
- Contest: AICC Round 2

## Task Statement

You are given a dataset of celebrity portrait images and a file, `ref_img.csv`, containing one reference image ID for each celebrity identity. For every reference image, find the other dataset images that show the same celebrity.

Do not include the reference image itself in that reference's predicted matches.

The downloaded Kaggle archive contains 109 `.jpg` images and 15 reference rows. The AICC article prose describes the same task as a small holiday portrait set with 15 distinct VIP guests. Each celebrity appears a variable number of times, and each celebrity has at least 5 photos.

## Input Format

Expected files after extracting the Kaggle archive:

- `face-matching/images/*.jpg`: 109 image files named with 3-digit IDs, such as `000.jpg`
- `face-matching/ref_img.csv`: reference image IDs

`ref_img.csv` contains:

- `ref_img`: 3-digit image ID for a reference celebrity image

## Evaluation

Submissions are evaluated with F1 score for each reference image, then averaged across the 15 references. F1 rewards finding the correct matching images while avoiding false matches.

## Submission Format

Create `submission.csv` with exactly 15 rows:

- `ref_img`: 3-digit reference image ID
- `photos`: pipe-separated list of matching image IDs

Example shape:

```csv
ref_img,photos
042,001|002|003
089,014|018
```

Important formatting rules:
 `photos`.
- Do not include the reference image in
- Use 3-digit IDs, such as `005`, not `5`.
- Use `|` between predicted image IDs.

## Contest Restrictions

- Face-specific pretrained models are prohibited, including FaceNet, ArcFace, VGGFace, and DeepFace.
- Face-specific libraries are prohibited, including `face_recognition`, `dlib`, and MediaPipe Face.
- General-purpose pretrained models are allowed, including CLIP, ResNet, and ViT.
- OpenCV is allowed, but without additional downloads.
- Manual labeling or manually grouping the images is prohibited.
- Solutions must use less than 16 GB VRAM.
- The full notebook must run in under 30 minutes.
