# Face Matching Summary

## Source

- AICC/problem page: [AICC page](https://aicc-official.org/solutions/round-2/face-matching)
- AICC contests page: [AICC contests](https://aicc-official.org/contests)
- Kaggle competition: [Kaggle](https://www.kaggle.com/competitions/face-matching-aicc-round-2)
- GitHub solution notebook: [face-matching.ipynb](https://github.com/AI-Community-Contest/solutions/blob/main/round-2/face-matching.ipynb)
- Raw AICC URL: https://aicc-official.org/solutions/round-2/face-matching
- Raw Kaggle URL: https://www.kaggle.com/competitions/face-matching-aicc-round-2
- Platform: Kaggle
- Contest: AICC Round 2
- Difficulty: Easy
- Author listed by AICC: Stefan Asandei
- Imported material: official AICC solution notebook and Kaggle competition data archive.

## Local Artifacts

- Original notebook: `source/face-matching.ipynb`
- Working notebook copy: none; solution notebooks are preserved in `source/` only.
- Data directory: D:\projects\Supervised-Learning-Experiments\olympiads\aicc\face-matching\data
- Dataset status: downloaded
- Submission script: `submission_script.bat`

## Task Shape

- Task type: computer vision / image retrieval / identity grouping by visual similarity
- Inputs: `face-matching/images/*.jpg` containing 109 images, plus `face-matching/ref_img.csv` containing 15 reference image IDs.
- Outputs/submission format: `submission.csv` with columns `ref_img` and `photos`; `photos` is a pipe-separated list of matched image IDs.
- Metric: F1, based on the AICC article noting the reference top-5 heuristic reaches F1=0.83.

## IOAI Syllabus Coverage

- Primary coverage: Computer Vision.
- Secondary coverage: PyTorch basics, tensor manipulation, pre-trained vision encoders, vision-text encoders such as CLIP, image preprocessing, feature extraction, cosine similarity, and submission-format validation.
- Why this maps to the syllabus: the solution uses a general-purpose pretrained CLIP image encoder to embed portraits, then performs nearest-neighbor retrieval in embedding space. It directly practices using pretrained vision models without training a face-specific model.

## Notebook Data-Flow Check

- Installs packages: no
- Downloads/prepares dataset: no
- Manual download needed: no
- Evidence: solution notebook inspection found local reads for `ref_img.csv` and image files under `root_dir/images`, no Kaggle API calls, no direct downloads, and no archive extraction inside the notebook. `kaggle competitions download -c face-matching-aicc-round-2` downloaded the archive successfully after competition rules were accepted.

## Next Action

Use `notes/overview.md` and the downloaded data to create your own practice notebook when you are ready. Generate a root-level `submission.csv` with columns `ref_img,photos`, then submit with `submission_script.bat`. The AICC solution notebook is preserved in `source/face-matching.ipynb` for reference only.
