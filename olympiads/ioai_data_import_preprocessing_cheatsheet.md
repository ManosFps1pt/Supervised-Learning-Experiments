# IOAI Data Import & Preprocessing Cheatsheet

The IOAI 2026 syllabus explicitly says that tasks may use **tabular data, text, images, audio, video, and time-series data**. It also expects practical handling of missing and irregular data, normalization, splitting, augmentation, tokenization, padding, embeddings and image patching. citeturn737349view0turn205780view0turn205780view1

The contest environment uses Python in JupyterLab and includes NumPy, pandas, Polars, PyArrow, h5py, scikit-learn, PyTorch, torchvision, torchaudio, Transformers, OpenCV, Pillow and Albumentations. TensorFlow and Keras are unavailable, and extra packages cannot be installed. citeturn205780view2turn950422view0

---

## 1. First identify the data

### Quick file-type decision table

| What you see | Likely content | Import with |
|---|---|---|
| `.csv`, `.tsv` | Tabular data, text metadata, image paths, labels | `pandas.read_csv()` |
| `.parquet` | Large tabular data | `pandas.read_parquet()` |
| `.json` | Nested records, annotations, configuration | `json.load()` |
| `.jsonl` | One record per line, often NLP | `pandas.read_json(..., lines=True)` |
| `.txt` | Raw text, labels, vocabulary | `Path.read_text()` |
| `.jpg`, `.png`, `.webp` | Images or masks | Pillow/OpenCV |
| Image folders | Classification or multimodal dataset | `Path.rglob()` |
| `.wav`, `.flac`, `.mp3` | Audio | `torchaudio.load()` |
| `.mp4`, `.avi`, `.mov` | Video | `torchvision.io` or OpenCV |
| `.npy` | One NumPy array | `np.load()` |
| `.npz` | Multiple NumPy arrays | `np.load()` |
| `.pt`, `.pth` | PyTorch tensors or model weights | `torch.load()` |
| `.pkl`, `.joblib` | Serialized Python/sklearn object | `joblib.load()` |
| `.h5`, `.hdf5` | Large arrays or hierarchical data | `h5py.File()` |
| `.zip` | Compressed dataset | `zipfile` or `unzip` |
| `.tar`, `.tar.gz` | Compressed dataset/models | `tarfile` |
| `.yaml`, `.yml` | Configuration or annotation data | `yaml.safe_load()` |

A CSV is not necessarily tabular model input. It may contain image filenames, audio paths, bounding boxes, text, IDs or labels.

---

# 2. Universal first cell

Run something like this before touching the model:

```python
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")

files = [p for p in DATA_DIR.rglob("*") if p.is_file()]

print("Number of files:", len(files))
print("Extensions:")
print(Counter(p.suffix.lower() for p in files))

for path in files[:30]:
    print(path, path.stat().st_size)
```

Then locate likely important files:

```python
for pattern in [
    "*train*", "*test*", "*valid*", "*val*",
    "*sample*", "*submission*", "*label*",
    "*.csv", "*.json", "*.jsonl"
]:
    print(f"\n--- {pattern} ---")
    for path in DATA_DIR.rglob(pattern):
        print(path)
```

## Questions to answer immediately

```text
What is one sample?
Where are the labels?
What is the target column?
What is the evaluation metric?
What is the required prediction format?
Are train and test stored differently?
Are there groups, users, videos or sequences that must remain together?
Is there a sample_submission file?
```

---

# 3. Compressed datasets

## ZIP

Portable Python method:

```python
from pathlib import Path
import zipfile

zip_path = Path("dataset.zip")
output_dir = Path("data")

output_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(zip_path) as z:
    z.extractall(output_dir)
```

In the Linux/JupyterLab environment:

```python
!unzip -q dataset.zip -d data
```

Inspect the archive without extracting:

```python
with zipfile.ZipFile("dataset.zip") as z:
    for name in z.namelist()[:30]:
        print(name)
```

## TAR or TAR.GZ

```python
import tarfile

with tarfile.open("dataset.tar.gz") as archive:
    archive.extractall("data")
```

Do not repeatedly extract large archives every time the notebook runs. Extraction and other repeated preprocessing can waste runtime; the technical appendix currently describes a maximum notebook evaluation runtime of 20 minutes unless a task states otherwise. citeturn950422view2

---

# 4. Tabular data

## Import

```python
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(train.shape, test.shape)
display(train.head())
train.info()
```

Other formats:

```python
tsv = pd.read_csv("data/train.tsv", sep="\t")
parquet = pd.read_parquet("data/train.parquet")
json_data = pd.read_json("data/train.json")
jsonl_data = pd.read_json("data/train.jsonl", lines=True)
```

## Essential inspection

```python
print(train.dtypes)
print(train.isna().sum().sort_values(ascending=False).head(20))
print("Duplicates:", train.duplicated().sum())

for column in train.select_dtypes(include="object"):
    print(column, train[column].nunique(), train[column].head().tolist())
```

Target separation:

```python
TARGET = "target"

X = train.drop(columns=TARGET)
y = train[TARGET]

X_test = test.copy()
```

## Typical preprocessing

| Problem | Treatment |
|---|---|
| Missing numeric values | Median or mean imputation |
| Missing categorical values | Most-frequent or `"missing"` category |
| Categorical columns | One-hot encoding, ordinal encoding or CatBoost |
| Different numerical scales | Standardization |
| Skewed positive feature | `np.log1p()` where justified |
| Date column | Extract year, month, weekday, elapsed time |
| ID column | Usually remove unless it contains meaningful structure |
| Class imbalance | Stratification, class weights or suitable metric |
| Outliers | Robust scaler, clipping or robust model |
| Duplicate rows | Investigate before removing |
| Constant columns | Remove |

## Safe sklearn pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_columns = X.select_dtypes(include="number").columns
categorical_columns = X.select_dtypes(exclude="number").columns

numeric_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_columns),
    ("cat", categorical_pipeline, categorical_columns),
])
```

Use it directly with the model:

```python
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=2000)
)
```

### When scaling matters

Usually scale for:

- Linear and logistic regression with regularization
- KNN
- SVM
- PCA and clustering
- Neural networks

Usually unnecessary for:

- Decision trees
- Random forests
- XGBoost, LightGBM and CatBoost

---

# 5. Text data

Text may be provided as:

```text
train.csv: id, text, label
train.jsonl: one text record per line
documents/*.txt
```

## Import

```python
import pandas as pd

train = pd.read_csv("data/train.csv")

texts = train["text"].fillna("").astype(str)
labels = train["label"]
```

Raw text files:

```python
from pathlib import Path

documents = [
    path.read_text(encoding="utf-8", errors="replace")
    for path in Path("data/documents").glob("*.txt")
]
```

## Fast classical baseline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000
    ),
    LogisticRegression(max_iter=2000)
)
```

## Transformer tokenization

Use only an organizer-provided or locally cached model; external model downloads are prohibited during the contest. citeturn950422view4

```python
from transformers import AutoTokenizer

MODEL_PATH = "/path/to/provided/model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

encoded = tokenizer(
    texts.tolist(),
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

print(encoded["input_ids"].shape)
print(encoded["attention_mask"].shape)
```

## Text preprocessing rules

For TF-IDF, simple normalization may help:

```python
texts = (
    texts.str.replace(r"\s+", " ", regex=True)
         .str.strip()
)
```

For pretrained transformers:

- Do not remove punctuation automatically.
- Do not remove stop words automatically.
- Do not lowercase unless the model is uncased.
- Use the model’s own tokenizer.
- Check maximum sequence length.
- Keep an attention mask.
- Consider dynamic padding for efficiency.

Expected shape:

```text
input_ids:      [batch_size, sequence_length]
attention_mask: [batch_size, sequence_length]
```

---

# 6. Image classification

Common layouts:

```text
train/
    cat/
        image1.jpg
    dog/
        image2.jpg
```

or:

```text
train.csv
id,filename,label
1,images/001.jpg,cat
```

## Import one image

```python
from PIL import Image

image = Image.open("data/images/001.jpg").convert("RGB")

print(image.size)
display(image)
```

OpenCV alternative:

```python
import cv2

image = cv2.imread("data/images/001.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
```

## Basic PyTorch preprocessing

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

The normalization values and resolution must match the pretrained model being used. Do not assume every model uses ImageNet preprocessing.

## Image checks

```python
import numpy as np

array = np.asarray(image)

print("Shape:", array.shape)
print("Dtype:", array.dtype)
print("Range:", array.min(), array.max())
```

Look for:

- Grayscale images
- RGBA images with alpha channels
- Corrupted files
- Different resolutions
- Incorrect orientation
- Duplicate images
- Class imbalance
- Train/test preprocessing mismatch

Expected model shape:

```text
[batch_size, channels, height, width]
```

---

# 7. Object detection

Detection labels usually contain:

```text
image_id, class_id, x_min, y_min, x_max, y_max
```

or YOLO format:

```text
class_id x_center y_center width height
```

YOLO coordinates are commonly normalized to the range `[0, 1]`, but verify the task documentation.

## Group annotations per image

```python
import pandas as pd

annotations = pd.read_csv("data/annotations.csv")

for image_id, rows in annotations.groupby("image_id"):
    boxes = rows[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
    labels = rows["class_id"].to_numpy()

    print(image_id, boxes.shape, labels.shape)
    break
```

Typical PyTorch target:

```python
import torch

target = {
    "boxes": torch.tensor(boxes, dtype=torch.float32),
    "labels": torch.tensor(labels, dtype=torch.int64),
}
```

## Preprocessing requirements

- Resize bounding boxes whenever the image is resized.
- Flip bounding boxes whenever the image is flipped.
- Ensure `x_min < x_max` and `y_min < y_max`.
- Clip boxes to image boundaries.
- Remove zero-area boxes.
- Confirm whether coordinates are pixels or normalized.
- Confirm whether the format is `xyxy`, `xywh` or `cxcywh`.

Never augment the image without applying the exact corresponding transformation to its boxes.

---

# 8. Image segmentation

A segmentation dataset normally has:

```text
images/001.jpg
masks/001.png
```

## Import

```python
from PIL import Image
import numpy as np

image = Image.open("data/images/001.jpg").convert("RGB")
mask = Image.open("data/masks/001.png")

image_array = np.asarray(image)
mask_array = np.asarray(mask)

print(image_array.shape)
print(mask_array.shape)
print("Mask classes:", np.unique(mask_array))
```

## Critical preprocessing rules

- Use bilinear interpolation when resizing normal images.
- Use nearest-neighbour interpolation when resizing masks.
- Do not normalize the mask.
- Do not convert mask class IDs into ordinary RGB values.
- Apply the same crop, rotation and flip to image and mask.
- Check whether masks use `0/1`, `0/255` or multiple class IDs.
- Check whether an ignore index such as `255` exists.

Typical shapes:

```text
Image: [batch, channels, height, width]
Mask:  [batch, height, width]
```

Binary segmentation usually uses one output channel. Multiclass segmentation generally uses one output channel per class.

---

# 9. Audio

Common files include WAV, FLAC and MP3, often accompanied by a metadata CSV.

## Import

```python
import torchaudio

waveform, sample_rate = torchaudio.load("data/audio/example.wav")

print("Waveform shape:", waveform.shape)
print("Sample rate:", sample_rate)
print("Duration:", waveform.shape[-1] / sample_rate)
```

`waveform.shape` is normally:

```text
[channels, number_of_samples]
```

## Convert stereo to mono

```python
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
```

## Resample

```python
TARGET_SAMPLE_RATE = 16_000

if sample_rate != TARGET_SAMPLE_RATE:
    waveform = torchaudio.functional.resample(
        waveform,
        orig_freq=sample_rate,
        new_freq=TARGET_SAMPLE_RATE
    )
```

Always use the sample rate expected by the selected audio encoder.

## Pad or crop to fixed duration

```python
import torch

duration_seconds = 5
required_length = TARGET_SAMPLE_RATE * duration_seconds

if waveform.shape[-1] < required_length:
    padding = required_length - waveform.shape[-1]
    waveform = torch.nn.functional.pad(waveform, (0, padding))
else:
    waveform = waveform[..., :required_length]
```

## Spectrogram

```python
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE,
    n_fft=400,
    hop_length=160,
    n_mels=80
)

mel = mel_transform(waveform)
log_mel = torch.log(mel + 1e-6)

print(log_mel.shape)
```

## Audio preprocessing checklist

- Inspect sample rates.
- Convert channel count consistently.
- Pad or crop variable-length recordings.
- Avoid removing meaningful silence without evidence.
- Normalize amplitude cautiously.
- Use noise/time-shift augmentation only on training data.
- Check whether labels refer to the whole clip or timestamps.
- Use the exact processor supplied with Whisper, HuBERT or another pretrained encoder.

The 2026 syllabus explicitly includes audio embeddings and pretrained audio models such as HuBERT, Whisper, Qwen-Audio and Voxtral. citeturn205780view1

---

# 10. Video

Video preprocessing combines image preprocessing with temporal sampling.

## Import using torchvision

```python
from torchvision.io import read_video

video, audio, info = read_video(
    "data/video/example.mp4",
    pts_unit="sec"
)

print("Video:", video.shape)
print("Audio:", audio.shape)
print(info)
```

The video will commonly be:

```text
[frames, height, width, channels]
```

Convert to PyTorch model order:

```python
video = video.permute(0, 3, 1, 2)

print(video.shape)  # [frames, channels, height, width]
```

## Uniformly sample frames

```python
import torch

number_of_frames = 16
indices = torch.linspace(
    0,
    len(video) - 1,
    steps=number_of_frames
).long()

sampled_video = video[indices]
```

## Typical preprocessing

- Sample a fixed number of frames.
- Preserve temporal order.
- Resize every frame consistently.
- Normalize using the encoder’s required values.
- Decide whether audio is also needed.
- Use clips rather than processing every frame when runtime is limited.
- Split by original video, not by extracted frames, to prevent leakage.

Expected batched shape:

```text
[batch, time, channels, height, width]
```

Some models expect:

```text
[batch, channels, time, height, width]
```

Check the model contract.

---

# 11. Time-series data

Common format:

```text
timestamp,sensor_1,sensor_2,target
2026-01-01 10:00,1.2,4.1,0
```

## Import and sort

```python
import pandas as pd

data = pd.read_csv(
    "data/train.csv",
    parse_dates=["timestamp"]
)

data = data.sort_values("timestamp").reset_index(drop=True)

print(data.head())
print(data["timestamp"].min(), data["timestamp"].max())
```

## Typical preprocessing

```python
data["hour"] = data["timestamp"].dt.hour
data["weekday"] = data["timestamp"].dt.weekday
data["month"] = data["timestamp"].dt.month
```

Missing values:

```python
feature_columns = ["sensor_1", "sensor_2"]

data[feature_columns] = (
    data[feature_columns]
    .ffill()
    .bfill()
)
```

Create lag features:

```python
for lag in [1, 2, 3, 6, 12]:
    data[f"sensor_1_lag_{lag}"] = data["sensor_1"].shift(lag)
```

Rolling statistics:

```python
data["sensor_1_mean_6"] = (
    data["sensor_1"]
    .rolling(6)
    .mean()
)

data["sensor_1_std_6"] = (
    data["sensor_1"]
    .rolling(6)
    .std()
)
```

## Critical rules

- Never randomly shuffle time-series splits.
- Train on earlier periods and validate on later periods.
- Do not calculate rolling features using future values.
- Fit scalers on training time periods only.
- Preserve separate sequences, devices, patients or users.
- Check whether timestamps are regular or irregular.

Example split:

```python
split_index = int(len(data) * 0.8)

train_data = data.iloc[:split_index]
validation_data = data.iloc[split_index:]
```

---

# 12. Variable-length or ragged sequences

Examples:

- Text sentences of different lengths
- Audio clips of different durations
- Videos with different frame counts
- Sensor sequences with different lengths

PyTorch padding:

```python
from torch.nn.utils.rnn import pad_sequence

sequences = [
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 5]),
    torch.tensor([6, 7, 8, 9])
]

padded = pad_sequence(
    sequences,
    batch_first=True,
    padding_value=0
)

print(padded)
```

Also create an attention or validity mask:

```python
lengths = torch.tensor([len(x) for x in sequences])

mask = (
    torch.arange(padded.shape[1])[None, :]
    < lengths[:, None]
)
```

---

# 13. Multimodal data

Examples:

- image + text
- audio + transcript
- video + audio
- tabular metadata + image
- question + image

A metadata table might look like:

```text
id,image_path,description,age,label
1,images/001.jpg,"red vehicle",12,car
```

Import each modality separately:

```python
row = train.iloc[0]

image = Image.open(DATA_DIR / row["image_path"]).convert("RGB")
text = str(row["description"])
numeric_features = row[["age"]].to_numpy(dtype="float32")
```

Typical preprocessing:

- Image → resized and normalized tensor
- Text → token IDs and attention mask
- Audio → resampled waveform or spectrogram
- Numeric data → imputed and possibly standardized
- Categorical data → encoded
- Combine modality embeddings after encoding

Always verify that records line up by ID. Never assume that filesystem order matches CSV row order.

---

# 14. NumPy arrays and embeddings

## NPY

```python
import numpy as np

X = np.load("data/features.npy", allow_pickle=False)

print(X.shape, X.dtype)
```

## NPZ

```python
archive = np.load("data/dataset.npz", allow_pickle=False)

print(archive.files)

X = archive["X"]
y = archive["y"]
```

## Precomputed embeddings

Usually shaped:

```text
[number_of_samples, embedding_dimension]
```

Check:

```python
assert len(X) == len(y)
assert np.isfinite(X).all()
```

Typical preprocessing:

- Standardize for linear models, SVM or KNN.
- Normalize vectors for cosine similarity:

```python
from sklearn.preprocessing import normalize

X_normalized = normalize(X, norm="l2")
```

- Do not apply image or text preprocessing again if the supplied data already consists of embeddings.
- Check whether training and test embeddings came from the same encoder.

---

# 15. PyTorch tensors and checkpoints

## Tensor or state dictionary

```python
import torch

checkpoint = torch.load(
    "model.pth",
    map_location="cpu",
    weights_only=True
)

print(type(checkpoint))
```

Load weights:

```python
model.load_state_dict(checkpoint)
model.eval()
```

Some checkpoints contain nested dictionaries:

```python
print(checkpoint.keys())

model.load_state_dict(checkpoint["model_state_dict"])
```

## Important distinction

```text
Model architecture: defines the network
State dictionary: contains learned parameters
Checkpoint: may contain model weights, optimizer state, epoch and metadata
```

After loading:

```python
model.eval()

with torch.inference_mode():
    output = model(example_batch)

print(output.shape)
```

---

# 16. Pickle, Joblib and HDF5

## Sklearn model

```python
import joblib

model = joblib.load("model.joblib")
predictions = model.predict(X_test)
```

## HDF5

```python
import h5py

with h5py.File("data/data.h5", "r") as file:
    print(list(file.keys()))

    X = file["X"][:]
    y = file["y"][:]
```

Avoid loading arbitrary pickle files from unknown sources because pickle can execute code. Organizer-provided files are a different trust situation, but inspect their expected purpose first.

---

# 17. Data splitting rules

## Ordinary classification

```python
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

## Regression

```python
X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

## Grouped data

Use when multiple samples belong to the same person, patient, speaker, video or object:

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_indices, validation_indices = next(
    splitter.split(X, y, groups=group_ids)
)
```

## Time-series

Split chronologically; never shuffle.

The official contest structure normally distinguishes training data with labels, validation data without labels used for scoreboard feedback, and test data used for final scoring. You should still create your own labelled validation split from the provided training data. citeturn205780view3

---

# 18. The biggest preprocessing mistake: leakage

## Wrong

```python
scaler.fit(pd.concat([train, test]))
```

```python
pca.fit(all_data)
```

```python
imputer.fit(X_validation)
```

## Correct

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)
```

The same rule applies to:

- Imputation
- Scaling
- PCA
- Feature selection
- TF-IDF vocabulary
- Target encoding
- Audio normalization statistics
- Learned image normalization
- Clustering used as feature engineering

---

# 19. Shape-contract cheatsheet

| Data | Common model input |
|---|---|
| Tabular | `[N, features]` |
| Text token IDs | `[N, sequence_length]` |
| Image classification | `[N, channels, height, width]` |
| Detection images | List of `[channels, height, width]` tensors |
| Detection boxes | `[number_of_boxes, 4]` |
| Segmentation images | `[N, channels, height, width]` |
| Segmentation masks | `[N, height, width]` |
| Audio waveform | `[N, channels, samples]` |
| Audio spectrogram | `[N, channels, frequencies, time]` |
| Video | `[N, time, channels, height, width]` |
| Time-series | `[N, time, features]` |
| Embeddings | `[N, embedding_dimension]` |

Print the shape immediately before calling the model:

```python
print(batch.shape)
output = model(batch)
print(output.shape)
```

---

# 20. Submission validation

The provided sample submission is the authoritative output contract.

```python
import pandas as pd

sample_submission = pd.read_csv("data/sample_submission.csv")
display(sample_submission.head())

print(sample_submission.shape)
print(sample_submission.columns)
print(sample_submission.dtypes)
```

Construct predictions without changing the row order:

```python
submission = sample_submission.copy()
submission["target"] = predictions
```

Validate:

```python
assert len(submission) == len(test)
assert list(submission.columns) == list(sample_submission.columns)
assert submission["target"].notna().all()
```

Save and reload:

```python
submission.to_csv("submission.csv", index=False)

check = pd.read_csv("submission.csv")

assert check.shape == sample_submission.shape
assert list(check.columns) == list(sample_submission.columns)

display(check.head())
print(check.dtypes)
```

For JSONL:

```python
submission.to_json(
    "submission.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)
```

Never finish a task without checking:

```text
Correct filename
Correct number of rows
Correct column names
Correct row order
Correct prediction type
No index column
No NaN or infinite values
Probabilities in the correct range
Correct JSON or CSV structure
File reloads successfully
```

---

# 21. IOAI preprocessing reflexes

| Situation | First reaction |
|---|---|
| Unknown dataset | Inspect files, shapes, labels and sample submission |
| Tabular classification | SimpleImputer + baseline tree/linear model |
| Tabular regression | Median imputation + linear/tree baseline |
| Text classification | TF-IDF baseline before transformer |
| Image classification | Pretrained encoder + correct transforms |
| Detection | Validate box format before training |
| Segmentation | Check mask class IDs and interpolation |
| Audio | Check sample rate, channels and duration |
| Video | Sample frames rather than loading everything |
| Time-series | Chronological split and lag features |
| Embeddings | Check shape and cosine/L2 normalization |
| Variable sequence lengths | Pad and create a validity mask |
| Multimodal task | Match all modalities by ID |
| Poor score | Verify metric and output contract before model complexity |

The fastest reliable contest workflow is:

```text
Inspect → load one sample → print shape → create split →
build simple baseline → calculate the exact metric →
predict validation/test → validate file contract → save and reload
```
