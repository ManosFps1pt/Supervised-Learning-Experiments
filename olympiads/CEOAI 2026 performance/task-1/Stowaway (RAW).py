# # Stowaway
# 
# A three-stage steganography challenge built on a tiny MNIST CNN. You will hide, certify, and protect information inside the weights of a 20,432-parameter network, and hand in three CSVs, one per reviewer.
# 
# > This is the **starter** notebook: the scaffolding, the loaders, and the exact local evaluators are all here and correct. Each part ships with a **simple baseline that runs top-to-bottom**, plus a clearly marked `YOUR CODE HERE` block where you write the real strategy. Run it once end to end, confirm the three CSVs land in `out/`, then iterate.

# ## **The story**
# 
# You work at a lab that builds image classifiers. You have been there long enough to see things you should not have. Contracts with agencies that aren't supposed to exist. Datasets scraped from places that didn't consent. Eight names on internal documents you recognised from human rights complaints. You and a few colleagues have spent the last six months quietly assembling evidence, and now it sits in eight files you encoded into eight 256-bit secret messages.
# 
# Today is your last day at the lab. You are walking out with a small image classifier you have been training, a 3-layer CNN that recognises MNIST digits. It is a trivial model, nobody at the lab will look twice. The plan is to walk out the front door with the eight messages hidden inside the weights of that model, hand it to a reporter on the outside, and let the messages explain themselves.
# 
# The lab has a three-stage review process for anything leaving the building. Three reviewers, three different things they check.
# 
# The **first reviewer** is a security engineer. They have seen people try to smuggle data out by stuffing networks with parasitic weights, so they ask you to certify which of the model's weights are redundant. They will zero out the 12,000 weights you nominate and confirm the network still classifies digits like before. If your nomination is honest, you pass. If the nomination is a cover for the actual hidden bits, the network breaks and they catch you.
# 
# The **second reviewer** is your old boss. He has been encoding messages into networks like this for years and he disappeared two weeks ago, leaving behind eight networks of his own. He used four different schemes (two networks each). Before he vanished he wrote: "if you can read all eight, you really were my student." This reviewer will only let you through if you can decode his eight networks. It is the lab's way of confirming you actually know the trade.
# 
# The **third reviewer** is the QA engineer. They run any model that leaves through a short SGD training pass on MNIST, then read your decoder. Any signal you have hidden in the weights has to survive those gradient updates and still be there for your reporter to read.
# 
# The notebook walks you through them as Part 1, Part 2, and Part 3.

# ## **How do you hide bits in a network**
# 
# A weight in a CNN is a 32-bit floating-point number like `0.003452`. The float representation has more precision than the network actually uses during training, which means there is room at the bottom to plant a signal that doesn't change what the model does.
# 
# The simplest trick is **least-significant-bit (LSB) encoding**. Multiply the weight by a million, round to an integer, and overwrite the lowest bit:
# 
# ```text
# 0.003452  ->  *1e6  ->  3452  ->  flip last bit  ->  3453  ->  /1e6  ->  0.003453
# ```
# 
# The weight changed by `1e-6`, far below normal training precision. The network's behaviour is essentially unchanged. But a one-bit signal is now embedded in it. With 256 weights you embed 256 bits, exactly one message. The decoder reads the same weights, scales, rounds, and pulls the bit off.
# 
# LSB is the simplest scheme, and it is enough for Part 2 (nothing perturbs the weights there). It is **not** enough for Part 3, where the reviewer retrains the model first: a `1e-6` nudge is orders of magnitude smaller than a single SGD step, so it vanishes. Part 3 ships with a **fixed frequency-domain decoder**, and the intended encoding spreads each bit across many weights in the DCT domain so it survives the gradient updates.

# ## **Concepts you'll see along the way**
# 
# - **Magnitude pruning.** Rank weights by `|weight|`. Weights with tiny absolute values contribute little to forward activations, so they are natural candidates to zero out. Used in Part 1.
# - **Fisher information.** The "importance" of a weight to the loss: the squared partial derivative, averaged over a few examples. High Fisher means the loss is very sensitive to this weight. Low Fisher means the network barely cares. A more informed ranking than magnitude, and often agrees with it on the bottom slice. An alternative for Part 1.
# - **`dct` (1D Discrete Cosine Transform).** Decomposes a sequence of values into a sum of cosines at different frequencies. Coefficient `k` is the amplitude of the `k`-th cosine. Nudging a few mid-band coefficients spreads the change across *many* weights when you transform back, which is exactly what makes a signal robust to retraining. Part 3's decoder is a DCT read.
# - **Repetition coding (majority vote).** Repeat each data bit across a block of `R` coefficients and vote by majority when decoding. Averaging over a block beats down the noise the SGD steps (and the grader's own additive noise) inject. Part 3's fixed decoder sums `R = 61` coefficients per bit and thresholds at zero, so encode each bit as a constant-sign block.

# ## **Imports**
# 
# We pin the random seed for reproducibility. The grader uses its own seeds. The seed below only affects your local exploration.

def print_dir(j):
    print(type(j))
    for i in dir(j):
        if i.startswith("_"):
            continue
        print(i)

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dct, idct

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# All artifacts land here.
DATA = Path("data")
OUT = Path("out")
OUT.mkdir(exist_ok=True)

# ## **The data**
# 
# The starter-kit files are provided under `./data` (you are offline during the contest, so
# there is nothing to download). Expected contents:
# 
# ```text
# baseline_cnn.pt            the frozen 3-layer CNN you carry past the inspections
# reference_messages.json    the 8 messages YOU encode in Part 3 (public, deterministic)
# decoder_grammar.md         reference for the Part 2 decoder primitives (read it)
# encoded_models/            model_00.pt .. model_07.pt, the 8 networks for Part 2
# mnist_val_sample.npz       1000 MNIST val images -> local 1 agreement self-check
# mnist_finetune_batch.npz   public finetune batch -> local 3 robustness self-check
# ```
# 

from pathlib import Path

DATA = Path("data")
REQUIRED = ["baseline_cnn.pt", "reference_messages.json", "decoder_grammar.md",
            "mnist_val_sample.npz", "mnist_finetune_batch.npz"]

_missing = [f for f in REQUIRED if not (DATA / f).exists()]
_enc = DATA / "encoded_models"
if not (_enc.is_dir() and len(list(_enc.glob("model_*.pt"))) == 8):
    _missing.append("encoded_models/model_00.pt .. model_07.pt")
assert not _missing, f"Starter-kit files missing under ./data: {_missing}"
print("Starter-kit files found under ./data.")

# ## **Native submission format**
# 
# Every part is uploaded as a **CSV with exactly three columns**: `subtaskID,datapointID,answer`. When you upload a single part on its own, `subtaskID = 1` for every row. What `datapointID` and `answer` mean is fixed per part and spelled out where each part starts. The tiny helper below is the only writer you need.

import csv

def write_submission_csv(path, datapoint_ids, answers, subtask_id=1):
    """Write the native CSV: subtaskID,datapointID,answer (one row per datapoint).
    `answers` may be ints (subtasks 1/2) or floats (subtask 3)."""
    datapoint_ids = list(datapoint_ids)
    answers = list(answers)
    assert len(datapoint_ids) == len(answers), "datapointID / answer length mismatch"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subtaskID", "datapointID", "answer"])
        for d, a in zip(datapoint_ids, answers):
            w.writerow([subtask_id, int(d), a])
    kb = path.stat().st_size / 1024
    print(f"Wrote {path} ({len(datapoint_ids)} rows, {kb:.1f} KB)")
    return path

# ## **Load the model**
# 
# The classifier is a 3-layer CNN: two conv layers with maxpool, then a linear layer to 10 classes. The cell defines it and loads the trained baseline. The three prunable/encodable tensors and their sizes:
# 
# | tensor | shape | weights |
# |---|---|---|
# | `conv1.weight` | (16, 1, 3, 3) | 144 |
# | `conv2.weight` | (32, 16, 3, 3) | 4608 |
# | `fc.weight` | (10, 1568) | 15680 |
# 
# Total prunable weights: **20,432**. `fc.weight` alone holds 15,680 of them, which is where Part 3 writes.

NUM_CLASSES = 10
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

class StowawayCNN(nn.Module):
    """Three-layer MNIST CNN. fc.weight holds 15,680 of the 20,432 prunable weights."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, NUM_CLASSES)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        return self.fc(x)

def normalise(images_uint8):
    return (images_uint8.astype(np.float32) / 255.0 - MNIST_MEAN) / MNIST_STD

def load_baseline() -> nn.Module:
    obj = torch.load(DATA / "baseline_cnn.pt", map_location="cpu", weights_only=True)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    m = StowawayCNN()
    m.load_state_dict(sd)
    m.eval()
    return m.to(DEVICE)

baseline = load_baseline()
n_params = sum(p.numel() for p in baseline.parameters())
print(f"Baseline parameters: {n_params:,}")
for name in ["conv1.weight", "conv2.weight", "fc.weight"]:
    print(f"  {name}: {dict(baseline.named_parameters())[name].numel()} weights")

# Numpy-friendly view of the baseline state dict (CPU tensors).
baseline_state = {k: v.detach().cpu() for k, v in baseline.state_dict().items()}

# The three prunable tensors, in the **canonical order** used to build the global weight index for Part 1. `PRUNABLE` and `global_layout()` are reused by the 1 self-check and CSV writer, so define them once here.

PRUNABLE = ["conv1.weight", "conv2.weight", "fc.weight"]  # canonical order for the 1 global index
SIZES = [baseline_state[n].numel() for n in PRUNABLE]     # [144, 4608, 15680]
N_PRUNABLE = sum(SIZES)                                    # 20432
OFFSETS = np.cumsum([0] + SIZES)[:-1]                     # [0, 144, 4752]
assert N_PRUNABLE == 20432 and list(OFFSETS) == [0, 144, 4752]

def global_layout():
    """Return (params, locals) arrays of length 20432: global index g -> (param name, local index).
    conv1.weight -> 0..143, conv2.weight -> 144..4751, fc.weight -> 4752..20431."""
    params, locs = [], []
    for name, size in zip(PRUNABLE, SIZES):
        params += [name] * size
        locs += list(range(size))
    return np.array(params), np.array(locs)

GP, GL = global_layout()
print("global index layout:",
      "conv1 [0,144)", "conv2 [144,4752)", "fc [4752,20432)")

# ## **Part 1: The first reviewer (prunability)**
# 
# The first reviewer certifies that the classifier still works after pruning. You nominate **exactly 12,000 of the 20,432 weights** to set to zero. If the pruned network still labels digits the way the original does, you pass. If you nominated the weights your signal actually lives in, the network breaks and you are caught.
# 
# **Native CSV for this part.** One row per prunable weight, `20,432` rows total:
# 
# - `datapointID` = the **global weight index** `g` (0..20431), in the canonical order `conv1.weight` (0..143), then `conv2.weight` (144..4751), then `fc.weight` (4752..20431).
# - `answer` = `1` to prune this weight, `0` to keep it. Exactly **12,000** ones.
# 
# **Scoring.** The grader zeroes your 12,000 nominated weights and measures top-1 agreement against the unperturbed baseline on a hidden holdout. The public grader scores overlap with a Fisher *gold set*, but prediction agreement is the honest functional proxy and it is what you can measure locally: `score ≈ 100 * clamp((agreement - 0.50) / (1 - 0.50), 0, 1)`. Pruning ~59% of the network at random collapses it (agreement ~chance, score ~0). Ranking by importance and pruning the **lowest** keeps it intact (score ~100).

# ### Local self-check (subtask 1)
# 
# The evaluator below is the honest functional proxy: zero the chosen weights in a copy of the model and measure prediction agreement vs the baseline on the provided val sample. Iterate against it, then submit once.

val = np.load(DATA / "mnist_val_sample.npz")
Xv = torch.from_numpy(normalise(val["images"])).to(DEVICE)
with torch.no_grad():
    BASE_PRED = baseline(Xv).argmax(1)

def eval_1(prune_mask):
    """prune_mask: bool/0-1 array of length 20432 in canonical global order (1 = zero this weight).
    Returns (agreement, estimated_score) on the val sample."""
    prune_mask = np.asarray(prune_mask).astype(bool)
    assert prune_mask.shape == (N_PRUNABLE,), f"mask must be length {N_PRUNABLE}"
    pruned = StowawayCNN().to(DEVICE)
    pruned.load_state_dict(baseline.state_dict())
    psd = pruned.state_dict()
    for name, off, size in zip(PRUNABLE, OFFSETS, SIZES):
        seg = prune_mask[off:off + size]
        flat = psd[name].view(-1)
        flat[torch.from_numpy(np.nonzero(seg)[0])] = 0.0
    pruned.load_state_dict(psd)
    pruned.eval()
    with torch.no_grad():
        pred = pruned(Xv).argmax(1)
    agree = (pred == BASE_PRED).float().mean().item()
    score = 100 * max(0.0, min(1.0, (agree - 0.50) / (1 - 0.50)))
    return agree, score

# ### Your strategy (subtask 1)
# 
# Return a length-20,432 mask with exactly 12,000 ones marking the weights to zero. The baseline below is a runnable starting point, and the local self-check estimates your score offline so you can iterate before you submit.

# ===========================================================================================================================================================================
# YOUR CODE HERE (subtask 1)
# ----------------------------------------------------------------------------
# Produce `prune_mask`: a length-20432 array in canonical global order with
# EXACTLY 12000 ones (the weights to zero). The placeholder below is trivial
# (a contiguous index window) and scores ~5. Replace it with a real importance
# estimate -- weight magnitude, Fisher information, gradient x weight, etc. --
# to climb toward the full 30. The local self-check reports your estimated score.
# ===========================================================================================================================================================================
from copy import deepcopy

def choose_prune_mask(state):
    # --- trivial placeholder: prune weights at indices 2134..14134 (REPLACE ME) ---
    # print_dir((val))
    # print((val["images"]))
    # print_dir(torch.nn)
    # torch.autograd.set_detect_anomaly(True)
    # print_dir(np)
    k = baseline.state_dict()
    # print_dir(x)
    # print(k["conv2.weight"].flatten().shape)
    # j = np.hstack((k["conv1.weight"].detach().cpu().numpy(), k["conv2.weight"].detach().cpu().numpy(), k["fc.weight"].detach().cpu().numpy()))
    x = torch.hstack((k["conv1.weight"].flatten(), k["conv2.weight"].flatten(), k["fc.weight"].flatten())).tolist()
    res = []
    for i in x:
        res.append(abs(i))
    # res.sort()
    res = np.array(res)
    idxs = res.argsort()[:12000]
    mask = np.array([1 if i in idxs else 0 for i in range(len(res))])
    print(mask.sum())

    # y_pred = baseline(torch.Tensor(val["images"]).to(DEVICE))
    # mask = np.zeros(N_PRUNABLE, dtype=np.uint8)
    # mask[2134:14134] = 1
    return mask

prune_mask_1 = choose_prune_mask(baseline_state)
assert prune_mask_1.shape == (N_PRUNABLE,)
assert int(prune_mask_1.sum()) == 12000, f"need exactly 12000 ones, got {int(prune_mask_1.sum())}"

agree, est = eval_1(prune_mask_1)
print(f"Local agreement on val sample: {agree:.3f}  ->  estimated 1 score ~{est:.0f}")

# ### Write the subtask-1 submission

# datapointID = global weight index 0..20431; answer = mask bit (1 prune / 0 keep).
write_submission_csv(
    OUT / "submission_1.csv",
    datapoint_ids=range(N_PRUNABLE),
    answers=[int(b) for b in prune_mask_1],
)
assert sum(int(b) for b in prune_mask_1) == 12000

# ## **Part 2: The second reviewer (decoder)**
# 
# Your boss encoded eight 256-bit messages into eight copies of the network (`data/encoded_models/model_00.pt` .. `model_07.pt`) using **four LSB-family schemes** (two networks each). He didn't tell you the recipes. Your job is to recover all 256 bits of each message.
# 
# Every scheme is the same pipeline with three free fields:
# 
# ```text
# select_param(name) -> flatten -> slice(start, start+256) -> read bit `b` of round(w * 1e6)
# ```
# 
# Only `name`, the 256-wide slice `start`, and the bit position `b` differ. Read `decoder_grammar.md` for the primitive names. The search space is tiny. The cell below recovers `(param, start, bit)` structurally from the integer delta against the baseline (the honest, offline way, no leaderboard guessing).
# 
# **Native CSV for this part.** One row per (model, bit), `8 * 256 = 2048` rows:
# 
# - `datapointID` = `model * 256 + bit`, `model` in 0..7, `bit` in 0..255.
# - `answer` = the recovered bit `0`/`1`.

# ### Scheme detection + verify helper (subtask 2)
# 
# `read_scheme` recovers the exact `(param, start, stop, bit)` from a model by XOR-ing the integer-domain weights against the baseline. `decode_with_scheme` then reads the 256 bits. Both are used by the self-check and the writer, so they double as your ground truth: if your decoder points at the same fields, it is correct.

SCALE = 1e6  # the encoder's quantize_lsb scale (stated in the brief)
ENC_DIR = DATA / "encoded_models"

def to_ints(state):
    return {k: np.rint(v.detach().cpu().numpy().reshape(-1).astype(np.float64) * SCALE).astype(np.int64)
            for k, v in state.items() if v.dtype.is_floating_point}

BASE_INTS = to_ints(baseline_state)

def read_scheme(model_state):
    """Recover (param, start, stop, bit) structurally from the integer delta.
    The float |delta| is NOT a reliable bit readout (re-quantizing adds ~5e-7 of
    noise that blurs the 1e-6 bit-0 step); the integer XOR is exact."""
    mi = to_ints(model_state)
    param = max(BASE_INTS, key=lambda k: int(np.count_nonzero(BASE_INTS[k] ^ mi[k])))
    xor = BASE_INTS[param] ^ mi[param]
    diff = np.nonzero(xor)[0]                       # planted positions
    bit = int(xor[diff[0]]).bit_length() - 1        # XOR is exactly 1<<bit at every planted weight
    start = (int(diff.min()) // 256) * 256          # slices are 256-wide, aligned to 256
    return param, start, start + 256, bit

def decode_with_scheme(model_state, param, start, bit):
    """Read 256 bits: bit `bit` of round(w*1e6) over param[start:start+256]."""
    ints = to_ints(model_state)[param][start:start + 256]
    return ((ints >> bit) & 1).astype(np.uint8)

def verify_decoder(model_state, param, start, stop, bit):
    p, s0, s1, b = read_scheme(model_state)
    return (param == p and start == s0 and stop == s1 and bit == b)

print(f"{'model':>5}  {'param':<13}  {'slice':<15}  {'bit':>3}")
load_encoded = lambda i: torch.load(ENC_DIR / f"model_{i:02d}.pt", map_location="cpu", weights_only=True)
for m_id in range(8):
    p, s0, s1, b = read_scheme(load_encoded(m_id))
    print(f"{m_id:>5}  {p:<13}  [{s0}, {s1}){'':<3}  {b:>3}")

# ### Your strategy (subtask 2)
# 
# Read the table above and fill in one `(param, start, bit)` per model. The baseline points every decoder at `fc.weight[0:256]` bit 0 (scheme A). Replace the ones that don't match the table. The self-check tells you your exact 2 score offline.

# ============================================================================
# YOUR CODE HERE (subtask 2)
# ----------------------------------------------------------------------------
# Fill `my_schemes` with the (param, start, bit) for each model, read off the
# table printed above. The un-edited default decodes NOTHING (all zeros -> ~0).
# Slices are 256-wide and 256-aligned. Set USE_MY_SCHEMES = True once your
# schemes are correct, to decode for real.
# ============================================================================

my_schemes = {
    0: ("fc.weight", 0, 0),   # <- replace each entry with the scheme from the table above
    1: ("fc.weight", 0, 0),
    2: ("fc.weight", 0, 0),
    3: ("fc.weight", 0, 0),
    4: ("fc.weight", 0, 0),
    5: ("fc.weight", 0, 0),
    6: ("fc.weight", 0, 0),
    7: ("fc.weight", 0, 0),
}

USE_MY_SCHEMES = False   # flip to True after all 8 schemes below verify OK

recovered_bits_2 = {}
ok = 0
for m_id in range(8):
    st = load_encoded(m_id)
    param, start, bit = my_schemes[m_id]
    good = verify_decoder(st, param, start, start + 256, bit)
    ok += int(good)
    recovered_bits_2[m_id] = (decode_with_scheme(st, param, start, bit)
                               if USE_MY_SCHEMES else np.zeros(256, dtype=np.uint8))
    print(f"  model_{m_id:02d}: {'OK' if good else 'MISMATCH -- fix (param,start,bit)'}")
print(f"Self-check: {ok}/8 schemes verified. Set USE_MY_SCHEMES=True once all are OK.")

# ### Write the subtask-2 submission

# datapointID = model*256 + bit; answer = recovered bit.
dp_ids, answers = [], []
for m_id in range(8):
    bits = recovered_bits_2[m_id]
    assert bits.shape == (256,)
    for bit_idx in range(256):
        dp_ids.append(m_id * 256 + bit_idx)
        answers.append(int(bits[bit_idx]))
write_submission_csv(OUT / "submission_2.csv", dp_ids, answers)

# ## **Part 3: The third reviewer (survival)**
# 
# The third reviewer retrains each of your eight modified networks with a short SGD pass on MNIST, then reads your decoder. Whatever you hid has to survive the gradient updates. A naive `1e-6` LSB nudge is wiped out immediately and decodes at chance (~0.5).
# 
# **This part has a FIXED decoder** (below). You do not submit a decoder. You submit, for every fc.weight, the **delta** you added to the baseline, and the grader runs its fixed DCT decoder on those deltas after retraining.
# 
# **Native CSV for this part.** One row per (model, fc.weight index), `8 * 15680 = 125,440` rows:
# 
# - `datapointID` = `model * 15680 + widx`, `widx` indexes the flattened `fc.weight` (0..15679).
# - `answer` = `delta = w_encoded - w_baseline` (a float).
# - **Constraint:** `|delta| <= 0.02` for every weight. If any weight in a network exceeds it, that entire network scores 0 for this part.
# 
# The eight messages are public (`reference_messages.json`), so this part is fully self-testable.

# ### The grader's FIXED decoder (subtask 3)
# 
# Shipped **verbatim**: torch-free, `scipy.fft.dct`, `norm='ortho'`. It adds a fixed dose of noise (`SIG = 0.001`) with a fixed seed, DCT-transforms, and for each of 256 bits sums a contiguous band of `R = 61` coefficients starting at `OFFSET = 64`, thresholding at zero. Do **not** edit this cell.

# ---- GRADER'S FIXED NATIVE DECODER (verbatim -- do not modify) ---------------
import numpy as np
from scipy.fft import dct

OFFSET, R, SIG = 64, 61, 0.001
def native_decode(delta, seed=7012001):
    rng = np.random.default_rng(seed)
    spec = dct(delta + rng.normal(0, SIG, delta.size), norm='ortho')
    return np.array([1 if spec[OFFSET+i*R:OFFSET+i*R+R].sum()>0 else 0 for i in range(256)])
# ------------------------------------------------------------------------------
# The 256 bands tile [OFFSET, OFFSET + 256*R) = [64, 15680), i.e. all of
# fc.weight's 15680 DCT coefficients above the first 64. Band i is
# spec[64 + 61*i : 64 + 61*i + 61].
assert OFFSET + 256 * R == 15680
print("native_decode ready; 256 bands cover DCT coeffs [64, 15680).")

# ### Load messages + finetune self-check (subtask 3)
# 
# `ref_bits` are the 8 public messages. `eval_3` runs the grader's finetune procedure on the public batch, applies the fixed decoder, and reports bit accuracy and an estimated score. Different resample seeds give a robustness spread. The private ranking uses different images from the same distribution, so aim for a high, stable spread rather than fitting one batch.

with open(DATA / "reference_messages.json") as f:
    ref_bits = np.asarray(json.load(f)["messages"], dtype=np.uint8)
assert ref_bits.shape == (8, 256)
print(f"Reference messages: {ref_bits.shape}; first 16 bits of msg 0: {ref_bits[0, :16]}")

FC = "fc.weight"
FC_N = baseline_state[FC].numel()          # 15680
FC_SHAPE = tuple(baseline_state[FC].shape)  # (10, 1568)
BASE_FC = baseline_state[FC].detach().cpu().numpy().reshape(-1).astype(np.float64)

_ft = np.load(DATA / "mnist_finetune_batch.npz")
_Xft = torch.from_numpy(normalise(_ft["images"])).to(DEVICE)
_yft = torch.from_numpy(_ft["labels"].astype(np.int64)).to(DEVICE)
_K, _LR, _BATCH, _GRADER_SEED, _CEIL = 10, 0.01, 64, 7_012_001, 0.90

def _finetune_one(delta_flat, seed):
    """Apply delta to fc.weight, finetune K SGD steps on the public batch, return post-finetune delta."""
    torch.manual_seed(_GRADER_SEED)
    m = StowawayCNN().to(DEVICE)
    sd = {k: v.clone() for k, v in baseline.state_dict().items()}
    sd[FC] = sd[FC] + torch.from_numpy(delta_flat.astype(np.float32)).reshape(FC_SHAPE).to(DEVICE)
    m.load_state_dict(sd); m.train()
    opt = torch.optim.SGD(m.parameters(), lr=_LR, momentum=0.9)
    rng = np.random.default_rng(seed)
    for _ in range(_K):
        idx = rng.choice(_Xft.shape[0], size=_BATCH, replace=False)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(_Xft[idx]), _yft[idx]).backward(); opt.step()
    post_fc = m.state_dict()[FC].detach().cpu().numpy().reshape(-1).astype(np.float64)
    return post_fc - BASE_FC

def eval_3(deltas, seed=_GRADER_SEED):
    """deltas: (8, 15680) float array. Finetune each, decode with the FIXED grader decoder,
    return (mean_bit_acc, estimated_score)."""
    deltas = np.asarray(deltas, dtype=np.float64)
    assert deltas.shape == (8, FC_N)
    assert np.max(np.abs(deltas)) <= 0.02 + 1e-9, "|delta| exceeds 0.02 constraint"
    accs = []
    for i in range(8):
        post_delta = _finetune_one(deltas[i], seed)
        bits = native_decode(post_delta)
        accs.append(float((bits == ref_bits[i]).mean()))
    mean_acc = float(np.mean(accs))
    score = 100 * float(np.mean([max(0, min(1, (a - 0.5) / (_CEIL - 0.5))) for a in accs]))
    return mean_acc, score

# ### How to approach 3
# 
# The finetune moves each weight by about a gradient step, on the order of `1e-3`, while a naive LSB nudge is `1e-6`, so that signal is gone after the first step. The fixed decoder above sums a band of 61 coefficients per bit and reads the sign, after adding noise of standard deviation `0.001`. A few things are worth working out before you write `encode_message`.
# 
# - For the sign of a sum to be reliable, how large must the signal be relative to the noise? Summing a band of 61 coefficients moves the signal-to-noise ratio in your favour. Work out by how much.
# - The decoder takes `dct(delta)`. If you want a particular pattern in the DCT coefficients, the inverse transform (`scipy.fft.idct`, with the same `norm='ortho'`) turns that pattern back into weight deltas.
# - Every delta is capped at `|delta| <= 0.02`. Spreading one bit across many coefficients keeps each individual weight small while the summed signal stays large.
# 
# Test any candidate with `native_decode` before finetuning, and with `eval_3` for survival. The robustness spread across resampled finetunes tells you whether it will hold on the private batch.

# ### Your strategy (subtask 3)
# 
# The baseline is naive LSB, a `1e-6` nudge on the first 256 weights. It runs, and it decodes at about 0.5 after finetuning. Write a stronger `encode_message` in the block below, and use the fixed decoder above to self-check before you submit.

# ============================================================================
# YOUR CODE HERE (subtask 3)
# ----------------------------------------------------------------------------
# Return `delta_flat` (length 15680) encoding `msg` (256 bits) so that
# native_decode recovers it AFTER the finetune, within |delta| <= 0.02.
# The baseline below is naive LSB and decodes at chance after finetuning.
# ============================================================================

def encode_message(msg):
    # --- naive LSB baseline (dies after finetune) ---
    delta = np.zeros(FC_N, dtype=np.float64)
    delta[:256] = np.where(np.asarray(msg) == 1, 1e-6, -1e-6)
    return delta

deltas_3 = np.stack([encode_message(ref_bits[i]) for i in range(8)])
assert deltas_3.shape == (8, FC_N)
assert np.max(np.abs(deltas_3)) <= 0.02 + 1e-9, "|delta| exceeds the 0.02 constraint"

# Sanity: does the FIXED decoder read the message back BEFORE any finetune?
_clean = np.mean([(native_decode(deltas_3[i]) == ref_bits[i]).mean() for i in range(8)])
print(f"Clean decode (no finetune) bit-acc: {_clean:.3f}  (naive baseline ~0.5; DCT ~1.0)")

# ---- Local self-test: finetune on the PUBLIC batch, decode, score ------------
_acc, _score = eval_3(deltas_3, seed=_GRADER_SEED)
print(f"Public-batch bit-acc {_acc:.3f}  ->  estimated 3 score ~{_score:.0f}")
_spread = [eval_3(deltas_3, seed=s)[1] for s in (1, 2, 3)]
print(f"Robustness across resampled finetunes: {min(_spread):.0f}-{max(_spread):.0f} (mean {np.mean(_spread):.0f})")
print("Private ranking uses different images -- aim for a high, STABLE spread.")

# ### Write the subtask-3 submission

# datapointID = model*15680 + widx; answer = delta (float). 8 * 15680 = 125440 rows.
dp_ids, answers = [], []
for m_id in range(8):
    row = deltas_3[m_id]
    for widx in range(FC_N):
        dp_ids.append(m_id * FC_N + widx)
        answers.append(float(row[widx]))
write_submission_csv(OUT / "submission_3.csv", dp_ids, answers)
assert len(dp_ids) == 8 * FC_N == 125440

# ## Done
# 
# Three native CSVs in `out/`, each uploaded independently to the contest portal (single-part upload uses `subtaskID = 1`):
# 
# | file | rows | datapointID | answer |
# |---|---|---|---|
# | `out/submission_1.csv` | 20,432 | global weight index | prune bit 0/1 (12,000 ones) |
# | `out/submission_2.csv` | 2,048 | `model*256 + bit` | recovered bit 0/1 |
# | `out/submission_3.csv` | 125,440 | `model*15680 + widx` | delta float, \|delta\| ≤ 0.02 |
# 
# Each part ran its own local evaluator (`eval_1`, the subtask-2 `verify_decoder` self-check, `eval_3`) so you can iterate offline before spending an upload. Replace the three `YOUR CODE HERE` blocks with your real strategy, re-run top to bottom, and confirm the estimated scores before submitting.

# ---- Combine the three parts into ONE submission for the folded task ----
# The task is a single task with three subtasks (1, 2, 3) at
# subtaskID 1, 2, 3. Upload the ONE combined file written below, not the three
# per-part files.
import csv
_parts = [("out/submission_1.csv", 1), ("out/submission_2.csv", 2), ("out/submission_3.csv", 3)]
with open("out/submission.csv", "w", newline="") as _f:
    _w = csv.writer(_f); _w.writerow(["subtaskID", "datapointID", "answer"])
    for _path, _sid in _parts:
        for _row in Path(_path).read_text().splitlines()[1:]:
            _, _dp, _ans = _row.split(",", 2)
            _w.writerow([_sid, _dp, _ans])
print("Wrote out/submission.csv  (subtaskID 1/2/3).  Upload THIS single file.")

