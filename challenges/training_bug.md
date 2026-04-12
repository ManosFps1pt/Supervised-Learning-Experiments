<pre class="overflow-visible! px-0!" data-start="0" data-end="551"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span># PyTorch Training Performance Notes (CIFAR-10)</span><br/><br/><span>## Hardware</span><br/><span>- CPU: Ryzen 7 5700X (8 cores / 16 threads)</span><br/><span>- GPU: NVIDIA GeForce RTX 4070</span><br/><span>- RAM: 32 GB</span><br/><br/><span>Dataset: CIFAR-10  </span><br/><span>Batch size: 256  </span><br/><span>Workers: initially 16 → later reduced</span><br/><br/><span>Observed behavior:</span><br/><span>- Epoch 1 ≈ 1 minute</span><br/><span>- Epochs 2+ ≈ 3 seconds</span><br/><span>- GPU utilization ~10% with spikes</span><br/><span>- Python process consuming ~19 GB RAM</span><br/><br/><span>This document explains **why this happens and what mechanisms inside PyTorch cause it**.</span><br/><br/><span>---</span><br/><br/><span># 1. GPU Underutilization (Original Problem)</span><br/><br/><span>GPU utilization appeared as:</span><br/></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

~10% average

Short spikes to 100%

<pre class="overflow-visible! px-0!" data-start="586" data-end="1010"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><br/><span>This pattern indicates that the **GPU is waiting for the CPU pipeline**.</span><br/><br/><span>The typical training pipeline consists of:</span><br/><br/><span>1. CPU loads data</span><br/><span>2. CPU transfers batch to GPU</span><br/><span>3. GPU computes forward/backward pass</span><br/><br/><span>If these steps run sequentially, the GPU will frequently idle.</span><br/><br/><span>---</span><br/><br/><span># 2. Importance of Asynchronous GPU Transfers</span><br/><br/><span>Original tensor transfer:</span><br/><br/><span>```python</span><br/><span>batch_X = batch_X.to(device)</span><br/><span>batch_y = batch_y.to(device)</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

This causes  **synchronous transfers** , meaning the CPU waits for the copy to complete before continuing.

Correct method:

<pre class="overflow-visible! px-0!" data-start="1135" data-end="1244"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">batch_X</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">batch_X</span><span class="ͼn">.</span><span>to(</span><span class="ͼt">device</span><span>, </span><span class="ͼt">non_blocking</span><span class="ͼn">=</span><span class="ͼq">True</span><span>)</span><br/><span class="ͼt">batch_y</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">batch_y</span><span class="ͼn">.</span><span>to(</span><span class="ͼt">device</span><span>, </span><span class="ͼt">non_blocking</span><span class="ͼn">=</span><span class="ͼq">True</span><span>)</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

This allows:

* CPU to prepare the next batch
* GPU to compute the current batch

at the same time.

Pipeline after the fix:

<pre class="overflow-visible! px-0!" data-start="1372" data-end="1439"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>Time →</span><br/><br/><span>CPU:  prepare batch N+1</span><br/><span>GPU:        compute batch N</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# 3. Why `pin_memory=True` Matters

Pinned (page-locked) memory allows the GPU to perform  **DMA transfers** .

Pinned memory is required for  **asynchronous copies** .

Correct combination:

<pre class="overflow-visible! px-0!" data-start="1634" data-end="1717"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">DataLoader</span><span>(..., </span><span class="ͼt">pin_memory</span><span class="ͼn">=</span><span class="ͼq">True</span><span>)</span><br/><span class="ͼt">tensor</span><span class="ͼn">.</span><span>to(</span><span class="ͼt">device</span><span>, </span><span class="ͼt">non_blocking</span><span class="ͼn">=</span><span class="ͼq">True</span><span>)</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

If either part is missing, transfers become synchronous again.

---

# 4. DataLoader Workers and Performance

Initial configuration:

<pre class="overflow-visible! px-0!" data-start="1853" data-end="1883"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">num_workers</span><span></span><span class="ͼn">=</span><span></span><span class="ͼq">16</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

System has:

* 8 physical CPU cores
* 16 threads

Problems caused by too many workers:

* heavy context switching
* duplicated dataset memory
* high RAM usage
* CPU contention

Better configuration:

<pre class="overflow-visible! px-0!" data-start="2085" data-end="2142"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">num_workers</span><span></span><span class="ͼn">=</span><span></span><span class="ͼq">6</span><span class="ͼt">–8</span><br/><span class="ͼt">persistent_workers</span><span></span><span class="ͼn">=</span><span></span><span class="ͼq">True</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

This reduces worker startup overhead and stabilizes throughput.

---

# 5. Why the First Epoch Is Slow

The first epoch performs  **initialization tasks that only happen once** .

These include:

### 1. CUDA Context Initialization

When GPU operations run for the first time, CUDA must:

* initialize GPU context
* allocate memory pools
* load CUDA kernels

This can take several seconds.

---

### 2. cuDNN Algorithm Benchmarking

If enabled:

<pre class="overflow-visible! px-0!" data-start="2586" data-end="2637"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">torch</span><span class="ͼn">.</span><span>backends</span><span class="ͼn">.</span><span>cudnn</span><span class="ͼn">.</span><span>benchmark </span><span class="ͼn">=</span><span></span><span class="ͼq">True</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

PyTorch tests multiple convolution algorithms and selects the fastest one for the given tensor shapes.

This happens only during the first forward pass.

---

### 3. DataLoader Worker Startup

When using:

<pre class="overflow-visible! px-0!" data-start="2845" data-end="2889"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">DataLoader</span><span>(..., </span><span class="ͼt">num_workers</span><span class="ͼn">=</span><span class="ͼq">8</span><span>)</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

PyTorch launches separate worker processes.

Each worker must:

* start a Python interpreter
* load dataset metadata
* initialize transforms
* allocate pinned buffers

This can take 10–30 seconds.

---

### 4. OS File Cache Warm-up

During the first epoch, data is read from disk.

Afterwards, the operating system caches the dataset in RAM.

Subsequent epochs read from memory instead of disk.

---

### Result

Typical behavior:

<pre class="overflow-visible! px-0!" data-start="3323" data-end="3440"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>Epoch 1</span><br/><span>  CUDA initialization</span><br/><span>  cuDNN benchmarking</span><br/><span>  worker startup</span><br/><span>  disk reads</span><br/><br/><span>Epoch 2+</span><br/><span>  cached execution</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# 6. Why Python Uses ~19 GB RAM

Large memory usage is mainly caused by  **DataLoader workers and pinned memory** .

Each worker prefetches batches.

Default PyTorch behavior:

<pre class="overflow-visible! px-0!" data-start="3622" data-end="3649"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>prefetch_factor = 2</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Total prefetched batches:

<pre class="overflow-visible! px-0!" data-start="3678" data-end="3715"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>num_workers × prefetch_factor</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Example configuration:

<pre class="overflow-visible! px-0!" data-start="3741" data-end="3801"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>num_workers = 8</span><br/><span>prefetch_factor = 2</span><br/><span>batch_size = 256</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Prefetched batches:

<pre class="overflow-visible! px-0!" data-start="3824" data-end="3850"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>8 × 2 = 16 batches</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# 7. CIFAR-10 Batch Memory Estimate

Image size:

<pre class="overflow-visible! px-0!" data-start="3907" data-end="3926"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>3 × 32 × 32</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Tensor representation (float32):

<pre class="overflow-visible! px-0!" data-start="3962" data-end="3999"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>3 × 32 × 32 × 4 bytes ≈ 12 KB</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Batch size:

<pre class="overflow-visible! px-0!" data-start="4014" data-end="4040"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>256 × 12 KB ≈ 3 MB</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Prefetched batches:

<pre class="overflow-visible! px-0!" data-start="4063" data-end="4088"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>16 × 3 MB ≈ 48 MB</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

This is the raw tensor data.

Actual memory usage is larger due to:

* Python process overhead
* transform buffers
* pinned memory allocation
* duplicated dataset state in workers

---

# 8. Why Pinned Memory Increases RAM Usage

Pinned memory has a special property:

* it **cannot be swapped to disk**

The operating system must keep it permanently in physical RAM.

With multiple workers and prefetched batches, pinned buffers accumulate quickly.

---

# 9. Recommended Stable Configuration

A balanced configuration for CIFAR-10:

<pre class="overflow-visible! px-0!" data-start="4625" data-end="4810"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">train_loader</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">DataLoader</span><span>(</span><br/><span></span><span class="ͼt">train_ds</span><span>,</span><br/><span></span><span class="ͼt">batch_size</span><span class="ͼn">=</span><span class="ͼq">256</span><span>,</span><br/><span></span><span class="ͼt">shuffle</span><span class="ͼn">=</span><span class="ͼq">True</span><span>,</span><br/><span></span><span class="ͼt">num_workers</span><span class="ͼn">=</span><span class="ͼq">6</span><span>,</span><br/><span></span><span class="ͼt">pin_memory</span><span class="ͼn">=</span><span class="ͼq">True</span><span>,</span><br/><span></span><span class="ͼt">persistent_workers</span><span class="ͼn">=</span><span class="ͼq">True</span><span>,</span><br/><span></span><span class="ͼt">prefetch_factor</span><span class="ͼn">=</span><span class="ͼq">2</span><br/><span>)</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

This typically provides:

* high GPU utilization
* reasonable RAM usage
* stable throughput

---

# 10. Expected Performance

With:

* RTX 4070
* Ryzen 7 5700X
* Batch size 256

Typical CIFAR-10 performance:

| Metric         | Expected      |
| -------------- | ------------- |
| Time per epoch | ~3–8 seconds |
| 60 epochs      | ~3–8 minutes |

---

# 11. Key Principle

Efficient GPU training depends on maintaining a  **continuous pipeline** :

<pre class="overflow-visible! px-0!" data-start="5223" data-end="5288"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>CPU prepares next batch</span><br/><span>while</span><br/><span>GPU processes current batch</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

When this pipeline breaks (for example due to synchronous transfers), GPU utilization collapses.

---

# 12. Useful Monitoring Tool

GPU activity can be monitored with:

<pre class="overflow-visible! px-0!" data-start="5460" data-end="5483"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>nvidia-smi -l 1</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Healthy training typically shows:

<pre class="overflow-visible! px-0!" data-start="5520" data-end="5571"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="w-full overflow-x-hidden overflow-y-auto pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>GPU utilization: 70–95%</span><br/><span>Memory usage stable</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

If GPU utilization repeatedly drops to 0%, the pipeline is starved again.

---

# 13. Core Takeaway

In many deep learning systems, the  **data pipeline is the real bottleneck** , not the neural network itself.

High-performance training requires overlapping:

1. CPU preprocessing
2. GPU data transfer
3. GPU computation
