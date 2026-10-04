# Architecture Deep Dive: Top-Down Technical & Mathematical Rationale

This document gives the top-down rationale for every architectural decision, hyperparameter and formulation in the clothes segmentation system (assignment CVCY002). It is the sibling of `Hands-on VAE/ARCHITECTURE_DEEP_DIVE.md` and `Hands-on DDPM/ARCHITECTURE_DEEP_DIVE.md`. The visual summary is `Architecture_visualization.jpg`; the catalogue of swappable components is `architectural_enhancements_taxonomy.html`.

> **Status**: Final. All numbers are measured on the reported run (`artifacts/runs/atr_resnet34_unet/`, copied to `docs/reports/figures/*.json`).

---

## 1. Task Formulation: Binary Semantic Segmentation

Every pixel $(i, j)$ of an image $x \in \mathbb{R}^{3 \times H \times W}$ gets a label $y_{ij} \in \{0, 1\}$: worn clothing (1) or anything else (0), including skin, hair, accessories and background.

- **Why binary**: the assignment asks to "segment the clothes from people" for virtual try-on. A single clothes mask and an RGBA cut-out answer that directly. Binary output also makes BCE + Dice the natural loss and IoU/Dice the natural metrics.
- **Why not per-garment classes**: possible with the same dataset (§10), but it adds a softmax head, class-imbalance handling and per-class metrics without serving the stated goal.

---

## 2. Dataset & Preprocessing: ATR Human Parsing

### 2.1 Why ATR?
- **17,706 real photos** of whole people in street and indoor scenes, close to the target input (a personal photo of a person).
- **18 pixel-level classes**, so the clothes/non-clothes rule is an explicit, documented choice instead of a fixed dataset property.
- **No login**: the public Hugging Face mirror `mattmdjaga/human_parsing_dataset` (two parquet shards, ~760 MB) avoids Kaggle tokens.
- Alternatives rejected: Clothing Co-Parsing (~1,000 catalogue-style images, too small); DeepFashion2 / ModaNet / iMaterialist (access forms or 20+ GB).

### 2.2 Binary Label Mapping
$$y_{ij} = \mathbb{1}\big[\,c_{ij} \in \{4, 5, 6, 7, 8, 17\}\,\big]$$
upper-clothes, skirt, pants, dress, belt, scarf. Excluded: hat, sunglasses and bag (accessories; a bag is carried, not worn), shoes (handled separately in try-on), hair, face, arms, legs and background. The id list lives in `configs/atr_unet.yaml` (`data.clothes_ids`), so the rule is reproducible and changeable with one command.

### 2.3 Split
ATR has no official split. `seg prepare-data` shuffles with seed 42 and writes **80 / 10 / 10 = 14,164 / 1,770 / 1,772** IDs to `data/splits/*.txt`, so the partition is fixed on disk, not re-drawn per run.

### 2.4 Resizing and Normalization
- **256 × 256** input: fits batch 16 in fp32 within the 3 GB budget (§9) and keeps garments well resolved for single-person photos.
- Images are bilinear-resized; masks are **nearest-neighbour**-resized, so they stay strictly binary.
- ImageNet mean/std normalization, because the encoder was pretrained with it.

### 2.5 Paired Augmentation (training only)
`PairedSegmentationTransform` draws **one** set of random parameters and applies it to image and mask together, so labels stay aligned:

| Augmentation | Range | Applies to | Purpose |
|---|---|---|---|
| Horizontal flip | p = 0.5 | image + mask | Left/right symmetry of people |
| Rotation | ±10° | image + mask | Slightly tilted cameras |
| Scale | 0.9 – 1.1 | image + mask | Distance to camera |
| Translation | ±5 % | image + mask | Off-centre framing |
| Brightness / contrast / saturation | ×0.8 – 1.2 | image only | Lighting and white balance |

Validation and test images get resize + normalization only. Blur, noise and low-resolution augmentation were deliberately **not** added; §8 shows this is the main robustness gap.

---

## 3. Network: U-Net with a Pretrained ResNet-34 Encoder

### 3.1 Overall Topology (`ResNet34UNet`, registered as `resnet34_unet` and `unet`)
```text
x [B,3,256,256] (ImageNet-normalized)
ENCODER  torchvision ResNet-34, ImageNet weights                  skip
  init_conv  conv7×7/2 + BN + ReLU         → [B, 64,128,128]  ── x0 ──────────┐
  maxpool/2 + layer1 (3 BasicBlocks)       → [B, 64, 64, 64]  ── x1 ────────┐ │
  layer2 (4 BasicBlocks, stride 2)         → [B,128, 32, 32]  ── x2 ──────┐ │ │
  layer3 (6 BasicBlocks, stride 2)         → [B,256, 16, 16]  ── x3 ────┐ │ │ │
  layer4 (3 BasicBlocks, stride 2)         → [B,512,  8,  8]  bottleneck │ │ │ │
DECODER  hand-written; DecoderBlock = Upsample×2 (bilinear) → concat skip → DoubleConv
  dec4  (512 + 256 → 256)                  → [B,256, 16, 16]  ◄─────────┘ │ │ │
  dec3  (256 + 128 → 128)                  → [B,128, 32, 32]  ◄───────────┘ │ │
  dec2  (128 +  64 →  64)                  → [B, 64, 64, 64]  ◄─────────────┘ │
  dec1  ( 64 +  64 →  32)                  → [B, 32,128,128]  ◄───────────────┘
  final_up  Upsample×2 + DoubleConv(32→16) → [B, 16,256,256]
HEAD   Conv1×1 (16→1)                       → logits [B,1,256,256]
       σ(·) → probabilities; ≥ 0.5 → binary mask
DoubleConv = 2 × [Conv3×3 (no bias) → BatchNorm → ReLU]
```
Measured size: **24,436,241 parameters** = 21.28 M encoder + 3.15 M decoder and head.

### 3.2 Why a U-Net?
- Dense prediction needs **context** (is this fabric on a person or a cushion?) and **pixel-exact boundaries**. The encoder supplies context at 8×8 (receptive field covers the whole person); the skip connections carry 128×128 to 16×16 detail straight to the decoder.
- It is the standard, well-understood architecture for binary segmentation, which matches the user decision of "a regular U-Net".

### 3.3 Why a Pretrained ResNet-34 Encoder?
- ImageNet features already separate textures, fabrics and skin. **Validation IoU was 0.825 after one epoch**, which removed the risk of the unstated "minimum performance" bar on a one-day schedule.
- ResNet-34 (BasicBlocks, 512-channel bottleneck) is the usual accuracy/memory sweet spot: ResNet-50 has a 2,048-channel bottleneck and would push the decoder and activations past the 3 GB budget at batch 16.
- The assignment has no from-scratch requirement (unlike GenCV003). A from-scratch alternative is still implemented (§3.7).

### 3.4 Decoder Channel Widths $[256, 128, 64, 32, 16]$
Each stage halves the channels as resolution doubles, mirroring the encoder and keeping the per-level information roughly constant. The two extra stages (dec1 at 128×128 and final_up at 256×256) recover full resolution, because ResNet's stem already downsamples by 4 before layer1.

### 3.5 Bilinear Upsampling + Convolution (no Transposed Convolutions)
Transposed convolutions with stride 2 overlap unevenly and produce **checkerboard artefacts**, which would show as grid patterns along garment edges. Fixed bilinear interpolation (`align_corners=True`) followed by two learnable 3×3 convs gives smooth masks at no extra parameter cost. A size guard re-interpolates when an odd input size makes the upsampled map and skip differ by one pixel.

### 3.6 Normalization and Attention
- **BatchNorm**: batch 16 gives stable statistics, and BN matches the pretrained encoder. (The DDPM needed GroupNorm because of micro-batching and mixed noise levels; neither applies here.)
- **No attention**: garments are large contiguous regions, so local convolutional context is enough, and attention at 64×64 or larger would cost memory the 4 GB GPU does not have.

### 3.7 Alternative: `StandardUNet` (registered as `standard_unet`)
The classic Ronneberger U-Net, trained from scratch: DoubleConv + MaxPool down path with features $[32, 64, 128, 256]$, a 512-channel bottleneck, **ConvTranspose2d** upsampling, skip concatenation and a 1×1 head. **7,763,041 parameters**. It is selectable by changing `model.name` in the config. Time did not allow a full training run; it is the baseline for the "pretraining vs. from scratch" comparison.

---

## 4. Loss: BCE + Soft Dice

$$\mathcal{L} = w_{\text{BCE}}\,\mathcal{L}_{\text{BCE}} + w_{\text{Dice}}\,\big(1 - \text{Dice}_{\text{soft}}\big), \qquad w_{\text{BCE}} = w_{\text{Dice}} = 1$$

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{ij}\big[\,y_{ij}\log\sigma(z_{ij}) + (1-y_{ij})\log(1-\sigma(z_{ij}))\,\big]$$

$$\text{Dice}_{\text{soft}} = \frac{1}{B}\sum_{b}\frac{2\sum_{ij} p_{ij} y_{ij} + s}{\sum_{ij} p_{ij} + \sum_{ij} y_{ij} + s}, \qquad p = \sigma(z),\ s = 1$$

| Term | Role | Why it is needed here |
|---|---|---|
| BCE (with logits) | Per-pixel log loss, numerically stable | Smooth gradients from the first step; calibrated probabilities, so τ = 0.5 is meaningful |
| Soft Dice (per image, then averaged) | Region-overlap objective | Clothes cover ~14 % of pixels on average; BCE alone is dominated by background and under-segments. Dice is independent of background size and matches the IoU/Dice metrics |

- The smoothing term $s$ keeps Dice defined for images with no clothes.
- An optional `loss.pos_weight` (BCE foreground weight) exists in the schema; it was not needed.

---

## 5. Optimization

| Setting | Value | Why |
|---|---|---|
| Optimizer | AdamW, lr 3·10⁻⁴, weight decay 10⁻⁴ | Standard for fine-tuning a pretrained CNN; decoupled weight decay |
| Warm-up | 500 linear steps | Protects pretrained encoder weights from large early updates by the randomly initialized decoder |
| Schedule | CosineAnnealingLR to 10⁻⁶ over the remaining steps | Smooth decay; validation IoU plateaus exactly as the schedule ends |
| Batch / precision | 16, fp32 | Fits 3 GB; fp16 was unstable on the T2000 in the DDPM project |
| Epochs | 15 × 885 steps | Train/val IoU gap 0.917 / 0.905; more epochs would add very little |
| Gradient clipping | 1.0 (global norm) | Guards against rare loss spikes from tiny or empty masks |
| Model selection | Best **validation IoU** | Selects on the reported metric, not on the loss |
| Seed | 42 (Python, NumPy, PyTorch, cuDNN deterministic) | Reproducibility |

---

## 6. Inference Pipeline (Personal Images)

`seg predict` (`src/inference/predict.py`):
1. `load_photo`: open JPEG / PNG / **HEIC** (pillow-heif), apply **EXIF transpose**, convert to RGB. Without the EXIF fix a portrait phone photo reached the model rotated 90° and the jeans were missed entirely.
2. Resize to 256×256, normalize with ImageNet statistics.
3. Forward pass → $\sigma$ → probability map $p \in [0,1]^{256 \times 256}$.
4. **Upsample the probability map** bilinearly to the original resolution, *then* threshold at τ = 0.5. Thresholding after upsampling gives smoother edges than upsampling a binary mask.
5. Write `*_mask.png`, `*_overlay.png`, `*_clothes.png` (RGBA cut-out, transparent background) and `*_panel.png`; report clothes coverage and latency (~0.15–0.2 s per photo on the T2000).

---

## 7. Evaluation Design

### 7.1 Metrics (threshold 0.5)
With TP, FP, FN, TN counted per image:
$$\text{IoU} = \frac{TP}{TP+FP+FN}, \quad \text{Dice} = \frac{2TP}{2TP+FP+FN}, \quad \text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}$$
plus pixel accuracy. Reported both **macro** (mean over images, the headline numbers) and **micro** (all pixels pooled). Pixel accuracy is inflated by background and is not used for decisions.

### 7.2 Target and Result
The assignment gives no number; the target **IoU ≥ 0.80, Dice ≥ 0.88** was fixed before training. Test (1,772 images): **IoU 0.9035, Dice 0.9470**, accuracy 0.986, precision 0.943, recall 0.955. Validation IoU 0.9052 agrees closely.

### 7.3 Galleries and Breakdown
`seg evaluate` writes the best and worst test images by IoU (input, ground truth, prediction, error map) and a breakdown by clothes area (< 15 %, 15–40 %, > 40 %). `--exclude-id` removes an image from the gallery only; it still counts in all metrics.

### 7.4 Robustness Protocol
`seg robustness` re-evaluates the full test set under ten deterministic capture conditions:

| Condition | IoU | Condition | IoU |
|---|---|---|---|
| Clean | 0.904 | Overexposed ×1.8 | 0.837 |
| Dim ×0.4 | 0.894 | Grayscale | 0.798 |
| Low contrast ×0.4 | 0.891 | Noise σ = 0.08 | 0.689 |
| 30° rotation | 0.859 | ¼ resolution | 0.597 |
| Upper-body crop 2× | 0.846 | Gaussian blur σ = 3 | 0.248 |

The model relies on fine texture and edges: lighting is well tolerated, while blur, low resolution and noise (absent from training augmentation) break it.

---

## 8. Software Architecture

### 8.1 Decorator Registry (as in the VAE and DDPM)
`@register_model("resnet34_unet")`, `@register_model("unet")`, `@register_model("standard_unet")`. `build_model_from_config` looks up `model.name` and passes the model section as keyword arguments. Adding an encoder (§10) needs no change to the trainer or CLI. Every model subclasses `BaseSegmentationModel` (`forward` → logits, `predict_proba`, `predict_mask`).

### 8.2 Typed Config Schema
`src/configs/schema.py`: one dataclass per YAML section (`experiment`, `data`, `model`, `loss`, `training`, `evaluation`, `verify`). Unknown sections or keys and out-of-range values fail with a `ConfigError` before any GPU work starts. The resolved config is saved with every run and inside every checkpoint.

### 8.3 Checkpoints and Resume
- `latest.pt` every epoch, `best_checkpoint.pt` on validation-IoU improvement, `final_checkpoint.pt` at the end, `checkpoint_epoch_NNN.pt` every 5 epochs.
- Each holds model, optimizer and scheduler state, best IoU, full metric history, RNG states, config and provenance (torch version, device, timestamp, seed, git commit).
- Written **atomically** (`.tmp` then rename), so an interrupted save never corrupts the previous file. `seg train --resume` continues exactly where training stopped.

### 8.4 CLI (`seg`, Typer)
`verify` (shapes, finite loss, gradient flow, one optimizer step, 3 GB budget) → `prepare-data` → `train` → `evaluate` → `robustness` → `predict`. Option names mirror the DDPM (`-c`, `-k`, `--resume`, `--output-dir`).

### 8.5 Recorded Deviations
Differences from the sibling repositories are justified in `docs/adr/0001-deviations-from-hands-on-ddpm-conventions.md`.

---

## 9. Hardware Budget (Measured)

NVIDIA Quadro T2000, 4 GB, budget 3,072 MiB:

| Workload | Configuration | Peak GPU memory | Time |
|---|---|---|---|
| Training | batch 16, 256², fp32, AdamW | **2,640 MiB** | ~5.8 min per epoch, 86.7 min for 15 epochs |
| Inference | 1 photo, any resolution | < 500 MiB | ~0.15–0.2 s per photo |

---

## 10. Swappable Components & Possible Enhancements

Full table with formulations and trade-offs: `architectural_enhancements_taxonomy.html`.

| Component | Current choice | Alternatives | Expected effect |
|---|---|---|---|
| Label rule | Binary, ids [4,5,6,7,8,17] | 18-class softmax head, merge garment classes at inference | Explicit "bag" class removes the most frequent false positive |
| Augmentation | Flip, ±10°, scale, translate, colour jitter | + Gaussian blur, noise, JPEG, down-sampling | Closes the blur / low-res / noise gap (§7.4) |
| Encoder | ResNet-34, ImageNet | `standard_unet` from scratch; MobileNetV3 (edge); ResNet-50 / ConvNeXt (capacity) | Speed vs. accuracy vs. memory |
| Skip connections | Concatenation | Attention gates | Suppresses clutter near the person; small overhead |
| Upsampling | Bilinear + DoubleConv | Transposed conv | Learnable, but checkerboard risk |
| Head | 1×1 conv, sigmoid | Multi-class softmax; deep supervision on dec2 / dec3 | Garment types; better intermediate features |
| Loss | BCE + Dice (1 : 1) | Focal, Lovász / Jaccard, `pos_weight` | Imbalance handling; direct IoU optimization |
| Resolution | 256 × 256 | 384 × 256 portrait, 512 × 512 | Sharper boundaries on large photos; more memory |
| Inference | Single pass, τ = 0.5 | Flip test-time augmentation; τ ∈ [0.3, 0.7] | +0.5–1.5 % IoU at 2× cost; precision / recall trade-off without retraining |
| Data | ATR | LIP, CIHP (multi-person) | Reduces domain bias (mostly young women, standing poses) |
