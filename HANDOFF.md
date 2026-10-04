# HANDOFF: Hands-on Segmentation (Clothes Segmentation, Assignment CVCY002)

**Repository**: `/home/amousa1/Projects/Hands-on Segmentation` (git, branch `main`, remote `origin` = https://github.com/Abd-Elfattah5/Hands-on-Segmentation)
**Assignment**: `docs/CYCV002.pdf` (task code CVCY002, "Computer Vision Engineer")
**Sibling repos to copy conventions from**: `/home/amousa1/Projects/Hands-on DDPM` (most recent, preferred) and `/home/amousa1/Projects/Hands-on VAE`
**Created**: 2026-10-03 · **Deadline**: 2026-10-04 night (about 28 h from creation)
**Status (2026-10-04)**: Implemented, trained and evaluated (test IoU 0.9035, Dice 0.9470). Report and README written. Personal photos done (EXIF-orientation and HEIC fixes added). Architecture docs added (`ARCHITECTURE_DEEP_DIVE.md`, `Architecture_visualization.jpg`, `architectural_enhancements_taxonomy.html`, `CONTEXT.md`, ADR 0001). Graphify knowledge graph committed. Published to https://github.com/Abd-Elfattah5/Hands-on-Segmentation (public); `best_checkpoint.pt` is the v1.0.0 release asset. The local T2000 is free for the DDPM benchmark.

---

## 1. Mission

Build a clothes segmentation system that separates worn clothes from people in photos, and deliver it as a GitHub repository with:
- **(a)** documented code for dataset preprocessing, model implementation, and training and evaluation;
- **(b)** a report (Markdown, `docs/reports/clothes-segmentation-report.md`) covering dataset choice and reason, model architecture, loss choice and reason, performance analysis and metrics, and limitations;
- **(c)** a `README.md` with setup, run and reproduction instructions.

The PDF also requires:
- a **pipeline that takes a personal image and segments the clothes** (task 2b);
- a "Discover System Capabilities" section (task 3): strengths, drawbacks and wrong segmentations, and limitations (capture conditions).

It says the system must reach "the minimum required performance" **but gives no number**: define and justify a target (see §5).

## 2. Requirements Checklist (from the PDF)

| PDF item | Planned implementation | Done |
|---|---|---|
| 1a Choose a proper dataset | ATR human parsing via Hugging Face mirror (no Kaggle token needed), 14,164/1,770/1,772 split | ☑ |
| 1b Preprocess + augmentation | Binary clothes mask (ids 4,5,6,7,8,17); resize 256; flip, rotate, scale, translate, colour jitter (train only) | ☑ |
| 2a Train a DL segmentation model | ResNet-34 U-Net, BCE + Dice, 15 epochs, 87 min, best val IoU 0.9052 | ☑ |
| 2b Personal-image pipeline | `seg predict --image photo.jpg` → mask, overlay, RGBA cut-out, panel (verified on test photos) | ☑ |
| 3a Strengths | Best-case gallery, area breakdown, robustness table (report §7.1) | ☑ |
| 3b Drawbacks / wrong segmentation | Worst-case gallery + 5 failure modes (report §7.2) | ☑ |
| 3c Limitations / capture conditions | `seg robustness` (10 simulated conditions) + operating-conditions table (report §7.3) + 5 personal photos (report §8.1) | ☑ |
| Deliverable: documented script | Docstrings + CLI `--help`; modular `src/`; 15 unit tests pass | ☑ |
| Deliverable: report | `docs/reports/clothes-segmentation-report.md` | ☑ |
| Deliverable: README | Setup, data download, train, evaluate, robustness, predict, tests | ☑ |
| Minimum performance | Target **IoU ≥ 0.80 / Dice ≥ 0.88**; achieved test **IoU 0.9035 / Dice 0.9470** | ☑ |

## 3. Decisions Already Made (user)

- **Keep it simple and fast**: one-day project. No from-scratch requirement in this assignment (unlike GenCV003).
- **Model**: a regular **U-Net, no attention**. Clothes are large regions, and attention would waste memory on the 4 GB GPU.
- **Loss**: **BCE + Dice** (binary: clothes vs. everything else).
- **Metrics**: **IoU** and **Dice** (plus pixel accuracy, precision, recall).
- **Architecture/conventions**: same project layout and style as Hands-on VAE/DDPM (config YAML + schema validation, registry, Typer CLI, trainer with `latest.pt`/`best_checkpoint.pt`/`final_checkpoint.pt` + `--resume`, `metrics.json`, seeding, logging, pytest).
- **Venv**: `clothes-unet/` (Python 3.12.13, torch 2.6.0+cu124, torchvision 0.21.0; gitignored).

### Recommended (agent may adopt; record in the report)
- **Pretrained encoder**: U-Net with an ImageNet-pretrained **torchvision ResNet-34** encoder and a hand-written decoder with skip connections. It is still a "regular U-Net", converges in a few epochs and is safer for the unstated performance bar. A from-scratch U-Net is acceptable if time allows comparison.
- **Binary task** (clothes vs. non-clothes) rather than per-garment classes. The PDF asks to "segment clothes from people", and binary keeps BCE + Dice natural.
- **Resolution**: 256×256 (or 384×256 portrait). Mixed precision is **not** needed; the T2000 was unstable with fp16 in the DDPM project.
- **Lightweight process**: do **not** run the full Spec Kit cycle (constitution, specify, clarify, plan, tasks, analyze took most of a day for the DDPM). This handoff is the spec. If documentation is wanted, keep one short `specs/001-clothes-unet/plan.md`.

## 4. Dataset Options

| Dataset | Size | Labels | Fit | Risk |
|---|---|---|---|---|
| **ATR** human parsing (Kaggle mirrors, search "ATR human parsing") | ~17,700 images | 18 classes: background, hat, hair, sunglasses, upper-clothes, skirt, pants, dress, belt, shoes, face, legs, arms, bag, scarf | Primary: varied people photos generalize better to personal images | Larger download; train ~2-4 h on the T2000 |
| **Clothing Co-Parsing** / Kaggle "People Clothing Segmentation" | ~1,000 images | ~59 labels | Fallback: fastest to download and train | Small, fashion-style photos, weaker generalization |
| DeepFashion2, ModaNet, iMaterialist | large | — | Avoid | Access forms / scraping / 20+ GB |

**Binary mask rule**: clothes = upper-clothes, skirt, pants, dress, belt, scarf (decide on hat, shoes, bag, sunglasses and **document the choice**, e.g. accessories excluded). Everything else is 0. Write the exact label-id list into the config so the mapping is reproducible.

**Kaggle access**: needs `~/.kaggle/kaggle.json` (API token from kaggle.com → Settings → API). `kaggle.json` is gitignored. Ask the user for it if missing.

**Split**: seeded train/val/test (e.g. 80/10/10) if the dataset has no official split. Save the split index lists to `data/splits/*.txt` so results are reproducible.

## 5. Performance Target ("minimum required performance")

The PDF gives no number. Procedure:
1. Train a short baseline (few epochs) to see realistic IoU.
2. State the target in the README and report, e.g. **test IoU ≥ 0.80 and Dice ≥ 0.88** for the binary clothes mask, citing that published human-parsing models reach high-0.8x/0.9x for large garment regions.
3. Report the final test metrics against it, plus per-condition breakdowns (e.g. plain vs. busy background, full body vs. cropped) for task 3.

## 6. Planned Repository Layout

```text
configs/
  atr_unet.yaml                 # data paths, label→clothes mapping, image size, model, loss weights, training
src/
  cli/main.py                   # Typer app `seg`: verify, prepare-data, train, evaluate, predict (stub exists)
  configs/schema.py             # dataclass schema + validate_config (copy pattern from Hands-on DDPM)
  data/
    download.py                 # Kaggle download helper
    dataset.py                  # Dataset: image + binary mask, splits, transforms
    transforms.py               # paired augmentation (same geometric op on image and mask)
  models/
    registry.py                 # register_model / build_model_from_config (DDPM pattern)
    unet.py                     # U-Net (ResNet-34 encoder or plain), 1-channel logits
  training/
    losses.py                   # BCE + Dice (weights from config), soft Dice on sigmoid
    checkpoint.py               # save/load (port from DDPM: atomic save, history, rng)
    trainer.py                  # train/val loop, best by val IoU, latest/best/final, --resume, metrics.json, loss curves
  evaluation/
    metrics.py                  # IoU, Dice, pixel acc, precision, recall (threshold 0.5), per-image + aggregate
    evaluator.py                # test-split evaluation → JSON
    visualizer.py               # overlays, best/worst galleries
  inference/
    predict.py                  # personal-image pipeline: load → resize/pad → predict → upsample → mask/overlay/cutout
  utils/
    seeding.py, logging.py, memory.py   # ported from Hands-on DDPM (already copied)
tests/unit/                     # shapes, loss values, metrics on known masks, paired transforms, predict pipeline
notebooks/                      # optional Colab notebook (CLI calls only), only if local GPU is busy
samples/personal/               # user's own test photos (gitignored contents)
docs/
  CYCV002.pdf                   # the assignment (gitignored by *.pdf; local reference)
  reports/clothes-segmentation-report.md
  adr/                          # optional
artifacts/                      # runs/, eval/, predictions/ (gitignored)
data/                           # datasets (gitignored)
```

### CLI to implement (mirror DDPM option names)
- `seg verify -c configs/atr_unet.yaml`: checks shapes, a finite loss, gradients and one training step within 3 GB.
- `seg prepare-data -c ...`: downloads, converts masks to binary and writes the splits.
- `seg train -c ... [--resume latest.pt] [--epochs N] [--batch-size N] [--output-dir DIR]`.
- `seg evaluate -k artifacts/runs/<run>/best_checkpoint.pt [--split test]`: writes `eval_metrics.json` and best/worst galleries.
- `seg predict -k <ckpt> --image photo.jpg [--out-dir artifacts/predictions]`: writes `<name>_mask.png`, `<name>_overlay.png` and `<name>_clothes.png` (RGBA cut-out).

## 7. Hardware & Scheduling Constraints

- **Local GPU**: NVIDIA Quadro T2000, 4 GB (treat **3 GB** as the budget, as in the DDPM project). U-Net with a ResNet-34 encoder at 256×256 and batch 8-16 in fp32 should fit; verify with `src/utils/memory.py` (`measured_total_mib`).
- **GPU contention**: the Hands-on DDPM official training currently runs on **Colab** (free tier, typically one GPU session), and its 5,000-image benchmark will later need the **local T2000** for ~3 h. Train this project **locally**, and coordinate so the DDPM benchmark runs after this training (e.g. overnight).
- WSL2 Linux; CIFAR-style local paths; datasets go in `data/` (gitignored).

## 8. Time Budget (~28 h total)

| Hours | Work |
|---|---|
| 0-3 | Dataset download, binary masks, splits |
| 3-7 | Port skeleton code (schema, registry, trainer, checkpoint, CLI) + U-Net + BCE+Dice + metrics; start training |
| 7-10 | (training runs) personal-image `predict` pipeline, unit tests |
| 10-13 | Evaluation, best/worst galleries, personal photos, per-condition metrics |
| 13-16 | Report + README |
| 16-28 | Buffer; push to GitHub |

## 9. Setup Commands (already done for the venv)

```bash
cd "/home/amousa1/Projects/Hands-on Segmentation"
source clothes-unet/bin/activate
# already installed:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
seg --version          # Hands-on Segmentation version: 0.1.0
```

## 10. Resume Prompt for the Implementing Agent

> *"Read `/home/amousa1/Projects/Hands-on Segmentation/HANDOFF.md` and `docs/CYCV002.pdf`. Implement the clothes segmentation project following §2-§6, reusing code patterns from `/home/amousa1/Projects/Hands-on DDPM/src` (schema, registry, checkpoint, trainer, CLI). Start with the dataset (§4), confirm Kaggle access, then port the skeleton, verify the 3 GB budget, and train locally. Keep the process lightweight (no full Spec Kit cycle). Track progress by ticking the checklist in §2."*
