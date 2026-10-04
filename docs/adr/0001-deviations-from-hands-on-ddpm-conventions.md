# 0001. Deviations from Hands-on DDPM Conventions

**Status**: Accepted (2026-10-04) · **Project**: CVCY002 clothes segmentation

This repository mirrors the `Hands-on DDPM` (and `Hands-on VAE`) module layout, decorator registry, typed config schema, Typer CLI vocabulary, atomic checkpoint convention with `--resume`, seeding and pytest suite. This ADR records every place where it intentionally does not.

| # | DDPM convention | Segmentation decision | Justification |
|---|---|---|---|
| D1 | Full Spec Kit cycle (`.specify/`, `specs/001-*`, constitution, tasks) | Not used; `HANDOFF.md` is the specification | One-day deadline. The DDPM spec cycle took most of a day (user decision, HANDOFF §3). |
| D2 | Everything from scratch | ImageNet-pretrained torchvision ResNet-34 encoder; hand-written decoder | CVCY002 has no from-scratch requirement. Pretraining reached val IoU 0.825 after one epoch. A from-scratch `standard_unet` remains registered. |
| D3 | GroupNorm, EMA weights, best checkpoint by EMA validation loss | BatchNorm, no EMA, best checkpoint by **validation IoU** | Batch 16 gives stable BN statistics and matches the pretrained encoder. EMA matters for diffusion sampling, not for a 15-epoch fine-tune. Selecting on IoU optimizes the reported metric. |
| D4 | Linear warm-up then constant LR | 500-step warm-up then cosine decay to 1e-6 | Short fine-tuning run; cosine decay gives the final plateau in 15 epochs. |
| D5 | Shared FID/IS benchmark protocol | IoU, Dice, precision, recall, pixel accuracy (macro + micro), galleries, robustness conditions | Different task; metrics follow the assignment. |
| D6 | Two configs (local / Colab) and a tracked Colab notebook | One config, trained locally; no notebook | The run fits the T2000 (2,640 MiB peak, 87 min). |
| D7 | Checkpoint published as a release asset; `artifacts/` gitignored | Same release convention (`best_checkpoint.pt`, release `v1.0.0`); `artifacts/` **tracked** except `*.pt` | 280 MB checkpoints exceed the GitHub 100 MB file limit. Tracking logs, metrics, galleries and predictions lets a reviewer inspect the reported run without retraining. |
| D8 | Personal data never in the repository | Five processed personal photos in `docs/reports/figures/personal_*.jpg`; originals in `samples/personal/` stay gitignored | Required by assignment task 2b and published with the owner's consent. |
| D9 | `graphify-out/` not tracked (DDPM) | Added after implementation, as in `Hands-on VAE` | The user runs graphify before the first push. |

**Consequences**: The code reads the same way as the sibling repositories (registry, schema, CLI, checkpoints), while the training recipe follows standard practice for fine-tuning a pretrained segmentation network.
