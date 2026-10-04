# Graph Report - Hands-on Segmentation  (2026-10-04)

## Corpus Check
- 54 files · ~180,717 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: (none) 2)

## Summary
- 504 nodes · 882 edges · 37 communities (21 shown, 16 thin omitted)
- Extraction: 89% EXTRACTED · 10% INFERRED · 1% AMBIGUOUS · INFERRED: 86 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- seg CLI Commands
- Data Prep & Augmentation
- Personal-Image Pipeline
- Model Architecture & Registry
- Segmentation Metrics
- Training Recipe & Results
- Base Model Interface
- Docs, Config & Taxonomy
- Design Rationale & Alternatives
- Best/Worst Case Galleries
- GPU Memory Utilities
- ATR Download
- Clothes Dataset Loader
- Logging Utility
- ATR Dataset & Split
- Loss & Overlap Metrics
- Performance Target
- Robustness & Capture Conditions
- Decoder & Skips
- Threshold & Probabilities
- Optimization Schedule
- Checkpoint Convention
- Encoder & Bottleneck
- Evaluation Metrics Summary
- Typer CLI Concept
- Macro vs Micro
- Package Metadata
- Environment Setup
- Unit Test Suite

## God Nodes (most connected - your core abstractions)
1. `BaseSegmentationModel` - 25 edges
2. `predict_image()` - 22 edges
3. `load_checkpoint()` - 19 edges
4. `SegmentationMetricTracker` - 16 edges
5. `evaluate_split()` - 15 edges
6. `ResNet34UNet` - 15 edges
7. `BCEDiceLoss` - 15 edges
8. `verify()` - 14 edges
9. `build_model_from_config()` - 14 edges
10. `ClothesSegmentationTrainer` - 14 edges

## Surprising Connections (you probably didn't know these)
- `model: resnet34_unet, pretrained` --references--> `ResNet34UNet`  [INFERRED]
  configs/atr_unet.yaml → src/models/unet.py
- `Paired Augmentation` --references--> `PairedSegmentationTransform`  [INFERRED]
  CONTEXT.md → src/data/transforms.py
- `evaluate_split()` --shares_data_with--> `Worst-case IoU 0.404-0.487 / Dice 0.576-0.655 (samples 06833,15393,04695,14420,04911,05035,04452,08379)`  [INFERRED]
  src/evaluation/evaluator.py → docs/reports/figures/worst_cases_gallery.jpg
- `create_comparison_panel()` --shares_data_with--> `Best Segmentation Cases Gallery (Top IoU on Test Set)`  [INFERRED]
  src/evaluation/visualizer.py → docs/reports/figures/best_cases_gallery.jpg
- `Layer 4 Bottleneck [B,512,16,16]` --conceptually_related_to--> `ResNet34UNet`  [AMBIGUOUS]
  Architecture_visualization.jpg → src/models/unet.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **U-Net encoder-skip-decoder flow** — context_encoder, context_bottleneck, context_skip_connection, context_decoder_block, context_logits_probability_map, architecture_deep_dive_resnet34_unet [EXTRACTED 1.00]
- **Blur/noise/low-res robustness gap and augmentation fix** — architecture_deep_dive_robustness_protocol, context_capture_condition, architectural_enhancements_taxonomy_heavy_blur_augmentation, docs_reports_clothes_segmentation_report_operating_conditions [INFERRED 0.85]
- **BCE+Dice loss aligned with IoU/Dice metrics under class imbalance** — architecture_deep_dive_bce_dice_loss, context_soft_dice, context_iou, context_dice, docs_reports_clothes_segmentation_report_class_imbalance [EXTRACTED 1.00]
- **Encoder-skip-decoder-head forward pass** — architecture_visualization_resnet34_encoder, architecture_visualization_skip_connections, architecture_visualization_bottleneck, architecture_visualization_decoder, architecture_visualization_binary_mask_head [EXTRACTED 1.00]

## Communities (37 total, 16 thin omitted)

### Community 0 - "seg CLI Commands"
Cohesion: 0.05
Nodes (63): callback, command, dataclasses, functools, _check(), evaluate(), _handle_errors(), main() (+55 more)

### Community 1 - "Data Prep & Augmentation"
Cohesion: 0.07
Nodes (48): Paired Augmentation, io, json, math, matplotlib, matplotlib_pyplot, numpy, pathlib (+40 more)

### Community 2 - "Personal-Image Pipeline"
Cohesion: 0.05
Nodes (53): Test-Time Augmentation (flip TTA), EXIF Transpose / HEIC Support, Personal-Image Inference Pipeline, Personal-Image Pipeline, D8: Personal photos published in figures, Personal Photos Evaluation, Arm Across Torso Correctly Excluded, Night Street Capture Condition (low light, street-light glare) (+45 more)

### Community 3 - "Model Architecture & Registry"
Cohesion: 0.06
Nodes (37): BaseSegmentationModel, Decorator Model Registry, Augmentations & Formatting (flip, rotation, seed 42), Layer 4 Bottleneck [B,512,16,16], Data Preparation Pipeline (ATR Reduced Binary Mask), Decoder Network (Blocks 4-1: Bilinear Upsample + Concat Skip + 2x Conv), Full ResNet-34 U-Net Architecture Diagram, Reduction Operator: human parsing to clothes mask (+29 more)

### Community 4 - "Segmentation Metrics"
Cohesion: 0.06
Nodes (29): pytest, RuntimeError, compute_metrics_batch(), compute_per_image_metrics(), ndarray, Tensor, Compute mean segmentation metrics across a batch of predictions and targets.…, Compute aggregated metrics. (+21 more)

### Community 5 - "Training Recipe & Results"
Cohesion: 0.08
Nodes (26): Loss: BCE + Dice, Binary Mask Head (1x1 conv, sigmoid, threshold 0.5), Multi-Class Parsing Head (softmax, modular option), Optimizer: AdamW lr=3e-4, Cosine Annealing, 15 epochs, Test Results: Mean IoU 0.9035, Mean Dice 0.9470; heavy blur weakness (IoU 0.248), loss: bce_weight 1.0, dice_weight 1.0, smooth 1.0, Learning Curves Figure, Loss converges: train 0.81->~0.09, val 0.40->~0.105 over 15 epochs (+18 more)

### Community 6 - "Base Model Interface"
Cohesion: 0.11
Nodes (25): ABC, datetime, Optimizer, BaseSegmentationModel, Tensor, Abstract base class for clothes segmentation models., Forward pass. Args: x: Input tensor of shape [B, in_channels, H, W] Returns:…, Compute sigmoid probability maps. (+17 more)

### Community 7 - "Docs, Config & Taxonomy"
Cohesion: 0.11
Nodes (27): Architectural Enhancements Taxonomy, Multi-Class Softmax Head, Architecture Deep Dive, Hands-on DDPM (sibling repo), Hands-on VAE (sibling repo), Swappable Components & Enhancements, Typed Config Schema, atr_unet.yaml config (+19 more)

### Community 8 - "Design Rationale & Alternatives"
Cohesion: 0.09
Nodes (25): Attention Gates (Attention U-Net), Focal Loss, MobileNetV3 Encoder, ResNet-50 / ConvNeXt Encoder, Transposed Convolution Upsampling, BatchNorm, No Attention, BCE + Soft Dice Loss Formulation, Bilinear Upsampling + Convolution (+17 more)

### Community 9 - "Best/Worst Case Galleries"
Cohesion: 0.09
Nodes (24): Best Segmentation Cases Gallery (Top IoU on Test Set), Dice (F1), Easy cases: full-body, plain background, solid dresses/garments, Error Map (Green=TP, Red=FP, Blue=FN), IoU (Jaccard), Top test IoU 0.983-0.987 / Dice 0.991-0.993, Worst Segmentation Cases / Failure Modes gallery (lowest IoU on test set), Failure mode: bags segmented as clothes (+16 more)

### Community 10 - "GPU Memory Utilities"
Cohesion: 0.11
Nodes (20): os, random, measured_total_mib(), peak_reserved_mib(), process_total_mib(), device, _query(), GPU memory measurement helpers used by `ddpm verify` and the memory-budget… (+12 more)

### Community 11 - "ATR Download"
Cohesion: 0.24
Nodes (9): shutil, download_atr_dataset(), download_file_with_progress(), DownloadProgressBar, Path, Dataset download utilities for the ATR Human Parsing dataset. Supports direct…, Download a file with progress bar and atomic rename., Ensure ATR dataset files exist in ``raw_dir``, downloading if necessary.… (+1 more)

### Community 12 - "Clothes Dataset Loader"
Cohesion: 0.25
Nodes (6): Dataset, ClothesDataset, Image, Path, Tensor, PyTorch Dataset loading paired RGB image and binary clothes mask.

### Community 13 - "Logging Utility"
Cohesion: 0.29
Nodes (6): Logger, get_logger(), Path, Structured logging utility with console and file handlers., Configures and returns a structured logger. Args: name: Name of the logger.…, sys

### Community 14 - "ATR Dataset & Split"
Cohesion: 0.33
Nodes (6): ATR Human Parsing Dataset, Hugging Face mirror mattmdjaga/human_parsing_dataset, Seeded Split, Domain Bias of ATR, LIP / CIHP datasets, Dataset Options (ATR, Clothing Co-Parsing, DeepFashion2...)

### Community 15 - "Loss & Overlap Metrics"
Cohesion: 0.40
Nodes (5): Lovasz-Softmax / Jaccard Loss, BCE + Dice Loss, Dice (F1), IoU (Jaccard), Soft Dice

### Community 16 - "Performance Target"
Cohesion: 0.40
Nodes (5): evaluation: threshold 0.5, target_iou 0.80, target_dice 0.88, Minimum Performance Target, Test Results (IoU 0.9035, Dice 0.9470), Performance Target Procedure, Test Results (IoU 0.904, Dice 0.947)

### Community 17 - "Robustness & Capture Conditions"
Cohesion: 0.50
Nodes (4): Heavy Blur Augmentation, Robustness Protocol (10 capture conditions), Capture Condition, Supported Operating Conditions

### Community 18 - "Decoder & Skips"
Cohesion: 0.67
Nodes (3): Deep Supervision Head, Decoder Block, Skip Connection

### Community 19 - "Threshold & Probabilities"
Cohesion: 0.67
Nodes (3): Threshold Tuning (tau in [0.3,0.7]), Logits / Probability Map, Threshold (tau)

### Community 20 - "Optimization Schedule"
Cohesion: 0.67
Nodes (3): Optimization Recipe (AdamW, warm-up, cosine), training: 15 epochs, lr 3e-4, warmup 500, D4: Warm-up then cosine decay

## Ambiguous Edges - Review These
- `ResNet34UNet` → `Layer 4 Bottleneck [B,512,16,16]`  [AMBIGUOUS]
  Architecture_visualization.jpg · relation: conceptually_related_to
- `Binary Mask Head (1x1 conv, sigmoid, threshold 0.5)` → `Multi-Class Parsing Head (softmax, modular option)`  [AMBIGUOUS]
  Architecture_visualization.jpg · relation: conceptually_related_to
- `Worst Segmentation Cases / Failure Modes gallery (lowest IoU on test set)` → `Failure mode: clothes-like objects touching the person`  [AMBIGUOUS]
  docs/reports/figures/worst_cases_gallery.jpg · relation: references
- `Predict demo panel: ATR test image 02219 (input, overlay, RGBA cut-out)` → `Clean cut-out boundary with minor arm/hem edge roughness`  [AMBIGUOUS]
  docs/reports/figures/predict_demo_02219.jpg · relation: references
- `Personal photo 2 prediction panel (input / predicted mask overlay / clothes RGBA cut-out)` → `Small leak above the collar (neck/head region) and minor gaps near the hand`  [AMBIGUOUS]
  docs/reports/figures/personal_2.jpg · relation: references
- `Night scene with harsh flash lighting (real-world capture condition)` → `Small leak above the collar (neck/head region) and minor gaps near the hand`  [AMBIGUOUS]
  docs/reports/figures/personal_2.jpg · relation: conceptually_related_to
- `Personal photo 4 prediction panel (lift mirror selfie)` → `Phone/hand over shirt partly included in mask`  [AMBIGUOUS]
  docs/reports/figures/personal_4.jpg · relation: conceptually_related_to

## Knowledge Gaps
- **88 isolated node(s):** `hands-on-segmentation`, `ExperimentSectionConfig`, `DataSectionConfig`, `ModelSectionConfig`, `LossSectionConfig` (+83 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 243 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `ResNet34UNet` and `Layer 4 Bottleneck [B,512,16,16]`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Binary Mask Head (1x1 conv, sigmoid, threshold 0.5)` and `Multi-Class Parsing Head (softmax, modular option)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Worst Segmentation Cases / Failure Modes gallery (lowest IoU on test set)` and `Failure mode: clothes-like objects touching the person`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Predict demo panel: ATR test image 02219 (input, overlay, RGBA cut-out)` and `Clean cut-out boundary with minor arm/hem edge roughness`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Personal photo 2 prediction panel (input / predicted mask overlay / clothes RGBA cut-out)` and `Small leak above the collar (neck/head region) and minor gaps near the hand`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Night scene with harsh flash lighting (real-world capture condition)` and `Small leak above the collar (neck/head region) and minor gaps near the hand`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Personal photo 4 prediction panel (lift mirror selfie)` and `Phone/hand over shirt partly included in mask`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._