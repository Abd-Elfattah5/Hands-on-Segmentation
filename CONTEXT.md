# Clothes Segmentation Context

The domain of binary clothes segmentation from photos of people, its training pipeline and its evaluation. Sibling glossary to `Hands-on VAE/CONTEXT.md` and `Hands-on DDPM/CONTEXT.md`.

## Language

### Task & Data

**Clothes Mask**:
The binary per-pixel target: 1 for worn clothing (upper-clothes, skirt, pants, dress, belt, scarf), 0 for everything else.
_Avoid_: Garment map, parsing map (the 18-class source labels)

**Clothes Label Rule**:
The list of ATR class ids mapped to 1 (`data.clothes_ids: [4, 5, 6, 7, 8, 17]`); hat, sunglasses, bag, shoes, hair, skin and background map to 0.
_Avoid_: Class filter, label whitelist

**Human Parsing**:
Multi-class pixel labelling of body parts and garments (ATR's original 18-class task), from which the clothes mask is derived.
_Avoid_: Pose estimation, instance segmentation

**Seeded Split**:
The 80/10/10 partition (14,164 / 1,770 / 1,772) drawn once with seed 42 and stored as ID lists in `data/splits/*.txt`.
_Avoid_: Random split (implies re-drawing per run)

**Paired Augmentation**:
Geometric augmentation applied with identical random parameters to an image and its mask; photometric augmentation touches the image only.
_Avoid_: Data augmentation (when the pairing matters)

### Model

**Encoder**:
The ImageNet-pretrained ResNet-34 that turns the image into features at 1/2 to 1/32 resolution.
_Avoid_: Backbone (acceptable in prose), feature extractor

**Decoder Block**:
Bilinear ×2 upsampling, concatenation with the matching encoder feature map and a DoubleConv.
_Avoid_: Up block, deconvolution

**Skip Connection**:
The encoder feature map passed directly to the decoder block at the same resolution.
_Avoid_: Residual connection (ResNet's internal shortcut)

**Bottleneck**:
The 512 × 8 × 8 output of ResNet-34 layer4, the coarsest and most semantic representation.
_Avoid_: Latent code (VAE term)

**Logits / Probability Map**:
The raw 1 × H × W head output and its sigmoid; the probability map is thresholded at τ to get the predicted mask.
_Avoid_: Prediction (ambiguous between the two)

**Threshold (τ)**:
The probability cut-off for the binary mask, 0.5 by default.
_Avoid_: Confidence

**Component Registry**:
The decorator factory binding `model.name` in the config to a model class (`resnet34_unet` / `unet`, `standard_unet`).
_Avoid_: Hardcoded switch

### Training

**BCE + Dice Loss**:
Weighted sum of binary cross-entropy with logits and one minus soft Dice (1 : 1).
_Avoid_: Combined loss, IoU loss

**Soft Dice**:
Dice computed on probabilities instead of the thresholded mask, so it is differentiable; per image, then averaged.
_Avoid_: Dice score (the evaluation metric)

**Checkpoint Convention**:
`latest.pt` every epoch, `best_checkpoint.pt` on validation-IoU improvement, `final_checkpoint.pt` at the end, `checkpoint_epoch_NNN.pt` periodically; any can be passed to `train --resume`.
_Avoid_: Snapshot

### Evaluation

**IoU (Jaccard)**:
TP / (TP + FP + FN) of the predicted vs. true clothes mask; the headline metric.
_Avoid_: Accuracy

**Dice (F1)**:
2TP / (2TP + FP + FN); always ≥ IoU for the same mask.
_Avoid_: Soft Dice (the loss term)

**Macro vs. Micro**:
Macro averages the metric over images (headline numbers); micro pools all pixels of the split first.
_Avoid_: Mean metric (unspecified)

**Minimum Performance Target**:
Test IoU ≥ 0.80 and Dice ≥ 0.88, fixed before training because the assignment states no number.
_Avoid_: Baseline

**Capture Condition**:
A simulated photo-quality change (lighting, contrast, grayscale, blur, low resolution, noise, rotation, crop) used by `seg robustness`.
_Avoid_: Corruption, attack

**Failure Mode**:
A recurring error pattern in the worst-case gallery, such as bags segmented as clothes or ambiguous tights.
_Avoid_: Bug

**Personal-Image Pipeline**:
`seg predict`: EXIF-corrected load, resize, forward pass, probability upsampling to the original size, threshold, and mask / overlay / RGBA cut-out / panel outputs.
_Avoid_: Demo, inference script
