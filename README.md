# Hands-on Clothes Segmentation (U-Net, BCE + Dice)

Clothes segmentation model that separates worn clothes from people in photos, built for assignment `CVCY002`.

> **Status**: Repository initialized. Implementation is in progress; see `HANDOFF.md` for the plan, decisions and remaining work. This README is completed with setup, training, evaluation and prediction instructions once the code exists.

## Planned Overview

- **Dataset**: public human-parsing dataset (ATR, or Clothing Co-Parsing as fallback) converted to a binary clothes mask.
- **Model**: U-Net (no attention), optionally with an ImageNet-pretrained ResNet-34 encoder.
- **Loss**: Binary cross-entropy + Dice.
- **Metrics**: IoU, Dice, pixel accuracy, precision, recall.
- **Pipeline**: `seg predict --image photo.jpg` produces a mask, an overlay and a transparent clothes cut-out.

## Setup

```bash
python3 -m venv clothes-unet
source clothes-unet/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
seg --version
```

## License

This project is licensed under the MIT License.
