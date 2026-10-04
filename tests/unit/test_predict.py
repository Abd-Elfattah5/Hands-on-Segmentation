"""Unit tests for checkpointing and inference pipeline."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from src.inference.predict import load_photo, predict_image
from src.models.unet import StandardUNet
from src.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path):
    model = StandardUNet(in_channels=3, out_channels=1, features=[16, 32])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "test_ckpt.pt"

    config = {
        "model": {"name": "standard_unet", "in_channels": 3, "out_channels": 1, "features": [16, 32]},
        "experiment": {"name": "test_exp", "seed": 42},
    }

    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=opt,
        scheduler=None,
        epoch=3,
        global_step=150,
        best_val_iou=0.825,
        history=[{"epoch": 1, "val_iou": 0.825}],
        config=config,
        seed=42,
    )

    assert ckpt_path.exists()

    loaded = load_checkpoint(ckpt_path, device="cpu")
    assert loaded["epoch"] == 3
    assert loaded["global_step"] == 150
    assert loaded["best_val_iou"] == 0.825
    assert isinstance(loaded["model"], StandardUNet)


def test_load_photo_applies_exif_orientation(tmp_path: Path):
    # Landscape pixel data tagged "rotate 90 CW" (Orientation=6), as phone cameras write portrait shots.
    img = Image.new("RGB", (400, 300), color=(10, 20, 30))
    exif = img.getexif()
    exif[0x0112] = 6
    path = tmp_path / "portrait.jpg"
    img.save(path, exif=exif)

    loaded = load_photo(path)
    assert loaded.size == (300, 400)
    assert loaded.mode == "RGB"


def test_predict_pipeline(tmp_path: Path):
    model = StandardUNet(in_channels=3, out_channels=1, features=[16, 32])
    # Create test image
    test_img = Image.new("RGB", (300, 400), color=(180, 100, 50))
    img_path = tmp_path / "test_sample.jpg"
    test_img.save(img_path)

    out_dir = tmp_path / "predictions"
    res = predict_image(
        model=model,
        image_path=img_path,
        device="cpu",
        target_size=(128, 128),
        threshold=0.5,
        output_dir=out_dir,
    )

    assert res["original_size"] == (300, 400)
    assert Path(res["mask_path"]).exists()
    assert Path(res["overlay_path"]).exists()
    assert Path(res["cutout_path"]).exists()
    assert Path(res["panel_path"]).exists()

    # Verify cutout is RGBA
    cutout = Image.open(res["cutout_path"])
    assert cutout.mode == "RGBA"
    assert cutout.size == (300, 400)
