"""Dataset implementation, preprocessing, binary mask conversion, and DataLoader factories."""

import io
import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.transforms import PairedSegmentationTransform

# Default ATR garment class IDs corresponding to worn clothes:
# 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 8: Belt, 17: Scarf
# Accessories (hat, sunglasses, bag) and footwear (shoes) and body parts are excluded (label 0).
DEFAULT_CLOTHES_LABEL_IDS = [4, 5, 6, 7, 8, 17]


def prepare_atr_data(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    splits_dir: str | Path = "data/splits",
    clothes_ids: Sequence[int] = DEFAULT_CLOTHES_LABEL_IDS,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> dict[str, int]:
    """Extract raw ATR parquet files into processed images, binary masks, and reproducible splits.

    Args:
        raw_dir: Directory containing downloaded raw parquet files.
        processed_dir: Output directory for images/ and masks/.
        splits_dir: Output directory for train.txt, val.txt, test.txt.
        clothes_ids: Garment label IDs to map to binary 1 (all others map to 0).
        train_ratio: Ratio of dataset for training (e.g. 0.8).
        val_ratio: Ratio for validation (e.g. 0.1).
        test_ratio: Ratio for testing (e.g. 0.1).
        seed: Random seed for split shuffling.
        max_samples: Optional cap on processed samples (for testing/debugging).

    Returns:
        Dictionary with sample counts for train, val, and test splits.
    """
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    splits_path = Path(splits_dir)

    images_dir = proc_path / "images"
    masks_dir = proc_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    splits_path.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(raw_path.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No train-*.parquet files found in {raw_path}")

    clothes_id_set = set(clothes_ids)
    sample_ids: List[str] = []

    print(f"Extracting and preprocessing ATR dataset from {len(parquet_files)} parquet file(s)...")
    idx = 0
    stopped = False

    for pf_file in parquet_files:
        if stopped:
            break
        pf = pq.ParquetFile(pf_file)
        num_row_groups = pf.num_row_groups

        for rg_idx in range(num_row_groups):
            if stopped:
                break
            batch = pf.read_row_group(rg_idx, columns=["image", "mask"])
            images_col = batch["image"]
            masks_col = batch["mask"]
            batch_len = len(images_col)

            for i in range(batch_len):
                sample_id = f"{idx:05d}"
                img_path = images_dir / f"{sample_id}.jpg"
                mask_path = masks_dir / f"{sample_id}.png"

                # Check if already processed to allow resuming
                if not (img_path.exists() and mask_path.exists()):
                    img_bytes = images_col[i]["bytes"].as_py()
                    mask_bytes = masks_col[i]["bytes"].as_py()

                    # Save image directly from jpeg bytes
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                    # Convert multi-class mask to binary clothes mask
                    raw_mask = Image.open(io.BytesIO(mask_bytes))
                    raw_mask_arr = np.array(raw_mask)
                    binary_mask_arr = np.isin(raw_mask_arr, list(clothes_id_set)).astype(np.uint8) * 255
                    Image.fromarray(binary_mask_arr).save(mask_path)

                sample_ids.append(sample_id)
                idx += 1
                if max_samples is not None and idx >= max_samples:
                    stopped = True
                    break

    total = len(sample_ids)
    print(f"Total processed samples: {total}")

    # Create reproducible deterministic splits
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(total)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    train_ids = [sample_ids[i] for i in shuffled_indices[:n_train]]
    val_ids = [sample_ids[i] for i in shuffled_indices[n_train : n_train + n_val]]
    test_ids = [sample_ids[i] for i in shuffled_indices[n_train + n_val :]]

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_file = splits_path / f"{split_name}.txt"
        with open(split_file, "w", encoding="utf-8") as f:
            for s_id in ids:
                f.write(f"{s_id}\n")

    info = {
        "total_samples": total,
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "test_samples": len(test_ids),
        "clothes_ids": list(clothes_ids),
        "seed": seed,
    }
    with open(proc_path / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Splits saved: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    return {
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
    }


class ClothesDataset(Dataset):
    """PyTorch Dataset loading paired RGB image and binary clothes mask."""

    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path | Sequence[str],
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.transform = transform

        if isinstance(split_file, (str, Path)):
            with open(split_file, "r", encoding="utf-8") as f:
                self.sample_ids = [line.strip() for line in f if line.strip()]
        else:
            self.sample_ids = list(split_file)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample_id = self.sample_ids[idx]
        img_path = self.images_dir / f"{sample_id}.jpg"
        mask_path = self.masks_dir / f"{sample_id}.png"

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(image, mask)
        else:
            # Fallback simple transform
            img_tensor = TF.to_tensor(image)
            mask_tensor = (TF.to_tensor(mask) > 0.5).float()

        return img_tensor, mask_tensor, sample_id


def build_dataloaders(
    config: dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train, validation, and test DataLoaders from configuration dict."""
    data_cfg = config.get("data", {})
    data_dir = Path(data_cfg.get("data_dir", "data/processed"))
    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 4))
    image_size = data_cfg.get("image_size", 256)
    random_flip = bool(data_cfg.get("random_flip", True))
    color_jitter = bool(data_cfg.get("color_jitter", True))

    train_transform = PairedSegmentationTransform(
        image_size=image_size,
        is_train=True,
        random_flip_prob=0.5 if random_flip else 0.0,
        color_jitter=color_jitter,
    )
    eval_transform = PairedSegmentationTransform(
        image_size=image_size,
        is_train=False,
    )

    train_dataset = ClothesDataset(data_dir, splits_dir / "train.txt", transform=train_transform)
    val_dataset = ClothesDataset(data_dir, splits_dir / "val.txt", transform=eval_transform)
    test_dataset = ClothesDataset(data_dir, splits_dir / "test.txt", transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
