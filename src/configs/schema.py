"""Typed experiment configuration schema, loader, and validator."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, List, Optional
import yaml

from src.data.dataset import DEFAULT_CLOTHES_LABEL_IDS


class ConfigError(ValueError):
    """Raised when an experiment configuration is malformed or violates constraints."""


@dataclass
class ExperimentSectionConfig:
    name: str = "atr_resnet34_unet"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "artifacts/runs/atr_resnet34_unet"


@dataclass
class DataSectionConfig:
    dataset: str = "atr"
    data_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    raw_dir: str = "data/raw"
    batch_size: int = 16
    num_workers: int = 4
    image_size: int = 256
    random_flip: bool = True
    color_jitter: bool = True
    clothes_ids: List[int] = field(default_factory=lambda: list(DEFAULT_CLOTHES_LABEL_IDS))


@dataclass
class ModelSectionConfig:
    name: str = "resnet34_unet"
    in_channels: int = 3
    out_channels: int = 1
    pretrained: bool = True
    features: List[int] = field(default_factory=lambda: [32, 64, 128, 256])


@dataclass
class LossSectionConfig:
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    smooth: float = 1.0
    pos_weight: Optional[float] = None


@dataclass
class TrainingSectionConfig:
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_accum_steps: int = 1
    gradient_clip_val: float = 1.0
    save_every: int = 5
    eval_every: int = 1


@dataclass
class EvaluationSectionConfig:
    batch_size: int = 16
    threshold: float = 0.5
    target_iou: float = 0.80
    target_dice: float = 0.88


@dataclass
class VerifySectionConfig:
    memory_budget_mib: int = 3072


@dataclass
class ExperimentConfig:
    experiment: ExperimentSectionConfig = field(default_factory=ExperimentSectionConfig)
    data: DataSectionConfig = field(default_factory=DataSectionConfig)
    model: ModelSectionConfig = field(default_factory=ModelSectionConfig)
    loss: LossSectionConfig = field(default_factory=LossSectionConfig)
    training: TrainingSectionConfig = field(default_factory=TrainingSectionConfig)
    evaluation: EvaluationSectionConfig = field(default_factory=EvaluationSectionConfig)
    verify: VerifySectionConfig = field(default_factory=VerifySectionConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS: dict[str, type] = {
    "experiment": ExperimentSectionConfig,
    "data": DataSectionConfig,
    "model": ModelSectionConfig,
    "loss": LossSectionConfig,
    "training": TrainingSectionConfig,
    "evaluation": EvaluationSectionConfig,
    "verify": VerifySectionConfig,
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration into a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def _build_section(name: str, raw: Any) -> Any:
    cls = _SECTIONS[name]
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Section '{name}' must be a mapping")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in section '{name}': {unknown}. Allowed: {sorted(allowed)}")
    return cls(**raw)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_config(config: dict[str, Any]) -> ExperimentConfig:
    """Validate a raw config dict and return the typed ExperimentConfig."""
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a mapping")
    unknown_sections = sorted(set(config) - set(_SECTIONS))
    if unknown_sections:
        raise ConfigError(f"Unknown section(s): {unknown_sections}. Allowed: {sorted(_SECTIONS)}")

    cfg = ExperimentConfig(**{name: _build_section(name, config.get(name)) for name in _SECTIONS})
    e, d, m, l, t, ev, v = (
        cfg.experiment,
        cfg.data,
        cfg.model,
        cfg.loss,
        cfg.training,
        cfg.evaluation,
        cfg.verify,
    )

    # experiment
    _require(isinstance(e.name, str) and e.name != "", "experiment.name must be a non-empty string")
    _require(_is_int(e.seed) and e.seed >= 0, "experiment.seed must be an int >= 0")
    _require(e.device in ("auto", "cuda", "cpu"), "experiment.device must be one of 'auto' / 'cuda' / 'cpu'")

    # data
    _require(_is_int(d.batch_size) and d.batch_size >= 1, "data.batch_size must be >= 1")
    _require(_is_int(d.num_workers) and d.num_workers >= 0, "data.num_workers must be >= 0")
    _require(_is_int(d.image_size) and d.image_size >= 32, "data.image_size must be >= 32")

    # model
    _require(isinstance(m.name, str) and m.name != "", "model.name must be non-empty string")
    _require(_is_int(m.in_channels) and m.in_channels >= 1, "model.in_channels must be >= 1")
    _require(_is_int(m.out_channels) and m.out_channels >= 1, "model.out_channels must be >= 1")

    # training
    _require(_is_int(t.epochs) and t.epochs >= 1, "training.epochs must be >= 1")
    _require(_is_num(t.lr) and t.lr > 0, "training.lr must be > 0")
    _require(_is_int(t.grad_accum_steps) and t.grad_accum_steps >= 1, "training.grad_accum_steps must be >= 1")

    # verify
    _require(_is_int(v.memory_budget_mib) and v.memory_budget_mib > 0, "verify.memory_budget_mib must be > 0")

    return cfg


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate raw config and return as a full dictionary with all defaults populated."""
    return validate_config(config).to_dict()
