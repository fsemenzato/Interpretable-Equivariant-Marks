from __future__ import annotations
import json
import os
import re
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

TRAINER_KEYS = frozenset(
    {
        "_comment",
        "_model_file",
        "_model_class",
        "_cell_tokens",
        "_grid",
        "_grid_cell",
        "n_sims",
        "stage",
        "code_hash",
        "resume_from",
        "_cell",
        "seed",
    }
)


@dataclass
class TrainingConfig:
    save_dir: str = "./runs/run"
    _model_class: str = "HessianMark"
    bootstrap_pretrain_from: str = ""
    mas_scheme: str = "PCS"
    box_size: float = 1000.0
    grid_dim: int = 128
    data_root: str = "${DATA}/BSQ_128"
    param_file: str = "BSQ_params.txt"
    data_name: str = "dm_z0_PCS_128.npy"
    num_workers: int = 8
    device: str = "cuda"
    batch_size: int = 8
    num_epochs: int = 700
    learning_rate: float = 0.0005
    marker_lr_multiplier: float = 2.0
    theta_projector_lr_multiplier: float = 0.25
    pdd_projector_lr_multiplier: float = 0.0
    weight_decay: float = 0.0001
    grad_clip_norm: float = 10.0
    patience: int = 20
    min_delta: float = 1e-05
    use_lr_scheduler: bool = True
    lr_patience: int = 8
    lr_gamma: float = 0.5
    lr_min: float = 1e-07
    seed: int = 42
    split_seed: Optional[int] = None
    loss_function: str = "conditional_infonce"
    marked_features: str = "pmm"
    similarity_mode: str = "mahalanobis"
    ortho_mode: str = "embedding"
    embedding_dim: int = 16
    projector_hidden_dim: int = 64
    projector_depth: int = 2
    theta_projector_type: str = "linear"
    summary_projector_type: str = "mlp"
    temperature: float = 0.05
    k_min_idx: int = 0
    k_max_idx: int = 47
    mahal_log_diag_clamp: Optional[float] = None
    mahal_diagonal: bool = False
    metric_trainable: bool = True
    metric_lr_multiplier: float = 1.0
    feature_standardize: bool = False
    feature_stats_file: str = ""
    load_bootstrap_metric: bool = False
    mark_reg_weight: float = 0.0
    legacy_clip_frozen_head: bool = False
    head_hidden_dim: int = 128
    head_depth: int = 2
    head_dropout: float = 0.0
    regression_features: str = "pmm"
    keep_every_n_epochs: int = 0
    n_real: int = 128
    n_far: int = 384
    n_close: int = 32
    neighborhood_scale: float = 0.15
    min_distance_scale: float = 0.03
    negative_exclusion_scale: float = 0.03
    local_all_anchor_exclusion_scale: float = 0.03
    val_negative_scheme: str = "train_like"
    val_fraction: float = 0.15
    n_val_synthetic: int = 512
    save_last_every_n_epochs: int = 5
    auto_resume: bool = False

    def to_dict(self) -> dict:
        from dataclasses import asdict

        d = asdict(self)
        for k, v in vars(self).items():
            if k not in d and (not k.startswith("__")):
                if k == "_provided_keys":
                    continue
                d[k] = v
        return d


_ENV_RE = re.compile("\\$\\{([A-Za-z_][A-Za-z0-9_]*)\\}")


def expand_path(value: str, base: Path) -> str:
    def _sub(m):
        name = m.group(1)
        if name not in os.environ:
            raise KeyError(
                f"config path uses ${{{name}}} but it is not set in the environment"
            )
        return os.environ[name]

    v = _ENV_RE.sub(_sub, str(value))
    p = Path(v)
    return str(p if p.is_absolute() else (base / p).resolve())


def read_config(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Configuration inheritance cycle at {path}")
    data = json.loads(path.read_text())
    parent = data.pop("extends", None)
    base = read_config(path.parent / parent, (*seen, path)) if parent else {}
    for key in (
        "data_root",
        "save_dir",
        "bootstrap_pretrain_from",
        "feature_stats_file",
    ):
        if data.get(key):
            data[key] = expand_path(data[key], path.parent)
    return {**base, **data}


def load_config(
    config_path: str,
    n_sims: int = -1,
    stage: Optional[str] = None,
    warn_unknown: bool = True,
) -> "TrainingConfig":
    path = Path(config_path).resolve()
    data = read_config(path)
    cfg = TrainingConfig()
    from marks import get_mark_class

    mark_cls = get_mark_class(data.get("_model_class", cfg._model_class))
    for key, value in getattr(mark_cls, "DEFAULTS", {}).items():
        setattr(cfg, key, value)
    known = {f.name for f in fields(cfg)}
    for k, v in data.items():
        setattr(cfg, k, v)
    cfg._provided_keys = set(data.keys())
    cfg.n_sims = int(n_sims) if n_sims is not None else -1
    if stage is not None:
        cfg.stage = stage
    for attr in (
        "data_root",
        "save_dir",
        "bootstrap_pretrain_from",
        "feature_stats_file",
    ):
        val = getattr(cfg, attr)
        if val:
            setattr(cfg, attr, expand_path(val, path.parent))
    if cfg._model_class == "MarkSH":
        cfg.sh_convention = (
            "physics" if cfg.sh_basis == "physical" else "e3nn_component"
        )
    if warn_unknown:
        from marks import get_mark_class

        try:
            mark_cls = get_mark_class(cfg._model_class)
            mark_keys = set(getattr(mark_cls, "KNOWN_KEYS", ()))
        except KeyError:
            mark_keys = set()
        unknown = sorted(
            (
                k
                for k in data
                if k not in known
                and k not in mark_keys
                and (k not in TRAINER_KEYS)
                and (not k.startswith("_comment"))
            )
        )
        if unknown:
            warnings.warn(
                f"{path.name}: {len(unknown)} config key(s) are read by neither the trainer nor {cfg._model_class}: {unknown}",
                RuntimeWarning,
                stacklevel=2,
            )
    return cfg
