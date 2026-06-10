from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class TrainingConfig:
    
    # wiring
    save_dir: str = "./runs/fsp"
    _model_file: str = "/u/fsemenzato/negtrail/marks/mark.py"
    _model_class: str = "MarkSH"
    bootstrap_pretrain_from: str = "/u/fsemenzato/negtrail/marks/runs/precomp_d16/best_bootstrap.pt"

    # mark architecture
    sh_basis: str = "physical"            # physical | e3nn
    radial: str = "gaussian"              # gaussian | fourier_mlp
    taper: str = "cosine"                 # cosine | mas
    mark_positivity: str = "free"         # free | softplus
    l_max: int = 2
    
    dc_normalize: bool = True
    l2_normalize_high_l: bool = False     # per-l L2-normalize W_l for l>=1
    l2_normalize_l0: bool = False         # L2-normalize the l=0 channel
    subtract_dc_high_l: bool = False      # fourier_mlp: DC-subtract l>=1 FALSE, no DC component
    
    gaussian_high_l_init: str = "match_l0"  # match_l0 | zero | noise (l>=1 seed)
    n_radial_basis: int = 12              # gaussian radial
    k0_frac: float = 0.2                  # gaussian radial
    radial_hidden: int = 16               # fourier_mlp radial
    radial_n_fourier: int = 8             # fourier_mlp radial
    nyquist_taper_kpass: float = 0.85
    nyquist_taper_floor: float = 0.0
    mas_taper_power: float = 4.0          # mas taper exponent
    input_transform: str = "log1p"
    sh_convention: str = "physics"   
    gam_hidden_dim: int = 16
    cross_hidden_dim: int = 8
    cross_inputs: tuple = ("E0", "E2", "I3_norm")
    

    # data
    mas_scheme: str = "PCS"
    box_size: float = 1000.0
    grid_dim: int = 128
    data_root: str = "/path/to/BSQ_128"
    param_file: str = "BSQ_params.txt"
    data_name: str = "dm_z0_PCS_128.npy"
    num_workers: int = 8

    # optimization
    device: str = "cuda"
    batch_size: int = 16
    num_epochs: int = 700
    learning_rate: float = 5e-4
    marker_lr_multiplier: float = 2.0
    theta_projector_lr_multiplier: float = 0.25
    pdd_projector_lr_multiplier: float = 0.0
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0
    patience: int = 20
    min_delta: float = 1e-5
    use_lr_scheduler: bool = True
    lr_patience: int = 8
    lr_gamma: float = 0.5
    lr_min: float = 1e-7

    # loss
    loss_function: str = "conditional_infonce_adaptive"
    similarity_mode: str = "mahalanobis"
    ortho_mode: str = "embedding"
    embedding_dim: int = 16
    projector_hidden_dim: int = 64
    projector_depth: int = 2
    theta_projector_type: str = "linear"
    summary_projector_type: str = "mlp"
    temperature: float = 0.05
    k_min_idx: int = 0
    k_max_idx: int = 47 # kmax=0.3 h/Mpc

    # negative sampling
    n_real: int = 128
    n_far: int = 384
    n_close: int = 32
    neighborhood_scale: float = 0.15
    min_distance_scale: float = 0.03
    negative_exclusion_scale: float = 0.03

    # validation
    val_negative_scheme: str = "train_like"       # global_fixed | train_like | val_local_global
    val_fraction: float = 0.15
    val_every_n_epochs: int = 1
    n_val_synthetic: int = 512
    save_every_n_epochs: int = 1

    # bootstrap wiring
    bootstrap_warmup_epochs: int = 20
    bootstrap_freeze_pdd_projector: bool = True
    reuse_pdd_feature_cache: bool = True

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)



def _resolve_paths(cfg: "TrainingConfig", config_path: Optional[Path]) -> None:
    base = config_path.parent if config_path else Path.cwd()
    for attr in ("data_root", 
                "save_dir", 
                "bootstrap_pretrain_from",):
        val = getattr(cfg, attr)
        if val and not Path(val).is_absolute():
            setattr(cfg, attr, str((base / val).resolve()))


def load_config(config_path: str, n_sims: int = -1, stage: Optional[str] = None) -> "TrainingConfig":
    path = Path(config_path).resolve()
    data = json.loads(path.read_text())
    cfg = TrainingConfig()
    
    known = {f.name for f in fields(cfg)}
    for k, v in data.items():
        if k in known:
            setattr(cfg, k, v)

    if n_sims is not None:
        cfg.n_sims = n_sims
    if stage is not None:
        cfg.stage = stage
    cfg.sh_convention = "physics" if cfg.sh_basis == "physical" else "e3nn_component"
    _resolve_paths(cfg, path)
    return cfg
