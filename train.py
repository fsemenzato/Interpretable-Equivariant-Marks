from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from sched import scheduler
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainingConfig, load_config
from data import (QuijoteDataset, scan_available_sims, get_mixed_distractors, setup_validation_negatives)
from losses import ConditionalInfoNCEAdaptive, PddThetaInfoNCEAdaptive, load_bootstrap_projectors
from tqdm import tqdm

import importlib.util
import sys

from pk import PkEstimator







def set_global_determinism(seed: int, *, full: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if full:
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  
            pass


def _seed_worker(worker_id: int) -> None:
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


def build_dataloader_kwargs(config, *, shuffle: bool) -> dict:
    
    num_workers = int(getattr(config, "num_workers", 0))
    pin_memory = bool(getattr(config, "pin_memory", str(getattr(config, "device", "cpu")).startswith("cuda")))
    persistent_workers = bool(getattr(config, "persistent_workers", num_workers > 0))
    prefetch_factor = getattr(config, "prefetch_factor", 2)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if shuffle:
        loader_kwargs["shuffle"] = True
        seed = int(getattr(config, "seed", 42))
        g = torch.Generator()
        g.manual_seed(seed)
        loader_kwargs["generator"] = g
        loader_kwargs["worker_init_fn"] = _seed_worker
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return loader_kwargs















def _add_param_group(
    param_groups: list[dict],
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    group_name: str,
    seen: set[int],
) -> None:
    unique_params = []
    for param in params:
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        unique_params.append(param)
    if not unique_params:
        return
    param_groups.append({"params": unique_params, "lr": float(lr), "group_name": group_name})


def build_optimizer(
    model: Optional[nn.Module],
    loss_fn: nn.Module,
    config,
) -> tuple[torch.optim.Optimizer, list[tuple[str, float, int]]]:
    base_lr = float(config.learning_rate)
    seen: set[int] = set()
    param_groups: list[dict] = []

    if model is not None:
        _add_param_group(
            param_groups,
            model.parameters(),
            lr=base_lr * float(getattr(config, "marker_lr_multiplier", 1.0)),
            group_name="marker",
            seen=seen,
        )
    if hasattr(loss_fn, "mark_projector"):
        _add_param_group(
            param_groups,
            loss_fn.mark_projector.parameters(),
            lr=base_lr * float(getattr(config, "mark_projector_lr_multiplier", 1.0)),
            group_name="mark_projector",
            seen=seen,
        )
    if hasattr(loss_fn, "pdd_projector"):
        _add_param_group(
            param_groups,
            loss_fn.pdd_projector.parameters(),
            lr=base_lr * float(getattr(config, "pdd_projector_lr_multiplier", 1.0)),
            group_name="pdd_projector",
            seen=seen,
        )
    if hasattr(loss_fn, "pk_projector"):
        _add_param_group(
            param_groups,
            loss_fn.pk_projector.parameters(),
            lr=base_lr * float(getattr(config, "pk_projector_lr_multiplier", 1.0)),
            group_name="pk_projector",
            seen=seen,
        )
    if hasattr(loss_fn, "theta_projector"):
        _add_param_group(
            param_groups,
            loss_fn.theta_projector.parameters(),
            lr=base_lr * float(getattr(config, "theta_projector_lr_multiplier", 1.0)),
            group_name="theta_projector",
            seen=seen,
        )
    _add_param_group(
        param_groups,
        loss_fn.parameters(),
        lr=base_lr,
        group_name="loss_misc",
        seen=seen,
    )

    if not param_groups:
        raise RuntimeError("No optimizer parameter groups were created.")

    summary = []
    for group in param_groups:
        summary.append((group["group_name"], float(group["lr"]), len(group["params"])))
        group.pop("group_name", None)

    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=config.weight_decay)
    return optimizer, summary























def _config_to_dict(config) -> dict:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return {k: v for k, v in vars(config).items() if not k.startswith("_")}


def _write_config_json(config, save_dir: Path) -> None:
    with open(save_dir / "config.json", "w") as f:
        json.dump(_config_to_dict(config), f, indent=2)


def _param_file_path(config) -> Path:
    pf = Path(config.param_file)
    return pf if pf.is_absolute() else Path(config.data_root) / pf


def _setup_data(config):
    data_root = Path(config.data_root)
    param_path = _param_file_path(config)
    all_params = np.loadtxt(param_path).astype(np.float32)
    if all_params.ndim == 1:
        all_params = all_params[:, None]
    n_params = len(all_params)

    available = np.array(scan_available_sims(data_root, config.data_name, n_params))
    if len(available) == 0:
        raise RuntimeError(f"No simulations found in {data_root} with data_name={config.data_name}")

    seed = int(getattr(config, "seed", 42))
    np.random.seed(seed)
    np.random.shuffle(available)

    n_cap = int(getattr(config, "n_sims", -1))
    if n_cap is not None and n_cap > 0 and n_cap < len(available):
        available = available[:n_cap]

    n_val = int(config.val_fraction * len(available))
    val_sim_indices = available[:n_val]
    train_sim_indices = available[n_val:]

    train_params_phys = torch.from_numpy(all_params[train_sim_indices]).float()
    val_params_phys = torch.from_numpy(all_params[val_sim_indices]).float()
    normalizer = (train_params_phys.mean(dim=0), train_params_phys.std(dim=0))

    pf = str(param_path)
    train_dataset = QuijoteDataset(data_root, config.data_name, pf, train_sim_indices, normalizer)
    val_dataset = QuijoteDataset(data_root, config.data_name, pf, val_sim_indices, normalizer)

    train_loader = DataLoader(train_dataset, **build_dataloader_kwargs(config, shuffle=True))
    val_loader = DataLoader(val_dataset, **build_dataloader_kwargs(config, shuffle=False))
    theta_dim = train_dataset.params.shape[1]
    
    return (train_loader, val_loader, normalizer, train_params_phys, val_params_phys, theta_dim)



def load_model_class(file_path: str, class_name: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Mark file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("dynamic_mark_mod", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_mark_mod"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {file_path}")

    return getattr(module, class_name)




















def _mixed_distractors(config, theta_anchor, train_params_phys, normalizer, n_real):
    return get_mixed_distractors(
        theta_anchor, train_params_phys, batch_indices=None,
        n_real=int(n_real),
        n_global_synthetic=int(getattr(config, "n_far", 256)),
        n_local_synthetic=int(getattr(config, "n_close", 32)),
        neighborhood_scale=float(getattr(config, "neighborhood_scale", 0.03)),
        min_distance_scale=float(getattr(config, "min_distance_scale", 0.01)),
        negative_exclusion_scale=float(getattr(config, "negative_exclusion_scale", 0.0)),
        device=config.device, theta_mean=normalizer[0], theta_std=normalizer[1],
        theta_is_normalized=True)


def _batch_distractors(config, theta_real, train_params_phys, normalizer):
    return _mixed_distractors(config, theta_real, train_params_phys, normalizer,
                              n_real=int(getattr(config, "n_real", 64)))



def _build_fixed_val_pool(config, val_params_phys, train_params_phys, normalizer, device):
    scheme = str(getattr(config, "val_negative_scheme", "train_like"))
    if scheme == "global_fixed":
        return setup_validation_negatives(
            val_params=val_params_phys, train_params=train_params_phys,
            theta_mean=normalizer[0], theta_std=normalizer[1],
            n_synthetic=int(getattr(config, "n_val_synthetic", 512)),
            seed=int(getattr(config, "seed", 42)),
            negative_exclusion_scale=float(getattr(config, "negative_exclusion_scale", 0.0)),
        ).to(device)
    if scheme in ("train_like", "val_local_global"):
        return None
    raise ValueError(
        f"Unknown val_negative_scheme: {scheme!r} "
        "(expected 'global_fixed', 'train_like' or 'val_local_global')")



def _val_distractors(config, theta_anchor, neg_params, normalizer, batch_idx):
    scheme = str(getattr(config, "val_negative_scheme", "train_like"))
    n_real = int(getattr(config, "n_real", 64)) if scheme == "train_like" else 0
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(int(getattr(config, "seed", 42)) * 100003 + int(batch_idx))
    try:
        return _mixed_distractors(config, theta_anchor, neg_params, normalizer, n_real)
    finally:
        torch.random.set_rng_state(rng_state)































def _build_main_loss(config, theta_dim: int) -> ConditionalInfoNCEAdaptive:
    return ConditionalInfoNCEAdaptive(
        k_min=config.k_min_idx, k_max=config.k_max_idx, theta_dim=theta_dim,
        embedding_dim=config.embedding_dim,
        temperature=float(getattr(config, "temperature", 0.07)),
        distance_mask_threshold=float(getattr(config, "distance_mask_threshold", 0.0)),
        theta_projector_type=getattr(config, "theta_projector_type", "mlp"),
        summary_projector_type=getattr(config, "summary_projector_type", "mlp"),
        similarity_mode=getattr(config, "similarity_mode", "cosine"),
        ortho_mode=getattr(config, "ortho_mode", "embedding"),
        projector_hidden_dim=int(getattr(config, "projector_hidden_dim", 0)),
        projector_depth=int(getattr(config, "projector_depth", 2)),
    )



def _marked_pk(model, pk_estimator, delta, mas_scheme):
    outputs = model(delta)
    final_field = outputs["final_field"]
    return pk_estimator.compute(delta, final_field, MAS=(mas_scheme, mas_scheme))


def _save_curve(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if history["train"]:
            ax.plot(history["train"], label="train")
        if history["val"]:
            ax.plot(history["val"], label="val")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
        fig.savefig(path, dpi=80)
        plt.close(fig)
    except Exception:
        pass

























def train_main(config) -> None:
    
    set_global_determinism(int(getattr(config, "seed", 42)))
    device = config.device
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _write_config_json(config, save_dir)

    (train_loader, val_loader, normalizer, train_params_phys, val_params_phys, theta_dim) = _setup_data(config)

    MarkModelClass = load_model_class(config._model_file, config._model_class)
    model = MarkModelClass(config).to(device)
    pk_estimator = PkEstimator(BoxSize=config.box_size, device=device)
    loss_fn = _build_main_loss(config, theta_dim).to(device)
    
    

    boot = getattr(config, "bootstrap_pretrain_from", "")
    if boot and Path(boot).exists():
        load_bootstrap_projectors(loss_fn, boot)

    optimizer, optimizer_groups = build_optimizer(model, loss_fn, config)
    scheduler = None
    if bool(getattr(config, "use_lr_scheduler", True)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(getattr(config, "lr_gamma", 0.5)),
            patience=int(getattr(config, "lr_patience", 8)),
            min_lr=float(getattr(config, "lr_min", 1e-7)))

    mas_scheme = str(getattr(config, "mas_scheme", "PCS"))
    theta_mean = normalizer[0].to(device)
    theta_std = normalizer[1].to(device)
    
    val_pool = _build_fixed_val_pool(config, val_params_phys, train_params_phys, normalizer, device)
    val_neg_params = (None if val_pool is not None else torch.cat([train_params_phys, val_params_phys], dim=0))

    history = {"train": [], "val": []}
    best_val = float("inf")
    patience_left = int(getattr(config, "patience", 20))
    min_delta = float(getattr(config, "min_delta", 1e-5))
    grad_clip = float(getattr(config, "grad_clip_norm", 10.0))
    
    
    start_epoch = 0
    resume_from = getattr(config, "resume_from", None)
    if resume_from or save_dir.joinpath("best_checkpoint.pt").exists():
        if resume_from is None:
            resume_from = save_dir / "best_checkpoint.pt"
        checkpoint = torch.load(resume_from, map_location=config.device, weights_only=False)
        start_epoch = checkpoint.get('epoch', 0)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_fn.load_state_dict(checkpoint['loss_fn'])
        history = checkpoint.get('history', history)
        boot = getattr(config, "bootstrap_pretrain_from", "")
        if boot and Path(boot).exists():
            load_bootstrap_projectors(loss_fn, boot)
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"Resumed from epoch {start_epoch}")




    for epoch in range(start_epoch, int(config.num_epochs)):
        model.train()
        loss_fn.train()
        epoch_losses = []
        
        # Train
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for bidx,batch in enumerate(pbar):
            delta, theta_real, _ = batch
            delta = delta.to(device); theta_real = theta_real.to(device)
            theta_dist = _batch_distractors(config, theta_real, train_params_phys, normalizer)
            optimizer.zero_grad()
            pk_results = _marked_pk(model, pk_estimator, delta, mas_scheme)
            metrics = loss_fn(pk_results, theta_real, theta_dist)
            loss = metrics["loss"]
            loss.backward()
            params = [p for p in list(model.parameters()) + list(loss_fn.parameters()) if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            
                
            if epoch == 0 and bidx == 0:
                n_real = getattr(config, 'n_real', 64)
                n_global = getattr(config, 'n_far', 256)
                n_local = getattr(config, 'n_close', 32)
                print(f"  Training distractors: {len(theta_dist)} "
                        f"({n_real} real + {n_global} global + {config.batch_size}*{n_local} local)")
                print(f"  theta_real  mean/std: {theta_real.mean().item():.3f} / {theta_real.std().item():.3f}")
                print(f"  theta_dist  mean/std: {theta_dist.mean().item():.3f} / {theta_dist.std().item():.3f}")

            
            
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history["train"].append(train_loss)
        


        # Validation
        val_loss = train_loss
        if (epoch % int(getattr(config, "val_every_n_epochs", 1))) == 0:
            model.eval(); loss_fn.eval()
            v_losses = []
            with torch.no_grad():
                for vb, (delta, theta_real, _) in enumerate(val_loader):
                    delta = delta.to(device); theta_real = theta_real.to(device)
                    pk_results = _marked_pk(model, pk_estimator, delta, mas_scheme)
                    pool = val_pool if val_pool is not None else _val_distractors(
                        config, theta_real, val_neg_params, normalizer, vb)
                    m = loss_fn(pk_results, theta_real, pool)
                    v_losses.append(float(m["loss"].item()))
            if v_losses:
                val_loss = float(np.mean(v_losses))
        history["val"].append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)
            
            
        print(f"Epoch {epoch+1}: Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Best val: {best_val:.4f} | Patience left: {patience_left}")

        if val_loss < best_val - min_delta:
            improvement_pct = (best_val - val_loss) / best_val * 100 if best_val > 0 else float("inf")
            print(f"  New best validation loss: {val_loss:.4f} (improved by {improvement_pct:.2f}%)")
            best_val = val_loss
            patience_left = int(getattr(config, "patience", 20))
            torch.save({
                "model": model.state_dict(),
                "loss": loss_fn.state_dict(),
                "config": _config_to_dict(config),
                "theta_mean": normalizer[0],
                "theta_std": normalizer[1],
                "epoch": epoch,
                'optimizer': optimizer.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'history': history,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'bootstrap_status': {
                    "train_mark_projector": hasattr(loss_fn, "mark_projector"),
                    "train_theta_projector": hasattr(loss_fn, "theta_projector"),
                    "freeze_pdd_projector": bool(getattr(config, "bootstrap_freeze_pdd_projector", True)),
                },
                "val_loss": val_loss,
            }, save_dir / "best_checkpoint.pt")
            _save_curve(history, save_dir / "loss_curve.png")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
            
        if (epoch + 1) % 5 == 0:
            torch.save({
                "model": model.state_dict(),
                "loss": loss_fn.state_dict(),
                "config": _config_to_dict(config),
                "theta_mean": normalizer[0],
                "theta_std": normalizer[1],
                "epoch": epoch,
                'optimizer': optimizer.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'history': history,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'bootstrap_status': {
                    "train_mark_projector": hasattr(loss_fn, "mark_projector"),
                    "train_theta_projector": hasattr(loss_fn, "theta_projector"),
                    "freeze_pdd_projector": bool(getattr(config, "bootstrap_freeze_pdd_projector", True)),
                },
                "val_loss": val_loss,
            }, save_dir / "last_epoch_checkpoint.pt")
            _save_curve(history, save_dir / "loss_curve.png")


    if not (save_dir / "best_checkpoint.pt").exists():
            torch.save({
                "model": model.state_dict(),
                "loss": loss_fn.state_dict(),
                "config": _config_to_dict(config),
                "theta_mean": normalizer[0],
                "theta_std": normalizer[1],
                "epoch": epoch,
                'optimizer': optimizer.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'history': history,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'bootstrap_status': {
                    "train_mark_projector": hasattr(loss_fn, "mark_projector"),
                    "train_theta_projector": hasattr(loss_fn, "theta_projector"),
                    "freeze_pdd_projector": bool(getattr(config, "bootstrap_freeze_pdd_projector", True)),
                },
                "val_loss": val_loss,
        }, save_dir / "best_checkpoint.pt")
    _save_curve(history, save_dir / "loss_curve.png")
    

















def _build_bootstrap_loss(config, theta_dim: int) -> PddThetaInfoNCEAdaptive:
    return PddThetaInfoNCEAdaptive(
        k_min=config.k_min_idx, k_max=config.k_max_idx, theta_dim=theta_dim,
        embedding_dim=config.embedding_dim,
        temperature=float(getattr(config, "temperature", 0.07)),
        distance_mask_threshold=float(getattr(config, "distance_mask_threshold", 0.0)),
        theta_projector_type=getattr(config, "theta_projector_type", "mlp"),
        summary_projector_type=getattr(config, "summary_projector_type", "mlp"),
        similarity_mode=getattr(config, "similarity_mode", "cosine"),
        projector_hidden_dim=int(getattr(config, "projector_hidden_dim", 0)),
    )







def _pk_cache_meta(config) -> dict:
    """Split-INDEPENDENT cache key: only the field geometry matters for P(k)."""
    return {
        "data_root": str(config.data_root), "data_name": str(config.data_name),
        "grid_dim": int(config.grid_dim), "box_size": float(config.box_size),
        "mas_scheme": str(config.mas_scheme),
    }


def _precompute_pk(config, sim_indices, normalizer, pk_estimator, device):
    from data import QuijoteDataset
    ds = QuijoteDataset(config.data_root, config.data_name,
                        str(_param_file_path(config)), sim_indices, normalizer)
    loader = DataLoader(ds, **build_dataloader_kwargs(config, shuffle=False))
    mas = str(getattr(config, "mas_scheme", "PCS"))
    
    pks = []
    with torch.no_grad():
        for delta, _, _ in tqdm(loader, desc="Precomputing P(k)", total=len(loader)):
            d = delta.to(device)
            pks.append(pk_estimator.compute(d, d, MAS=(mas, mas)).float().cpu())
    print(f"P(k) precompute: {len(sim_indices)} sims")
    return torch.cat(pks, dim=0)


def _build_pk_cache(config, sim_indices, normalizer, pk_estimator, device, save_dir):
    cache_path = save_dir / "pk_cache.npz"
    meta = _pk_cache_meta(config)
    sim_indices = np.asarray(sim_indices, dtype=np.int64)
    if bool(getattr(config, "reuse_pdd_feature_cache", True)) and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as b:
                if json.loads(str(b["meta"])) == meta:
                    lut = {int(s): i for i, s in enumerate(b["sim_indices"])}
                    if all(int(s) in lut for s in sim_indices):
                        rows = [lut[int(s)] for s in sim_indices]
                        print(f"reusing P(k) cache at {cache_path}")
                        return torch.from_numpy(b["pk"][rows]).to(device)
        except Exception as exc:  # pragma: no cover
            print(f"unreadable P(k) cache: {exc}")
    pk = _precompute_pk(config, sim_indices, normalizer, pk_estimator, device)
    np.savez(cache_path, meta=np.array(json.dumps(meta)),
             sim_indices=sim_indices, pk=pk.numpy())
    return pk


def train_bootstrap(config) -> None:
    set_global_determinism(int(getattr(config, "seed", 42)))
    device = config.device
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _write_config_json(config, save_dir)

    (train_loader, val_loader, normalizer, train_params_phys, val_params_phys, theta_dim) = _setup_data(config)

    pk_estimator = PkEstimator(BoxSize=config.box_size, device=device)
    train_sims = np.asarray(train_loader.dataset.sim_indices, dtype=np.int64)
    val_sims = np.asarray(val_loader.dataset.sim_indices, dtype=np.int64)
    all_sims = np.concatenate([train_sims, val_sims])
    all_pk = _build_pk_cache(config, all_sims, normalizer, pk_estimator, device, save_dir)
    n_tr = len(train_sims)
    train_pk, val_pk = all_pk[:n_tr], all_pk[n_tr:]
    eps = 1e-8
    train_theta = (train_params_phys - normalizer[0]) / (normalizer[1] + eps)
    val_theta = (val_params_phys - normalizer[0]) / (normalizer[1] + eps)

    loss_fn = _build_bootstrap_loss(config, theta_dim).to(device)
    optimizer, parsum = build_optimizer(None, loss_fn, config)
    scheduler = None
    if bool(getattr(config, "use_lr_scheduler", True)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(getattr(config, "lr_gamma", 0.5)),
            patience=int(getattr(config, "lr_patience", 8)),
            min_lr=float(getattr(config, "lr_min", 1e-7)))

    val_pool = _build_fixed_val_pool(config, val_params_phys, train_params_phys, normalizer, device)
    val_neg_params = (None if val_pool is not None else torch.cat([train_params_phys, val_params_phys], dim=0))

    from torch.utils.data import TensorDataset, DataLoader as _DL
    g = torch.Generator(); g.manual_seed(int(getattr(config, "seed", 42)))
    bs = int(config.batch_size)
    train_ds = TensorDataset(train_pk, train_theta)
    train_pk_loader = _DL(train_ds, batch_size=bs, shuffle=True, generator=g)

    history = {"train": [], "val": []}
    best_val = float("inf")
    grad_clip = float(getattr(config, "grad_clip_norm", 10.0))

    for epoch in range(int(config.num_epochs)):
        loss_fn.train()
        epoch_losses = []
        pbar = tqdm(train_pk_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for pk_batch, theta_batch in pbar:
            pk_batch = pk_batch.to(device); theta_batch = theta_batch.to(device)
            theta_dist = _batch_distractors(config, theta_batch, train_params_phys, normalizer)
            optimizer.zero_grad()
            loss = loss_fn(pk_batch, theta_batch, theta_dist)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in loss_fn.parameters() if p.requires_grad], max_norm=grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history["train"].append(train_loss)

        loss_fn.eval()
        v_losses = []
        with torch.no_grad():
            for bi, i in enumerate(range(0, val_pk.shape[0], bs)):
                pk_batch = val_pk[i:i + bs].to(device)
                theta_batch = val_theta[i:i + bs].to(device)
                pool = val_pool if val_pool is not None else _val_distractors(
                    config, theta_batch, val_neg_params, normalizer, bi)
                v_losses.append(float(loss_fn(pk_batch, theta_batch, pool)["loss"].item()))
        val_loss = float(np.mean(v_losses)) if v_losses else train_loss
        history["val"].append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{config.num_epochs} - Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val:
            print(f"  New best validation loss: {val_loss:.4f} (improved from {best_val:.4f})")
            best_val = val_loss
            torch.save({"loss_fn": loss_fn.state_dict(),
                        "config": _config_to_dict(config), "epoch": epoch},
                       save_dir / "bootstrap_checkpoint.pt")

    if not (save_dir / "bootstrap_checkpoint.pt").exists():
        torch.save({"loss_fn": loss_fn.state_dict(),
                    "config": _config_to_dict(config),
                    "epoch": int(config.num_epochs) - 1},
                   save_dir / "bootstrap_checkpoint.pt")
    _save_curve(history, save_dir / "bootstrap_loss_curve.png")
    
    print("Bootstrap training completed.")











def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["bootstrap", "main"], default="main")
    parser.add_argument("--n_sims", type=int, default=-1)
    args = parser.parse_args()
    
    
    config = load_config(args.config, n_sims=args.n_sims, stage=args.stage)
    if args.stage == "bootstrap":
        train_bootstrap(config)
    else:
        train_main(config)


if __name__ == "__main__":
    main()
  