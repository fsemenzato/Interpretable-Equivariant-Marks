import argparse
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pk import PkEstimator


EPSILON_STABILITY = 1e-9
DEFAULT_DATA_ROOT = Path("/work/nvme/bdne/fsemenzato/quijote_z0")
DEFAULT_DATA_NAME = "df_m_128_PCS_z=0.npy"
DEFAULT_PARAM_FILE = DEFAULT_DATA_ROOT / "latin_hypercube_params.txt"
DEFAULT_PARAM_LABELS = ("Om", "Ob", "h", "ns", "s8")


@dataclass(frozen=True)
class SimulationScan:
    paths: list[Path]
    sim_indices: np.ndarray
    missing_indices: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scan_bsq_simulations(data_root: Path, data_name: str, n_params: int) -> SimulationScan:
    data_root = Path(data_root)
    paths: list[Path] = []
    available: list[int] = []
    missing: list[int] = []
    for sim_idx in range(int(n_params)):
        field_path = data_root / str(sim_idx) / data_name
        if field_path.exists():
            available.append(sim_idx)
            paths.append(field_path)
        else:
            missing.append(sim_idx)
    return SimulationScan(
        paths=paths,
        sim_indices=np.asarray(available, dtype=np.int64),
        missing_indices=np.asarray(missing, dtype=np.int64),
    )


def select_params_for_sims(params: np.ndarray, sim_indices: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=np.float32)
    sim_indices = np.asarray(sim_indices, dtype=np.int64)
    return params[sim_indices].astype(np.float32, copy=False)


def resolve_param_file(data_root: Path, param_file: Path) -> Path:
    param_file = Path(param_file).expanduser()
    if param_file.is_absolute():
        return param_file
    return Path(data_root).expanduser() / param_file


def _coerce_param_indices(raw: Any) -> Optional[list[int]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, int):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"Parameter index list must be list/tuple/int/JSON string, got {type(raw)}")
    indices = [int(item) for item in raw]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Parameter subset contains duplicates: {indices}")
    return indices or None


def select_parameter_subset_from_config(params: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, Optional[list[int]]]:
    raw_indices = (
        config.get("resolved_target_param_indices")
        or config.get("target_param_indices")
        or config.get("selected_param_indices")
        or config.get("theta_indices")
    )
    indices = _coerce_param_indices(raw_indices)
    params = np.asarray(params, dtype=np.float32)
    if indices is None:
        return params, None
    for idx in indices:
        if idx < 0 or idx >= params.shape[1]:
            raise ValueError(f"Parameter index {idx} out of range for params with shape {params.shape}")
    return params[:, indices].astype(np.float32, copy=False), indices


def normalize_theta(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float32)
    return ((theta - theta.mean(axis=0, keepdims=True)) / (theta.std(axis=0, keepdims=True) + 1e-8)).astype(
        np.float32
    )


def build_raw_features(
    pdd: np.ndarray,
    pmm: np.ndarray,
    pxm: np.ndarray,
    *,
    k_min_idx: int,
    k_max_idx: int,
) -> dict[str, np.ndarray]:
    pdd = np.asarray(pdd, dtype=np.float32)
    pmm = np.asarray(pmm, dtype=np.float32)
    pxm = np.asarray(pxm, dtype=np.float32)
    if pdd.shape != pmm.shape or pdd.shape != pxm.shape:
        raise ValueError(f"Spectra shapes disagree: {pdd.shape}, {pmm.shape}, {pxm.shape}")
    if pdd.ndim != 2:
        raise ValueError(f"Expected spectra with shape [N, K], got {pdd.shape}")
    if not (0 <= int(k_min_idx) < int(k_max_idx) <= pdd.shape[1]):
        raise ValueError(f"Invalid k slice [{k_min_idx}:{k_max_idx}] for spectra width {pdd.shape[1]}")

    s = slice(int(k_min_idx), int(k_max_idx))
    pdd_k = pdd[:, s]
    pmm_k = pmm[:, s]
    pxm_k = pxm[:, s]
    log_pdd = np.log10(pdd_k + EPSILON_STABILITY).astype(np.float32)
    log_ratio = np.log10((pmm_k + EPSILON_STABILITY) / (pdd_k + EPSILON_STABILITY)).astype(np.float32)
    r_k = pxm_k / (np.sqrt(pdd_k * pmm_k) + EPSILON_STABILITY)
    r_k = np.clip(r_k, -1.0, 1.0).astype(np.float32)
    mark_features = np.concatenate([log_ratio, r_k], axis=1).astype(np.float32)
    raw_pk_features = np.concatenate([log_pdd, log_ratio, r_k], axis=1).astype(np.float32)
    return {
        "pdd_features": log_pdd,
        "mark_features": mark_features,
        "raw_pk_features": raw_pk_features,
    }


class ProjectionHead(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2):
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(depth)):
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.GELU()])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(output_dim)))
        super().__init__(*layers)


class LinearProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(int(input_dim), int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _state_has_prefix(state: Mapping[str, Any], prefix: str) -> bool:
    needle = f"{prefix}."
    return any(str(key).startswith(needle) for key in state)


def detect_projector_layout(loss_state: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    fusion_mode = str(config.get("fusion_mode", "embedding_ortho") or "embedding_ortho").lower()
    has_pdd = _state_has_prefix(loss_state, "pdd_projector")
    has_mark = _state_has_prefix(loss_state, "mark_projector")
    has_pk = _state_has_prefix(loss_state, "pk_projector")
    has_theta = _state_has_prefix(loss_state, "theta_projector")
    has_fusion = _state_has_prefix(loss_state, "fusion_projector")

    if has_pdd and has_mark and has_fusion:
        kind = "conditional_fusion"
    elif has_pdd and has_mark:
        kind = "conditional"
    elif has_pk:
        kind = "plain_infonce"
    elif has_pdd and has_theta:
        kind = "pdd_theta"
    else:
        raise ValueError(
            "Unrecognized projector layout"
        )

    return {
        "kind": kind,
        "fusion_mode": fusion_mode,
        "ortho_mode": str(config.get("ortho_mode", "embedding") or "embedding").lower(),
        "has_pdd_projector": has_pdd,
        "has_mark_projector": has_mark,
        "has_pk_projector": has_pk,
        "has_theta_projector": has_theta,
        "has_fusion_projector": has_fusion,
    }


def load_model_class(file_path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location("latent_geometry_model", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["latent_geometry_model"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class {class_name!r} not found in {file_path}")
    return getattr(module, class_name)


def resolve_model_file(raw_model_file: str, config_path: Path) -> Path:
    raw = Path(raw_model_file).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend(
        [
            raw,
            Path.cwd() / raw,
            Path.cwd() / raw.name,
            config_path.parent / raw,
            config_path.parent / raw.name,
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Could not resolve model file {raw_model_file!r}")


def load_model_from_run(run_dir: Path, checkpoint_name: str, device: torch.device):
    checkpoint_path = run_dir / checkpoint_name
    config_path = run_dir / "config.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_file = resolve_model_file(str(config["_model_file"]), config_path)
    model_cls = load_model_class(model_file, str(config["_model_class"]))

    class ConfigObj:
        def __init__(self, payload: Mapping[str, Any]):
            self.__dict__.update(payload)

    model = model_cls(ConfigObj(config)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        stripped = {key.replace("0.", "", 1): value for key, value in state.items() if key.startswith("0.")}
        model.load_state_dict(stripped or state, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model, config, checkpoint, checkpoint_path


def _extract_prefixed_state(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    start = f"{prefix}."
    return {key[len(start) :]: value for key, value in state.items() if key.startswith(start)}


def build_projection_head_from_state(state: Mapping[str, torch.Tensor], prefix: str) -> Optional[nn.Module]:
    sub = _extract_prefixed_state(state, prefix)
    if not sub:
        return None
    if "linear.weight" in sub:
        out_dim, in_dim = sub["linear.weight"].shape
        module = LinearProjectionHead(in_dim, out_dim)
        module.load_state_dict(sub, strict=True)
        return module
    linear_weight_keys = sorted(
        (
            key
            for key, value in sub.items()
            if key.endswith(".weight") and key.split(".")[0].isdigit() and getattr(value, "ndim", 0) == 2
        ),
        key=lambda item: int(item.split(".")[0]),
    )
    if not linear_weight_keys:
        return None
    first_key = linear_weight_keys[0]
    last_key = linear_weight_keys[-1]
    hidden_dim, input_dim = sub[first_key].shape
    output_dim, _ = sub[last_key].shape
    depth = len(linear_weight_keys) - 1
    module = ProjectionHead(input_dim, hidden_dim, output_dim, depth=depth)
    module.load_state_dict(sub, strict=True)
    return module


class LatentExtractor(nn.Module):
    def __init__(self, loss_state: Mapping[str, torch.Tensor], config: Mapping[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.layout = detect_projector_layout(loss_state, config)
        self.ortho_mode = self.layout["ortho_mode"]
        self.fusion_mode = self.layout["fusion_mode"]
        self.eps = EPSILON_STABILITY
        self.pdd_projector = build_projection_head_from_state(loss_state, "pdd_projector")
        self.mark_projector = build_projection_head_from_state(loss_state, "mark_projector")
        self.pk_projector = build_projection_head_from_state(loss_state, "pk_projector")
        self.theta_projector = build_projection_head_from_state(loss_state, "theta_projector")
        self.fusion_projector = build_projection_head_from_state(loss_state, "fusion_projector")

    @staticmethod
    def _feature_residual(mark_feats: torch.Tensor, pdd_feats: torch.Tensor, eps: float) -> torch.Tensor:
        ridge = pdd_feats.var() + eps
        xtx = pdd_feats.T @ pdd_feats + ridge * torch.eye(
            pdd_feats.shape[1], device=pdd_feats.device, dtype=pdd_feats.dtype
        )
        beta = torch.linalg.solve(xtx, pdd_feats.T @ mark_feats)
        return mark_feats - pdd_feats @ beta

    @staticmethod
    def _embedding_orthogonalize(z_mark: torch.Tensor, z_pdd: torch.Tensor, eps: float) -> torch.Tensor:
        coeff = torch.einsum("bd,bd->b", z_mark, z_pdd).unsqueeze(1)
        z = z_mark - coeff * z_pdd
        return z / z.norm(dim=1, keepdim=True).clamp(min=eps)

    @torch.no_grad()
    def extract_from_features(
        self,
        features: Mapping[str, np.ndarray],
        theta_normalized: Optional[np.ndarray],
        device: torch.device,
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        pdd_feats = torch.from_numpy(features["pdd_features"]).float().to(device)
        mark_feats = torch.from_numpy(features["mark_features"]).float().to(device)
        raw_feats = torch.from_numpy(features["raw_pk_features"]).float().to(device)

        if self.pdd_projector is not None:
            self.pdd_projector.to(device).eval()
            z_pdd = F.normalize(self.pdd_projector(pdd_feats), dim=1)
            out["z_pdd"] = z_pdd.cpu().numpy()
        else:
            z_pdd = None

        if self.mark_projector is not None:
            self.mark_projector.to(device).eval()
            mark_input = mark_feats
            if self.ortho_mode == "feature" and z_pdd is not None:
                mark_input = self._feature_residual(mark_feats, pdd_feats, self.eps)
            z_mark = F.normalize(self.mark_projector(mark_input), dim=1)
            out["z_mark"] = z_mark.cpu().numpy()
            if z_pdd is not None and self.ortho_mode == "embedding":
                z_mark_ortho = self._embedding_orthogonalize(z_mark, z_pdd, self.eps)
            else:
                z_mark_ortho = z_mark
            out["z_mark_orthogonalized"] = z_mark_ortho.cpu().numpy()
            if z_pdd is not None:
                z_concat = torch.cat([z_pdd, z_mark_ortho], dim=1)
                out["z_concat"] = z_concat.cpu().numpy()
                if self.fusion_projector is not None:
                    self.fusion_projector.to(device).eval()
                    z_fused = F.normalize(self.fusion_projector(torch.cat([z_mark, z_pdd], dim=1)), dim=1)
                    out["z_fused"] = z_fused.cpu().numpy()
                    out["z_query"] = out["z_fused"]
                else:
                    out["z_query"] = out["z_mark_orthogonalized"]

        if self.pk_projector is not None:
            self.pk_projector.to(device).eval()
            z_pk = F.normalize(self.pk_projector(raw_feats), dim=1)
            out["z_pk"] = z_pk.cpu().numpy()
            out["z_query"] = out["z_pk"]

        if self.theta_projector is not None and theta_normalized is not None:
            self.theta_projector.to(device).eval()
            theta_t = torch.from_numpy(theta_normalized).float().to(device)
            out["z_theta"] = F.normalize(self.theta_projector(theta_t), dim=1).cpu().numpy()

        return out


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got {x.shape}")
    return x


def compute_retrieval_metrics(
    query: np.ndarray,
    candidates: np.ndarray,
    ks: Iterable[int] = (1, 5),
    *,
    transform: Optional[np.ndarray] = None,
) -> dict[str, float]:
    query = _as_2d_float(query)
    candidates = _as_2d_float(candidates)
    if query.shape[0] != candidates.shape[0]:
        raise ValueError("Retrieval assumes aligned query/candidate rows")
    if transform is not None:
        transform = _as_2d_float(transform)
        if query.shape[1] != transform.shape[0] or candidates.shape[1] != transform.shape[0]:
            raise ValueError(
                f"Metric transform shape {transform.shape} is incompatible with "
                f"query/candidate dims {query.shape[1]}/{candidates.shape[1]}"
            )
        query = query @ transform
        candidates = candidates @ transform
    distances = pairwise_distances(query, candidates, metric="euclidean")
    order = np.argsort(distances, axis=1)
    target = np.arange(query.shape[0])[:, None]
    ranks = np.argmax(order == target, axis=1) + 1
    result = {
        "median_rank": float(np.median(ranks)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
    }
    for k in ks:
        result[f"retrieval_at_{int(k)}"] = float(np.mean(ranks <= int(k)))
    return result


def _to_numpy_1d_or_2d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def extract_similarity_transform(
    loss_state: Mapping[str, Any],
    config: Mapping[str, Any],
    dim: int,
) -> dict[str, Any]:
    mode = str(config.get("similarity_mode", "") or "").strip().lower()
    if mode in {"diag_mahalanobis", "diagonal_mahalanobis"}:
        mode = "anisotropic"
    if not mode:
        if "mahal_log_diag" in loss_state and "mahal_off_diag" in loss_state:
            mode = "mahalanobis"
        elif "log_lambda" in loss_state:
            mode = "anisotropic"
        else:
            mode = "cosine"

    dim = int(dim)
    if mode == "anisotropic" and "log_lambda" in loss_state:
        log_lambda = _to_numpy_1d_or_2d(loss_state["log_lambda"]).reshape(-1)
        if log_lambda.shape[0] != dim:
            raise ValueError(f"log_lambda has dim {log_lambda.shape[0]}, expected {dim}")
        transform = np.diag(np.sqrt(np.exp(log_lambda))).astype(np.float64)
        lambdas = np.exp(log_lambda)
        return {
            "kind": "anisotropic",
            "transform": transform,
            "summary": {
                "lambda_mean": float(lambdas.mean()),
                "lambda_min": float(lambdas.min()),
                "lambda_max": float(lambdas.max()),
                "lambda_std": float(lambdas.std()),
            },
        }

    if mode == "mahalanobis" and "mahal_log_diag" in loss_state and "mahal_off_diag" in loss_state:
        log_diag = _to_numpy_1d_or_2d(loss_state["mahal_log_diag"]).reshape(-1)
        off_diag = _to_numpy_1d_or_2d(loss_state["mahal_off_diag"]).reshape(-1)
        if log_diag.shape[0] != dim:
            raise ValueError(f"mahal_log_diag has dim {log_diag.shape[0]}, expected {dim}")
        rows, cols = np.tril_indices(dim, k=-1)
        if off_diag.shape[0] != rows.shape[0]:
            raise ValueError(f"mahal_off_diag has {off_diag.shape[0]} entries, expected {rows.shape[0]}")
        transform = np.diag(np.exp(log_diag)).astype(np.float64)
        transform[rows, cols] = off_diag
        eigs = np.linalg.svd(transform, compute_uv=False) ** 2
        eigs = np.clip(eigs, 1e-12, None)
        return {
            "kind": "mahalanobis",
            "transform": transform,
            "summary": {
                "eig_mean": float(eigs.mean()),
                "eig_min": float(eigs.min()),
                "eig_max": float(eigs.max()),
                "condition": float(eigs.max() / eigs.min()),
                "diag_mean": float(np.exp(log_diag).mean()),
                "off_absmean": float(np.abs(off_diag).mean()) if off_diag.size else 0.0,
                "off_absmax": float(np.abs(off_diag).max()) if off_diag.size else 0.0,
            },
        }

    return {
        "kind": "cosine_or_euclidean",
        "transform": None,
        "summary": {"reason": f"No learned metric transform found for similarity_mode={mode or 'unset'}"},
    }


class FieldDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = [Path(path) for path in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        arr = np.load(self.paths[idx], mmap_mode="r").astype(np.float32).copy()
        return torch.from_numpy(arr), idx


def spectra_cache_valid(path: Path, n_expected: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as bundle:
            return all(key in bundle for key in ("P_dd", "P_mm", "P_xm")) and bundle["P_dd"].shape[0] == n_expected
    except Exception:
        return False


@torch.no_grad()
def compute_or_load_spectra(
    *,
    cache_path: Path,
    paths: list[Path],
    model: nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    force_recompute: bool,
) -> dict[str, np.ndarray]:
    if spectra_cache_valid(cache_path, len(paths)) and not force_recompute:
        with np.load(cache_path, allow_pickle=False) as bundle:
            return {
                "P_dd": np.asarray(bundle["P_dd"], dtype=np.float32),
                "P_mm": np.asarray(bundle["P_mm"], dtype=np.float32),
                "P_xm": np.asarray(bundle["P_xm"], dtype=np.float32),
            }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = FieldDataset(paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pk_estimator = PkEstimator(BoxSize=float(config["box_size"]), device=device)
    mas = str(config.get("mas", "PCS"))
    pdd_chunks: list[np.ndarray] = []
    pmm_chunks: list[np.ndarray] = []
    pxm_chunks: list[np.ndarray] = []
    model.to(device).eval()
    for delta, _ in tqdm(loader, desc="Computing learned spectra"):
        delta = delta.to(device)
        out = model(delta.unsqueeze(1))
        if not isinstance(out, Mapping) or "final_field" not in out:
            raise KeyError("Model output must be a mapping containing 'final_field'")
        marked = out["final_field"]
        pk = pk_estimator.compute(delta, marked.squeeze(1), MAS=(mas, mas))
        pdd_chunks.append(pk[:, :, 1].detach().cpu().numpy())
        pmm_chunks.append(pk[:, :, 2].detach().cpu().numpy())
        pxm_chunks.append(pk[:, :, 3].detach().cpu().numpy())

    spectra = {
        "P_dd": np.concatenate(pdd_chunks, axis=0).astype(np.float32),
        "P_mm": np.concatenate(pmm_chunks, axis=0).astype(np.float32),
        "P_xm": np.concatenate(pxm_chunks, axis=0).astype(np.float32),
    }
    np.savez_compressed(cache_path, **spectra)
    return spectra


def compute_global_geometry(x: np.ndarray) -> dict[str, Any]:
    x = _as_2d_float(x)
    x_centered = x - x.mean(axis=0, keepdims=True)
    std = x_centered.std(axis=0)
    _, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    eig = s**2
    prob = eig / np.maximum(eig.sum(), 1e-12)
    entropy = -np.sum(prob * np.log(prob + 1e-12))
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    cos = x_norm @ x_norm.T
    mask = ~np.eye(cos.shape[0], dtype=bool)
    return {
        "n": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "effective_rank": float(np.exp(entropy)),
        "std_mean": float(std.mean()),
        "std_max": float(std.max()),
        "std_min": float(std.min()),
        "mean_offdiag_cosine": float(cos[mask].mean()) if mask.any() else float("nan"),
        "pca_explained_variance_ratio": [float(v) for v in prob[: min(20, len(prob))]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--cbase", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-name", default=DEFAULT_DATA_NAME)
    parser.add_argument("--param-file", type=Path, default=DEFAULT_PARAM_FILE)
    parser.add_argument("--output-dir", type=Path, default="/u/fsemenzato/negtrail/latent_geometry")
    parser.add_argument("--checkpoint-name", default="best_checkpoint.pt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-sims", type=int, default=0)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_run_dir(root: str, cbase: str) -> Path:
    direct = Path(root) / cbase
    if direct.exists():
        return direct
    alt = Path("./") / root / cbase
    if alt.exists():
        return alt
    return direct


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = resolve_run_dir(args.root, args.cbase)
    output_dir = args.output_dir 
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, checkpoint, checkpoint_path = load_model_from_run(run_dir, args.checkpoint_name, device)

    loss_state = checkpoint["loss_fn"]
    layout = detect_projector_layout(loss_state, config)
    k_min_idx = int(config.get("k_min_idx", 0))
    k_max_idx = int(config.get("k_max_idx", 47))

    param_path = resolve_param_file(args.data_root, args.param_file)
    params = np.loadtxt(param_path).astype(np.float32)
    if params.ndim != 2:
        raise ValueError(f"Expected 2D parameter file, got {params.shape}")
    params, param_indices = select_parameter_subset_from_config(params, config)

    scan = scan_bsq_simulations(args.data_root, args.data_name, n_params=params.shape[0])
    if scan.sim_indices.size == 0:
        raise RuntimeError(f"No simulations found in {args.data_root} with data_name={args.data_name}")
    paths = scan.paths
    sim_indices = scan.sim_indices
    if args.max_sims and args.max_sims > 0:
        paths = paths[: args.max_sims]
        sim_indices = sim_indices[: args.max_sims]

    theta = select_params_for_sims(params, sim_indices)
    theta_normalized = normalize_theta(theta)

    spectra = compute_or_load_spectra(
        cache_path=output_dir / "spectra.npz",
        paths=paths,
        model=model,
        config=config,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force_recompute=args.force_recompute,
    )
    features = build_raw_features(
        spectra["P_dd"], spectra["P_mm"], spectra["P_xm"],
        k_min_idx=k_min_idx, k_max_idx=k_max_idx,
    )

    extractor = LatentExtractor(loss_state, config).to(device).eval()
    latents = extractor.extract_from_features(features, theta_normalized, device)
    np.savez_compressed(
        output_dir / "latents.npz",
        sim_indices=sim_indices, theta=theta, theta_normalized=theta_normalized,
        **spectra, **features, **latents,
    )

    metric_info = {"kind": "cosine_or_euclidean", "summary": {}}
    if "z_theta" in latents:
        info = extract_similarity_transform(loss_state, config, dim=latents["z_theta"].shape[1])
        metric_info = {"kind": info.get("kind"), "summary": info.get("summary", {})}

    retrieval: dict[str, Any] = {}
    if "z_query" in latents and "z_theta" in latents:
        retrieval["z_query_to_z_theta"] = compute_retrieval_metrics(
            latents["z_query"], latents["z_theta"], ks=(1, 5))

    geometry = {key: compute_global_geometry(latents[key])
                for key in ("z_query", "z_theta") if key in latents}

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "data_root": str(args.data_root),
        "data_name": args.data_name,
        "param_file": str(param_path),
        "param_indices": param_indices,
        "n_available": int(len(sim_indices)),
        "k_min_idx": k_min_idx,
        "k_max_idx": k_max_idx,
        "layout": layout,
        "similarity_metric": metric_info,
        "retrieval": retrieval,
        "geometry": geometry,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote latent geometry analysis to {output_dir}")


if __name__ == "__main__":
    main()
