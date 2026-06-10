from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional

PK_AUTO_FIELD1 = 1
PK_AUTO_FIELD2 = 2
PK_CROSS = 3
EPSILON_STABILITY = 1e-8



class ProjectionHead(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2):
        layers = []
        in_dim = input_dim
        for _ in range(int(depth)):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        super().__init__(*layers)


class LinearProjectionHead(nn.Module):
    # For the parameter projector 

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _configure_theta_projector(
    theta_projector: nn.Module,
    projector_type: str,
    input_dim: int,
    output_dim: int,
    identity_init: bool = False,
    freeze: bool = False,
) -> None:
    

    if identity_init:
        linear = theta_projector.linear
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.zero_()
            eye_dim = min(input_dim, output_dim)
            linear.weight[:eye_dim, :eye_dim] = torch.eye(
                eye_dim, device=linear.weight.device, dtype=linear.weight.dtype
            )

    if freeze:
        for param in theta_projector.parameters():
            param.requires_grad_(False)


def _set_module_requires_grad(module: Optional[nn.Module], requires_grad: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def _clone_frozen_module(module: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(module)
    teacher.eval()
    _set_module_requires_grad(teacher, False)
    return teacher


def _extract_loss_state_dict(checkpoint_or_state) -> Dict[str, torch.Tensor]:
    raw = checkpoint_or_state
    if isinstance(raw, (str, bytes)) or hasattr(raw, "__fspath__"):
        raw = torch.load(raw, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "loss_fn" in raw and isinstance(raw["loss_fn"], dict):
        return raw["loss_fn"]
    if not isinstance(raw, dict):
        raise TypeError(f"Expected checkpoint path or state dict, got {type(raw)}")
    return raw


def _load_module_from_prefixed_state(
    module: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    source_prefixes: Tuple[str, ...],
) -> List[str]:
    current = module.state_dict()
    subset: Dict[str, torch.Tensor] = {}
    loaded: List[str] = []
    for source_prefix in source_prefixes:
        prefix = f"{source_prefix}."
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            inner_key = key[len(prefix):]
            if inner_key not in current:
                continue
            if current[inner_key].shape != value.shape:
                continue
            subset[inner_key] = value
            loaded.append(f"{source_prefix}.{inner_key}")
    if subset:
        module.load_state_dict(subset, strict=False)
    return sorted(set(loaded))


def load_bootstrap_projectors(
    loss_fn: nn.Module,
    checkpoint_or_state,
    *,
    load_pdd: bool = True,
    load_mark: bool = False,
    copy_pdd_to_mark: bool = False,
    load_theta: bool = True,
) -> Dict[str, List[str]]:
    state_dict = _extract_loss_state_dict(checkpoint_or_state)
    loaded: Dict[str, List[str]] = {"pdd": [], "mark": [], "theta": []}
    if load_pdd and hasattr(loss_fn, "pdd_projector"):
        loaded["pdd"] = _load_module_from_prefixed_state(
            loss_fn.pdd_projector,
            state_dict,
            ("pdd_projector", "pk_projector"),
        )
    if load_mark and hasattr(loss_fn, "mark_projector"):
        source_prefixes = ("pdd_projector", "pk_projector") if copy_pdd_to_mark else ("mark_projector",)
        loaded["mark"] = _load_module_from_prefixed_state(
            loss_fn.mark_projector,
            state_dict,
            source_prefixes,
        )
    if load_theta and hasattr(loss_fn, "theta_projector"):
        loaded["theta"] = _load_module_from_prefixed_state(
            loss_fn.theta_projector,
            state_dict,
            ("theta_projector",),
        )
    return loaded


def capture_bootstrap_teachers(
    loss_fn: nn.Module,
    checkpoint_or_state=None,
    *,
    include_pdd: bool = True,
    include_mark: bool = False,
    copy_pdd_to_mark: bool = False,
    include_theta: bool = True,
) -> Dict[str, bool]:
    state_dict = _extract_loss_state_dict(checkpoint_or_state) if checkpoint_or_state is not None else None
    pdd_teacher = None
    mark_teacher = None
    theta_teacher = None
    if include_pdd and hasattr(loss_fn, "pdd_projector"):
        pdd_teacher = _clone_frozen_module(loss_fn.pdd_projector)
        if state_dict is not None:
            _load_module_from_prefixed_state(pdd_teacher, state_dict, ("pdd_projector", "pk_projector"))
    if include_mark and hasattr(loss_fn, "mark_projector"):
        mark_teacher = _clone_frozen_module(loss_fn.mark_projector)
        if state_dict is not None:
            prefixes = ("pdd_projector", "pk_projector") if copy_pdd_to_mark else ("mark_projector",)
            _load_module_from_prefixed_state(mark_teacher, state_dict, prefixes)
    if include_theta and hasattr(loss_fn, "theta_projector"):
        theta_teacher = _clone_frozen_module(loss_fn.theta_projector)
        if state_dict is not None:
            _load_module_from_prefixed_state(theta_teacher, state_dict, ("theta_projector",))

    object.__setattr__(loss_fn, "_bootstrap_pdd_teacher", pdd_teacher)
    object.__setattr__(loss_fn, "_bootstrap_mark_teacher", mark_teacher)
    object.__setattr__(loss_fn, "_bootstrap_theta_teacher", theta_teacher)
    return {
        "pdd_teacher": pdd_teacher is not None,
        "mark_teacher": mark_teacher is not None,
        "theta_teacher": theta_teacher is not None,
    }


def clear_bootstrap_teachers(loss_fn: nn.Module) -> None:
    object.__setattr__(loss_fn, "_bootstrap_pdd_teacher", None)
    object.__setattr__(loss_fn, "_bootstrap_mark_teacher", None)
    object.__setattr__(loss_fn, "_bootstrap_theta_teacher", None)


def set_bootstrap_projector_trainability(
    loss_fn: nn.Module,
    *,
    train_pdd: Optional[bool] = None,
    train_theta: Optional[bool] = None,
    train_mark: Optional[bool] = None,
) -> None:
    if train_pdd is not None and hasattr(loss_fn, "pdd_projector"):
        _set_module_requires_grad(loss_fn.pdd_projector, train_pdd)
    if train_theta is not None and hasattr(loss_fn, "theta_projector"):
        _set_module_requires_grad(loss_fn.theta_projector, train_theta)
    if train_mark is not None and hasattr(loss_fn, "mark_projector"):
        _set_module_requires_grad(loss_fn.mark_projector, train_mark)


class InfoNCEAdaptive(nn.Module):

    def __init__(self,
                 k_min: int,
                 k_max: int,
                 theta_dim: int,
                 embedding_dim: int = 64,
                 temperature: float = 0.07,
                 distance_mask_threshold: float = 0.0,
                 theta_projector_type: str = "mlp",
                 summary_projector_type: str = "mlp",
                 similarity_mode: str = "cosine",
                 projector_hidden_dim: int = 0,
                 ortho_mode: str = "none",
                 theta_projector_identity_init: bool = False,
                 freeze_theta_projector: bool = False,
                 **kwargs):
        super().__init__()
        
        self.k_min = k_min
        self.k_max = k_max
        self.distance_mask_threshold = float(distance_mask_threshold)
        self.similarity_mode = similarity_mode
        self.summary_projector_type = summary_projector_type
        self.eps = 1e-9

        # dim = k_range * 3 (log_pdd, log_ratio, r_k)
        k_range = k_max - k_min
        self.pk_feat_dim = k_range * 3

        pk_hid = projector_hidden_dim if projector_hidden_dim > 0 else self.pk_feat_dim * 2
        if summary_projector_type == "linear":
            self.pk_projector = LinearProjectionHead(self.pk_feat_dim, embedding_dim)
        else:
            self.pk_projector = ProjectionHead(self.pk_feat_dim, pk_hid, embedding_dim)

        if theta_projector_type == "linear":
            self.theta_projector = LinearProjectionHead(theta_dim, embedding_dim)
        else:
            theta_hid = projector_hidden_dim if projector_hidden_dim > 0 else theta_dim * 2
            self.theta_projector = ProjectionHead(theta_dim, theta_hid, embedding_dim)
        _configure_theta_projector(
            self.theta_projector,
            projector_type=theta_projector_type,
            input_dim=theta_dim,
            output_dim=embedding_dim,
            identity_init=theta_projector_identity_init,
            freeze=freeze_theta_projector,
        )

        self.register_buffer('logit_scale', torch.ones([]) * np.log(1 / temperature))

        if self.similarity_mode == "anisotropic":
            self.log_lambda = nn.Parameter(torch.zeros(embedding_dim))
        elif self.similarity_mode == "mahalanobis":
            self.mahal_log_diag = nn.Parameter(torch.zeros(embedding_dim))
            n_off = embedding_dim * (embedding_dim - 1) // 2
            self.mahal_off_diag = nn.Parameter(torch.zeros(n_off))

    def _extract_features(self, pk_tensor: torch.Tensor) -> torch.Tensor:
        # log_pdd, log_ratio, r_k] 
        sub_pk = pk_tensor[:, self.k_min:self.k_max, :]
        P_dd = sub_pk[:, :, 1]
        P_mm = sub_pk[:, :, 2]
        P_dm = sub_pk[:, :, 3]
        eps = self.eps

        log_pdd = torch.log10(P_dd + eps)
        log_ratio = torch.log10((P_mm + eps) / (P_dd + eps))
        r_k = P_dm / (torch.sqrt(P_dd * P_mm) + eps)
        r_k = torch.clamp(r_k, -1.0, 1.0)
        return torch.cat([log_pdd, log_ratio, r_k], dim=1)  # [B, 3*k_range]

    def _build_distance_mask(self, theta_real, theta_all):
        if self.distance_mask_threshold <= 0:
            return None, 0.0
        with torch.no_grad():
            distances = torch.cdist(theta_real, theta_all, p=2)
            close_mask = distances < self.distance_mask_threshold
            bsz = theta_real.shape[0]
            diag_idx = torch.arange(bsz, device=theta_real.device)
            close_mask[diag_idx, diag_idx] = False
            masked_frac = close_mask.float().mean().item()
        return close_mask, masked_frac

    def _build_cholesky_L(self) -> torch.Tensor:
        D = self.mahal_log_diag.shape[0]
        device = self.mahal_log_diag.device
        dtype = self.mahal_log_diag.dtype
        L_diag = torch.diag_embed(self.mahal_log_diag.exp())
        tril_row, tril_col = torch.tril_indices(D, D, offset=-1, device=device)
        zero = torch.zeros(D, D, device=device, dtype=dtype)
        L_off = zero.index_put((tril_row, tril_col), self.mahal_off_diag)
        return L_diag + L_off

    def _similarity_diagnostics(self) -> Dict[str, float]:
        if self.similarity_mode == "anisotropic":
            lam = self.log_lambda.exp()
            return {
                "aniso_lambda_mean": lam.mean().item(),
                "aniso_lambda_std": lam.std().item(),
                "aniso_lambda_max": lam.max().item(),
                "aniso_lambda_min": lam.min().item(),
            }
        if self.similarity_mode == "mahalanobis":
            with torch.no_grad():
                L = self._build_cholesky_L()
                diag_vals = self.mahal_log_diag.to(torch.float64).clamp(min=-30.0, max=30.0).exp()
                off_abs = self.mahal_off_diag.to(torch.float64).abs()
                try:
                    sv = torch.linalg.svdvals(L.to(torch.float64))
                    eigs = sv.square().clamp(min=1e-12)
                except RuntimeError:
                    eigs = diag_vals.square().clamp(min=1e-12)
            return {
                "mahal_eig_mean": eigs.mean().item(),
                "mahal_eig_min": eigs.min().item(),
                "mahal_eig_max": eigs.max().item(),
                "mahal_cond": (eigs.max() / eigs.min()).item(),
                "mahal_diag_mean": diag_vals.mean().item(),
                "mahal_off_absmean": off_abs.mean().item() if off_abs.numel() > 0 else 0.0,
                "mahal_off_absmax": off_abs.max().item() if off_abs.numel() > 0 else 0.0,
            }
        return {}

    def _compute_logits(self, z_query, z_candidates):
        scale = self.logit_scale.exp()
        if self.similarity_mode == "anisotropic":
            lam = self.log_lambda.exp()
            diff = z_query.unsqueeze(1) - z_candidates.unsqueeze(0)
            logits = -(diff ** 2 * lam.unsqueeze(0).unsqueeze(0)).sum(dim=2)
            return scale * logits
        elif self.similarity_mode == "mahalanobis":
            L = self._build_cholesky_L()
            diff = z_query.unsqueeze(1) - z_candidates.unsqueeze(0)
            u = diff @ L
            return scale * (-(u ** 2).sum(dim=2))
        else:
            return scale * (z_query @ z_candidates.T)

    def monitor_embeddings(self, pk_marked, theta_real) -> Dict[str, torch.Tensor]:
        pk_feats = self._extract_features(pk_marked)
        z_summary = F.normalize(self.pk_projector(pk_feats), dim=1)
        z_theta = F.normalize(self.theta_projector(theta_real), dim=1)
        return {"z_summary": z_summary, "z_theta": z_theta}

    def forward(self, pk_marked, theta_real, theta_distractors=None) -> dict:
        pk_feats = self._extract_features(pk_marked)
        z_pk = F.normalize(self.pk_projector(pk_feats), dim=1)
        z_theta = F.normalize(self.theta_projector(theta_real), dim=1)

        if theta_distractors is not None:
            z_dist = F.normalize(self.theta_projector(theta_distractors), dim=1)
            z_all_candidates = torch.cat([z_theta, z_dist], dim=0)
            theta_all = torch.cat([theta_real, theta_distractors], dim=0)
        else:
            z_all_candidates = z_theta
            theta_all = theta_real

        logits = self._compute_logits(z_pk, z_all_candidates)
        distance_mask, masked_frac = self._build_distance_mask(theta_real, theta_all)
        if distance_mask is not None:
            logits = logits.masked_fill(distance_mask, -1e9)

        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            scale = self.logit_scale.exp()
            k_range = self.k_max - self.k_min
            log_ratio_part = pk_feats[:, k_range:2 * k_range]
            r_k_part = pk_feats[:, 2 * k_range:]
            log_ratio_var = log_ratio_part.var(dim=1).mean()
            r_k_deviation = (r_k_part - 1.0).abs().mean()

        result = {
            'loss': loss,
            'accuracy': acc.item(),
            'temperature': 1.0 / scale.item(),
            'logit_scale': scale.item(),
            'z_pk_norm': z_pk.norm(dim=1).mean().item(),
            'log_ratio_var': log_ratio_var.item(),
            'r_k_deviation': r_k_deviation.item(),
            'distance_masked_frac': masked_frac,
        }
        result.update(self._similarity_diagnostics())
        return result


class ConditionalInfoNCEAdaptive(nn.Module):

    def __init__(self,
                 k_min: int,
                 k_max: int,
                 theta_dim: int,
                 embedding_dim: int = 64,
                 temperature: float = 0.07,
                 distance_mask_threshold: float = 0.0,
                 theta_projector_type: str = "mlp",
                 summary_projector_type: str = "mlp",
                 similarity_mode: str = "cosine",
                 ortho_mode: str = "embedding",
                 projector_hidden_dim: int = 0,
                 projector_depth: int = 2):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.distance_mask_threshold = float(distance_mask_threshold)
        self.theta_projector_type = theta_projector_type
        self.summary_projector_type = summary_projector_type
        self.similarity_mode = similarity_mode

        self.ortho_mode = ortho_mode
        self.fusion_mode = "embedding_ortho"

        # Dim = (k_max-k_min)*2 (log(Pmm/Pdd), r_k)
        self.mark_feat_dim = (k_max - k_min) * 2
        # Dim = (k_max-k_min) (log(Pdd))
        self.pdd_feat_dim = k_max - k_min

        mark_hid = projector_hidden_dim if projector_hidden_dim > 0 else self.mark_feat_dim * 2
        pdd_hid = projector_hidden_dim if projector_hidden_dim > 0 else self.pdd_feat_dim * 2

        if summary_projector_type == "linear":
            self.mark_projector = LinearProjectionHead(self.mark_feat_dim, embedding_dim)
            self.pdd_projector = LinearProjectionHead(self.pdd_feat_dim, embedding_dim)
        else:
            self.mark_projector = ProjectionHead(self.mark_feat_dim, mark_hid, embedding_dim, depth=projector_depth)
            self.pdd_projector = ProjectionHead(self.pdd_feat_dim, pdd_hid, embedding_dim, depth=projector_depth)
        if theta_projector_type == "linear":
            self.theta_projector = LinearProjectionHead(theta_dim, embedding_dim)
        else:
            theta_hid = projector_hidden_dim if projector_hidden_dim > 0 else theta_dim * 2
            self.theta_projector = ProjectionHead(theta_dim, theta_hid, embedding_dim, depth=projector_depth)
        self.eps = 1e-9

        self.register_buffer('logit_scale', torch.ones([]) * np.log(1 / temperature))

        if self.similarity_mode == "anisotropic":
            self.log_lambda = nn.Parameter(torch.zeros(embedding_dim))
        elif self.similarity_mode == "mahalanobis":
            self.mahal_log_diag = nn.Parameter(torch.zeros(embedding_dim))
            n_off = embedding_dim * (embedding_dim - 1) // 2
            self.mahal_off_diag = nn.Parameter(torch.zeros(n_off))
        object.__setattr__(self, "_bootstrap_pdd_teacher", None)
        object.__setattr__(self, "_bootstrap_theta_teacher", None)

    def _extract_features(self, pk_tensor):
        sub_pk = pk_tensor[:, self.k_min:self.k_max, :]
        P_dd = sub_pk[:, :, 1]
        P_mm = sub_pk[:, :, 2]
        P_dm = sub_pk[:, :, 3]
        eps = self.eps

        log_pdd = torch.log10(P_dd + eps)
        log_ratio = torch.log10((P_mm + eps) / (P_dd + eps))
        r_k = P_dm / (torch.sqrt(P_dd * P_mm) + eps)
        r_k = torch.clamp(r_k, -1.0, 1.0)
        var_lr = log_ratio.var(dim=1).mean()
        mark_features = torch.cat([log_ratio, r_k], dim=1)
        return mark_features, log_pdd, var_lr

    def _build_distance_mask(self, theta_real, theta_all):
        if self.distance_mask_threshold <= 0:
            return None, 0.0
        with torch.no_grad():
            distances = torch.cdist(theta_real, theta_all, p=2)
            close_mask = distances < self.distance_mask_threshold
            bsz = theta_real.shape[0]
            diag_idx = torch.arange(bsz, device=theta_real.device)
            close_mask[diag_idx, diag_idx] = False
            masked_frac = close_mask.float().mean().item()
        return close_mask, masked_frac

    def _orthogonalize(self, z_mark, z_pdd):

        eps = self.eps
        if self.ortho_mode == "none":
            z_comp_pre_norm = z_mark.norm(dim=1)
            return z_mark, z_comp_pre_norm, z_pdd
        z_pdd_norm = z_pdd 
        proj_coeff = torch.einsum('bd,bd->b', z_mark, z_pdd_norm).unsqueeze(1)
        z_complementary = z_mark - proj_coeff * z_pdd_norm
        z_comp_pre_norm = z_complementary.norm(dim=1)
        z_comp_norm = z_comp_pre_norm.unsqueeze(1).clamp(min=eps)
        z_complementary = z_complementary / z_comp_norm
        return z_complementary, z_comp_pre_norm, z_pdd_norm

    def _build_cholesky_L(self) -> torch.Tensor:
        D = self.mahal_log_diag.shape[0]
        device = self.mahal_log_diag.device
        dtype = self.mahal_log_diag.dtype
        L_diag = torch.diag_embed(self.mahal_log_diag.exp())
        tril_row, tril_col = torch.tril_indices(D, D, offset=-1, device=device)
        zero = torch.zeros(D, D, device=device, dtype=dtype)
        L_off = zero.index_put((tril_row, tril_col), self.mahal_off_diag)
        return L_diag + L_off

    def _similarity_diagnostics(self) -> Dict[str, float]:
        if self.similarity_mode == "anisotropic":
            lam = self.log_lambda.exp()
            return {
                "aniso_lambda_mean": lam.mean().item(),
                "aniso_lambda_std": lam.std().item(),
                "aniso_lambda_max": lam.max().item(),
                "aniso_lambda_min": lam.min().item(),
            }
        if self.similarity_mode == "mahalanobis":
            with torch.no_grad():
                L = self._build_cholesky_L()
                diag_vals = self.mahal_log_diag.to(torch.float64).clamp(min=-30.0, max=30.0).exp()
                off_abs = self.mahal_off_diag.to(torch.float64).abs()
                try:
                    sv = torch.linalg.svdvals(L.to(torch.float64))
                    eigs = sv.square().clamp(min=1e-12)
                except RuntimeError:
                    eigs = diag_vals.square().clamp(min=1e-12)
            return {
                "mahal_eig_mean": eigs.mean().item(),
                "mahal_eig_min": eigs.min().item(),
                "mahal_eig_max": eigs.max().item(),
                "mahal_cond": (eigs.max() / eigs.min()).item(),
                "mahal_diag_mean": diag_vals.mean().item(),
                "mahal_off_absmean": off_abs.mean().item() if off_abs.numel() > 0 else 0.0,
                "mahal_off_absmax": off_abs.max().item() if off_abs.numel() > 0 else 0.0,
            }
        return {}

    def _compute_logits(self, z_query, z_candidates):
        scale = self.logit_scale.exp()
        if self.similarity_mode == "anisotropic":
            lam = self.log_lambda.exp()
            diff = z_query.unsqueeze(1) - z_candidates.unsqueeze(0)
            logits = -(diff ** 2 * lam.unsqueeze(0).unsqueeze(0)).sum(dim=2)
            return scale * logits
        elif self.similarity_mode == "mahalanobis":
            L = self._build_cholesky_L()
            diff = z_query.unsqueeze(1) - z_candidates.unsqueeze(0)
            u = diff @ L
            return scale * (-(u ** 2).sum(dim=2))
        else:
            return scale * (z_query @ z_candidates.T)

    def monitor_embeddings(self, pk_marked, theta_real) -> Dict[str, torch.Tensor]:
        mark_feats, pdd_feats, _ = self._extract_features(pk_marked)
        z_mark = F.normalize(self.mark_projector(mark_feats), dim=1)
        z_pdd = F.normalize(self.pdd_projector(pdd_feats), dim=1)
        z_theta = F.normalize(self.theta_projector(theta_real), dim=1)
        z_summary, _, _ = self._orthogonalize(z_mark, z_pdd)
        return {"z_summary": z_summary, "z_theta": z_theta}

    def _collect_diag_metrics(self, mark_feats, pdd_feats, z_mark, z_pdd_diag,
                              z_comp_pre_norm, masked_frac):
        with torch.no_grad():
            scale = self.logit_scale.exp()
            k_range = self.k_max - self.k_min
            log_ratio_var = mark_feats[:, :k_range].var(dim=1).mean()
            r_k_vals = mark_feats[:, k_range:]
            r_k_deviation = (r_k_vals - 1.0).abs().mean()
            cosine_per_sample = (z_mark * z_pdd_diag).sum(dim=1)
            result = {
                'temperature': 1.0 / scale.item(),
                'logit_scale': scale.item(),
                'z_comp_norm': z_comp_pre_norm.mean().item(),
                'log_ratio_var': log_ratio_var.item(),
                'r_k_deviation': r_k_deviation.item(),
                'mark_pdd_cosine': cosine_per_sample.mean().item(),
                'abs_mark_pdd_cosine': cosine_per_sample.abs().mean().item(),
                'mark_feat_var': mark_feats.var(dim=1).mean().item(),
                'pdd_feat_var': pdd_feats.var(dim=1).mean().item(),
                'distance_masked_frac': masked_frac,
            }
        result.update(self._similarity_diagnostics())
        return result

    def forward(self, pk_marked, theta_real, theta_distractors=None) -> dict:

        mark_feats, pdd_feats, var_lr = self._extract_features(pk_marked)

        # Project into the embedding space
        z_mark = F.normalize(self.mark_projector(mark_feats), dim=1)
        z_pdd = F.normalize(self.pdd_projector(pdd_feats), dim=1)
        z_theta = F.normalize(self.theta_projector(theta_real), dim=1)

        # orthogonalization
        z_complementary, z_comp_pre_norm, z_pdd_diag = self._orthogonalize(z_mark, z_pdd)

        # Embed distractors
        if theta_distractors is not None:
            z_dist = F.normalize(self.theta_projector(theta_distractors), dim=1)
            z_all_candidates = torch.cat([z_theta, z_dist], dim=0)
            theta_all = torch.cat([theta_real, theta_distractors], dim=0)
        else:
            z_all_candidates = z_theta
            theta_all = theta_real

        # Contrastive loss
        logits = self._compute_logits(z_complementary, z_all_candidates)
        distance_mask, masked_frac = self._build_distance_mask(theta_real, theta_all)
        if distance_mask is not None:
            logits = logits.masked_fill(distance_mask, -1e9)

        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()

        result = {
            'loss': loss,
            'accuracy': acc.item(),
            'fusion_mode': self.fusion_mode,
            'query_pdd_cosine': (z_complementary * z_pdd_diag).sum(dim=1).mean().item(),
        }
        result.update(
            self._collect_diag_metrics(
                mark_feats=mark_feats, pdd_feats=pdd_feats, z_mark=z_mark,
                z_pdd_diag=z_pdd_diag, z_comp_pre_norm=z_comp_pre_norm,
                masked_frac=masked_frac))
        return result


class PddThetaInfoNCEAdaptive(InfoNCEAdaptive):
    # pretraining 
    
    def __init__(self, k_min, k_max, theta_dim, embedding_dim=64, temperature=0.07,
                 distance_mask_threshold=0.0, theta_projector_type="mlp",
                 summary_projector_type="mlp", similarity_mode="cosine",
                 projector_hidden_dim=0, theta_projector_identity_init=False,
                 freeze_theta_projector=False, **kwargs):
        super().__init__(
            k_min=k_min, k_max=k_max, theta_dim=theta_dim, embedding_dim=embedding_dim,
            temperature=temperature, distance_mask_threshold=distance_mask_threshold,
            theta_projector_type=theta_projector_type,
            summary_projector_type=summary_projector_type,
            similarity_mode=similarity_mode, projector_hidden_dim=projector_hidden_dim,
            theta_projector_identity_init=theta_projector_identity_init,
            freeze_theta_projector=freeze_theta_projector, **kwargs)
        del self._modules["pk_projector"]
        self.pk_feat_dim = k_max - k_min
        pdd_hid = projector_hidden_dim if projector_hidden_dim > 0 else self.pk_feat_dim * 2
        if summary_projector_type == "linear":
            self.pdd_projector = LinearProjectionHead(self.pk_feat_dim, embedding_dim)
        else:
            self.pdd_projector = ProjectionHead(self.pk_feat_dim, pdd_hid, embedding_dim)

    @property
    def pk_projector(self) -> nn.Module:
        return self.pdd_projector

    def _extract_features(self, pk_tensor: torch.Tensor) -> torch.Tensor:
        sub_pk = pk_tensor[:, self.k_min:self.k_max, :]
        p_dd = sub_pk[:, :, PK_AUTO_FIELD1]
        return torch.log10(p_dd + self.eps)

    def monitor_embeddings(self, pk_marked, theta_real) -> Dict[str, torch.Tensor]:
        pdd_feats = self._extract_features(pk_marked)
        z_pdd = F.normalize(self.pdd_projector(pdd_feats), dim=1)
        z_theta = F.normalize(self.theta_projector(theta_real), dim=1)
        return {"z_summary": z_pdd, "z_theta": z_theta}

    def forward_pdd_features(self, pdd_feats, theta_real, theta_distractors=None) -> dict:
        z_pdd = F.normalize(self.pdd_projector(pdd_feats), dim=1)
        z_theta = F.normalize(self.theta_projector(theta_real), dim=1)

        if theta_distractors is not None:
            z_dist = F.normalize(self.theta_projector(theta_distractors), dim=1)
            z_all_candidates = torch.cat([z_theta, z_dist], dim=0)
            theta_all = torch.cat([theta_real, theta_distractors], dim=0)
        else:
            z_all_candidates = z_theta
            theta_all = theta_real

        logits = self._compute_logits(z_pdd, z_all_candidates)
        distance_mask, masked_frac = self._build_distance_mask(theta_real, theta_all)
        if distance_mask is not None:
            logits = logits.masked_fill(distance_mask, -1e9)

        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            scale = self.logit_scale.exp()

        result = {
            "loss": loss,
            "accuracy": acc.item(),
            "temperature": 1.0 / scale.item(),
            "logit_scale": scale.item(),
            "z_pdd_norm": z_pdd.norm(dim=1).mean().item(),
            "pdd_feat_var": pdd_feats.var(dim=1).mean().item(),
            "distance_masked_frac": masked_frac,
        }
        result.update(self._similarity_diagnostics())
        return result

    def forward(self, pk_marked, theta_real, theta_distractors=None) -> dict:
        pdd_feats = self._extract_features(pk_marked)
        return self.forward_pdd_features(pdd_feats, theta_real, theta_distractors)
