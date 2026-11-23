from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from marks._mark_base import CanonicalMark
from marks.tidal_base import (
    _SOFTPLUS_INV1,
    EigenvalueExtractor,
    HessianFromDeltaPlus,
    MultiThresholdWebGate,
    ScalarCurve,
    SingleChannelFilterBankPlus,
    signlog,
)

from marks._mark_base import ENVIRONMENTS as _ENV_NAMES

_EPS = 1e-08
KNOWN_KEYS = frozenset(
    {
        "grid_dim",
        "grid_size",
        "box_size",
        "mas_scheme",
        "mas",
        "use_log1p",
        "log_compression_kind",
        "filter_radial_kind",
        "filter_n_bins",
        "filter_init_sigma",
        "filter_n_basis",
        "filter_k0_frac",
        "filter_lowpass_R_mpc",
        "filter_dc_normalize",
        "filter_nyquist_taper_kpass",
        "filter_nyquist_taper_floor",
        "filter_k_norm_convention",
        "filter_keep_corner_modes",
        "disable_mas_filter",
        "filter_mas_before_log1p",
        "filter_order",
        "hessian_smooth_order",
        "gate_input_roll",
        "hessian_use_log1p",
        "hessian_operator",
        "disable_mas_hessian",
        "hessian_mas_before_log1p",
        "hessian_smooth_R_voxels",
        "hessian_k_grid_units",
        "hessian_corner_mask",
        "hessian_pre_taper_kpass",
        "hessian_pre_taper_floor",
        "hessian_nyquist_fix",
        "gate_init_tau",
        "gate_mode",
        "gate_learnable_tau",
        "gate_learnable_threshold",
        "gate_eig_scale",
        "gate_eig_transform",
        "gate_eig_box_normalize",
        "include_gated_experts",
        "include_cross_term",
        "include_ungated_term",
        "include_eigen_main_effects",
        "eigen_input_transform",
        "include_anisotropy_term",
        "anisotropy_norm",
        "anisotropy_input_transform",
        "gam_hidden_dim",
        "gam_act",
        "expert_output_bound",
        "mark_positivity",
        "mark_normalize",
    }
)


def _g(config, name, default):
    v = getattr(config, name, default)
    return default if v is None else v


class MorphologyMark(CanonicalMark):
    DEFAULTS = {}
    KNOWN_KEYS = KNOWN_KEYS
    feature_names = ("delta_filtered", "lam1", "lam2", "lam3")
    gate_names = _ENV_NAMES

    def __init__(self, config):
        super().__init__()
        from types import SimpleNamespace

        config = SimpleNamespace(**{**self.DEFAULTS, **vars(config)})
        N = int(_g(config, "grid_size", _g(config, "grid_dim", 128)))
        box = float(_g(config, "box_size", 1000.0))
        mas = str(_g(config, "mas_scheme", _g(config, "mas", "PCS")))
        self.use_log1p = bool(_g(config, "use_log1p", True))
        log_kind = str(_g(config, "log_compression_kind", "log1p"))
        self.shared_filter = SingleChannelFilterBankPlus(
            grid_size=N,
            box_size=box,
            mas_scheme=mas,
            disable_mas=bool(_g(config, "disable_mas_filter", True)),
            use_log1p=self.use_log1p,
            radial_kind=str(_g(config, "filter_radial_kind", "sigmoid_lut")),
            n_bins=int(_g(config, "filter_n_bins", 32)),
            init_sigma=float(_g(config, "filter_init_sigma", 0.2)),
            n_basis=int(_g(config, "filter_n_basis", 12)),
            k0_frac=float(_g(config, "filter_k0_frac", 0.0)),
            lowpass_R_mpc=float(_g(config, "filter_lowpass_R_mpc", 6.5)),
            dc_normalize=bool(_g(config, "filter_dc_normalize", True)),
            nyquist_taper_kpass=getattr(config, "filter_nyquist_taper_kpass", None),
            nyquist_taper_floor=float(_g(config, "filter_nyquist_taper_floor", 0.0)),
            k_norm_convention=str(_g(config, "filter_k_norm_convention", "kmax")),
            keep_corner_modes=bool(_g(config, "filter_keep_corner_modes", True)),
            mas_before_log1p=bool(_g(config, "filter_mas_before_log1p", True)),
            log_compression_kind=log_kind,
            filter_order=str(_g(config, "filter_order", "log_then_filter")),
        )
        hess_log1p = getattr(config, "hessian_use_log1p", None)
        self.hessian_operator = str(_g(config, "hessian_operator", "density_hessian"))
        self.hessian = HessianFromDeltaPlus(
            grid_size=N,
            box_size=box,
            mas_scheme=mas,
            disable_mas=bool(_g(config, "disable_mas_hessian", True)),
            use_log1p=self.use_log1p if hess_log1p is None else bool(hess_log1p),
            hessian_operator=self.hessian_operator,
            smooth_R_voxels=float(_g(config, "hessian_smooth_R_voxels", 0.0)),
            k_grid_units=str(_g(config, "hessian_k_grid_units", "dimensionless")),
            corner_mask=bool(_g(config, "hessian_corner_mask", False)),
            mas_before_log1p=bool(_g(config, "hessian_mas_before_log1p", True)),
            log_compression_kind=log_kind,
            pre_taper_kpass=getattr(config, "hessian_pre_taper_kpass", None),
            pre_taper_floor=float(_g(config, "hessian_pre_taper_floor", 0.0)),
            nyquist_fix=bool(_g(config, "hessian_nyquist_fix", True)),
            smooth_order=str(_g(config, "hessian_smooth_order", "log_then_smooth")),
        )
        self.eig_extractor = EigenvalueExtractor()
        self.gate_input_roll = int(_g(config, "gate_input_roll", 0))
        self.gate_mode = str(_g(config, "gate_mode", "soft"))
        self.web_gate = MultiThresholdWebGate(
            init_tau=float(_g(config, "gate_init_tau", 1.0)),
            learnable_threshold=bool(_g(config, "gate_learnable_threshold", False)),
            learnable_tau=bool(_g(config, "gate_learnable_tau", True)),
            mode=self.gate_mode,
        )
        self.gate_eig_scale = getattr(config, "gate_eig_scale", None)
        self.gate_eig_scale = (
            None if self.gate_eig_scale is None else float(self.gate_eig_scale)
        )
        self.gate_eig_box_normalize = bool(_g(config, "gate_eig_box_normalize", False))
        gt = getattr(config, "gate_eig_transform", None)
        self.gate_eig_transform = (
            "raw" if gt is None or str(gt).lower() in ("none", "null", "") else str(gt)
        )
        if self.gate_eig_transform not in ("raw", "signlog", "sigmoid", "tanh"):
            raise ValueError(
                f"gate_eig_transform must be raw/signlog/sigmoid/tanh; got {self.gate_eig_transform!r}"
            )
        gam_hidden = int(_g(config, "gam_hidden_dim", 16))
        gam_act = str(_g(config, "gam_act", "gelu"))
        bound = getattr(config, "expert_output_bound", None)
        bound = None if bound is None else float(bound)
        self.include_gated_experts = bool(
            _g(config, "include_gated_experts", _g(config, "include_cross_term", True))
        )
        self.include_ungated_term = bool(_g(config, "include_ungated_term", False))
        self.include_eigen_main_effects = bool(
            _g(config, "include_eigen_main_effects", False)
        )
        self.eigen_input_transform = str(_g(config, "eigen_input_transform", "signlog"))
        if self.eigen_input_transform not in ("raw", "signlog"):
            raise ValueError(
                f"eigen_input_transform must be 'raw' or 'signlog'; got {self.eigen_input_transform!r}"
            )
        self.include_anisotropy_term = bool(
            _g(config, "include_anisotropy_term", False)
        )
        self.anisotropy_norm = str(_g(config, "anisotropy_norm", "normalized"))
        if self.anisotropy_norm not in ("raw", "normalized"):
            raise ValueError(
                f"anisotropy_norm must be 'raw' or 'normalized'; got {self.anisotropy_norm!r}"
            )
        self.anisotropy_input_transform = str(
            _g(config, "anisotropy_input_transform", "raw")
        )
        if self.anisotropy_input_transform not in ("raw", "signlog"):
            raise ValueError(
                f"anisotropy_input_transform must be 'raw' or 'signlog'; got {self.anisotropy_input_transform!r}"
            )
        if not (
            self.include_gated_experts
            or self.include_ungated_term
            or self.include_eigen_main_effects
            or self.include_anisotropy_term
        ):
            raise ValueError(
                "enable at least one of include_gated_experts, include_ungated_term, include_eigen_main_effects, include_anisotropy_term"
            )
        self.density_experts = (
            nn.ModuleList(
                [ScalarCurve(gam_hidden, gam_act, output_bound=bound) for _ in range(4)]
            )
            if self.include_gated_experts
            else None
        )
        self.ungated_curve = (
            ScalarCurve(gam_hidden, gam_act, output_bound=bound)
            if self.include_ungated_term
            else None
        )
        self.eigen_curves = (
            nn.ModuleList([ScalarCurve(gam_hidden, gam_act) for _ in range(3)])
            if self.include_eigen_main_effects
            else None
        )
        self.anisotropy_curve = (
            ScalarCurve(gam_hidden, gam_act) if self.include_anisotropy_term else None
        )
        self.positivity = str(_g(config, "mark_positivity", "positive"))
        if self.positivity not in ("positive", "free"):
            raise ValueError(
                f"mark_positivity must be 'positive' or 'free'; got {self.positivity!r}"
            )
        self.offset = nn.Parameter(
            torch.tensor(
                _SOFTPLUS_INV1 if self.positivity == "positive" else 1.0,
                dtype=torch.float32,
            )
        )
        self.mark_normalize = bool(_g(config, "mark_normalize", False))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for old, new in (("logit_bias", "offset"), ("density_curves.", "density_experts.")):
            for key in list(state_dict):
                suffix = key.removeprefix(prefix)
                if key.startswith(prefix) and (
                    suffix == old or (old.endswith(".") and suffix.startswith(old))
                ):
                    state_dict[prefix + new + suffix[len(old):]] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def gate_input(self, eigs: torch.Tensor) -> torch.Tensor:
        if self.gate_eig_box_normalize:
            return self._gate_transform(
                eigs / eigs.std(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-08)
            )
        x = eigs if self.gate_eig_scale is None else eigs / self.gate_eig_scale
        return self._gate_transform(x)

    def _gate_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_eig_transform == "signlog":
            return signlog(x)
        if self.gate_eig_transform == "sigmoid":
            return 2.0 * torch.sigmoid(x) - 1.0
        if self.gate_eig_transform == "tanh":
            return torch.tanh(x)
        return x

    def forward(self, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        delta_f = self.shared_filter(delta)
        eigs, aniso = self.eig_extractor(self.hessian(delta))
        if self.gate_input_roll:
            r = self.gate_input_roll
            eigs = torch.roll(eigs, shifts=(r, r, r), dims=(-3, -2, -1))
            aniso = torch.roll(aniso, shifts=(r, r, r), dims=(-3, -2, -1))
        gates = self.web_gate(self.gate_input(eigs))
        logit = self.offset.to(delta.dtype)
        out: Dict[str, torch.Tensor] = {}
        if self.include_ungated_term:
            f0 = self.ungated_curve(delta_f)
            logit = logit + f0
            out["ungated_term"] = f0.detach()
        if self.include_gated_experts:
            terms: List[torch.Tensor] = [
                gates[:, i : i + 1] * self.density_experts[i](delta_f) for i in range(4)
            ]
            expert_terms = torch.cat(terms, dim=1)
            logit = logit + expert_terms.sum(dim=1, keepdim=True)
            for i, name in enumerate(_ENV_NAMES):
                out[f"expert_{name}"] = expert_terms[:, i : i + 1].detach()
        if self.include_eigen_main_effects:
            eig_in = signlog(eigs) if self.eigen_input_transform == "signlog" else eigs
            eig_terms = torch.cat(
                [self.eigen_curves[i](eig_in[:, i : i + 1]) for i in range(3)], dim=1
            )
            logit = logit + eig_terms.sum(dim=1, keepdim=True)
            for i, lam in enumerate(("lam1", "lam2", "lam3")):
                out[f"g_{lam}"] = eig_terms[:, i : i + 1].detach()
        if self.include_anisotropy_term:
            if self.anisotropy_norm == "normalized":
                aniso_in = (eigs[:, 2:3] - eigs[:, 0:1]) / (
                    eigs.abs().sum(dim=1, keepdim=True) + _EPS
                )
            else:
                aniso_in = eigs[:, 2:3] - eigs[:, 0:1]
            if self.anisotropy_input_transform == "signlog":
                aniso_in = signlog(aniso_in)
            g = self.anisotropy_curve(aniso_in)
            logit = logit + g
            out["g_anisotropy"] = g.detach()
        mark = F.softplus(logit) if self.positivity == "positive" else logit
        final, amp = self._marked_field_and_amplitude(mark, delta)
        out.update(
            {
                "mark_field": mark,
                "final_field": final,
                "mark_amplitude": amp.detach().reshape(delta.shape[0]),
                "delta_f": delta_f.detach(),
                "lam1": eigs[:, 0:1].detach(),
                "lam2": eigs[:, 1:2].detach(),
                "lam3": eigs[:, 2:3].detach(),
                "anisotropy_raw": aniso.detach(),
                "tau": self.web_gate.tau.detach(),
            }
        )
        for i, name in enumerate(_ENV_NAMES):
            out[f"w_{name}"] = gates[:, i : i + 1].detach()
        if self.web_gate.learnable_threshold:
            out["mu_thresholds"] = self.web_gate.mus.detach()
        return out

    def introspect(self, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            out = self.forward(delta)
        out["gate_names"] = list(_ENV_NAMES)
        out["feature_names"] = list(self.feature_names)
        out["introspection_metadata"] = {
            "hessian_operator": self.hessian_operator,
            "hessian_use_log1p": bool(self.hessian.use_log1p),
            "hessian_nyquist_fix": bool(self.hessian.nyquist_fix),
            "filter_order": self.shared_filter.filter_order,
            "hessian_smooth_order": self.hessian.smooth_order,
            "gate_mode": self.gate_mode,
            "gate_eig_transform": self.gate_eig_transform,
            "gate_eig_scale": self.gate_eig_scale,
            "gate_eig_box_normalize": self.gate_eig_box_normalize,
            "hessian_pre_taper_kpass": self.hessian.pre_taper_kpass,
            "mark_normalize": self.mark_normalize,
        }
        return out


class WebMark(MorphologyMark):
    DEFAULTS = {
        "hessian_operator": "potential_tweb",
        "gate_mode": "hard",
        "mark_positivity": "free",
    }


class HessianMark(MorphologyMark):
    DEFAULTS = {
        "hessian_operator": "density_hessian",
        "gate_mode": "soft",
        "gate_init_tau": 0.6,
        "gate_learnable_tau": False,
        "mark_positivity": "free",
    }
