from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from marks.filter_bank import SHFilterBank
from marks.invariants import InvariantExtractor
from marks.mixer import SHMixer


class MarkSH(nn.Module):
    def __init__(self, config):
        super().__init__()
        N = int(getattr(config, "grid_dim", getattr(config, "grid_size", 128)))
        box = float(getattr(config, "box_size", 1000.0))
        self.input_transform = str(getattr(config, "input_transform", "log1p"))

        self.filter_bank = SHFilterBank(
            grid_size=N,
            box_size=box,
            sh_basis=str(getattr(config, "sh_basis", "physical")),
            radial=str(getattr(config, "radial", "gaussian")),
            taper=str(getattr(config, "taper", "cosine")),
            l_max=int(getattr(config, "l_max", 2)),
            n_radial_basis=int(getattr(config, "n_radial_basis", 12)),
            k0_frac=float(getattr(config, "k0_frac", 0.2)),
            radial_hidden=int(getattr(config, "radial_hidden", 16)),
            radial_n_fourier=int(getattr(config, "radial_n_fourier", 8)),
            mas_scheme=str(getattr(config, "mas_scheme", "PCS")),
            dc_normalize=bool(getattr(config, "dc_normalize", True)),
            l2_normalize_high_l=bool(getattr(config, "l2_normalize_high_l", False)),
            l2_normalize_l0=bool(getattr(config, "l2_normalize_l0", False)),
            nyquist_taper_kpass=float(getattr(config, "nyquist_taper_kpass", 0.85)),
            nyquist_taper_floor=float(getattr(config, "nyquist_taper_floor", 0.0)),
            mas_taper_power=float(getattr(config, "mas_taper_power", 4.0)),
            gaussian_high_l_init=str(getattr(config, "gaussian_high_l_init", "match_l0")),
            subtract_dc_high_l=bool(getattr(config, "subtract_dc_high_l", False)),
        )
        self.extractor = InvariantExtractor(
            sh_convention=str(getattr(config, "sh_convention", "physics")))
        l_max = int(getattr(config, "l_max", 2))
        feature_order = ("E0", "E1", "E2", "I3_norm")[:(1, 2, 4)[l_max]]
        self.mixer = SHMixer(
            hidden_dim=int(getattr(config, "gam_hidden_dim", 16)),
            cross_hidden_dim=int(getattr(config, "cross_hidden_dim", 8)),
            cross_inputs=tuple(getattr(config, "cross_inputs", ("E0", "E2", "I3_norm"))),
            positivity=str(getattr(config, "mark_positivity", "free")),
            feature_order=feature_order,
        )

    def _apply_input_transform(self, invs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.input_transform == "none":
            return invs
        if self.input_transform != "log1p":
            raise ValueError(f"Unknown input_transform: {self.input_transform!r}")
        out = dict(invs)

        # Signed channel (E0): sign-preserving log1p
        x0 = invs["E0"]
        out["E0"] = x0.sign() * torch.log1p(x0.abs())
        # Positive channels (E1, E2): plain log1p (present only for l_max >= 1, 2)
        if "E1" in invs:
            out["E1"] = torch.log1p(invs["E1"])
        if "E2" in invs:
            out["E2"] = torch.log1p(invs["E2"])
        # I3_norm stays untransformed (dimensionless)
        return out

    def _make_final_field(self, mark: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        field = mark * (1.0 + delta)
        return field - field.mean(dim=(-3, -2, -1), keepdim=True)

    def forward(self, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        f_lm = self.filter_bank(delta)
        invs_raw = self.extractor(f_lm)
        invs = self._apply_input_transform(invs_raw)
        mark = self.mixer(invs)

        out = {
            "mark_field": mark,
            "final_field": self._make_final_field(mark, delta),
        }
        for key, v in invs_raw.items():
            out[key] = v.detach()
        out["f_lm"] = f_lm.detach()
        return out
