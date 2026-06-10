from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

_SOFTPLUS_INV1 = math.log(math.e - 1.0)  # softplus(_SOFTPLUS_INV1) == 1.0


class _ScalarCurve(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        return self.net(x.reshape(-1, 1)).reshape(shape)


class _CrossCurve(nn.Module):
    # Linear(n,h)→GELU→Linear(h,1)

    def __init__(self, hidden: int, n_inputs: int = 2):
        super().__init__()

        self.n_inputs = n_inputs
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        if len(xs) != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} inputs, got {len(xs)}")
        shape = xs[0].shape
        flats = torch.stack([x.reshape(-1) for x in xs], dim=1)
        return self.net(flats).reshape(shape)


class SHMixer(nn.Module):
    feature_order = ("E0", "E1", "E2", "I3_norm")

    def __init__(self,
                 hidden_dim: int = 16,
                 cross_hidden_dim: int = 8,
                 cross_inputs=("E0", "E2", "I3_norm"),
                 positivity: str = "free",
                 feature_order=("E0", "E1", "E2", "I3_norm")):
        super().__init__()

        if positivity not in ("softplus", "free"):
            raise ValueError(
                f"positivity must be 'softplus' or 'free', got {positivity!r}")
        self.positivity = str(positivity)
        self.feature_order = tuple(feature_order)  # available invariants (set by l_max)

        self.curves = nn.ModuleList( _ScalarCurve(hidden_dim) for _ in self.feature_order)

        # keep only cross inputs that are actually present
        cross_inputs = tuple(k for k in (cross_inputs or ()) if k in self.feature_order)
        self.cross_inputs = cross_inputs
        if len(cross_inputs) >= 2:
            self.cross = _CrossCurve(cross_hidden_dim, n_inputs=len(cross_inputs))
        else:
            self.cross = None

        bias_init = _SOFTPLUS_INV1 if self.positivity == "softplus" else 1.0
        self.logit_bias = nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))

    @property
    def cross_key(self) -> str:
        return "cross_" + "_".join(self.cross_inputs)

    def evaluate_contributions(self, invs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for i, key in enumerate(self.feature_order):
            out[f"g_{key}"] = self.curves[i](invs[key])
        if self.cross is not None:
            cross_args = [invs[k] for k in self.cross_inputs]
            out[self.cross_key] = self.cross(*cross_args)
        return out

    def forward(self, invs: Dict[str, torch.Tensor]) -> torch.Tensor:
        contribs = self.evaluate_contributions(invs)
        logit = self.logit_bias
        for key in self.feature_order:
            logit = logit + contribs[f"g_{key}"]
        if self.cross is not None:
            logit = logit + contribs[self.cross_key]
        if self.positivity == "softplus":
            return F.softplus(logit)
        return logit
