"""
ON SH ORDERING:
- physical bank: l=1 order [y, z, x]; l,m)=2,0 is z quadrupole (2z^2-x^2-y^2); last is (x^2-y^2).
- e3nn bank: l=1 order [x, y, z]; l,m=2,0 "m=0" is y quadrupole (2y^2-x^2-z^2); last is (z^2-x^2).

Q in physical ordering
e3nn builds a Q' = P Q P^T & sqrt(4pi) amplitude rescale
BUT 
Tr(Q^2)=E2^2 and Tr(Q^3)=I3 are invariant by any orthogonal P
so marker is robust, kernel visualizations need rotations

"""



from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

_EPS = 1e-9


class InvariantExtractor(nn.Module):

    def __init__(self, sh_convention: str = "physics",
                 compute_align: bool = False, eps: float = _EPS):
        super().__init__()
        self.compute_align = bool(compute_align)
        self.eps = eps
        if sh_convention not in ("physics", "e3nn_component"):
            raise ValueError(
                f"sh_convention must be 'physics' or 'e3nn_component, got {sh_convention!r}")
        self.sh_convention = sh_convention

        self._sh_scale = (1.0 if sh_convention == "physics"
                          else 1.0 / math.sqrt(4.0 * math.pi))

    def forward(self, f_lm: torch.Tensor) -> Dict[str, torch.Tensor]:
        n = f_lm.shape[1]  # (l_max+1)^2: 1 -> E0, 4 -> +E1, 9 -> +E2,I3_norm
        E0 = f_lm[:, 0:1] * self._sh_scale
        out = {"E0": E0}
        if n < 4:
            return out

        f1 = f_lm[:, 1:4]
        E1 = torch.sqrt((f1 * f1).sum(dim=1, keepdim=True) + self.eps) * self._sh_scale
        out["E1"] = E1
        if n < 9:
            return out

        f2 = f_lm[:, 4:9]
        pref_off  = math.sqrt(15.0 / (16.0 * math.pi)) * self._sh_scale
        pref_diag = math.sqrt(5.0  / (4.0  * math.pi)) * self._sh_scale
        pref_diff = math.sqrt(15.0 / (4.0  * math.pi)) * self._sh_scale
        Qxy = f2[:, 0:1] * pref_off
        Qyz = f2[:, 1:2] * pref_off
        Qzz = f2[:, 2:3] * pref_diag
        Qxz = f2[:, 3:4] * pref_off
        Qxx_mQyy = f2[:, 4:5] * pref_diff
        Qxx = (-Qzz + Qxx_mQyy) / 2.0
        Qyy = (-Qzz - Qxx_mQyy) / 2.0

        E2_sq = (Qxx * Qxx + Qyy * Qyy + Qzz * Qzz + 2.0 * (Qxy * Qxy + Qxz * Qxz + Qyz * Qyz))
        E2 = torch.sqrt(E2_sq + self.eps)

        I3 = (Qxx ** 3 + Qyy ** 3 + Qzz ** 3
              + 3.0 * Qxy * Qxy * (Qxx + Qyy)
              + 3.0 * Qxz * Qxz * (Qxx + Qzz)
              + 3.0 * Qyz * Qyz * (Qyy + Qzz)
              + 6.0 * Qxy * Qxz * Qyz)
        
        # I3 / Tr(Q²)^{3/2} = I3 / E2^3 (E2 = sqrt(Tr(Q²))).
        I3_norm = I3 / (E2.pow(3) + self.eps)

        out["E2"] = E2
        out["I3_norm"] = I3_norm

        if self.compute_align:
            n1 = f1 / (E1 + self.eps)                                        # [B, 3, ...]
            ny = n1[:, 0:1]; nz = n1[:, 1:2]; nx = n1[:, 2:3]
            nQn = (Qxx * nx * nx + Qyy * ny * ny + Qzz * nz * nz
                   + 2.0 * Qxy * nx * ny + 2.0 * Qxz * nx * nz + 2.0 * Qyz * ny * nz)
            out["align"] = nQn / (E2 + self.eps)
        return out
