from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def mas_deconv_kernel(N: int, scheme: str, box_size: float, device,
                      floor: float = 1e-3) -> torch.Tensor:

    p = {"NGP": 1, "CIC": 2, "TSC": 3, "PCS": 4}[scheme.upper()]
    cell = box_size / N
    kx = 2.0 * math.pi * torch.fft.fftfreq(N, d=cell, device=device)
    kz = 2.0 * math.pi * torch.fft.rfftfreq(N, d=cell, device=device)
    kxg, kyg, kzg = torch.meshgrid(kx, kx, kz, indexing="ij")

    def _inv_sinc_p(k):
        arg = k * cell / 2.0
        safe = torch.where(arg.abs() > 1e-9, arg, torch.ones_like(arg))
        s = torch.where(arg.abs() > 1e-9, torch.sin(arg) / safe, torch.ones_like(arg))
        return s.clamp(min=floor).pow(-p)

    return (_inv_sinc_p(kxg) * _inv_sinc_p(kyg) * _inv_sinc_p(kzg)).detach()


def k_mag_norm_grid(N: int, box_size: float, device) -> torch.Tensor:
    # |k| / k_Nyquist on the rfft grid 
    cell = box_size / N
    kx = 2.0 * math.pi * torch.fft.fftfreq(N, d=cell, device=device)
    kz = 2.0 * math.pi * torch.fft.rfftfreq(N, d=cell, device=device)
    kxg, kyg, kzg = torch.meshgrid(kx, kx, kz, indexing="ij")
    k_mag = torch.sqrt(kxg * kxg + kyg * kyg + kzg * kzg)
    k_nyq = math.pi / cell
    return (k_mag / k_nyq).detach()


def cosine_taper(x: torch.Tensor, kpass: float, floor: float = 0.0) -> torch.Tensor:
    # Tukey window on x = |k|/k_Nyq: flat to kpass, cosine roll-off to floor at x=1, hard zero past Nyquist
    kp = max(0.0, min(1.0, kpass))
    fl = max(0.0, min(1.0, floor))
    denom = max(1.0 - kp, 1e-8)
    t = torch.clamp((x - kp) / denom, 0.0, 1.0)
    win = fl + 0.5 * (1.0 - fl) * (1.0 + torch.cos(math.pi * t))
    win = torch.where(x <= kp, torch.ones_like(win), win)
    return torch.where(x <= 1.0, win, torch.zeros_like(win))


def mas_like_taper(x: torch.Tensor, power: float = 4.0, kpass: float = 0.0) -> torch.Tensor:
    # Isotropized MAS-sinc window on x = |k|/k_Nyq
   
    def _sinc(a):
        a_safe = torch.where(a == 0, torch.ones_like(a), a)
        return torch.where(a.abs() > 1e-8, torch.sin(a) / a_safe, torch.ones_like(a))
    s = _sinc((math.pi / 2.0) * x).clamp(min=0.0)
    if kpass > 0:
        norm = float(_sinc(torch.tensor((math.pi / 2.0) * float(kpass))).clamp(min=1e-6))
    else:
        norm = 1.0
    win = (s / norm).clamp(min=0.0) ** float(power)
    return torch.where(x <= float(kpass), torch.ones_like(win), win)


def build_taper(kind: str, k_mag_norm: torch.Tensor, *, kpass: float,
                floor: float, mas_power: float) -> torch.Tensor:
    if kind == "cosine":
        return cosine_taper(k_mag_norm, kpass=kpass, floor=floor).float()
    if kind == "mas":
        return mas_like_taper(k_mag_norm, power=mas_power, kpass=kpass).float()
    raise ValueError(f"taper must be 'cosine' or 'mas', got {kind!r}")


class GaussianRadial(nn.Module):
    # Per-l Gaussian-basis radial profiles W_l(k)

    def __init__(self, n_basis: int, k0_frac: float = 0.2, l_max: int = 2,
                 high_l_init: str = "match_l0"):
        super().__init__()
        if n_basis < 1:
            raise ValueError("n_radial_basis must be >= 1")
        if high_l_init not in ("match_l0", "zero", "noise"):
            raise ValueError( "high_l_init must be 'match_l0', 'zero' or 'noise', got {high_l_init!r}")
        self.n_basis = n_basis
        self.l_max = int(l_max)
        self.high_l_init = str(high_l_init)
        centers = torch.linspace(0.0, 1.0, n_basis)
        sigma = 1.0 / max(n_basis - 1, 1)
        self.register_buffer("centers", centers)
        self.sigma = float(sigma)

        # Coefficients per l 
        self.coeffs = nn.Parameter(torch.zeros(self.l_max + 1, n_basis))
        with torch.no_grad():
            # l=0: peak at k0_frac × k_Nyq 
            j = int(round(float(k0_frac) * (n_basis - 1)))
            j = max(0, min(n_basis - 1, j))
            self.coeffs[0, j] = 1.0

            for l in range(1, self.l_max + 1):
                if self.high_l_init == "match_l0":
                    self.coeffs[l, j] = 1.0
                elif self.high_l_init == "noise":
                    self.coeffs[l].normal_(0.0, 1e-3)

    def forward(self, k_norm: torch.Tensor) -> torch.Tensor:
        shape = k_norm.shape
        k_flat = k_norm.reshape(-1)  # [M]
        diff = self.centers.unsqueeze(1) - k_flat.unsqueeze(0)
        basis = torch.exp(-(diff * diff) / (2.0 * self.sigma * self.sigma))
        profiles = self.coeffs @ basis  # [l_max+1, M]
        return profiles.view(self.l_max + 1, *shape)


class FourierMLPRadial(nn.Module):
    # Per-l Fourier-feature MLP radial profiles W_l(k)
    def __init__(self, 
                 hidden: int = 16, 
                 n_fourier: int = 8, 
                 l_max: int = 2,
                 r0_init_per_l: tuple = (0.15, 0.25, 0.35),
                 sigma_init_per_l: tuple = (0.10, 0.12, 0.15),
                 residual_scale: float = 1.0, 
                 subtract_dc_high_l: bool = False):
        super().__init__()
        self.l_max = int(l_max)
        if self.l_max + 1 != len(r0_init_per_l):
            r0_init_per_l = tuple(0.15 + 0.10 * l for l in range(self.l_max + 1))
        if self.l_max + 1 != len(sigma_init_per_l):
            sigma_init_per_l = tuple(0.10 + 0.02 * l for l in range(self.l_max + 1))

        self.register_buffer(
            "freqs",
            torch.logspace(math.log10(1.0), math.log10(16.0), steps=int(n_fourier)),
        )
        in_dim = 1 + 2 * int(n_fourier)

        def _inv_sigmoid_for(target):
            t = (float(target) - 0.05) / 0.55
            t = max(1e-3, min(1.0 - 1e-3, t))
            return math.log(t / (1.0 - t))

        raw_r0 = torch.tensor(
            [_inv_sigmoid_for(t) for t in r0_init_per_l], dtype=torch.float32)
        self.r0_raw = nn.Parameter(raw_r0)

        def _inv_softplus(y):
            y = float(y)
            if y > 20.0:
                return y
            return float(math.log(math.expm1(y)))

        raw_sigma = torch.tensor(
            [_inv_softplus(s - 1e-3) for s in sigma_init_per_l],
            dtype=torch.float32)
        self.sigma_raw = nn.Parameter(raw_sigma)

        self.residual_scale = float(residual_scale)
        self.subtract_dc_high_l = bool(subtract_dc_high_l)

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            for _ in range(self.l_max + 1)
        ])
        for mlp in self.mlps:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def _r0(self) -> torch.Tensor:
        return 0.05 + 0.55 * torch.sigmoid(self.r0_raw)        # [L+1]

    def _sigma(self) -> torch.Tensor:
        return F.softplus(self.sigma_raw) + 1e-3               # [L+1]

    def _encode(self, r: torch.Tensor) -> torch.Tensor:
        r1 = r.unsqueeze(-1)
        ang = r1 * self.freqs
        return torch.cat([r1, torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, r_norm: torch.Tensor) -> torch.Tensor:

        enc = self._encode(r_norm)
        r0_l = self._r0()
        sig_l = self._sigma()
        zero_r = torch.zeros((), device=r_norm.device, dtype=r_norm.dtype)
        zero_enc = self._encode(zero_r.expand(1)) # [1, in_dim]

        outs = []
        for l, mlp in enumerate(self.mlps):
            base = torch.exp(
                -((r_norm - r0_l[l]) ** 2) / (2.0 * sig_l[l] * sig_l[l]))
            resid = mlp(enc).squeeze(-1)
            w = base + self.residual_scale * resid
            if l >= 1 and self.subtract_dc_high_l:
                base0 = torch.exp(
                    -((zero_r - r0_l[l]) ** 2) / (2.0 * sig_l[l] * sig_l[l]))
                resid0 = mlp(zero_enc).squeeze(-1).squeeze(0)
                w0 = base0 + self.residual_scale * resid0
                w = w - w0
            outs.append(w)
        return torch.stack(outs, dim=0)


def _cartesian_sh_from_unit_vec(xh, yh, zh):
    #[Y_0^0, Y_1^{-1}, Y_1^0, Y_1^{+1}, Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^{+1}, Y_2^{+2}].

    Y00 = torch.full_like(xh, 1.0 / (2.0 * math.sqrt(math.pi)))
    c1 = math.sqrt(3.0 / (4.0 * math.pi))
    c2a = 0.5 * math.sqrt(15.0 / math.pi)
    c2b = 0.25 * math.sqrt(5.0 / math.pi)
    c2c = 0.25 * math.sqrt(15.0 / math.pi)

    Y1m1 = c1 * yh
    Y10 = c1 * zh
    Y1p1 = c1 * xh
    Y2m2 = c2a * xh * yh
    Y2m1 = c2a * yh * zh
    Y20 = c2b * (2.0 * zh * zh - xh * xh - yh * yh)
    Y2p1 = c2a * xh * zh
    Y2p2 = c2c * (xh * xh - yh * yh)
    return torch.stack([Y00, Y1m1, Y10, Y1p1, Y2m2, Y2m1, Y20, Y2p1, Y2p2], dim=0)


def sh_kgrid_physical(N: int, box_size: float, device, l_max: int = 2) -> torch.Tensor:
    # Real Cartesian SH on the rfft k-grid: shape [(l_max+1)^2, N, N, N//2+1]
    cell = box_size / N
    kx = 2.0 * math.pi * torch.fft.fftfreq(N, d=cell, device=device)
    kz = 2.0 * math.pi * torch.fft.rfftfreq(N, d=cell, device=device)
    kxg, kyg, kzg = torch.meshgrid(kx, kx, kz, indexing="ij")
    k_mag = torch.sqrt(kxg * kxg + kyg * kyg + kzg * kzg)
    k_safe = torch.where(k_mag > 0, k_mag, torch.ones_like(k_mag))
    xh, yh, zh = kxg / k_safe, kyg / k_safe, kzg / k_safe
    sh = _cartesian_sh_from_unit_vec(xh, yh, zh)[:(l_max + 1) ** 2]
    dc = (k_mag == 0)
    sh[1:][:, dc] = 0.0
    return sh.detach()


def sh_kgrid_e3nn(N: int, box_size: float, device, l_max: int = 2) -> torch.Tensor:
    # Real e3nn SHon the rfft k-grid: [(l_max+1)^2, N, N, N//2+1]

    from e3nn import o3

    cell = box_size / N
    kx = 2.0 * math.pi * torch.fft.fftfreq(N, d=cell, device=device)
    kz = 2.0 * math.pi * torch.fft.rfftfreq(N, d=cell, device=device)
    kxg, kyg, kzg = torch.meshgrid(kx, kx, kz, indexing="ij")
    k_mag = torch.sqrt(kxg * kxg + kyg * kyg + kzg * kzg)

    k_safe = torch.clamp(k_mag, min=1e-12)
    k_hat = torch.stack([kxg, kyg, kzg], dim=-1) / k_safe.unsqueeze(-1)
    origin = (k_mag < 1e-10)
    k_hat[origin] = 0.0

    irreps = o3.Irreps.spherical_harmonics(l_max)
    Y = o3.spherical_harmonics(irreps, k_hat, normalize=False,
                               normalization="component") # [N, N, N//2+1, (l_max+1)^2]

    return Y.permute(3, 0, 1, 2).contiguous().detach()



class SHFilterBank(nn.Module):
    # f_lm(x) = irfftn( δ_k · MAS^{-1}(k) · i^l · W_l(|k|) · Y_lm(k̂) )
    

    def __init__(self, 
                 grid_size: int, 
                 box_size: float, 
                 sh_basis: str,
                 radial: str, 
                 taper: str, 
                 l_max: int = 2,
                 n_radial_basis: int = 12, 
                 k0_frac: float = 0.2,
                 radial_hidden: int = 16, 
                 radial_n_fourier: int = 8,
                 mas_scheme: str = "PCS", 
                 dc_normalize: bool = True,
                 l2_normalize_high_l: bool = False, 
                 l2_normalize_l0: bool = False,
                 nyquist_taper_kpass: float = 0.85, 
                 nyquist_taper_floor: float = 0.0,
                 mas_taper_power: float = 4.0, 
                 gaussian_high_l_init: str = "match_l0",
                 subtract_dc_high_l: bool = False):
        super().__init__()
        
        if l_max not in (0, 1, 2):
            raise ValueError("SHFilterBank supports l_max in {0, 1, 2}")
        self.N = int(grid_size)
        self.box_size = float(box_size)
        self.l_max = int(l_max)
        self.n_sh = (self.l_max + 1) ** 2
        self.dc_normalize = bool(dc_normalize)
        self.l2_normalize_high_l = bool(l2_normalize_high_l)
        self.l2_normalize_l0 = bool(l2_normalize_l0)

        cpu = torch.device("cpu")
        if sh_basis == "physical":
            ylm_k = sh_kgrid_physical(self.N, self.box_size, cpu, l_max)
        elif sh_basis == "e3nn":
            ylm_k = sh_kgrid_e3nn(self.N, self.box_size, cpu, l_max)
        else:
            raise ValueError(f"sh_basis must be 'physical' or 'e3nn', got {sh_basis!r}")

        l_mapping = torch.tensor(
            [l for l in range(self.l_max + 1) for _ in range(2 * l + 1)], dtype=torch.long)
        self.register_buffer("l_mapping", l_mapping)
        phase = (1j ** l_mapping.to(torch.complex64)).to(torch.complex64)   # [9]
        ylm_k_phased = ylm_k.to(torch.complex64) * phase.view(-1, 1, 1, 1)
        self.register_buffer("ylm_k_phased", ylm_k_phased, persistent=False)

        k_mag_norm = k_mag_norm_grid(self.N, self.box_size, cpu)
        self.register_buffer("k_mag_norm", k_mag_norm.float(), persistent=False)
        mas = mas_deconv_kernel(self.N, mas_scheme, self.box_size, cpu).float()
        self.register_buffer("mas_corr", mas, persistent=False)
        win = build_taper(taper, k_mag_norm, kpass=nyquist_taper_kpass,
                          floor=nyquist_taper_floor, mas_power=mas_taper_power)
        self.register_buffer("nyquist_taper", win.float(), persistent=False)

        if radial == "gaussian":
            self.radial = GaussianRadial(n_radial_basis, k0_frac, l_max=l_max,
                                         high_l_init=gaussian_high_l_init)
        elif radial == "fourier_mlp":
            self.radial = FourierMLPRadial(radial_hidden, radial_n_fourier, l_max=l_max,
                                           subtract_dc_high_l=subtract_dc_high_l)
        else:
            raise ValueError(
                f"radial must be 'gaussian' or 'fourier_mlp', got {radial!r}")

    def _dc_normalize_W(self, W_l: torch.Tensor) -> torch.Tensor:
        # l=0 channel with unit DC gain (W_0(k=0)=1)
        if not self.dc_normalize:
            return W_l
        w0_dc = W_l[0, 0, 0, 0].clamp(min=1e-4)
        scale = torch.ones_like(W_l[:, :1, :1, :1])
        scale[0] = 1.0 / w0_dc
        return W_l * scale

    def _l2_normalize_W(self, W_l: torch.Tensor) -> torch.Tensor:
        # Per-l L2 normalization Σ_rfft |MAS·W_l|² = N³/(2(2l+1))
        if not self.l2_normalize_high_l and not self.l2_normalize_l0:
            return W_l
        
        out = W_l.clone()
        N3 = float(self.N) ** 3
        weight2 = self.mas_corr ** 2
        l_start = 0 if self.l2_normalize_l0 else 1
        l_end = W_l.shape[0] if self.l2_normalize_high_l else 1
        
        for l in range(l_start, l_end):
            n_m = 2 * l + 1
            sq = ((W_l[l] ** 2) * weight2).sum()
            target_sq = N3 / (2.0 * n_m)
            scale = torch.sqrt(target_sq / sq.clamp(min=1e-12))
            out[l] = W_l[l] * scale
            
        return out

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        B, _, D, H, W = delta.shape
        
        delta_k = torch.fft.rfftn(delta.squeeze(1), dim=(-3, -2, -1)) # [B, D, H, W//2+1]
        delta_k_deconv = delta_k * self.mas_corr
        
        W_l = self.radial(self.k_mag_norm)                              # [l_max+1, ...]
        W_l = self._dc_normalize_W(W_l)
        W_l = W_l * self.nyquist_taper.unsqueeze(0)
        W_l = self._l2_normalize_W(W_l)
        
        W_ch = W_l[self.l_mapping].to(torch.complex64)                  # [n_sh, ...]
        T = W_ch * self.ylm_k_phased                                    # [n_sh, ...] complex

        f_lm_k = delta_k_deconv.unsqueeze(1) * T.unsqueeze(0)           # [B, n_sh, ...]
        f_lm_k = f_lm_k.reshape(B * self.n_sh, D, H, W // 2 + 1)
        f_lm = torch.fft.irfftn(f_lm_k, s=(D, H, W), dim=(-3, -2, -1))

        return f_lm.reshape(B, self.n_sh, D, H, W)
