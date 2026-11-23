from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_SOFTPLUS_INV1 = math.log(math.e - 1.0)
_EPS = 1e-08


def signlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def log_compress(
    delta: torch.Tensor, kind: str = "log1p", floor: float = -0.999
) -> torch.Tensor:
    if kind == "log1p":
        return torch.log1p(delta.clamp(min=floor))
    if kind == "log2p":
        return torch.log(delta.clamp(min=floor) + 2.0)
    raise ValueError(f"log_compression_kind must be 'log1p' or 'log2p'; got {kind!r}")


def rfft_kgrid(N: int, d: float = 1.0):
    k1 = 2.0 * math.pi * torch.fft.fftfreq(N, d=d)
    kz = 2.0 * math.pi * torch.fft.rfftfreq(N, d=d)
    return torch.meshgrid(k1, k1, kz, indexing="ij")


def mas_deconv_kernel(
    N: int, scheme: str, box_size: float, floor: float = 0.001
) -> torch.Tensor:
    p = {"NGP": 1, "CIC": 2, "TSC": 3, "PCS": 4}[scheme.upper()]
    cell = box_size / N
    kxg, kyg, kzg = rfft_kgrid(N, d=cell)

    def _inv_sinc_p(k):
        arg = k * cell / 2.0
        safe = torch.where(arg.abs() > 1e-09, arg, torch.ones_like(arg))
        s = torch.where(arg.abs() > 1e-09, torch.sin(arg) / safe, torch.ones_like(arg))
        return s.clamp(min=floor).pow(-p)

    return (_inv_sinc_p(kxg) * _inv_sinc_p(kyg) * _inv_sinc_p(kzg)).detach()


def tukey_window(x: torch.Tensor, kpass: float, floor: float = 0.0) -> torch.Tensor:
    kp = max(0.0, min(1.0, float(kpass)))
    fl = max(0.0, min(1.0, float(floor)))
    denom = max(1.0 - kp, 1e-08)
    t = torch.clamp((x - kp) / denom, 0.0, 1.0)
    win = fl + 0.5 * (1.0 - fl) * (1.0 + torch.cos(math.pi * t))
    win = torch.where(x <= kp, torch.ones_like(win), win)
    return torch.where(x <= 1.0, win, torch.full_like(win, fl))


class _LUTInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, profile: torch.Tensor, k_norm: torch.Tensor) -> torch.Tensor:
        pos = k_norm.clamp(0.0, 1.0) * (profile.numel() - 1)
        idx0 = torch.floor(pos).long()
        idx1 = torch.clamp(idx0 + 1, max=profile.numel() - 1)
        alpha = (pos - idx0).to(profile.dtype)
        ctx.save_for_backward(profile, k_norm, idx0, idx1, alpha)
        return profile[idx0] * (1.0 - alpha) + profile[idx1] * alpha

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        profile, k_norm, idx0, idx1, alpha = ctx.saved_tensors
        grad_profile = grad_k = None
        if ctx.needs_input_grad[0]:
            g = grad_out.reshape(-1)
            a = alpha.reshape(-1)
            grad_profile = torch.bincount(
                idx0.reshape(-1), weights=g * (1.0 - a), minlength=profile.numel()
            ) + torch.bincount(
                idx1.reshape(-1), weights=g * a, minlength=profile.numel()
            )
        if ctx.needs_input_grad[1]:
            grad_k = (
                grad_out
                * (profile[idx1] - profile[idx0])
                * (profile.numel() - 1)
                * ((k_norm >= 0.0) & (k_norm <= 1.0))
            )
        return (grad_profile, grad_k)


class SigmoidLUTRadial(nn.Module):
    def __init__(self, n_bins: int = 32, init_sigma: float = 0.2):
        super().__init__()
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.n_bins = int(n_bins)
        radii = torch.linspace(0.0, 1.0, self.n_bins)
        target = torch.exp(-0.5 * (radii / float(init_sigma)) ** 2).clamp(0.01, 0.99)
        self.logits = nn.Parameter(torch.log(target / (1.0 - target)))

    def get_profile(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def forward(self, k_norm: torch.Tensor) -> torch.Tensor:
        return _LUTInterpolate.apply(self.get_profile(), k_norm)


class PositiveGaussianBasis(nn.Module):
    _SOFTPLUS_INV_ONE = math.log(math.e - 1.0)

    def __init__(self, n_basis: int = 12, k0_frac: float = 0.0):
        super().__init__()
        if n_basis < 2:
            raise ValueError(f"n_basis must be >= 2, got {n_basis}")
        self.n_basis = int(n_basis)
        self.register_buffer("centres", torch.linspace(0.0, 1.0, self.n_basis))
        self.sigma = float(1.0 / (self.n_basis - 1))
        raw = torch.full((self.n_basis,), -8.0)
        j = max(
            0, min(self.n_basis - 1, int(round(float(k0_frac) * (self.n_basis - 1))))
        )
        raw[j] = self._SOFTPLUS_INV_ONE
        self.raw_coeffs = nn.Parameter(raw)

    @property
    def coeffs(self) -> torch.Tensor:
        return F.softplus(self.raw_coeffs)

    def get_profile(self) -> torch.Tensor:
        return self.coeffs.detach()

    def forward(self, k_norm: torch.Tensor) -> torch.Tensor:
        shape = k_norm.shape
        diff = self.centres.unsqueeze(1) - k_norm.reshape(-1).unsqueeze(0)
        basis = torch.exp(-(diff * diff) / (2.0 * self.sigma * self.sigma))
        return (self.coeffs @ basis).view(shape)


class GaussianLowPassRadial(nn.Module):
    def __init__(self, R_mpc: float = 6.5, k_ref: float = 1.0):
        super().__init__()
        R_mpc = float(R_mpc)
        if R_mpc <= 0.0:
            raise ValueError(f"lowpass_R_mpc must be > 0; got {R_mpc}")
        raw = math.log(math.expm1(R_mpc)) if R_mpc < 20.0 else R_mpc
        self.raw_R = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        self.k_ref = float(k_ref)

    @property
    def R(self) -> torch.Tensor:
        return F.softplus(self.raw_R)

    def get_profile(self) -> torch.Tensor:
        return self.R.detach().reshape(1)

    def forward(self, k_norm: torch.Tensor) -> torch.Tensor:
        k_phys = k_norm * self.k_ref
        return torch.exp(-0.5 * (k_phys * self.R.to(k_norm.dtype)) ** 2)


class SingleChannelFilterBankPlus(nn.Module):
    def __init__(
        self,
        grid_size: int,
        box_size: float,
        mas_scheme: str = "PCS",
        disable_mas: bool = False,
        use_log1p: bool = False,
        radial_kind: str = "sigmoid_lut",
        n_bins: int = 32,
        init_sigma: float = 0.2,
        n_basis: int = 12,
        k0_frac: float = 0.0,
        lowpass_R_mpc: float = 6.5,
        dc_normalize: bool = True,
        nyquist_taper_kpass: Optional[float] = None,
        nyquist_taper_floor: float = 0.0,
        k_norm_convention: str = "kmax",
        keep_corner_modes: bool = True,
        mas_before_log1p: bool = True,
        log_compression_kind: str = "log1p",
        filter_order: str = "log_then_filter",
    ):
        super().__init__()
        self.N = int(grid_size)
        self.box_size = float(box_size)
        self.disable_mas = bool(disable_mas)
        self.use_log1p = bool(use_log1p)
        if log_compression_kind not in ("log1p", "log2p"):
            raise ValueError(
                f"log_compression_kind must be 'log1p' or 'log2p'; got {log_compression_kind!r}"
            )
        self.log_compression_kind = str(log_compression_kind)
        if filter_order not in ("log_then_filter", "filter_then_log"):
            raise ValueError(
                f"filter_order must be 'log_then_filter' or 'filter_then_log'; got {filter_order!r}"
            )
        self.filter_order = str(filter_order)
        self.last_clamped_frac = 0.0
        self.dc_normalize = bool(dc_normalize)
        self.mas_before_log1p = bool(mas_before_log1p)
        self.nyquist_taper_kpass = (
            None if nyquist_taper_kpass is None else float(nyquist_taper_kpass)
        )
        self.nyquist_taper_floor = float(nyquist_taper_floor)
        if k_norm_convention not in ("kmax", "knyq"):
            raise ValueError(
                f"k_norm_convention must be 'kmax' or 'knyq'; got {k_norm_convention!r}"
            )
        self.k_norm_convention = str(k_norm_convention)
        self.keep_corner_modes = bool(keep_corner_modes)
        cell = self.box_size / self.N
        kxg, kyg, kzg = rfft_kgrid(self.N, d=cell)
        k_mag = torch.sqrt(kxg**2 + kyg**2 + kzg**2)
        k_nyq = math.pi / cell
        k_ref = float(k_mag.max()) if self.k_norm_convention == "kmax" else float(k_nyq)
        self.k_nyq, self.k_ref = (float(k_nyq), float(k_ref))
        self.register_buffer("k_mag_norm", (k_mag / k_ref).float(), persistent=False)
        self.register_buffer(
            "mas_corr",
            mas_deconv_kernel(self.N, mas_scheme, self.box_size).float(),
            persistent=False,
        )
        k_nyq_norm = k_nyq / k_ref
        if self.nyquist_taper_kpass is not None:
            win = tukey_window(
                self.k_mag_norm / k_nyq_norm,
                self.nyquist_taper_kpass,
                self.nyquist_taper_floor,
            )
            win = torch.where(
                self.k_mag_norm / k_nyq_norm <= 1.0, win, torch.zeros_like(win)
            )
        elif self.keep_corner_modes:
            win = torch.ones_like(self.k_mag_norm)
        else:
            win = (self.k_mag_norm <= k_nyq_norm).float()
        self.register_buffer("taper", win.float(), persistent=False)
        radial_kind = str(radial_kind)
        if radial_kind == "sigmoid_lut":
            self.radial = SigmoidLUTRadial(n_bins=n_bins, init_sigma=init_sigma)
        elif radial_kind == "gaussian_positive":
            self.radial = PositiveGaussianBasis(n_basis=n_basis, k0_frac=k0_frac)
        elif radial_kind == "gaussian_lowpass":
            self.radial = GaussianLowPassRadial(R_mpc=lowpass_R_mpc, k_ref=self.k_ref)
        else:
            raise ValueError(
                f"filter_radial_kind must be 'sigmoid_lut', 'gaussian_positive' or 'gaussian_lowpass'; got {radial_kind!r}"
            )
        self.radial_kind = radial_kind

    def get_W(self) -> torch.Tensor:
        W = self.radial(self.k_mag_norm)
        if self.dc_normalize:
            W = W / W.reshape(-1)[0].clamp(min=0.0001)
        return W * self.taper

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.dim() != 5 or delta.shape[1] != 1:
            raise ValueError(f"expected [B, 1, D, H, W], got {tuple(delta.shape)}")
        B, _, D, H, Wsz = delta.shape
        x = delta.squeeze(1)
        if self.use_log1p and self.filter_order == "filter_then_log":
            x_k = torch.fft.rfftn(x, dim=(-3, -2, -1))
            if not self.disable_mas:
                x_k = x_k * self.mas_corr
            x_k = x_k * self.get_W().to(x_k.dtype)
            y = torch.fft.irfftn(x_k, s=(D, H, Wsz), dim=(-3, -2, -1))
            self.last_clamped_frac = float((y < -0.999).float().mean().detach())
            return log_compress(y, self.log_compression_kind).unsqueeze(1)
        if not self.disable_mas and self.use_log1p and self.mas_before_log1p:
            x_k = torch.fft.rfftn(x, dim=(-3, -2, -1)) * self.mas_corr
            x = torch.fft.irfftn(x_k, s=(D, H, Wsz), dim=(-3, -2, -1))
            x_k = torch.fft.rfftn(
                log_compress(x, self.log_compression_kind), dim=(-3, -2, -1)
            )
        else:
            if self.use_log1p:
                x = log_compress(x, self.log_compression_kind)
            x_k = torch.fft.rfftn(x, dim=(-3, -2, -1))
            if not self.disable_mas:
                x_k = x_k * self.mas_corr
        x_k = x_k * self.get_W().to(x_k.dtype)
        return torch.fft.irfftn(x_k, s=(D, H, Wsz), dim=(-3, -2, -1)).unsqueeze(1)


class HessianFromDeltaPlus(nn.Module):
    def __init__(
        self,
        grid_size: int,
        box_size: float,
        mas_scheme: str = "PCS",
        disable_mas: bool = False,
        use_log1p: bool = False,
        hessian_operator: str = "density_hessian",
        smooth_R_voxels: float = 0.0,
        k_grid_units: str = "dimensionless",
        corner_mask: bool = False,
        mas_before_log1p: bool = True,
        log_compression_kind: str = "log1p",
        pre_taper_kpass: Optional[float] = None,
        pre_taper_floor: float = 0.0,
        nyquist_fix: bool = True,
        smooth_order: str = "log_then_smooth",
    ):
        super().__init__()
        if smooth_order not in ("log_then_smooth", "smooth_then_log"):
            raise ValueError(
                f"smooth_order must be 'log_then_smooth' or 'smooth_then_log'; got {smooth_order!r}"
            )
        self.smooth_order = str(smooth_order)
        self.N = int(grid_size)
        self.box_size = float(box_size)
        self.disable_mas = bool(disable_mas)
        self.use_log1p = bool(use_log1p)
        if log_compression_kind not in ("log1p", "log2p"):
            raise ValueError(
                f"log_compression_kind must be 'log1p' or 'log2p'; got {log_compression_kind!r}"
            )
        self.log_compression_kind = str(log_compression_kind)
        if hessian_operator not in ("density_hessian", "potential_tweb"):
            raise ValueError(
                f"hessian_operator must be 'density_hessian' or 'potential_tweb'; got {hessian_operator!r}"
            )
        self.hessian_operator = str(hessian_operator)
        self.mas_before_log1p = bool(mas_before_log1p)
        self.smooth_R_voxels = float(smooth_R_voxels)
        self.pre_taper_kpass = (
            None if pre_taper_kpass is None else float(pre_taper_kpass)
        )
        self.pre_taper_floor = float(pre_taper_floor)
        if k_grid_units not in ("dimensionless", "physical"):
            raise ValueError(
                f"k_grid_units must be 'dimensionless' or 'physical'; got {k_grid_units!r}"
            )
        self.k_grid_units = str(k_grid_units)
        self.use_corner_mask = bool(corner_mask)
        self.nyquist_fix = bool(nyquist_fix)
        cell = self.box_size / self.N
        d = 1.0 if self.k_grid_units == "dimensionless" else cell
        kxg, kyg, kzg = rfft_kgrid(self.N, d=d)
        hess_k = torch.stack(
            [kxg * kxg, kyg * kyg, kzg * kzg, kxg * kyg, kxg * kzg, kyg * kzg], dim=0
        ).float()
        if self.hessian_operator == "potential_tweb":
            k2 = kxg * kxg + kyg * kyg + kzg * kzg
            inv_k2 = torch.where(k2 > 0.0, 1.0 / k2, torch.zeros_like(k2))
            hess_k = hess_k * inv_k2.unsqueeze(0).float()
        # Mixed derivatives must preserve Hermitian symmetry on Nyquist planes.
        if self.nyquist_fix and self.N % 2 == 0:
            n2 = self.N // 2
            ix = torch.arange(self.N).view(-1, 1, 1) == n2
            iy = torch.arange(self.N).view(1, -1, 1) == n2
            iz = torch.arange(self.N // 2 + 1).view(1, 1, -1) == n2
            hess_k[3] = torch.where(ix | iy, torch.zeros_like(hess_k[3]), hess_k[3])
            hess_k[4] = torch.where(ix | iz, torch.zeros_like(hess_k[4]), hess_k[4])
            hess_k[5] = torch.where(iy | iz, torch.zeros_like(hess_k[5]), hess_k[5])
        self.register_buffer("hess_k", hess_k, persistent=False)
        self.register_buffer(
            "mas_corr",
            mas_deconv_kernel(self.N, mas_scheme, self.box_size).float(),
            persistent=False,
        )
        k_mag = torch.sqrt(kxg**2 + kyg**2 + kzg**2)
        k_nyq = math.pi / d
        if self.smooth_R_voxels > 0.0:
            R_units = self.smooth_R_voxels * (
                1.0 if self.k_grid_units == "dimensionless" else cell
            )
            self.register_buffer(
                "smooth",
                torch.exp(-0.5 * (k_mag * R_units) ** 2).float(),
                persistent=False,
            )
        else:
            self.smooth = None
        self.register_buffer(
            "corner_mask",
            (k_mag / k_nyq <= 1.0).float()
            if self.use_corner_mask
            else torch.ones_like(k_mag).float(),
            persistent=False,
        )
        if self.pre_taper_kpass is not None:
            x = (k_mag / k_nyq).clamp(0.0, math.sqrt(3.0))
            taper = tukey_window(x, self.pre_taper_kpass, self.pre_taper_floor)
            taper = torch.where(x <= 1.0, taper, torch.zeros_like(taper))
            self.register_buffer("pre_taper", taper.float(), persistent=False)
        else:
            self.pre_taper = None

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.dim() != 5 or delta.shape[1] != 1:
            raise ValueError(f"expected [B, 1, D, H, W], got {tuple(delta.shape)}")
        B, _, D, H, Wsz = delta.shape
        x = delta.squeeze(1)
        smooth_first = (
            self.smooth_order == "smooth_then_log"
            and self.smooth is not None
            and self.use_log1p
        )
        if smooth_first or (
            not self.disable_mas and self.use_log1p and self.mas_before_log1p
        ):
            x_k = torch.fft.rfftn(x, dim=(-3, -2, -1))
            if self.pre_taper is not None:
                x_k = x_k * self.pre_taper.to(x_k.dtype)
            if not self.disable_mas:
                x_k = x_k * self.mas_corr
            if smooth_first:
                x_k = x_k * self.smooth
            x = torch.fft.irfftn(x_k, s=(D, H, Wsz), dim=(-3, -2, -1))
            x_k = torch.fft.rfftn(
                log_compress(x, self.log_compression_kind), dim=(-3, -2, -1)
            )
        else:
            if self.use_log1p:
                x = log_compress(x, self.log_compression_kind)
            x_k = torch.fft.rfftn(x, dim=(-3, -2, -1))
            if self.pre_taper is not None:
                x_k = x_k * self.pre_taper.to(x_k.dtype)
            if not self.disable_mas:
                x_k = x_k * self.mas_corr
        if self.smooth is not None and (not smooth_first):
            x_k = x_k * self.smooth
        x_k = x_k * self.corner_mask
        comp_k = (self.hess_k.to(x_k.dtype).unsqueeze(0) * x_k.unsqueeze(1)).reshape(
            B * 6, D, H, Wsz // 2 + 1
        )
        return torch.fft.irfftn(comp_k, s=(D, H, Wsz), dim=(-3, -2, -1)).reshape(
            B, 6, D, H, Wsz
        )


def hessian_eigs_from_components(a, d, f, b, c, e, eps: float = 1e-07) -> torch.Tensor:
    tr = a + d + f
    q = tr / 3.0
    a1, d1, f1 = (a - q, d - q, f - q)
    p2 = a1 * a1 + d1 * d1 + f1 * f1 + 2.0 * (b * b + c * c + e * e)
    p = torch.sqrt(torch.clamp(p2 / 6.0, min=1e-30))
    mask = p <= eps
    p_safe = torch.where(mask, torch.ones_like(p), p)
    B11, B22, B33 = (a1 / p_safe, d1 / p_safe, f1 / p_safe)
    B12, B13, B23 = (b / p_safe, c / p_safe, e / p_safe)
    detB = (
        B11 * (B22 * B33 - B23 * B23)
        - B12 * (B12 * B33 - B23 * B13)
        + B13 * (B12 * B23 - B22 * B13)
    )
    r = torch.clamp(detB / 2.0, -1.0 + 1e-07, 1.0 - 1e-07)
    phi = torch.acos(r) / 3.0
    two_pi_3 = 2.0 * math.pi / 3.0
    lam1 = torch.where(mask, q, q + 2.0 * p * torch.cos(phi))
    lam2 = torch.where(mask, q, q + 2.0 * p * torch.cos(phi + two_pi_3))
    lam3 = torch.where(mask, q, q + 2.0 * p * torch.cos(phi + 2.0 * two_pi_3))
    lams, _ = torch.sort(torch.stack([lam1, lam2, lam3], dim=-1), dim=-1)
    return lams


class EigenvalueExtractor(nn.Module):
    def forward(self, hess: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if hess.dim() != 5 or hess.shape[1] != 6:
            raise ValueError(f"expected [B, 6, D, H, W], got {tuple(hess.shape)}")
        eigs = hessian_eigs_from_components(
            hess[:, 0], hess[:, 1], hess[:, 2], hess[:, 3], hess[:, 4], hess[:, 5]
        )
        eigs = eigs.permute(0, 4, 1, 2, 3).contiguous()
        return (eigs, eigs[:, 2:3] - eigs[:, 0:1])


class MultiThresholdWebGate(nn.Module):
    gate_names = ("void", "sheet", "filament", "cluster")
    _SOFTPLUS_INV_ZERO = math.log(math.expm1(0.001))

    def __init__(
        self,
        init_tau: float = 1.0,
        learnable_threshold: bool = False,
        learnable_tau: bool = True,
        mode: str = "soft",
    ):
        super().__init__()
        if mode not in ("soft", "hard", "hard_st"):
            raise ValueError(
                f"gate_mode must be 'soft', 'hard' or 'hard_st'; got {mode!r}"
            )
        self.mode = str(mode)
        self.learnable_tau = bool(learnable_tau)
        log_tau_init = torch.tensor(math.log(float(init_tau)), dtype=torch.float32)
        if self.learnable_tau:
            self.log_tau = nn.Parameter(log_tau_init)
        else:
            self.register_buffer("log_tau", log_tau_init)
        self.learnable_threshold = bool(learnable_threshold)
        if self.learnable_threshold:
            self.mu_base = nn.Parameter(torch.tensor(0.0))
            self.delta2_raw = nn.Parameter(torch.tensor(self._SOFTPLUS_INV_ZERO))
            self.delta3_raw = nn.Parameter(torch.tensor(self._SOFTPLUS_INV_ZERO))
        else:
            self.register_buffer("mu_base", torch.tensor(0.0))
            self.register_buffer("delta2_raw", torch.tensor(-1000000.0))
            self.register_buffer("delta3_raw", torch.tensor(-1000000.0))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    @property
    def mus(self) -> torch.Tensor:
        mu1 = self.mu_base
        mu2 = mu1 - F.softplus(self.delta2_raw)
        mu3 = mu2 - F.softplus(self.delta3_raw)
        return torch.stack([mu1, mu2, mu3])

    def forward(self, eigs: torch.Tensor) -> torch.Tensor:
        mu_b = self.mus.view(1, 3, *[1] * (eigs.dim() - 2))
        p = torch.sigmoid((eigs - mu_b) / self.tau)
        p1, p2, p3 = (p[:, 0:1], p[:, 1:2], p[:, 2:3])
        soft = torch.cat([1.0 - p3, p3 - p2, p2 - p1, p1], dim=1)
        if self.mode == "soft":
            return soft
        with torch.no_grad():
            count = (eigs > mu_b).sum(dim=1, keepdim=True).long()
            hard = torch.zeros_like(soft).scatter_(1, count, 1.0)
        if self.mode == "hard":
            return hard
        return hard + (soft - soft.detach())


class ScalarCurve(nn.Module):
    def __init__(
        self, hidden: int = 16, act: str = "gelu", output_bound: Optional[float] = None
    ):
        super().__init__()
        if act.lower() == "gelu":
            act_layer = nn.GELU()
        elif act.lower() == "silu":
            act_layer = nn.SiLU()
        else:
            raise ValueError(f"act must be 'gelu' or 'silu', got {act!r}")
        self.net = nn.Sequential(nn.Linear(1, hidden), act_layer, nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.output_bound = None if output_bound is None else float(output_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.reshape(-1, 1))
        if self.output_bound is not None:
            y = self.output_bound * torch.tanh(y / self.output_bound)
        return y.reshape(x.shape)
