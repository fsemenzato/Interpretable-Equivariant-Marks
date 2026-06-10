import torch, warnings, numpy as np
from typing import Dict, Tuple, Optional

class PkEstimator:
    def __init__(self, BoxSize: float, device: str | torch.device = "cpu"):
        if "cuda" in str(device) and not torch.cuda.is_available():
            warnings.warn("CUDA not available -- falling back to CPU.")
            device = "cpu"
        self.device  = torch.device(device)
        self.BoxSize = float(BoxSize)

        # (N, device)
        self._kgrid_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._stats_cache: Dict[Tuple[int, torch.device], Tuple]        = {}
        # (N, MAS, device)
        self._mas_cache:   Dict[Tuple[int, str, torch.device], torch.Tensor] = {}

    def compute(
        self,
        delta_r_1: torch.Tensor | np.ndarray,
        delta_r_2: Optional[torch.Tensor | np.ndarray] = None,
        MAS: Tuple[Optional[str], Optional[str]] = (None, None),
    ) -> torch.Tensor:
        
        # (B, 1, N, N, N) 
        if hasattr(delta_r_1, "ndim") and delta_r_1.ndim == 5 and delta_r_1.shape[1] == 1:
            delta_r_1 = delta_r_1.squeeze(1)
        d1 = self._as_tensor(delta_r_1)

        if delta_r_2 is not None:
            if delta_r_2.ndim == 5 and delta_r_2.shape[1] == 1:
                delta_r_2 = delta_r_2.squeeze(1)
            d2 = self._as_tensor(delta_r_2)
        else:
            d2 = d1

        if d1.shape != d2.shape:
            raise ValueError("delta_r_1 and delta_r_2 must have identical shapes.")

        B, N, _, _ = d1.shape
        d1k = torch.fft.rfftn(d1, dim=(-3, -2, -1))
        d2k = d1k if d1 is d2 else torch.fft.rfftn(d2, dim=(-3, -2, -1))

        return self._pk_from_fft(d1k, d2k, N, B, MAS, same=(d1 is d2))

    def compute_from_fft(
        self,
        d1k: torch.Tensor,
        d2k: torch.Tensor,
        N: int,
        MAS: Tuple[Optional[str], Optional[str]] = (None, None),
    ) -> torch.Tensor:

        if d1k.ndim == 3:
            d1k = d1k.unsqueeze(0)
        if d2k.ndim == 3:
            d2k = d2k.unsqueeze(0)
        B = d1k.shape[0]
        return self._pk_from_fft(d1k, d2k, N, B, MAS, same=(d1k is d2k))

    def _pk_from_fft(self, d1k, d2k, N, B, MAS, *, same):

        mas0 = MAS[0] if (MAS[0] and str(MAS[0]).upper() != "NONE") else None
        mas1 = MAS[1] if (MAS[1] and str(MAS[1]).upper() != "NONE") else None

        if mas0 is not None:
            d1k = d1k * self._get_mas(N, mas0).unsqueeze(0)
        if same:
            d2k = d1k
        elif mas1 is not None:
            d2k = d2k * self._get_mas(N, mas1).unsqueeze(0)

        V    = self.BoxSize ** 3
        norm = V / (N ** 6)

        P1 = (d1k.conj() * d1k).real * norm
        P2 = (d2k.conj() * d2k).real * norm
        PX = (d1k.conj() * d2k).real * norm

        # bin assignment, k-shell centers, per-shell mode count.
        valid_mask, k_idx_valid0, k_center, n_modes, w_valid, num_bins = \
            self._get_geometry(N)

        Pk1 = self._bin_weighted(P1, valid_mask, k_idx_valid0, w_valid,
                                  n_modes, num_bins, B)
        Pk2 = self._bin_weighted(P2, valid_mask, k_idx_valid0, w_valid,
                                  n_modes, num_bins, B)
        PkX = self._bin_weighted(PX, valid_mask, k_idx_valid0, w_valid,
                                  n_modes, num_bins, B)

        kcen_b = k_center.unsqueeze(0).expand(B, -1)
        Nm_b   = n_modes .unsqueeze(0).expand(B, -1)
        return torch.stack([kcen_b, Pk1, Pk2, PkX, Nm_b], dim=2)  # (B, Nk, 5)

    def _bin_weighted(self, P, valid_mask, k_idx_valid0, w_valid, n_modes,
                      num_bins, B):
        
        # P shape (B, N, N, N//2+1)
        P_flat = P.permute(1, 2, 3, 0)[valid_mask].permute(1, 0)   # (B, M)
        weighted = P_flat * w_valid.unsqueeze(0)
        sums = torch.zeros(B, num_bins, device=self.device, dtype=P.dtype)
        idx  = k_idx_valid0.unsqueeze(0).expand(B, -1)
        sums.scatter_add_(1, idx, weighted)
        n_safe = torch.where(n_modes == 0, torch.ones_like(n_modes), n_modes)
        return sums / n_safe.unsqueeze(0)

    def _as_tensor(self, arr):
        if arr is None:
            raise ValueError("Input cannot be None.")
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr.astype(np.float32, copy=False))
        arr = arr.to(self.device, dtype=torch.float32)
        if arr.ndim == 3:
            arr = arr.unsqueeze(0)
        if arr.ndim != 4:
            raise ValueError(f"Input must be 3- or 4-D. Got {arr.ndim}-D.")
        return arr

    def _get_geometry(self, N):
        key = (N, self.device)
        if key in self._stats_cache:
            return self._stats_cache[key]

        middle = N // 2
        _, kF = self._get_kgrid(N)


        n_axis = torch.arange(N, device=self.device, dtype=torch.long)
        nx = torch.where(n_axis > middle, n_axis - N, n_axis)
        nz = torch.arange(N // 2 + 1, device=self.device, dtype=torch.long)
        n2 = (nx.view(N, 1, 1) ** 2
              + nx.view(1, N, 1) ** 2
              + nz.view(1, 1, N // 2 + 1) ** 2) # (N, N, N//2+1) int64

        k_mag_kF = torch.sqrt(n2.to(torch.float64))
        k_idx    = torch.floor(k_mag_kF).long()
        
        # |k| = kF * sqrt(n^2)
        k_mag_phys = (k_mag_kF * kF.to(torch.float64))

        # drop k=0 (DC) 
        kmax_bin = int(np.floor(np.sqrt(3.0) * middle + 1e-12))
        num_bins = kmax_bin  # bins 1..kmax_bin -> output indices 0..kmax_bin-1

        valid_mask   = (k_idx >= 1) & (k_idx <= kmax_bin)
        k_idx_valid0 = (k_idx[valid_mask] - 1)
        k_flat       = k_mag_phys[valid_mask]

        # Independent-mode weight (corrects rfft Hermitian double-counting).
        w = self._build_indep_weight(N)
        w_valid = w[valid_mask]

        n_modes = torch.zeros(num_bins, device=self.device, dtype=torch.float64)
        k_sum   = torch.zeros(num_bins, device=self.device, dtype=torch.float64)
        n_modes.scatter_add_(0, k_idx_valid0, w_valid.to(torch.float64))
        k_sum  .scatter_add_(0, k_idx_valid0, (k_flat * w_valid.to(torch.float64)))

        n_safe   = torch.where(n_modes == 0, torch.ones_like(n_modes), n_modes)
        k_center = (k_sum / n_safe).to(torch.float32)

        n_modes_f32 = n_modes.to(torch.float32)
        w_valid_f32 = w_valid.to(torch.float32)

        self._stats_cache[key] = (valid_mask, k_idx_valid0,
                                  k_center.detach(), n_modes_f32.detach(),
                                  w_valid_f32.detach(), num_bins)
        return self._stats_cache[key]

    def _get_kgrid(self, N):
        key = (N, self.device)
        if key in self._kgrid_cache:
            return self._kgrid_cache[key]
        cell = self.BoxSize / N
        kx = 2 * np.pi * torch.fft.fftfreq(N,  d=cell, device=self.device)
        ky = kx
        kz = 2 * np.pi * torch.fft.rfftfreq(N, d=cell, device=self.device)
        kxg, kyg, kzg = torch.meshgrid(kx, ky, kz, indexing="ij")
        k_mag = torch.sqrt(kxg * kxg + kyg * kyg + kzg * kzg).detach()
        kF = torch.tensor(2 * np.pi / self.BoxSize, device=self.device)
        self._kgrid_cache[key] = (k_mag, kF)
        return k_mag, kF

    def _build_indep_weight(self, N):

        device = self.device
        w = torch.ones((N, N, N // 2 + 1), device=device, dtype=torch.float32)
        middle = N // 2

        special_planes = [0]
        if N % 2 == 0:
            special_planes.append(middle)

        sc_idx = [0]
        if N % 2 == 0:
            sc_idx.append(middle)

        for kz_s in special_planes:
            w[:, :, kz_s] = 0.5
            for kxs in sc_idx:
                for kys in sc_idx:
                    w[kxs, kys, kz_s] = 1.0
        return w

    def _get_mas(self, N, scheme: str):
        key = (N, scheme.upper(), self.device)
        if key in self._mas_cache:
            return self._mas_cache[key]
        p = {"NGP": 1, "CIC": 2, "TSC": 3, "PCS": 4}.get(scheme.upper())
        if p is None:
            raise ValueError(f"Unknown MAS scheme '{scheme}'")

        cell = self.BoxSize / N
        kx = 2 * np.pi * torch.fft.fftfreq(N,  d=cell, device=self.device)
        ky = kx
        kz = 2 * np.pi * torch.fft.rfftfreq(N, d=cell, device=self.device)
        kxg, kyg, kzg = torch.meshgrid(kx, ky, kz, indexing="ij")
        fac = lambda k: k * cell / 2

        def inv_sinc_pow(x):
            s = torch.ones_like(x)
            nz = torch.abs(x) > 1e-9
            s[nz] = torch.sin(x[nz]) / x[nz]
            return s.pow(-p)

        mas = (inv_sinc_pow(fac(kxg))
               * inv_sinc_pow(fac(kyg))
               * inv_sinc_pow(fac(kzg)))
        self._mas_cache[key] = mas.detach()
        return mas
