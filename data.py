from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

EPSILON_STABILITY = 1e-8


class QuijoteDataset(Dataset):
    
    def __init__(self, data_root, data_name: str, param_file: str, sim_indices, normalizer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        self.data_root = Path(data_root)
        self.data_name = data_name
        self.sim_indices = np.asarray(sim_indices)
        params = np.loadtxt(param_file)
        if params.ndim == 1:
            params = params[:, None]
        self.params = torch.from_numpy(params[self.sim_indices]).float()

        if normalizer is not None:
            self.theta_mean, self.theta_std = normalizer
        else:
            self.theta_mean = torch.mean(self.params, dim=0)
            self.theta_std = torch.std(self.params, dim=0)

    def __len__(self) -> int:
        return len(self.sim_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sim_index = int(self.sim_indices[idx])
        field_path = self.data_root / str(sim_index) / self.data_name
        delta_field = torch.from_numpy(np.load(field_path)).float().unsqueeze(0)  # [1, N, N, N]
        theta = self.params[idx]
        normalized_theta = (theta - self.theta_mean) / (self.theta_std + EPSILON_STABILITY)
        return delta_field, normalized_theta, sim_index








def scan_available_sims(data_root, data_name: str, n_params: int) -> list:
    data_root = Path(data_root)
    available = []
    missing = []
    for i in range(n_params):
        if (data_root / str(i) / data_name).exists():
            available.append(i)
        else:
            missing.append(i)
    if missing:
        print(f"  Warning: {len(missing)} simulations missing out of {n_params}")
    return available


# 







def sample_lhs_vectors_torch_global(min_p, max_p, n_samples, device=None, dtype=None):
    # LHS over [min_p, max_p]
    
    min_vals = torch.as_tensor(min_p, device=device, dtype=dtype)
    max_vals = torch.as_tensor(max_p, device=device, dtype=dtype)

    D = min_vals.size(0)
    if dtype is None and not min_vals.is_floating_point():
        calc_dtype = torch.float32
    else:
        calc_dtype = dtype if dtype is not None else min_vals.dtype
    min_vals = min_vals.to(dtype=calc_dtype)
    max_vals = max_vals.to(dtype=calc_dtype)

    unit_step = 1.0 / n_samples
    base_grid = torch.arange(n_samples, device=device, dtype=calc_dtype) * unit_step
    base_grid = base_grid.unsqueeze(0).expand(D, n_samples)
    jitter = torch.rand(D, n_samples, device=device, dtype=calc_dtype) * unit_step
    permutations = torch.rand(D, n_samples, device=device, dtype=calc_dtype).argsort(dim=-1)
    stratified_base = torch.gather(base_grid, 1, permutations)
    lhs_01 = (stratified_base + jitter).t()
    diff = max_vals.unsqueeze(0) - min_vals.unsqueeze(0)
    
    return min_vals.unsqueeze(0) + lhs_01 * diff








def sample_lhs_batched(min_vals, max_vals, n_samples, device=None, dtype=None):
    # n_samples for EACH item in the batch
   
    B, D = min_vals.shape
    if dtype is None:
        dtype = min_vals.dtype
    unit_step = 1.0 / n_samples
    base_grid = torch.arange(n_samples, device=device, dtype=dtype) * unit_step
    base_grid = base_grid.view(1, 1, n_samples).expand(B, D, n_samples)
    jitter = torch.rand(B, D, n_samples, device=device, dtype=dtype) * unit_step
    permutations = torch.rand(B, D, n_samples, device=device, dtype=dtype).argsort(dim=-1)
    stratified_base = torch.gather(base_grid, 2, permutations)
    lhs_01 = stratified_base + jitter
    local_min = min_vals.unsqueeze(-1)
    local_max = max_vals.unsqueeze(-1)
    samples = local_min + lhs_01 * (local_max - local_min)
    
    return samples.permute(0, 2, 1).reshape(-1, D)








def _reject_too_close(candidates, anchors_phys, inv_range, exclusion_radius_norm, global_min, global_max, max_retries: int = 3):
    
    if exclusion_radius_norm <= 0:
        return candidates
    D = candidates.shape[1]
    dev = candidates.device
    for _ in range(max_retries):
        delta = (candidates.unsqueeze(1) - anchors_phys.unsqueeze(0)) * inv_range.unsqueeze(0)
        dists = delta.norm(dim=-1)
        min_dist_to_any_anchor = dists.min(dim=1).values
        too_close = min_dist_to_any_anchor < exclusion_radius_norm
        if not too_close.any():
            break
        n_replace = int(too_close.sum().item())
        replacements = global_min + torch.rand(n_replace, D, device=dev) * (global_max - global_min)
        candidates[too_close] = replacements
    delta = (candidates.unsqueeze(1) - anchors_phys.unsqueeze(0)) * inv_range.unsqueeze(0)
    dists = delta.norm(dim=-1)
    still_too_close = dists.min(dim=1).values < exclusion_radius_norm
    if still_too_close.any():
        raise RuntimeError(
            f"Failed to generate valid negatives outside the exclusion radius after {max_retries} retries; {int(still_too_close.sum().item())}.")
    return candidates








def _sample_shell_offsets_normalized(n_samples, dim, r_min_norm, r_max_norm, *, device=None, dtype=None):

    # uniform sampling spherical shell in normalized space

    if n_samples <= 0:
        return torch.empty((0, dim), device=device, dtype=dtype or torch.float32)
    if r_min_norm < 0 or r_max_norm <= 0 or r_min_norm >= r_max_norm:
        raise ValueError(f"Invalid shell radii: r_min_norm={r_min_norm}, r_max_norm={r_max_norm}")
    if dtype is None:
        dtype = torch.float32
    directions = torch.randn(n_samples, dim, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    u = torch.rand(n_samples, device=device, dtype=dtype)
    r_min_pow = float(r_min_norm) ** dim
    r_max_pow = float(r_max_norm) ** dim
    radii = (u * (r_max_pow - r_min_pow) + r_min_pow).pow(1.0 / dim)
    return directions * radii.unsqueeze(1)








def _sample_local_annulus_exact(anchors_phys, 
                                global_min, 
                                global_max, 
                                global_range,
                                *, 
                                n_samples_per_anchor, 
                                r_min_norm, 
                                r_max_norm,
                                all_anchor_exclusion_norm: float = 0.0,
                                device=None, 
                                dtype=None, 
                                oversample_factor: int = 8,
                                max_rounds: int = 32):
   
    # local negatives from a true spherical annulus in normalized space
    
    if n_samples_per_anchor <= 0:
        dim = anchors_phys.shape[1]
        return torch.empty((0, dim), device=device or anchors_phys.device,
                           dtype=dtype or anchors_phys.dtype)
    if r_min_norm < 0 or r_max_norm <= 0 or r_min_norm >= r_max_norm:
        raise ValueError(f"Invalid annulus radii: r_min_norm={r_min_norm}, r_max_norm={r_max_norm}")
    if device is None:
        device = anchors_phys.device
    if dtype is None:
        dtype = anchors_phys.dtype
        
        
    anchors_phys = anchors_phys.to(device=device, dtype=dtype)
    global_min = global_min.to(device=device, dtype=dtype)
    global_max = global_max.to(device=device, dtype=dtype)
    global_range = global_range.to(device=device, dtype=dtype)
    inv_range = 1.0 / global_range
    dim = anchors_phys.shape[1]
    accepted_per_anchor = []
    for anchor in anchors_phys:
        accepted = []
        n_collected = 0
        for _ in range(max_rounds):
            n_needed = n_samples_per_anchor - n_collected
            if n_needed <= 0:
                break
            n_proposals = max(n_needed * oversample_factor, n_needed)
            offsets_norm = _sample_shell_offsets_normalized(
                n_proposals, dim, r_min_norm, r_max_norm, device=device, dtype=dtype)
            candidates = anchor.unsqueeze(0) + offsets_norm * global_range.unsqueeze(0)
            in_bounds = ((candidates >= global_min) & (candidates <= global_max)).all(dim=1)
            valid = in_bounds
            if all_anchor_exclusion_norm > 0:
                delta = (candidates.unsqueeze(1) - anchors_phys.unsqueeze(0)) * inv_range.unsqueeze(0)
                min_dist = delta.norm(dim=-1).min(dim=1).values
                valid = valid & (min_dist >= all_anchor_exclusion_norm)
            if valid.any():
                chunk = candidates[valid][:n_needed]
                accepted.append(chunk)
                n_collected += len(chunk)
        if n_collected < n_samples_per_anchor:
            raise RuntimeError(
                f"Failed to sample enough valid local annulus negatives for anchor after {max_rounds} rounds: got {n_collected}, needed {n_samples_per_anchor}.")
        accepted_per_anchor.append(torch.cat(accepted, dim=0)[:n_samples_per_anchor])
    return torch.cat(accepted_per_anchor, dim=0)








def get_mixed_distractors(theta_real, 
                          all_train_params, 
                          batch_indices=None, 
                          *,
                          n_real: int = 64, 
                          n_global_synthetic: int = 256,
                          n_local_synthetic: int = 32, 
                          neighborhood_scale: float = 0.03,
                          min_distance_scale: float = 0.01,
                          negative_exclusion_scale: float = 0.0,
                          local_all_anchor_exclusion_scale: float = 0.0,
                          device="cpu", 
                          theta_mean=None, 
                          theta_std=None,
                          theta_is_normalized: bool = True, 
                          gen_device: str = "cpu"):
    
    
    # real training negatives + global LHC + local annulus negatives
    
    eps = EPSILON_STABILITY
    B, D = theta_real.shape

    theta_real_cpu = theta_real.to(gen_device, dtype=torch.float32)
    all_params_cpu = all_train_params.to(gen_device, dtype=torch.float32)

    global_min = all_params_cpu.min(dim=0).values
    global_max = all_params_cpu.max(dim=0).values
    global_range = (global_max - global_min).clamp_min(eps)
    inv_range = 1.0 / global_range

    if theta_is_normalized:
        if theta_mean is None or theta_std is None:
            raise ValueError("theta_mean/theta_std must be provided when theta_is_normalized=True")
        mean_cpu = theta_mean.to(gen_device, dtype=torch.float32)
        std_cpu = theta_std.to(gen_device, dtype=torch.float32)
        theta_real_phys = theta_real_cpu * (std_cpu + eps) + mean_cpu
    else:
        mean_cpu = None
        std_cpu = None
        theta_real_phys = theta_real_cpu

    exclusion_radius_norm = negative_exclusion_scale * (D ** 0.5)
    all_negatives = []

    # REAL NEGATIVES
    if n_real > 0:
        N_train = len(all_params_cpu)
        if batch_indices is not None:
            mask = torch.ones(N_train, dtype=torch.bool, device=gen_device)
            batch_idx_cpu = batch_indices.to(gen_device)
            mask[batch_idx_cpu] = False
            valid_indices = torch.where(mask)[0]
        else:
            valid_indices = torch.arange(N_train, device=gen_device)
        n_sample = min(n_real, len(valid_indices))
        perm = torch.randperm(len(valid_indices), device=gen_device)[:n_sample]
        real_neg_idx = valid_indices[perm]
        theta_real_neg = all_params_cpu[real_neg_idx]
        if exclusion_radius_norm > 0:
            delta = (theta_real_neg.unsqueeze(1) - theta_real_phys.unsqueeze(0)) * inv_range.unsqueeze(0)
            dists = delta.norm(dim=-1).min(dim=1).values
            keep = dists >= exclusion_radius_norm
            theta_real_neg = theta_real_neg[keep]
            n_deficit = n_sample - len(theta_real_neg)
            if n_deficit > 0:
                remaining = torch.ones(N_train, dtype=torch.bool, device=gen_device)
                remaining[real_neg_idx] = False
                if batch_indices is not None:
                    remaining[batch_idx_cpu] = False
                extra_idx = torch.where(remaining)[0]
                if len(extra_idx) > 0:
                    extra_pool = all_params_cpu[extra_idx]
                    extra_delta = (extra_pool.unsqueeze(1) - theta_real_phys.unsqueeze(0)) * inv_range.unsqueeze(0)
                    extra_dists = extra_delta.norm(dim=-1).min(dim=1).values
                    extra_keep = extra_dists >= exclusion_radius_norm
                    extra_pool = extra_pool[extra_keep]
                    if len(extra_pool) > 0:
                        extra_perm = torch.randperm(len(extra_pool), device=gen_device)[:n_deficit]
                        theta_real_neg = torch.cat([theta_real_neg, extra_pool[extra_perm]], dim=0)
        all_negatives.append(theta_real_neg)

    # GLOBAL SYNTHETIC
    if n_global_synthetic > 0:
        n_generate = int(n_global_synthetic * 1.5) if exclusion_radius_norm > 0 else n_global_synthetic
        theta_global = sample_lhs_vectors_torch_global(
            global_min, global_max, n_samples=n_generate, device=gen_device, dtype=torch.float32)
        theta_global = _reject_too_close(
            theta_global, theta_real_phys, inv_range, exclusion_radius_norm, global_min, global_max)
        all_negatives.append(theta_global[:n_global_synthetic])

    # LOCAL SYNTHETIC 
    if n_local_synthetic > 0:
        max_distance_normalized = neighborhood_scale * (D ** 0.5)
        min_distance_normalized = min_distance_scale * (D ** 0.5)
        local_all_anchor_exclusion_norm = local_all_anchor_exclusion_scale * (D ** 0.5)
        if max_distance_normalized <= min_distance_normalized:
            raise ValueError(
                "Invalid local annulus: neighborhood_scale must exceed min_distance_scale.")
        theta_local = _sample_local_annulus_exact(
            theta_real_phys, global_min, global_max, global_range,
            n_samples_per_anchor=n_local_synthetic,
            r_min_norm=min_distance_normalized, r_max_norm=max_distance_normalized,
            all_anchor_exclusion_norm=local_all_anchor_exclusion_norm,
            device=gen_device, dtype=torch.float32)
        all_negatives.append(theta_local)

    theta_phys = torch.cat(all_negatives, dim=0)
    if theta_is_normalized:
        theta_dist = (theta_phys - mean_cpu) / (std_cpu + eps)
    else:
        theta_dist = theta_phys
    return theta_dist.to(device, dtype=torch.float32)








def setup_validation_negatives(val_params, 
                               train_params, 
                               theta_mean, 
                               theta_std,
                               n_synthetic: int = 512, 
                               n_real_from_train: int = 0,
                               include_val_params: bool = False, 
                               seed: int = 42,
                               n_local_per_val: int = 0, 
                               neighborhood_scale: float = 0.05,
                               min_distance_scale: float = 0.01,
                               negative_exclusion_scale: float = 0.0,
                               local_all_anchor_exclusion_scale: float = 0.0):


    eps = EPSILON_STABILITY
    global_min = train_params.min(dim=0).values
    global_max = train_params.max(dim=0).values
    global_range = (global_max - global_min).clamp_min(eps)
    inv_range = 1.0 / global_range
    dev = val_params.device
    D = val_params.shape[1]
    exclusion_radius_norm = negative_exclusion_scale * (D ** 0.5)

    cpu_rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    pool_phys = []
    n_generate = int(n_synthetic * 1.5) if exclusion_radius_norm > 0 else n_synthetic
    theta_synthetic = sample_lhs_vectors_torch_global(
        global_min, global_max, n_samples=n_generate, device=dev, dtype=torch.float32)
    if exclusion_radius_norm > 0:
        theta_synthetic = _reject_too_close(
            theta_synthetic, val_params, inv_range, exclusion_radius_norm, global_min, global_max)
    pool_phys.append(theta_synthetic[:n_synthetic])

    if n_real_from_train > 0:
        n_sample = min(n_real_from_train, len(train_params))
        full_perm = torch.randperm(len(train_params), device=train_params.device)
        sample_idx = full_perm[:n_sample]
        theta_real = train_params[sample_idx]
        if exclusion_radius_norm > 0:
            delta = (theta_real.unsqueeze(1) - val_params.unsqueeze(0)) * inv_range.unsqueeze(0)
            dists = delta.norm(dim=-1).min(dim=1).values
            keep = dists >= exclusion_radius_norm
            theta_real = theta_real[keep]
        pool_phys.append(theta_real)

    if include_val_params:
        pool_phys.append(val_params)

    if n_local_per_val > 0:
        max_dist_norm = neighborhood_scale * (D ** 0.5)
        min_dist_norm = min_distance_scale * (D ** 0.5)
        local_all_anchor_exclusion_norm = local_all_anchor_exclusion_scale * (D ** 0.5)
        if max_dist_norm <= min_dist_norm:
            raise ValueError(
                "Invalid validation annulus: neighborhood_scale must exceed min_distance_scale.")
        theta_local = _sample_local_annulus_exact(
            val_params, global_min, global_max, global_range,
            n_samples_per_anchor=n_local_per_val,
            r_min_norm=min_dist_norm, r_max_norm=max_dist_norm,
            all_anchor_exclusion_norm=local_all_anchor_exclusion_norm,
            device=dev, dtype=torch.float32)
        pool_phys.append(theta_local)

    torch.random.set_rng_state(cpu_rng_state)
    val_pool_phys = torch.cat(pool_phys, dim=0)
    return (val_pool_phys - theta_mean) / (theta_std + eps)
