import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pk import PkEstimator

OUTPUT_DIR = Path("./mark_cache_span")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

DATA_ROOT_FID = Path("/work/nvme/bdne/fsemenzato/quijote_z0_fid")
DATA_ROOT_DIFF = Path("/work/nvme/bdne/fsemenzato/quijote_z0_diffs")
DATA_ROOT_LHC = Path("/work/nvme/bdne/fsemenzato/quijote_z0")
DATA_ROOT_BSQ = Path("/work/hdd/bdne/fsemenzato/BSQ_Processing/")
DATA_NAME = "df_m_128_PCS_z=0.npy"
DATA_NAME_BSQ = "dm_z0_PCS_128.npy"

GRID_DIM = 128
BOX_SIZE = 1000.0
MAS = 'PCS' 

N_SIMS_FID = 10000
N_SIMS_LHC = 2000
N_SIMS_BSQ = 2000
N_SIMS_VARIED = 500

FOLDER_NAME_MAP = {"Om": "Om", "Ob": "Ob2", "h": "h", "ns": "ns", "s8": "s8"}
PARAM_LABELS = ['Om', 's8', 'ns', 'h', 'Ob']


class NumpyDataset(Dataset):
    def __init__(self, file_paths: List[Path]):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = np.load(path)
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        return torch.from_numpy(data).float()
    
def get_paths(dataroot, n_sims, data_name) -> List[Path]:
    paths = []
    print(f"Scanning {dataroot}...")
    for i in range(n_sims):
        p = dataroot / str(i) / data_name
        if p.exists():
            paths.append(p)
    print(f"Found {len(paths)}/{n_sims} simulations in {dataroot}.")
    return paths


def get_paths_diff(param: str, direction: str) -> List[Path]:
    folder_base = FOLDER_NAME_MAP.get(param, param)
    folder_name = f"{folder_base}_{direction}" 
    root = DATA_ROOT_DIFF / folder_name
    
    paths = []
    if not root.exists():
        warnings.warn(f"Difference root does not exist: {root}")
        return []

    for i in range(N_SIMS_VARIED):
        p = root / str(i) / DATA_NAME
        if p.exists():
            paths.append(p)
    
    return paths


def gaussian_filter_fft(delta: torch.Tensor, R: float, box_size: float) -> torch.Tensor:
    D, H, W = delta.shape[-3:]
    dk = fft.rfftn(delta, dim=(-3, -2, -1))
    
    kx = torch.fft.fftfreq(D, d=box_size/D, device=delta.device) * 2 * np.pi
    ky = torch.fft.fftfreq(H, d=box_size/H, device=delta.device) * 2 * np.pi
    kz = torch.fft.rfftfreq(W, d=box_size/W, device=delta.device) * 2 * np.pi
    
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    KK_sq = KX**2 + KY**2 + KZ**2
    
    G = torch.exp(-0.5 * KK_sq * (R**2)).to(delta.dtype)
    
    dk_f = dk * G
    df = fft.irfftn(dk_f, s=(D, H, W), dim=(-3, -2, -1))
    return df

def classical_mw_mark(delta: torch.Tensor, filt: torch.Tensor, s: float, p: float, box_size: float) -> torch.Tensor:
    #  m = ((1 + filt + ds) / (1 + ds))^p
    base = (1.0 + filt + s) / (1.0 + s)
    base = torch.clamp(base, min=1e-6) 
    mark = base ** p
    mkd = mark * (1.0 + delta)
    
    return mkd - mkd.mean(dim=(-3, -2, -1), keepdim=True)

def make_torch_interp1d(xp: np.ndarray, fp: np.ndarray, device):
    """Linear interpolation helper for HTM."""
    xp_t = torch.from_numpy(xp).float().to(device)
    fp_t = torch.from_numpy(fp).float().to(device)
    order = torch.argsort(xp_t)
    xp_t, fp_t = xp_t[order], fp_t[order]
    dx = xp_t[1:] - xp_t[:-1]
    slope = (fp_t[1:] - fp_t[:-1]) / dx

    def f(x: torch.Tensor) -> torch.Tensor:
        i = torch.searchsorted(xp_t, x, right=True) - 1
        i = i.clamp(0, xp_t.numel() - 2)
        x0, y0, m_slope = xp_t[i], fp_t[i], slope[i]
        return y0 + m_slope * (x - x0)
    return f

def classical_htm_mark(delta: torch.Tensor, R: float, mark_func_interp, box_size: float) -> torch.Tensor:
    filt = gaussian_filter_fft(delta, R, box_size)
    m = mark_func_interp(filt)
    return m * (1.0 + delta) - 1.0


def compute_and_save(
    sim_paths: List[Path],
    output_filename: str,
    args: argparse.Namespace,
    pk_estimator: PkEstimator,
):
    if not sim_paths:
        print(f"No paths found for {output_filename}, skipping.")
        return

    out_path = OUTPUT_DIR / output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(sim_paths)} sims -> {out_path}...")
    
    dataset = NumpyDataset(sim_paths)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    
    
    k_bins = None

    RR = [10,20,30]
    ss = np.linspace(0,0.5,6)
    pp = [-2.5,-2,-1.5,-1,1,1.5,2,2.5]
    
    res = {
        'unmarked': {'pu': [], 'pm': [], 'px': []},
        'mw':       {'pu': [], 'pm': [], 'px': []},
        'htm':      {'pu': [], 'pm': [], 'px': []}
    }
    
    for R in RR:
        for s in ss:
            for p in pp:
                res[f'mw_R{R}_s{s}_p{p}'] = {'pu': [], 'pm': [], 'px': []}
    
    with torch.no_grad():
        for delta_batch in tqdm(loader, desc=output_filename, leave=False):
            delta = delta_batch.to(DEVICE) # [B, 1, D, H, W]
            
            # --- 1. Unmarked ---
            d_sq = delta.squeeze(1)
            pk_un = pk_estimator.compute(d_sq, d_sq, MAS=(MAS, MAS))
            
            if k_bins is None:
                k_bins = pk_un[0, :, 0].cpu().numpy()
            
            res['unmarked']['pu'].append(pk_un[:, :, 1].cpu())
            res['unmarked']['pm'].append(pk_un[:, :, 2].cpu())
            res['unmarked']['px'].append(pk_un[:, :, 3].cpu())

            # MW
            for R in (RR):
                filt = gaussian_filter_fft(delta, R, BOX_SIZE)
                for s in ss:
                    for p in pp:
                        marked_mw = classical_mw_mark(delta, filt, s, p, BOX_SIZE)
                        mw_sq = marked_mw.squeeze(1)
                        pk_mw = pk_estimator.compute(d_sq, mw_sq, MAS=(MAS, MAS))
                        
                        res[f'mw_R{R}_s{s}_p{p}']['pu'].append(pk_mw[:, :, 1].cpu())
                        res[f'mw_R{R}_s{s}_p{p}']['pm'].append(pk_mw[:, :, 2].cpu())
                        res[f'mw_R{R}_s{s}_p{p}']['px'].append(pk_mw[:, :, 3].cpu())

    output_npz = {'k': k_bins}

    output_npz['P_dd_unmarked'] = torch.cat(res['unmarked']['pu'], dim=0).numpy()
    output_npz['P_mm_unmarked'] = torch.cat(res['unmarked']['pm'], dim=0).numpy()
    output_npz['P_xm_unmarked'] = torch.cat(res['unmarked']['px'], dim=0).numpy()
    
    for key in res.keys():
        if key.startswith("mw_"):
            output_npz[f'P_dd_{key}'] = torch.cat(res[key]['pu'], dim=0).numpy()
            output_npz[f'P_mm_{key}'] = torch.cat(res[key]['pm'], dim=0).numpy()
            output_npz[f'P_xm_{key}'] = torch.cat(res[key]['px'], dim=0).numpy()
    
    np.savez(out_path, **output_npz)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)    
    args = parser.parse_args()

    pk_estimator = PkEstimator(BoxSize=BOX_SIZE, device=DEVICE)

    fid_paths = get_paths(DATA_ROOT_BSQ, N_SIMS_BSQ, DATA_NAME_BSQ)
    compute_and_save(fid_paths, "bsq_span.npz", args, pk_estimator)

    lhc_paths = get_paths(DATA_ROOT_LHC, N_SIMS_LHC, DATA_NAME)
    compute_and_save(lhc_paths, "lhc_span.npz", args, pk_estimator)

    fid_paths = get_paths(DATA_ROOT_FID, N_SIMS_FID, DATA_NAME)
    compute_and_save(fid_paths, "fiducial_span.npz", args, pk_estimator)

    print("Scanning derivatives...")
    for param in PARAM_LABELS:
        for direction in ['p', 'm']:
            diff_paths = get_paths_diff(param, direction)
            if diff_paths:
                out_name = f"{param}_{direction}_span.npz"
                compute_and_save(diff_paths, out_name, args, pk_estimator)


if __name__ == "__main__":
    main()