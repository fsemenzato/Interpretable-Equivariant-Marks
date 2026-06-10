import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast
from tqdm import tqdm
import warnings
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from getdist import plots
from getdist.gaussian_mixtures import GaussianND
import copy

from pk import PkEstimator

import importlib.util
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)



def compute_fisher_from_spectra(spectra_fid: dict, spectra_varied: dict, params_info: dict, k_mask: np.ndarray, summary_type: str) -> np.ndarray:

    def build_summary(spectra):
        if summary_type == 'unmarked':
            return spectra['P_dd'][:, k_mask]
        else:
            dd = spectra['P_dd'][:, k_mask]
            mm = spectra['P_mm'][:, k_mask]
            xm = spectra['P_xm'][:, k_mask]
            return np.concatenate([dd, mm, xm], axis=1)

    S_fid = build_summary(spectra_fid)
    if S_fid.shape[0] < 2: return np.zeros((len(params_info), len(params_info)))
    cov = np.cov(S_fid, rowvar=False)
    cov_inv = np.linalg.pinv(cov, rcond=1e-10, hermitian=True)

    derivs = []
    for param, info in params_info.items():
        S_p = build_summary(spectra_varied[f"{param}_p"])
        S_m = build_summary(spectra_varied[f"{param}_m"])
        deriv = (S_p.mean(axis=0) - S_m.mean(axis=0)) / (2 * info['step'])
        derivs.append(deriv)

    J = np.stack(derivs, axis=1) # Jacobian [D_summary, D_params]
    F = J.T @ cov_inv @ J
    return F




class MemoryMappedDataset(Dataset):
    def __init__(self, paths: List[Path], params: Optional[np.ndarray] = None):
        self.paths = [str(p) for p in paths]
        self.params = params

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        try:
            arr = np.load(self.paths[i], mmap_mode='r')
            arr = arr.astype(np.float32).copy()
            tensor = torch.from_numpy(arr).unsqueeze(0)

            if self.params is not None:
                p_tensor = torch.from_numpy(self.params[i].astype(np.float32))
                return (tensor, p_tensor)
            return (tensor,)

        except Exception as e:
            print(f"Error loading file {self.paths[i]}: {e}")
            raise e


def load_model_class(file_path: str, class_name: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Mark file not found: {file_path}")
    
    spec = importlib.util.spec_from_file_location("dynamic_mark_mod", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_mark_mod"] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {file_path}")
        
    return getattr(module, class_name)


def load_sh_model_from_checkpoint(checkpoint_path: Path, device: str) -> Tuple[nn.Module, Dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in checkpoint directory {checkpoint_path.parent}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    class ConfigObj:
        def __init__(self, d): self.__dict__.update(d)
    configobj = ConfigObj(config)

    mf = Path("./") / configobj._model_file
    refmark = load_model_class(mf, configobj._model_class)

    # Land on final 
    model = refmark(configobj).to(device)
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in checkpoint_data:
        model.load_state_dict(checkpoint_data['model'])
    elif 'model_state_dict' in checkpoint_data: 
        model_state_dict = {k.replace('0.', ''): v for k, v in checkpoint_data['model_state_dict'].items() if k.startswith('0.')}
        if not model_state_dict: 
            model_state_dict = checkpoint_data['model_state_dict']
        model.load_state_dict(model_state_dict, strict=False) 
    else:
        model.load_state_dict(checkpoint_data)

    model.eval()

    return model, config

def get_simulation_paths_direct(root: Path, data_name: str, n_sims: int) -> List[Path]:
    if not root.exists():
        warnings.warn(f"Root path not found: {root}!")
        return []
    return [root / str(idx) / data_name for idx in range(n_sims)]



def plot_fisher_triangle(fisher_list, param_names, param_labels=None, means_list=None, legend_labels=None):

    if param_labels is None:
        param_labels = param_names
        
    gaussians = []
    
    for i, F in enumerate(fisher_list):
        try:
            cov = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            raise ValueError(f"Fisher matrix at index {i} is singular and cannot be inverted.")
            
        if means_list is not None:
            mu = means_list[i]
        else:
            mu = np.zeros(len(param_names))
            
        gauss = GaussianND(mu, cov, names=param_names, labels=param_labels)
        gaussians.append(gauss)

    g = plots.get_subplot_plotter()
    
    g.triangle_plot(
        gaussians, 
        filled=True, 
        legend_labels=legend_labels,
        contour_colors=['#1f77b4', '#ff7f0e', '#2ca02c'], 
        contour_lws=[2, 2, 2]
    )
    
    return g


def run_fisher_analysis(learned_model: nn.Module, args: argparse.Namespace, analysis_cfg: Dict, train_cfg: Dict):
    device = torch.device(analysis_cfg.DEVICE)

    print(fr"MODEL: {args.cbase}")
    output_dir = Path(f"results/{args.config_name}/fisher_sh_integrated")
    learned_spectra_cache = output_dir / "learned_spectra_cache"
    learned_spectra_cache.mkdir(parents=True, exist_ok=True)

    pk_estimator = PkEstimator(BoxSize=train_cfg['box_size'], device=device)

    precomputed_spectra = {}
    sets_to_process = {"fiducial": None, **{f"{p}_{d}": None for p in analysis_cfg.PARAM_LABELS for d in ['p', 'm']}}

    for name in tqdm(sets_to_process, desc="Loading classical spectra"):
        cache_file = args.precomputed_cache_root / f"{name}_all_marks.npz"
        if not cache_file.exists():
            raise FileNotFoundError(f"Missing precomputed cache: {cache_file}. ")
        precomputed_spectra[name] = np.load(cache_file)


    k_vec = precomputed_spectra['fiducial']['k']
    k_mask = (k_vec >= args.k_min) & (k_vec <= args.k_max)
    data_name = f"df_m_{train_cfg['grid_dim']}_{train_cfg.get('mas', 'PCS')}_z=0.npy"

    path_map = {
        "fiducial": get_simulation_paths_direct(
            Path(analysis_cfg.DATA_ROOT_FID[0]),
            data_name,
            analysis_cfg.N_SIMS_FID
        ),
        **{f"{p}_{d}": get_simulation_paths_direct(
            Path(analysis_cfg.DATA_ROOT_DIFF[0]) / f"{analysis_cfg.FOLDER_NAME_MAP[p]}_{d}",
            data_name,
            analysis_cfg.N_SIMS_VARIED
        )
        for p in analysis_cfg.PARAM_LABELS for d in ['p', 'm']
        }
    }

    learned_spectra = {}
    with torch.set_grad_enabled(False):
        for name, paths in tqdm(path_map.items(), desc="Processing learned simulation sets"):
            cache_file = learned_spectra_cache / f"{name}_learned_mark.npz"
            if cache_file.exists() and not args.force_recompute:
                learned_spectra[name] = np.load(cache_file)
            else:
                if not paths:
                    warnings.warn(f"No simulation paths found for {name}, skipping.")
                    learned_spectra[name] = {}
                    continue

                loader = DataLoader(MemoryMappedDataset(paths), batch_size=args.batch_size, num_workers=args.num_workers)

                all_pu, all_pm, all_px = [], [], []
                for (delta_batch,) in tqdm(loader, desc=f"Computing learned spectra for {name}", leave=False):
                    delta = delta_batch.to(device) 

                    out = learned_model(delta)
                    marked_delta = out['final_field'] 
                    pk = pk_estimator.compute(
                        delta.squeeze(1),
                        marked_delta.squeeze(1),
                        MAS=(train_cfg.get('mas', 'PCS'), train_cfg.get('mas', 'PCS'))
                    )
                    all_pu.append(pk[:, :, 1].cpu().numpy())
                    all_pm.append(pk[:, :, 2].cpu().numpy())
                    all_px.append(pk[:, :, 3].cpu().numpy())

                if all_pu:
                    result = {
                        'k': pk[0, :, 0].cpu().numpy(),
                        'P_dd': np.concatenate(all_pu, axis=0),
                        'P_mm': np.concatenate(all_pm, axis=0),
                        'P_xm': np.concatenate(all_px, axis=0)
                    }
                    np.savez(cache_file, **result)
                    learned_spectra[name] = result

    fishers = {}
    param_info = analysis_cfg.PARAM_INFO

    spectra_fid_unmarked = {'P_dd': precomputed_spectra['fiducial']['P_dd_classical_mw']} 
    spectra_varied_unmarked = {
        k: {'P_dd': v['P_dd_classical_mw']}
        for k, v in precomputed_spectra.items() if k != 'fiducial'
    }
    fishers['unmarked'] = compute_fisher_from_spectra(
        spectra_fid_unmarked, spectra_varied_unmarked, param_info, k_mask, 'unmarked'
    )

    for mark in ['classical_mw', 'classical_htm']:
        if f'P_dd_{mark}' not in precomputed_spectra['fiducial']: continue
        spectra_fid = {
            'P_dd': precomputed_spectra['fiducial'][f'P_dd_{mark}'],
            'P_mm': precomputed_spectra['fiducial'][f'P_mm_{mark}'],
            'P_xm': precomputed_spectra['fiducial'][f'P_xm_{mark}']
        }
        spectra_varied = {
            k: {
                'P_dd': v[f'P_dd_{mark}'],
                'P_mm': v[f'P_mm_{mark}'],
                'P_xm': v[f'P_xm_{mark}']
            }
            for k, v in precomputed_spectra.items() if k != 'fiducial'
        }
        fishers[mark] = compute_fisher_from_spectra(
            spectra_fid, spectra_varied, param_info, k_mask, 'marked'
        )

    if learned_spectra and 'fiducial' in learned_spectra and learned_spectra['fiducial']:
        fishers['learned'] = compute_fisher_from_spectra(
            learned_spectra['fiducial'],
            {k: v for k, v in learned_spectra.items() if k != 'fiducial' and v},
            param_info, k_mask, 'marked'
        )



    sigmas = {name: np.sqrt(np.diag(np.linalg.inv(F))) for name, F in fishers.items() if np.linalg.det(F) != 0}
    sigmas_fix = {name: np.sqrt(np.diag(np.linalg.inv(F[[0, -1]][:, [0, -1]]))) for name, F in fishers.items() if np.linalg.det(F) != 0}

    header = f"{ 'Parameter':<10} |";
    marks_to_show = [m for m in ['unmarked', 'learned', 'classical_mw', 'classical_htm'] if m in sigmas]
    for mark in marks_to_show: header += f" { 'σ ('+mark+')':<15} |";
    # print(header)
    # print("-" * len(header))

    # for i, p in enumerate(param_info.keys()):
    #     row = f"{ p:<10} |";
    #     for mark in marks_to_show:
    #         val = sigmas.get(mark, [np.nan] * len(param_info.keys()))[i]
    #         row += f" {val:15.5f} |";
    #     print(row)

    print("\n--- Improvement Factor (σ_unmarked / σ_marked) ---")
    header = f"{ 'Parameter':<10} |";
    marks_to_show_imp = [m for m in marks_to_show if m != 'unmarked']
    for mark in marks_to_show_imp: header += f" {mark:<15} |";
    print(header); print("-" * len(header))
    for i, p in enumerate(param_info.keys()):
        row = f"{ p:<10} |";

        unmarked_sigma = sigmas['unmarked'][i] if 'unmarked' in sigmas else np.nan
        row = f"{ p:<10} |";
        for mark in marks_to_show_imp:
            marked_sigma = sigmas[mark][i] if mark in sigmas else np.nan
            if not np.isnan(unmarked_sigma) and not np.isnan(marked_sigma) and marked_sigma > 1e-12:
                imp = unmarked_sigma / marked_sigma
                row += f" {imp:15.5f}x |";
            else:
                row += f" { 'N/A':>15} |";
        print(row)
    print(r"\n")
    
    




def regression_precomp(model, args, analysis_cfg, train_cfg):
    print("LHC PRECOMP")
    device = torch.device(analysis_cfg.DEVICE)
    
    precomp_lhc_file = args.precomputed_cache_root / "lhc_all_marks.npz"
    precomp_data = np.load(precomp_lhc_file)
    k_vec = precomp_data['k']

    cache_dir = Path(f"results/{args.config_name}/regression_lhc_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    pk_estimator = PkEstimator(train_cfg['box_size'], device=device)

    data_name = f"df_m_{train_cfg['grid_dim']}_{train_cfg.get('mas', 'PCS')}_z=0.npy"
    lhc_paths = get_simulation_paths_direct(Path(analysis_cfg.DATA_ROOT_LHC), data_name, analysis_cfg.N_SIMS_LHC)

    spectra_components = {}

    learned_cache_file = cache_dir / "lhc_learned_mark.npz"
    if learned_cache_file.exists() and not args.force_recompute:
        print("Learned mark spectra for LHC set precomputed")
    else:
        loader = DataLoader(MemoryMappedDataset(lhc_paths), batch_size=args.batch_size, num_workers=args.num_workers)
        all_pu, all_pm, all_px = [], [], []
        with torch.no_grad():
            for (delta_batch,) in tqdm(loader, desc="Computing learned LHC spectra"):
                delta = delta_batch.to(device)
                marked_delta = model(delta)['final_field']
                pk = pk_estimator.compute(
                    delta.squeeze(1), marked_delta,
                    MAS=(train_cfg.get('mas', 'PCS'), train_cfg.get('mas', 'PCS'))
                )
                all_pu.append(pk[:, :, 1].cpu().numpy())
                all_pm.append(pk[:, :, 2].cpu().numpy())
                all_px.append(pk[:, :, 3].cpu().numpy())

        P_dd_l = np.concatenate(all_pu, axis=0)
        P_mm_l = np.concatenate(all_pm, axis=0)
        P_xm_l = np.concatenate(all_px, axis=0)
        np.savez(learned_cache_file, k=k_vec, P_dd=P_dd_l, P_mm=P_mm_l, P_xm=P_xm_l)
        spectra_components['learned'] = (P_dd_l, P_mm_l, P_xm_l)
        print("Learned mark spectra for LHC set computed and cached.")
    
        
    






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbase", type=Path, default="f_aa_spectral")
    parser.add_argument("--root", type=Path, default="runs")
    parser.add_argument("--precomputed_cache_root", type=Path, default=Path("/u/fsemenzato/precomps/mark_cache_allmarks"))
    parser.add_argument("--config", type=str, default="postprocess.config_fish")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action='store_true')
    parser.add_argument("--k_min", type=float, default=0.0)
    parser.add_argument("--k_max", type=float, default=0.3)
    parser.add_argument("--force_recompute", action='store_true')
    parser.add_argument("--reg_epochs", type=int, default=2000)
    parser.add_argument("--reg_patience", type=int, default=100)
    parser.add_argument("--reg_lr", type=float, default=1e-3)
    parser.add_argument("--reg_batch_size", type=int, default=128)
    parser.add_argument("--corner", action='store_true')
    parser.add_argument("--infofield", action='store_true')
    parser.add_argument("--task", type=str, default="fisher", choices=['fisher', 'regression'])
    
    args = parser.parse_args()
    args.checkpoint = Path(f"./{args.root}/{args.cbase}/best_checkpoint.pt")
    
    
    analysis_cfg = importlib.import_module(args.config)
    ckp_dir = Path(args.checkpoint).parent
    args.config_name = ckp_dir
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    learned_model, train_cfg = load_sh_model_from_checkpoint(Path(args.checkpoint), device)

    if args.task == "fisher":
        run_fisher_analysis(learned_model, args, analysis_cfg, train_cfg)
    elif args.task == "regression":
        regression_precomp(learned_model, args, analysis_cfg, train_cfg)

