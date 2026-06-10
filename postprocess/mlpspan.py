import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
import importlib



class PhysicsFeatureExtractor(nn.Module):
    def __init__(self, epsilon: float = 1e-9):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pk_raw: torch.Tensor) -> torch.Tensor:

        P_dd = torch.clamp(pk_raw[:, :, 0], min=self.epsilon)
        P_mm = torch.clamp(pk_raw[:, :, 1], min=self.epsilon)
        P_dm = pk_raw[:, :, 2]
        transfer = torch.log10(P_mm / P_dd)
        denom = torch.sqrt(P_dd * P_mm)
        r_coeff = torch.clamp(P_dm / denom, -1.0, 1.0)
        return torch.stack([torch.log10(P_dd), transfer, r_coeff], dim=2)

    
    
   
   
   
    

class LogSpaceMSELoss(nn.Module):
    def __init__(self, param_mean: torch.Tensor, param_std: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer('p_mean', param_mean)
        self.register_buffer('p_std', param_std)
        self.eps = eps

    def forward(self, pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        pred_phys = (pred_z * self.p_std) + self.p_mean
        target_phys = (target_z * self.p_std) + self.p_mean
        mse_per_param = torch.mean((pred_phys.float() - target_phys.float())**2, dim=0)
        return torch.sum(torch.log(mse_per_param + self.eps))





class PyramidMLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 base_dim: int,
                 num_layers: int = 5,
                 dropout: float = 0.1):
        
        super().__init__()
        n_hidden = num_layers - 1
        if n_hidden < 4 or n_hidden % 2 != 0:
            raise ValueError(
                f"Constraints: (num_layers-1) even and >=4; got {num_layers}"
            )
        half = n_hidden // 2
        up = [base_dim * (2 ** i) for i in range(half)]
        widths = up + up[::-1]  #[B, 2B, 4B, 4B, 2B, B] 
        
        layers = []
        in_d = input_dim
        for w in widths:
            layers.extend([
                nn.Linear(in_d, w),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = w
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PyramidCosmologyHead(nn.Module):
    def __init__(self, n_k: int, n_params: int, base_dim: int | None = None,  num_layers: int = 5):
        super().__init__()
        if base_dim is None:
            base_dim = n_k * 3
        self.feature_extractor = PhysicsFeatureExtractor()

        self.register_buffer('feat_mean', torch.zeros(1, n_k, 3))
        self.register_buffer('feat_std',  torch.ones(1, n_k, 3))
        self.register_buffer('feat_initialized', torch.tensor(0.0))

        self.mlp = PyramidMLP(
            input_dim=n_k * 3,
            output_dim=n_params,
            base_dim=base_dim,
            num_layers=num_layers,
        )

        nn.init.constant_(self.mlp.net[-1].bias, 0.0)
        nn.init.uniform_(self.mlp.net[-1].weight, -0.001, 0.001)

    @torch.no_grad()
    def fit_normalizer(self, X_train_raw: torch.Tensor, eps: float = 1e-8):
        features = self.feature_extractor(X_train_raw)
        self.feat_mean = features.mean(dim=0, keepdim=True)
        self.feat_std  = features.std(dim=0, keepdim=True) + eps
        self.feat_initialized.fill_(1.0)

    def forward(self, pk_raw):
        features = self.feature_extractor(pk_raw)
        features = (features - self.feat_mean) / self.feat_std
        B, Nk, C = features.shape
        pred = self.mlp(features.view(B, Nk * C))
        return pred, pred, pred



def train_eval_mlp_exact(
    X_raw, 
    Y_raw, 
    train_idx, 
    val_idx, 
    test_idx, 
    device,
    epochs=200, 
    patience=12, 
    lr=1e-3, 
    batch_size=128, 
    seed=1,
    lr_patience=6, 
    lr_gamma=0.5, 
    lr_min=1e-7,
):

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_train = torch.from_numpy(X_raw[train_idx]).float().to(device)
    X_val   = torch.from_numpy(X_raw[val_idx]).float().to(device)
    X_test  = torch.from_numpy(X_raw[test_idx]).float().to(device)

    Y_full = torch.from_numpy(Y_raw).float()
    y_mean = Y_full[train_idx].mean(dim=0, keepdim=True).to(device)
    y_std  = (Y_full[train_idx].std(dim=0, keepdim=True) + 1e-8).to(device)

    normalize_y = lambda y: (y.to(device) - y_mean) / y_std
    Y_train_n = normalize_y(Y_full[train_idx])
    Y_val_n   = normalize_y(Y_full[val_idx])
    Y_test    = Y_full[test_idx].to(device)

    n_k = X_train.shape[1]
    n_params = Y_train_n.shape[1]

    model = PyramidCosmologyHead(n_k=n_k, n_params=n_params, base_dim=None, num_layers=7).to(device)


    model.fit_normalizer(X_train)

    criterion = LogSpaceMSELoss(param_mean=y_mean, param_std=y_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_gamma,
        patience=lr_patience,
        min_lr=lr_min,
    )
    
    loader = DataLoader(TensorDataset(X_train, Y_train_n), batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            pred_total, _, _ = model(xb)
            loss = criterion(pred_total, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val, _, _ = model(X_val)
            val_loss = criterion(pred_val, Y_val_n).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_norm_test, _, _ = model(X_test)
        pred_phys_test = (pred_norm_test * y_std) + y_mean
        mse = torch.mean((pred_phys_test - Y_test)**2, dim=0).cpu().numpy()
        
    return mse








def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbase", type=Path, default="gam_pos_TAP0")
    parser.add_argument("--root", type=Path, default="../fin/fruns")

    parser.add_argument("--precomputed_cache_root", type=Path, default=Path("../precompo/mark_cache_span"))
    parser.add_argument("--params_lhc", type=Path, default=Path("../data/lhc_params.txt"))
    parser.add_argument("--config", type=str, default="config_fish")
    parser.add_argument("--learned_cache_file", type=Path, default=None)
    parser.add_argument("--out_json", type=Path, default=Path("mlp_lhc_mse_history"))
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=6742)
    parser.add_argument("--lr_patience", type=int, default=8)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--lr_min", type=float, default=1e-7)
    args = parser.parse_args()
    
    
    args.checkpoint = Path(f"../{args.root}/{args.cbase}/best_checkpoint.pt")
    
    ckp_dir = Path(args.checkpoint).parent
    args.config_name = ckp_dir

    cache_dir = Path(f"../results/{args.root}/{args.cbase}/regression_lhc_cache")
    args.learned_cache_file = cache_dir / "lhc_learned_mark.npz"

    analysis_cfg = importlib.import_module(args.config)
    args.params_lhc = analysis_cfg.PARAMS_LHC

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); np.random.seed(42)
    
    
    Y_full = np.loadtxt(args.params_lhc).astype(np.float32)
    n_sims = Y_full.shape[0]
    
    idx = np.arange(n_sims)
    np.random.shuffle(idx)
    train_idx, val_idx, test_idx = np.split(idx, [int(0.8*n_sims), int(0.9*n_sims)])

    precomp_lhc_file = args.precomputed_cache_root / "lhc_span.npz"
    if not precomp_lhc_file.exists():
        raise FileNotFoundError(f"Missing classical marks cache: {precomp_lhc_file}")
    
    pre = np.load(precomp_lhc_file)
    k_vec = pre['k']

    spectra_components = {}
    if not args.learned_cache_file.exists():
        raise FileNotFoundError(f"Missing learned mark cache: {args.learned_cache_file}")
    learned_spectra = np.load(args.learned_cache_file)
    spectra_components['learned'] = (
        learned_spectra['P_dd'][:n_sims], 
        learned_spectra['P_mm'][:n_sims], 
        learned_spectra['P_xm'][:n_sims]
    )
    
    RR = [10]
    ss = np.linspace(0,0.5,6)
    pp = [-2.5,-2,-1.5,-1,1,1.5,2,2.5]

    for R in RR:
        for p in pp[::-1]:
            for s in ss:
                mark_key = f"mw_R{R}_s{s}_p{p}"
                if f"P_mm_{mark_key}" in pre:
                    spectra_components[mark_key] = (
                        pre.get(f"P_dd_{mark_key}")[:n_sims],
                        pre[f"P_mm_{mark_key}"][:n_sims],
                        pre[f"P_xm_{mark_key}"][:n_sims]
                    )

    

    k_targets = [0.2]
    param_labels = ["Om", "Ob", "h", "ns", "s8"]
    seeds = [args.seed_base + i for i in range(args.n_runs)]
    history_mse = {
        m: {p: {"mean": [], "std": [], "raw": []} for p in param_labels}
        for m in spectra_components.keys()
    }


    for k_max in k_targets:
        print(f"k_max = {k_max}")
        k_mask = (k_vec >= 0.0) & (k_vec <= k_max)

        for mark, (p_dd_full, p_mm_full, p_xm_full) in spectra_components.items():

            p_dd = p_dd_full[:, k_mask]
            p_mm = p_mm_full[:, k_mask]
            p_xm = p_xm_full[:, k_mask]
            X_raw_numpy = np.stack([p_dd, p_mm, p_xm], axis=2)

            all_mse = []
            for seed in seeds:
                mse_vals = train_eval_mlp_exact(
                    X_raw=X_raw_numpy, Y_raw=Y_full,
                    train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                    device=device, seed=seed,
                    lr_patience=args.lr_patience,
                    lr_gamma=args.lr_gamma,
                    lr_min=args.lr_min,
                )
                all_mse.append(mse_vals)
            all_mse = np.stack(all_mse, axis=0) 
            mean_mse = all_mse.mean(axis=0)
            std_mse = all_mse.std(axis=0)

            mean_total = float(np.mean(mean_mse))
            mean_total_std = float(np.mean(std_mse))
            per_param_str = (
                f"[Om: {mean_mse[0]:.6f}±{std_mse[0]:.6f}, "
                f"s8: {mean_mse[-1]:.6f}±{std_mse[-1]:.6f}]"
            )

            if mark == 'learned':
                mses8 = mean_mse[-1]
                mseom = mean_mse[0]
                msetot = mean_total
                print(f"  {mark:>15}: Mean MSE = {mean_total:.6f}±{mean_total_std:.6f} | Per Param MSE = {per_param_str}")
            else:
                print(f" Ratios: Om Ratio = {mean_mse[0]/mseom:.3f} | s8 Ratio = {mean_mse[-1]/mses8:.3f}")
                outstring = f"  {mark:>15}: Mean MSE = {mean_total:.6f}±{mean_total_std:.6f} | Per Param MSE = {per_param_str}"
                print(outstring)

            for i, p in enumerate(param_labels):
                history_mse[mark][p]["mean"].append(float(mean_mse[i]))
                history_mse[mark][p]["std"].append(float(std_mse[i]))
                history_mse[mark][p]["raw"].append([float(x) for x in all_mse[:, i]])
                history_mse[mark][p]["mse_per_param"] = {param_labels[j]: float(mean_mse[j]) for j in range(len(param_labels))}

    outfile = f"{args.out_json}_{args.cbase}_kmax{int(k_max*1000)}.json"
    with open(outfile, 'w') as f:
        json.dump(history_mse, f, indent=4)
    print(f"Saved history to {outfile}")

if __name__ == "__main__":
    main()
