from __future__ import annotations
import argparse
import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from postprocess import fisher_common as FC
from postprocess import regression_cache as RC

PARAMS = list(RC.PARAM_LABELS)
SPLIT_SEED = 42
DEFAULT_SEED_BASE = 6742


class PhysicsFeatureExtractor(nn.Module):
    def __init__(self, epsilon: float = 1e-09):
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


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ResMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU()
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.head(self.blocks(self.input_proj(x)))


class _CosmologyHeadBase(nn.Module):
    def _init_norm(self, n_k: int):
        self.feature_extractor = PhysicsFeatureExtractor()
        self.register_buffer("feat_mean", torch.zeros(1, n_k, 3))
        self.register_buffer("feat_std", torch.ones(1, n_k, 3))
        self.register_buffer("feat_initialized", torch.tensor(0.0))

    @torch.no_grad()
    def fit_normalizer(self, X_train_raw: torch.Tensor, eps: float = 1e-08):
        features = self.feature_extractor(X_train_raw)
        self.feat_mean = features.mean(dim=0, keepdim=True)
        self.feat_std = features.std(dim=0, keepdim=True) + eps
        self.feat_initialized.fill_(1.0)

    def _features(self, pk_raw):
        features = self.feature_extractor(pk_raw)
        features = (features - self.feat_mean) / self.feat_std
        B, Nk, C = features.shape
        return features.view(B, Nk * C)


class ResidualCosmologyHead(_CosmologyHeadBase):
    def __init__(
        self,
        n_k: int,
        n_params: int,
        hidden_mult: int | None = None,
        num_blocks: int = 3,
    ):
        super().__init__()
        hidden_dim = n_k * hidden_mult
        self._init_norm(n_k)
        self.residual_net = ResMLP(
            input_dim=n_k * 3,
            output_dim=n_params,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )
        nn.init.constant_(self.residual_net.head.bias, 0.0)
        nn.init.uniform_(self.residual_net.head.weight, -0.001, 0.001)

    def forward(self, pk_raw):
        pred = self.residual_net(self._features(pk_raw))
        return (pred, pred, pred)


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StandardCosmologyHead(_CosmologyHeadBase):
    def __init__(
        self, n_k: int, n_params: int, hidden_mult: int = 6, num_layers: int = 6
    ):
        super().__init__()
        self._init_norm(n_k)
        self.mlp = SimpleMLP(
            input_dim=n_k * 3,
            output_dim=n_params,
            hidden_dim=n_k * hidden_mult,
            num_layers=num_layers,
        )
        nn.init.constant_(self.mlp.net[-1].bias, 0.0)
        nn.init.uniform_(self.mlp.net[-1].weight, -0.001, 0.001)

    def forward(self, pk_raw):
        pred = self.mlp(self._features(pk_raw))
        return (pred, pred, pred)


class LogSpaceMSELoss(nn.Module):
    def __init__(
        self, param_mean: torch.Tensor, param_std: torch.Tensor, eps: float = 1e-06
    ):
        super().__init__()
        self.register_buffer("p_mean", param_mean)
        self.register_buffer("p_std", param_std)
        self.eps = eps

    def forward(self, pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        pred_phys = pred_z * self.p_std + self.p_mean
        target_phys = target_z * self.p_std + self.p_mean
        mse_per_param = torch.mean(
            (pred_phys.float() - target_phys.float()) ** 2, dim=0
        )
        return torch.sum(torch.log(mse_per_param + self.eps))


class PyramidMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_dim: int,
        num_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        n_hidden = num_layers - 1
        if n_hidden < 4 or n_hidden % 2 != 0:
            raise ValueError(
                f"num_layers must satisfy (num_layers-1) even and >=4; got {num_layers}"
            )
        widths = [
            base_dim,
            base_dim * 2,
            base_dim * 4,
            base_dim * 4,
            base_dim * 2,
            base_dim,
        ]
        layers, in_d = ([], input_dim)
        for w in widths:
            layers.extend([nn.Linear(in_d, w), nn.GELU(), nn.Dropout(dropout)])
            in_d = w
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PyramidCosmologyHead(_CosmologyHeadBase):
    def __init__(
        self, n_k: int, n_params: int, base_dim: int | None = None, num_layers: int = 5
    ):
        super().__init__()
        if base_dim is None:
            base_dim = n_k * 3
        self._init_norm(n_k)
        self.mlp = PyramidMLP(
            input_dim=n_k * 3,
            output_dim=n_params,
            base_dim=base_dim,
            num_layers=num_layers,
        )
        nn.init.constant_(self.mlp.net[-1].bias, 0.0)
        nn.init.uniform_(self.mlp.net[-1].weight, -0.001, 0.001)

    def forward(self, pk_raw):
        pred = self.mlp(self._features(pk_raw))
        return (pred, pred, pred)


class LinearCosmologyHead(nn.Module):
    def __init__(self, n_k: int, n_params: int):
        super().__init__()
        self.feature_extractor = PhysicsFeatureExtractor()
        self.register_buffer("feat_mean", torch.zeros(1, n_k, 3))
        self.register_buffer("feat_std", torch.ones(1, n_k, 3))
        self.lin = nn.Linear(n_k * 3, n_params)
        nn.init.constant_(self.lin.bias, 0.0)
        nn.init.uniform_(self.lin.weight, -0.001, 0.001)

    @torch.no_grad()
    def fit_normalizer(self, X_train_raw, eps: float = 1e-08):
        f = self.feature_extractor(X_train_raw)
        self.feat_mean = f.mean(dim=0, keepdim=True)
        self.feat_std = f.std(dim=0, keepdim=True) + eps

    def forward(self, pk_raw):
        f = (self.feature_extractor(pk_raw) - self.feat_mean) / self.feat_std
        B, Nk, C = f.shape
        p = self.lin(f.view(B, Nk * C))
        return (p, p, p)


class StandardisedMSELoss(nn.Module):
    def forward(self, pred_z, target_z):
        return torch.mean((pred_z.float() - target_z.float()) ** 2)


class PddOnlyPyramidHead(nn.Module):
    def __init__(
        self,
        n_k: int,
        n_params: int,
        base_dim: int | None = None,
        num_layers: int = 5,
        eps: float = 1e-09,
    ):
        super().__init__()
        if base_dim is None:
            base_dim = n_k * 3
        self.eps = eps
        self.register_buffer("feat_mean", torch.zeros(1, n_k))
        self.register_buffer("feat_std", torch.ones(1, n_k))
        self.mlp = PyramidMLP(
            input_dim=n_k, output_dim=n_params, base_dim=base_dim, num_layers=num_layers
        )
        nn.init.constant_(self.mlp.net[-1].bias, 0.0)
        nn.init.uniform_(self.mlp.net[-1].weight, -0.001, 0.001)

    def _feat(self, pk_raw):
        return torch.log10(torch.clamp(pk_raw[:, :, 0], min=self.eps))

    @torch.no_grad()
    def fit_normalizer(self, X_train_raw, eps: float = 1e-08):
        f = self._feat(X_train_raw)
        self.feat_mean = f.mean(dim=0, keepdim=True)
        self.feat_std = f.std(dim=0, keepdim=True) + eps

    def forward(self, pk_raw):
        p = self.mlp((self._feat(pk_raw) - self.feat_mean) / self.feat_std)
        return (p, p, p)


VARIANTS = ["pyramid", "linear", "small", "resmlp", "pyramid_wide", "pyramid_mse"]
LC_FRACS = [0.5, 0.25]
CLASSICAL_PANEL = [
    (0.1, -2.5),
    (0.5, -2.5),
    (0.0, -2.0),
    (0.0, -1.5),
    (0.2, -1.5),
    (0.1, -1.5),
    (0.3, -1.5),
    (0.2, -2.0),
]


def make_head(variant: str, n_k: int, n_params: int) -> nn.Module:
    if variant in ("pyramid", "pyramid_mse"):
        return PyramidCosmologyHead(
            n_k=n_k, n_params=n_params, base_dim=None, num_layers=5
        )
    if variant == "pyramid_wide":
        return PyramidCosmologyHead(
            n_k=n_k, n_params=n_params, base_dim=6 * n_k, num_layers=5
        )
    if variant == "small":
        return StandardCosmologyHead(
            n_k=n_k, n_params=n_params, hidden_mult=2, num_layers=3
        )
    if variant == "resmlp":
        return ResidualCosmologyHead(
            n_k=n_k, n_params=n_params, hidden_mult=2, num_blocks=3
        )
    if variant == "linear":
        return LinearCosmologyHead(n_k=n_k, n_params=n_params)
    raise ValueError(variant)


def default_head(n_k: int, n_params: int) -> nn.Module:
    return PyramidCosmologyHead(n_k=n_k, n_params=n_params, base_dim=None, num_layers=5)


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_eval_mlp_exact(
    X_raw,
    Y_raw,
    train_idx,
    val_idx,
    test_idx,
    device,
    epochs=200,
    patience=12,
    lr=0.001,
    batch_size=128,
    seed=1,
    lr_patience=6,
    lr_gamma=0.5,
    lr_min=1e-07,
    head_factory=None,
    return_val: bool = False,
):
    seed_all(seed)
    X_train = torch.from_numpy(X_raw[train_idx]).float().to(device)
    X_val = torch.from_numpy(X_raw[val_idx]).float().to(device)
    X_test = torch.from_numpy(X_raw[test_idx]).float().to(device)
    Y_full = torch.from_numpy(Y_raw).float()
    y_mean = Y_full[train_idx].mean(dim=0, keepdim=True).to(device)
    y_std = (Y_full[train_idx].std(dim=0, keepdim=True) + 1e-08).to(device)
    normalize_y = lambda y: (y.to(device) - y_mean) / y_std
    Y_train_n = normalize_y(Y_full[train_idx])
    Y_val_n = normalize_y(Y_full[val_idx])
    Y_val = Y_full[val_idx].to(device)
    Y_test = Y_full[test_idx].to(device)
    n_k, n_params = (X_train.shape[1], Y_train_n.shape[1])
    model = (head_factory or default_head)(n_k, n_params).to(device)
    model.fit_normalizer(X_train)
    criterion = LogSpaceMSELoss(param_mean=y_mean, param_std=y_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_gamma, patience=lr_patience, min_lr=lr_min
    )
    loader = DataLoader(
        TensorDataset(X_train, Y_train_n), batch_size=batch_size, shuffle=True
    )
    best_val_loss, epochs_no_improve = (float("inf"), 0)
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
        pred_phys_test = model(X_test)[0] * y_std + y_mean
        mse = torch.mean((pred_phys_test - Y_test) ** 2, dim=0).cpu().numpy()
        if return_val:
            pred_phys_val = model(X_val)[0] * y_std + y_mean
            mse_val = torch.mean((pred_phys_val - Y_val) ** 2, dim=0).cpu().numpy()
            return (mse, mse_val)
    return mse


def make_split(
    n: int, seed: int = SPLIT_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    tr, va, te = np.split(idx, [int(0.8 * n), int(0.9 * n)])
    return (tr, va, te)


def mlp_dir(cache_root: Path, dataset: str) -> Path:
    d = Path(cache_root) / dataset / "mlp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def classical_cache_path(cache_root: Path, dataset: str) -> Path:
    return Path(cache_root) / dataset / "classical_panel.npz"


def learned_cache_path(cache_root: Path, dataset: str, tag: str) -> Path:
    return Path(cache_root) / dataset / f"learned_{tag}.npz"


def classical_mark_names(cl: dict) -> List[str]:
    from preprocess.classical_baseline import parse_mark_key

    keys = [k[len("Pmm__") :] for k in cl if k.startswith("Pmm__")]
    return sorted(
        keys,
        key=lambda m: (
            parse_mark_key(m)[2],
            parse_mark_key(m)[1],
            parse_mark_key(m)[0],
        ),
    )


def load_suite(
    dataset: str,
    tags: Sequence[str],
    cache_root: Path,
    params_file: Optional[Path] = None,
    with_classical: bool = True,
):
    cl = RC.load_regression_cache(classical_cache_path(cache_root, dataset))
    ids = np.asarray(cl["sim_ids"])
    Y = np.asarray(cl["params"], dtype=np.float32)
    pf = Path(params_file) if params_file else Path(cl["meta"]["params_file"])
    master = np.loadtxt(pf).astype(np.float32)
    if not np.array_equal(master[ids], Y):
        raise RuntimeError(f"classical cache params != {pf.name}[sim_ids]")
    RC.alignment_report(cl["k"], cl["Pdd"], Y, f"{dataset} classical cache")
    spectra: Dict[str, tuple] = {}
    learned_meta = {}
    for tag in tags:
        f = learned_cache_path(cache_root, dataset, tag)
        if not f.exists():
            raise FileNotFoundError(
                f"missing learned cache {f} (run postprocess.regression_cache first)"
            )
        lc = RC.load_regression_cache(f)
        RC.check_pair(cl, lc, name=f"{dataset} learned {tag}")
        spectra[f"learned:{tag}"] = (lc["P_dd"], lc["P_mm"], lc["P_dm"])
        learned_meta[tag] = {
            "file": str(f),
            "ckpt_sha": lc["meta"].get("ckpt_sha"),
            "run": lc["meta"].get("run"),
            "overrides": lc["meta"].get("overrides"),
        }
    names = classical_mark_names(cl)
    if with_classical:
        for m in names:
            spectra[m] = (cl["Pdd"], cl[f"Pmm__{m}"], cl[f"Pdm__{m}"])
    info = {
        "classical_marks": names,
        "learned": learned_meta,
        "params_file": str(pf),
        "classical_meta": cl["meta"],
    }
    return (np.asarray(cl["k"], float), ids, Y, spectra, info, cl)


def kmax_mask(k: np.ndarray, kmax: float) -> np.ndarray:
    return FC.kmask(k, 0.0, kmax)


def stack_X(spec: tuple, mask: np.ndarray) -> np.ndarray:
    p_dd, p_mm, p_dm = spec
    return np.stack([p_dd[:, mask], p_mm[:, mask], p_dm[:, mask]], axis=2)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


RESUME_KEYS = (
    "dataset",
    "n_sims",
    "k_targets",
    "seeds",
    "epochs",
    "patience",
    "lr_patience",
    "n_test",
)


def run_long5(args) -> List[Path]:
    device = _device()
    k_vec, ids, Y, spectra, info, _ = load_suite(
        args.dataset, args.runs, args.cache_root, args.params_file
    )
    n = len(Y)
    tr, va, te = make_split(n)
    seeds = [args.seed_base + i for i in range(args.n_runs)]
    marks = list(spectra)
    if args.max_marks is not None:
        learned = [m for m in marks if m.startswith("learned:")]
        marks = (
            learned
            + [m for m in marks if not m.startswith("learned:")][: args.max_marks]
        )
    print(
        f"{args.dataset}: {n} sims, ids {ids.min()}..{ids.max()}, split {len(tr)}/{len(va)}/{len(te)}, {len(marks)} summaries, seeds {seeds}",
        flush=True,
    )
    outs = []
    for k_max in args.k_max:
        out = (
            Path(args.out)
            if args.out and len(args.k_max) == 1
            else mlp_dir(args.cache_root, args.dataset) / f"long5_kmax{k_max:g}.json"
        )
        mask = kmax_mask(k_vec, k_max)
        meta = {
            "dataset": args.dataset,
            "classical_cache": str(classical_cache_path(args.cache_root, args.dataset)),
            "params_file": info["params_file"],
            "param_labels": PARAMS,
            "param_var_test": {p: float(Y[te, i].var()) for i, p in enumerate(PARAMS)},
            "param_var_val": {p: float(Y[va, i].var()) for i, p in enumerate(PARAMS)},
            "param_var_all": {p: float(Y[:, i].var()) for i, p in enumerate(PARAMS)},
            "learned_caches": info["learned"],
            "n_sims": int(n),
            "sim_id_min": int(ids.min()),
            "sim_id_max": int(ids.max()),
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_test": int(len(te)),
            "k_targets": [k_max],
            "n_k": int(mask.sum()),
            "seeds": seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr_patience": args.lr_patience,
            "head": "PyramidCosmologyHead(base 3 n_k)",
            "selection": "test MSE (raw) and validation MSE (val_raw) of the best-validation weights",
        }
        hist: Dict[str, dict] = {"_meta": meta}
        if out.exists() and (not args.force):
            old = json.loads(out.read_text())
            if all((old.get("_meta", {}).get(k) == meta.get(k) for k in RESUME_KEYS)):
                hist = {**old, "_meta": meta}
                print(
                    f"  resuming {out}: {len(hist) - 1} summaries already done",
                    flush=True,
                )
            else:
                print(f"  {out} has different settings -> recomputing", flush=True)
        for mark in marks:
            if mark in hist:
                continue
            X = stack_X(spectra[mark], mask)
            t0 = time.time()
            te_all, va_all = ([], [])
            for seed in seeds:
                m_te, m_va = train_eval_mlp_exact(
                    X,
                    Y,
                    tr,
                    va,
                    te,
                    device,
                    seed=seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    lr_patience=args.lr_patience,
                    return_val=True,
                )
                te_all.append(m_te)
                va_all.append(m_va)
            te_all, va_all = (np.stack(te_all), np.stack(va_all))
            hist[mark] = {
                p: {
                    "mean": [float(te_all[:, i].mean())],
                    "std": [float(te_all[:, i].std())],
                    "raw": [te_all[:, i].tolist()],
                    "val_raw": [va_all[:, i].tolist()],
                }
                for i, p in enumerate(PARAMS)
            }
            nm = te_all.mean(0)
            print(
                f"  [{args.dataset} k<={k_max:g}] {mark:<42} test MSE Om {nm[0]:.3e} s8 {nm[4]:.3e}  ({time.time() - t0:.0f}s)",
                flush=True,
            )
            _atomic_json(out, hist)
        _atomic_json(out, hist)
        print(f"saved {out}", flush=True)
        outs.append(out)
    return outs


def run_pdd_only(args) -> List[Path]:
    device = _device()
    k_vec, ids, Y, _, info, cl = load_suite(
        args.dataset, [], args.cache_root, args.params_file, with_classical=False
    )
    tr, va, te = make_split(len(Y))
    seeds = [args.seed_base + i for i in range(args.n_runs)]
    outs = []
    for k_max in args.k_max:
        out = mlp_dir(args.cache_root, args.dataset) / f"pdd_only_kmax{k_max:g}.json"
        mask = kmax_mask(k_vec, k_max)
        P = np.asarray(cl["Pdd"])[:, mask]
        X = np.stack([P, P, P], axis=2)
        te_all, va_all = ([], [])
        t0 = time.time()
        for seed in seeds:
            m_te, m_va = train_eval_mlp_exact(
                X,
                Y,
                tr,
                va,
                te,
                device,
                seed=seed,
                epochs=args.epochs,
                patience=args.patience,
                lr_patience=args.lr_patience,
                head_factory=lambda nk, npar: PddOnlyPyramidHead(nk, npar),
                return_val=True,
            )
            te_all.append(m_te)
            va_all.append(m_va)
        te_all, va_all = (np.stack(te_all), np.stack(va_all))
        res = {
            "_meta": {
                "dataset": args.dataset,
                "k_max": k_max,
                "seeds": seeds,
                "epochs": args.epochs,
                "patience": args.patience,
                "lr_patience": args.lr_patience,
                "n_test": int(len(te)),
                "n_k": int(mask.sum()),
                "head": "PddOnlyPyramidHead(base 3 n_k)",
                "param_var_test": {
                    p: float(Y[te, i].var()) for i, p in enumerate(PARAMS)
                },
            },
            "pdd_only": {
                p: {"raw": te_all[:, i].tolist(), "val_raw": va_all[:, i].tolist()}
                for i, p in enumerate(PARAMS)
            },
        }
        _atomic_json(out, res)
        nm = te_all.mean(0) / Y[te].var(0)
        print(
            f"{args.dataset} k<={k_max:g}: P_dd-only test nMSE "
            + " ".join((f"{p} {nm[i]:.4f}" for i, p in enumerate(PARAMS)))
            + f"  ({time.time() - t0:.0f}s) -> {out}",
            flush=True,
        )
        outs.append(out)
    return outs


def train_one(
    X,
    Y,
    tr,
    va,
    device,
    variant,
    seed,
    epochs,
    patience,
    lr=0.001,
    batch_size=128,
    lr_patience=8,
    lr_gamma=0.5,
    lr_min=1e-07,
):
    seed_all(seed)
    Xt = torch.from_numpy(X).float().to(device)
    Yt = torch.from_numpy(Y).float().to(device)
    y_mean = Yt[tr].mean(dim=0, keepdim=True)
    y_std = Yt[tr].std(dim=0, keepdim=True) + 1e-08
    Ytr = (Yt[tr] - y_mean) / y_std
    Yva = (Yt[va] - y_mean) / y_std
    model = make_head(variant, X.shape[1], Y.shape[1]).to(device)
    model.fit_normalizer(Xt[tr])
    crit = (
        StandardisedMSELoss()
        if variant == "pyramid_mse"
        else LogSpaceMSELoss(y_mean, y_std)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_gamma, patience=lr_patience, min_lr=lr_min
    )
    loader = DataLoader(TensorDataset(Xt[tr], Ytr), batch_size=batch_size, shuffle=True)
    best, bad, best_ep, ep = (float("inf"), 0, 0, 0)
    best_state = copy.deepcopy(model.state_dict())
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            loss = crit(model(xb)[0], yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xt[va])[0], Yva).item()
        sch.step(vl)
        if vl < best:
            best, bad, best_ep = (vl, 0, ep)
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = (model(Xt)[0] * y_std + y_mean).cpu().numpy()
    n_params = sum((p.numel() for p in model.parameters()))
    return (pred, best_ep, ep, n_params)


def make_folds(n, n_folds, val_frac=0.125, seed=42):
    perm = np.random.RandomState(seed).permutation(n)
    tests = np.array_split(perm, n_folds)
    folds = []
    for f, te in enumerate(tests):
        rest = np.concatenate([t for g, t in enumerate(tests) if g != f])
        rest = np.random.RandomState(1000 + f).permutation(rest)
        n_val = int(round(val_frac * len(rest)))
        folds.append((rest[n_val:], rest[:n_val], te))
    return folds


def run_cv(args) -> List[Path]:
    from preprocess.classical_baseline import mark_key

    device = _device()
    k_vec, ids, Y, spectra_all, info, _ = load_suite(
        args.dataset, args.runs, args.cache_root, args.params_file
    )
    panel = {
        f"classical:s{s:g}_p{p:g}": spectra_all[mark_key(10, s, p)]
        for s, p in CLASSICAL_PANEL
    }
    spectra = {m: v for m, v in spectra_all.items() if m.startswith("learned:")}
    spectra.update(panel)
    outs = []
    for k_max in args.k_max:
        out = (
            Path(args.out)
            if args.out and len(args.k_max) == 1
            else mlp_dir(args.cache_root, args.dataset) / f"cv_kmax{k_max:g}.pkl"
        )
        marks = list(spectra)[: args.max_marks] if args.max_marks else list(spectra)
        mask = kmax_mask(k_vec, k_max)
        folds = make_folds(len(Y), args.n_folds)
        fold_id = np.empty(len(Y), dtype=int)
        for f, (_, _, te) in enumerate(folds):
            fold_id[te] = f
        if out.exists():
            res = pickle.load(open(out, "rb"))
            if not (
                np.array_equal(res["ids"], ids)
                and np.array_equal(res["fold_id"], fold_id)
            ):
                raise RuntimeError(
                    f"{out} was made with different sims or folds; move it away"
                )
        else:
            res = {
                "dataset": args.dataset,
                "k_max": k_max,
                "ids": ids,
                "Y": Y,
                "fold_id": fold_id,
                "n_k": int(mask.sum()),
                "param_labels": PARAMS,
                "entries": {},
                "settings": dict(
                    epochs=args.epochs,
                    patience=args.patience,
                    n_folds=args.n_folds,
                    seed_base=args.seed_base,
                    val_frac=0.125,
                    lr_patience=8,
                ),
            }
        print(
            f"{args.dataset} k_max={k_max:g}: {len(Y)} sims, n_k={mask.sum()}, {len(marks)} marks, folds {[len(te) for _, _, te in folds]}, out {out}",
            flush=True,
        )
        jobs = [(v, 1.0) for v in args.variants] + [
            ("pyramid", fr) for fr in args.lc_fracs
        ]
        for variant, frac in jobs:
            for mark in marks:
                key = f"{variant}|{frac:g}|{mark}"
                if key in res["entries"]:
                    continue
                X = stack_X(spectra[mark], mask)
                oof = np.full_like(Y, np.nan)
                tr_mse, va_mse, te_mse, best_ep, stop_ep = ([], [], [], [], [])
                t0 = time.time()
                for f, (tr, va, te) in enumerate(folds):
                    tr_use = tr[: max(int(round(frac * len(tr))), 32)]
                    pred, be, se, n_par = train_one(
                        X,
                        Y,
                        tr_use,
                        va,
                        device,
                        variant,
                        args.seed_base + f,
                        args.epochs,
                        args.patience,
                    )
                    oof[te] = pred[te]
                    tr_mse.append(((pred[tr_use] - Y[tr_use]) ** 2).mean(0))
                    va_mse.append(((pred[va] - Y[va]) ** 2).mean(0))
                    te_mse.append(((pred[te] - Y[te]) ** 2).mean(0))
                    best_ep.append(be)
                    stop_ep.append(se)
                res["entries"][key] = dict(
                    variant=variant,
                    frac=frac,
                    mark=mark,
                    oof=oof,
                    train_mse=np.array(tr_mse),
                    val_mse=np.array(va_mse),
                    test_mse=np.array(te_mse),
                    best_epoch=np.array(best_ep),
                    stop_epoch=np.array(stop_ep),
                    n_params=n_par,
                )
                tmp = out.with_suffix(".pkl.tmp")
                pickle.dump(res, open(tmp, "wb"))
                os.replace(tmp, out)
                nm = ((oof - Y) ** 2).mean(0) / Y.var(0)
                print(
                    f"  {key:<56} nMSE Om {nm[0]:.4f} s8 {nm[4]:.4f} ns {nm[3]:.4f} | best ep {int(np.median(best_ep))} | {n_par} params | {time.time() - t0:.0f}s",
                    flush=True,
                )
        print(f"done: {len(res['entries'])} entries in {out}", flush=True)
        outs.append(out)
    return outs


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, runs: bool):
        p.add_argument("--dataset", choices=sorted(RC.DATASETS), required=True)
        if runs:
            p.add_argument(
                "--runs",
                nargs="+",
                required=True,
                help="run tags (caches learned_<tag>.npz)",
            )
        p.add_argument("--k_max", type=float, nargs="+", default=[0.2, 0.3])
        p.add_argument("--cache_root", type=Path, default=RC.DEFAULT_OUT_ROOT)
        p.add_argument(
            "--params_file",
            type=Path,
            default=None,
            help="override the parameter file in the cache meta",
        )
        p.add_argument("--seed_base", type=int, default=DEFAULT_SEED_BASE)
        p.add_argument("--epochs", type=int, default=1000)
        p.add_argument("--patience", type=int, default=60)
        p.add_argument(
            "--out", type=Path, default=None, help="output file (single k_max only)"
        )
        p.add_argument("--max_marks", type=int, default=None, help="smoke tests only")

    p1 = sub.add_parser("long5", help="learned runs + 48 classical marks, 5 seeds")
    common(p1, True)
    p1.add_argument("--n_runs", type=int, default=5)
    p1.add_argument("--lr_patience", type=int, default=8)
    p1.add_argument("--force", action="store_true")
    p2 = sub.add_parser("pdd_only", help="P(k)-only baseline")
    common(p2, False)
    p2.add_argument("--n_runs", type=int, default=5)
    p2.add_argument("--lr_patience", type=int, default=8)
    p3 = sub.add_parser("cv", help="5-fold OOF capacity study")
    common(p3, True)
    p3.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p3.add_argument("--lc_fracs", nargs="*", type=float, default=LC_FRACS)
    p3.add_argument("--n_folds", type=int, default=5)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    {"long5": run_long5, "pdd_only": run_pdd_only, "cv": run_cv}[args.cmd](args)


if __name__ == "__main__":
    main()
