from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from marks._mark_base import marked_field
from pk import PkEstimator
from postprocess import config_fish as CF
from postprocess.simdata import make_loader, pick_device

CONVENTION = "m = ((1 + delta_R + delta_s)/(1 + delta_s))^p ; this p equals -p of White 2016 / Massara 2021 (reciprocal base)"


def mark_key(R: float, ds: float, p: float) -> str:
    return f"R{R:g}_s{ds:g}_p{p:g}"


def parse_mark_key(key: str):
    r, s, p = key.split("_")
    return (float(r[1:]), float(s[1:]), float(p[1:]))


def _kgrid(delta: torch.Tensor, box_size: float):
    D, H, W = delta.shape[-3:]
    kx = 2 * np.pi * torch.fft.fftfreq(D, d=box_size / D, device=delta.device)
    ky = 2 * np.pi * torch.fft.fftfreq(H, d=box_size / H, device=delta.device)
    kz = 2 * np.pi * torch.fft.rfftfreq(W, d=box_size / W, device=delta.device)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
    return torch.sqrt(KX**2 + KY**2 + KZ**2)


def filter_kernel(kind: str, kmag: torch.Tensor, R: float) -> torch.Tensor:
    x = kmag * R
    if kind == "gaussian":
        return torch.exp(-0.5 * x**2)
    if kind == "tophat":
        xs = torch.where(x < 1e-06, torch.ones_like(x), x)
        return torch.where(
            x < 1e-06,
            torch.ones_like(x),
            3.0 * (torch.sin(xs) - xs * torch.cos(xs)) / xs**3,
        )
    raise ValueError(f"unknown filter {kind!r}")


def smooth(delta: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    dk = torch.fft.rfftn(delta, dim=(-3, -2, -1))
    return torch.fft.irfftn(
        dk * kernel.to(dk.dtype), s=delta.shape[-3:], dim=(-3, -2, -1)
    )


def classical_marked_field(
    delta: torch.Tensor,
    delta_R: torch.Tensor,
    ds: float,
    p: float,
    normalize: str = "none",
) -> torch.Tensor:
    base = torch.clamp((1.0 + delta_R + ds) / (1.0 + ds), min=1e-06)
    return marked_field(base**p, delta, normalize == "mean", amplitude_floor=1e-12)[0]


@torch.no_grad()
def compute_set(
    paths: List[Path], est: PkEstimator, args, device: torch.device, label: str = ""
) -> dict:
    loader = make_loader(paths, args.batch_size, args.num_workers, device)
    grid = [(R, ds, p) for R in args.R for ds in args.ds for p in args.p]
    acc: Dict[str, List[np.ndarray]] = {"Pdd": []}
    for R, ds, p in grid:
        acc[f"Pmm__{mark_key(R, ds, p)}"] = []
        acc[f"Pdm__{mark_key(R, ds, p)}"] = []
    kernels = None
    k_vec = None
    t0 = time.time()
    for ib, batch in enumerate(loader):
        d = batch.to(device, non_blocking=True)
        if kernels is None:
            kmag = _kgrid(d, args.box_size)
            kernels = {R: filter_kernel(args.filter, kmag, R) for R in args.R}
        base = est.compute(d, MAS=(args.mas, args.mas)).cpu().numpy()
        if k_vec is None:
            k_vec = base[0, :, 0].astype(np.float64)
        acc["Pdd"].append(base[:, :, 1])
        for R in args.R:
            dR = smooth(d, kernels[R])
            for ds in args.ds:
                for p in args.p:
                    f = classical_marked_field(d, dR, ds, p, args.normalize)
                    r = est.compute(d, f, MAS=(args.mas, args.mas)).cpu().numpy()
                    key = mark_key(R, ds, p)
                    acc[f"Pmm__{key}"].append(r[:, :, 2])
                    acc[f"Pdm__{key}"].append(r[:, :, 3])
        if (ib + 1) % 20 == 0:
            print(
                f"    {label}: {min((ib + 1) * args.batch_size, len(paths))}/{len(paths)} boxes, {time.time() - t0:.0f}s",
                flush=True,
            )
    out = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    out["k"] = k_vec
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sets", nargs="+", default=["all"], help="set names or 'all'")
    ap.add_argument(
        "--R",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 15.0],
        help="smoothing scale(s) [Mpc/h]",
    )
    ap.add_argument(
        "--ds", type=float, nargs="+", default=[0.1, 0.25, 0.5], help="delta_s value(s)"
    )
    ap.add_argument(
        "--p",
        type=float,
        nargs="+",
        default=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        help="exponent(s); see CONVENTION (sign flipped vs White/Massara)",
    )
    ap.add_argument("--filter", choices=["gaussian", "tophat"], default="gaussian")
    ap.add_argument(
        "--normalize",
        choices=["none", "mean"],
        default="none",
        help="'none': m(1+d) - <.>  (default); 'mean': m(1+d)/<.> - 1",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_fid", type=int, default=CF.N_SIMS_FID)
    ap.add_argument("--n_var", type=int, default=CF.N_SIMS_VAR)
    ap.add_argument("--grid_dim", type=int, default=CF.GRID_DIM)
    ap.add_argument("--box_size", type=float, default=CF.BOX_SIZE)
    ap.add_argument("--mas", type=str, default=CF.DEFAULT_MAS)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    sets = CF.SETS if args.sets == ["all"] else args.sets
    unknown = [s for s in sets if s not in CF.SETS]
    if unknown:
        raise SystemExit(f"unknown set(s) {unknown}; known: {CF.SETS}")
    device = pick_device(args.device)
    dname = CF.data_name(args.grid_dim, args.mas)
    args.out.mkdir(parents=True, exist_ok=True)
    est = PkEstimator(BoxSize=args.box_size, device=device)
    print(
        f"classical marks: filter={args.filter} R={args.R} ds={args.ds} p={args.p} normalize={args.normalize}"
    )
    print(
        f"  {CONVENTION}\n  roots: fid={CF.fid_root()} diff={CF.diff_root()}  data={dname}  device={device}"
    )
    for name in sets:
        f = args.out / f"{name}.npz"
        root = CF.set_root(name)
        n = CF.default_n_sims(name, args.n_fid, args.n_var)
        paths = CF.sim_paths(root, dname, n)
        if not paths:
            raise SystemExit(f"{name}: no boxes under {root}/*/{dname}")
        meta = {
            "set": name,
            "n_sims": len(paths),
            "data_name": dname,
            "mas": args.mas,
            "box_size": args.box_size,
            "grid_dim": args.grid_dim,
            "filter": args.filter,
            "R": list(args.R),
            "ds": list(args.ds),
            "p": list(args.p),
            "normalize": args.normalize,
            "convention": CONVENTION,
            "sim_ids_first_last": [CF.sim_ids(paths)[0], CF.sim_ids(paths)[-1]],
        }
        if f.exists() and (not args.force):
            old = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
            if all((old.get(k) == meta[k] for k in meta)):
                print(f"  {name}: cached ({f})", flush=True)
                continue
            print(f"  {name}: cache meta differs -> recomputing", flush=True)
        t0 = time.time()
        res = compute_set(paths, est, args, device, name)
        res["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(f, **res)
        print(
            f"  {name}: {len(paths)} boxes, {len(args.R) * len(args.ds) * len(args.p)} marks in {time.time() - t0:.0f}s -> {f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
