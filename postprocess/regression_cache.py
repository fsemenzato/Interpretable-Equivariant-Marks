from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from postprocess import config_fish as CF
from postprocess import fisher_common as FC

PARAM_LABELS: List[str] = list(CF.PARAMS)
DEFAULT_OUT_ROOT = ROOT / "results" / "regression"
DATASETS: Dict[str, dict] = {
    "lhc": dict(
        label="LHC",
        env="REGRESSION_LHC_ROOT",
        root=Path("data/quijote_z0"),
        params_name="latin_hypercube_params.txt",
        id_range=(0, 2000),
        name_fmt="df_m_{grid}_{mas}_z=0.npy",
    ),
    "bsq_extra": dict(
        label="BSQ_EXTRA",
        env="REGRESSION_BSQ_ROOT",
        root=Path("data/BSQ_EXTRA"),
        params_name="BSQ_params.txt",
        id_range=(28000, 32582),
        name_fmt="dm_z0_{mas}_{grid}.npy",
    ),
}
PANEL_R = [10.0]
PANEL_DS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
PANEL_P = [-2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5]
PANEL_FILTER = "gaussian"
PANEL_NORMALIZE = "none"


def panel_keys(R=PANEL_R, ds=PANEL_DS, p=PANEL_P) -> List[str]:
    from preprocess.classical_baseline import mark_key

    return [mark_key(r, s, q) for r in R for s in ds for q in p]


def old_to_new_key(old: str) -> str:
    from preprocess.classical_baseline import mark_key

    body = old[len("mw_") :] if old.startswith("mw_") else old
    r, s, p = body.split("_")
    return mark_key(float(r[1:]), round(float(s[1:]), 6), float(p[1:]))


def dataset_spec(
    name: str,
    root: Optional[Path] = None,
    params_file: Optional[Path] = None,
    id_range: Optional[Tuple[int, int]] = None,
) -> dict:
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    d = dict(DATASETS[name])
    d["name"] = name
    d["root"] = Path(root) if root else Path(os.environ.get(d["env"], d["root"]))
    d["params_file"] = (
        Path(params_file) if params_file else d["root"] / d["params_name"]
    )
    if id_range:
        d["id_range"] = tuple(id_range)
    return d


def dataset_sims(spec: dict, grid: int, mas: str, max_sims: Optional[int] = None):
    dname = spec["name_fmt"].format(grid=int(grid), mas=mas)
    lo, hi = spec["id_range"]
    root = spec["root"]
    ids, paths = ([], [])
    for i in range(lo, hi):
        p = root / str(i) / dname
        if p.exists():
            ids.append(i)
            paths.append(p)
    print(
        f"  {spec['name']}: found {len(ids)}/{hi - lo} sims in {root} (ids {lo}..{hi - 1}, file {dname})",
        flush=True,
    )
    if not ids:
        raise SystemExit(f"{spec['name']}: no boxes under {root}/<id>/{dname}")
    if max_sims is not None:
        ids, paths = (ids[:max_sims], paths[:max_sims])
    master = np.loadtxt(spec["params_file"])
    if master.ndim != 2 or master.shape[1] != len(PARAM_LABELS):
        raise RuntimeError(
            f"{spec['params_file']}: expected (N, {len(PARAM_LABELS)}), got {master.shape}"
        )
    if master.shape[0] <= max(ids):
        raise RuntimeError(
            f"{spec['params_file']} has {master.shape[0]} rows but sim id {max(ids)} is on disk"
        )
    ids = np.asarray(ids, dtype=np.int64)
    return (paths, ids, master[ids].astype(np.float32), dname)


def alignment_report(
    k_vec, P_dd, Y, label: str, k_ref: float = 0.1, seed: int = 0, min_corr: float = 0.5
) -> float:
    k_vec = np.asarray(k_vec, float)
    i = int(np.argmin(np.abs(k_vec - k_ref)))
    lp = np.log10(np.clip(np.asarray(P_dd)[:, i], 1e-12, None))
    c_s8 = float(np.corrcoef(lp, np.log(Y[:, PARAM_LABELS.index("s8")]))[0, 1])
    c_om = float(np.corrcoef(lp, Y[:, 0])[0, 1])
    rng = np.random.default_rng(seed)
    c_shuf = float(
        np.corrcoef(lp, np.log(rng.permutation(Y[:, PARAM_LABELS.index("s8")])))[0, 1]
    )
    print(
        f"[align] {label}: corr(log10 P_dd(k={k_vec[i]:.3f}), log s8) = {c_s8:+.3f}  (Om {c_om:+.3f}; scrambled-rows control {c_shuf:+.3f}; Quijote LHC reference +0.84)",
        flush=True,
    )
    if not c_s8 >= min_corr:
        raise RuntimeError(
            f"{label}: P_dd amplitude barely correlates with sigma_8 ({c_s8:.3f}) -- parameter rows are probably misaligned"
        )
    return c_s8


def load_regression_cache(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"])) if "meta" in out else {}
    return out


def save_regression_cache(path: Path, res: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        **{k: np.asarray(v) for k, v in res.items()},
        meta=np.array(json.dumps(meta)),
    )
    os.replace(tmp, path)


def check_pair(
    classical: dict, learned: dict, name: str = "learned", rtol: float = 0.0001
) -> None:
    if not np.array_equal(
        np.asarray(classical["sim_ids"]), np.asarray(learned["sim_ids"])
    ):
        raise RuntimeError(f"{name}: sim_ids differ from the classical cache")
    if not np.array_equal(
        np.asarray(classical["params"], np.float32),
        np.asarray(learned["params"], np.float32),
    ):
        raise RuntimeError(f"{name}: params differ from the classical cache")
    FC.assert_same_k(classical["k"], learned["k"], what=f"({name} vs classical cache)")
    a, b = (np.asarray(classical["Pdd"]), np.asarray(learned["P_dd"]))
    if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=0.0):
        worst = (
            float(np.max(np.abs(a - b) / np.abs(a)))
            if a.shape == b.shape
            else float("nan")
        )
        raise RuntimeError(
            f"{name}: learned P_dd differs from the classical P_dd (max rel diff {worst:.3g}); different sims or estimator"
        )


def _first_box_check(path: Path) -> dict:
    arr = np.load(path, mmap_mode="r")
    mean, mn = (float(np.mean(arr)), float(np.min(arr)))
    if mn < -1.0 - 0.001 or abs(mean) > 0.1:
        raise RuntimeError(
            f"{path}: not a density-contrast box (mean {mean:.3g}, min {mn:.3g})"
        )
    return {
        "first_box_mean": mean,
        "first_box_min": mn,
        "first_box_shape": list(arr.shape),
    }


def _meta_common(spec, ids, dname, mas, grid, box) -> dict:
    return {
        "dataset": spec["name"],
        "root": str(spec["root"]),
        "params_file": str(spec["params_file"]),
        "data_name": dname,
        "mas": mas,
        "grid_dim": int(grid),
        "box_size": float(box),
        "n_sims": int(len(ids)),
        "sim_ids_first_last": [int(ids[0]), int(ids[-1])],
        "sim_ids_sum": int(np.sum(ids)),
        "param_labels": PARAM_LABELS,
    }


def _same(cached: dict, wanted: dict, keys: Sequence[str]) -> bool:
    return all((cached.get(k) == wanted.get(k) for k in keys))


def classical_stage(spec, args, device, grid, mas, box) -> Path:
    from pk import PkEstimator
    from preprocess import classical_baseline as CB

    out = Path(args.out_root) / spec["name"] / "classical_panel.npz"
    paths, ids, params, dname = dataset_sims(spec, grid, mas, args.max_sims)
    meta = _meta_common(spec, ids, dname, mas, grid, box)
    meta.update(
        {
            "stage": "classical",
            "filter": PANEL_FILTER,
            "R": PANEL_R,
            "ds": args.ds,
            "p": args.p,
            "normalize": PANEL_NORMALIZE,
            "convention": CB.CONVENTION,
            "marks": panel_keys(PANEL_R, args.ds, args.p),
        }
    )
    if out.exists() and (not args.force):
        old = load_regression_cache(out)
        if _same(old["meta"], meta, list(meta)):
            print(f"  classical panel cached ({out})", flush=True)
            return out
        print("  classical panel: meta differs -> recomputing", flush=True)
    if args.dry_run:
        return out
    est = PkEstimator(BoxSize=box, device=device)
    cb_args = SimpleNamespace(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        R=PANEL_R,
        ds=args.ds,
        p=args.p,
        box_size=box,
        filter=PANEL_FILTER,
        mas=mas,
        normalize=PANEL_NORMALIZE,
    )
    meta["first_box"] = _first_box_check(paths[0])
    t0 = time.time()
    res = CB.compute_set(paths, est, cb_args, device, f"{spec['name']}/classical")
    res.update(sim_ids=ids, params=params, param_labels=np.array(PARAM_LABELS))
    alignment_report(res["k"], res["Pdd"], params, f"{spec['name']} classical cache")
    save_regression_cache(out, res, meta)
    print(
        f"  classical panel: {len(ids)} sims x {len(meta['marks'])} marks in {time.time() - t0:.0f}s -> {out}",
        flush=True,
    )
    return out


def learned_stage(
    spec, args, cfg, run_dir, tag, device, grid, mas, box, model_holder: dict
) -> Path:
    from pk import PkEstimator
    from postprocess import fisher as FM

    out = Path(args.out_root) / spec["name"] / f"learned_{tag}.npz"
    paths, ids, params, dname = dataset_sims(spec, grid, mas, args.max_sims)
    meta = _meta_common(spec, ids, dname, mas, grid, box)
    meta.update(
        {
            "stage": "learned",
            "tag": tag,
            "run": str(run_dir),
            "model_class": cfg["_model_class"],
            "code_hash": cfg.get("code_hash"),
            "ckpt_sha": FM.checkpoint_fingerprint(run_dir),
            "overrides": dict(sorted(args.overrides.items())),
            "mark_normalize": bool(cfg.get("mark_normalize", False)),
        }
    )
    keys = [
        "dataset",
        "root",
        "params_file",
        "data_name",
        "mas",
        "n_sims",
        "sim_ids_first_last",
        "sim_ids_sum",
        "model_class",
        "code_hash",
        "ckpt_sha",
        "overrides",
        "mark_normalize",
        "grid_dim",
    ]
    if out.exists() and (not args.force):
        old = load_regression_cache(out)
        if _same(old["meta"], meta, keys):
            print(
                f"  {spec['name']}: learned cache for {tag} cached ({out})", flush=True
            )
            return out
        print(f"  {spec['name']}: cache meta differs -> recomputing", flush=True)
    if args.dry_run:
        return out
    if "model" not in model_holder:
        model_holder["model"] = FM.build_model(cfg, run_dir, device)
    est = PkEstimator(BoxSize=box, device=device)
    meta["first_box"] = _first_box_check(paths[0])
    t0 = time.time()
    r = FM.compute_set(
        model_holder["model"],
        est,
        paths,
        mas,
        False,
        args.batch_size,
        args.num_workers,
        device,
        f"{spec['name']}/{tag}",
    )
    res = {
        "k": r["k"],
        "P_dd": r["Pdd"],
        "P_mm": r["Pmm"],
        "P_dm": r["Pdm"],
        "sim_ids": ids,
        "params": params,
        "param_labels": np.array(PARAM_LABELS),
    }
    alignment_report(res["k"], res["P_dd"], params, f"{spec['name']} learned {tag}")
    cpath = Path(args.out_root) / spec["name"] / "classical_panel.npz"
    if cpath.exists():
        check_pair(
            load_regression_cache(cpath), res, name=f"{spec['name']} learned {tag}"
        )
        meta["pdd_equals_classical_pdd"] = True
    else:
        print(
            "  (classical cache not present yet: the P_dd-equality guard is deferred to regress_mlp)",
            flush=True,
        )
        meta["pdd_equals_classical_pdd"] = None
    save_regression_cache(out, res, meta)
    print(
        f"  {spec['name']}: {len(ids)} sims in {time.time() - t0:.0f}s -> {out}",
        flush=True,
    )
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", choices=["learned", "classical", "all"], default="all")
    ap.add_argument(
        "--datasets", nargs="+", default=["lhc", "bsq_extra"], choices=sorted(DATASETS)
    )
    ap.add_argument(
        "--run",
        type=Path,
        default=None,
        help="run dir with config.json + best_checkpoint.pt (learned stage)",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="label of the run in the cache name (default: basename of --run)",
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a model setting",
    )
    ap.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--lhc_root", type=Path, default=None)
    ap.add_argument("--bsq_root", type=Path, default=None)
    ap.add_argument(
        "--bsq_ids", type=int, nargs=2, default=None, metavar=("START", "STOP")
    )
    ap.add_argument(
        "--ds",
        type=float,
        nargs="+",
        default=PANEL_DS,
        help="classical delta_s grid",
    )
    ap.add_argument(
        "--p",
        type=float,
        nargs="+",
        default=PANEL_P,
        help="classical exponent grid",
    )
    ap.add_argument(
        "--grid_dim",
        type=int,
        default=None,
        help="default: the run's grid_dim (classical stage: 128)",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument(
        "--max_sims",
        type=int,
        default=None,
        help="cap the sims per dataset (smoke tests only)",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--dry-run", action="store_true", help="list the sims found and exit"
    )
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args(argv)
    from postprocess.fisher import parse_overrides

    a.overrides = parse_overrides(a.override)
    return a


def main(argv=None) -> None:
    args = parse_args(argv)
    import torch
    from postprocess import fisher as FM
    from postprocess.simdata import pick_device

    device = pick_device(args.device)
    stages = ["classical", "learned"] if args.stage == "all" else [args.stage]
    cfg = run_dir = tag = None
    grid, mas, box = (args.grid_dim or CF.GRID_DIM, CF.DEFAULT_MAS, CF.BOX_SIZE)
    if "learned" in stages:
        if args.run is None:
            raise SystemExit("--run is required for the learned stage")
        run_dir = args.run.resolve()
        tag = args.tag or run_dir.name
        cfg = FM.read_run_config(run_dir)
        if args.overrides:
            print("config overrides:")
            cfg = FM.apply_overrides(cfg, args.overrides)
        mas = FM.mas_of(cfg)
        grid = int(args.grid_dim or cfg.get("grid_dim", CF.GRID_DIM))
        box = float(cfg.get("box_size", CF.BOX_SIZE))
        print(
            f"run: {run_dir}  tag={tag}  model={cfg['_model_class']}  MAS={mas}  grid={grid}  device={device}"
        )
    roots = {"lhc": args.lhc_root, "bsq_extra": args.bsq_root}
    holder: dict = {}
    for ds in args.datasets:
        spec = dataset_spec(
            ds, root=roots.get(ds), id_range=args.bsq_ids if ds == "bsq_extra" else None
        )
        print(f"=== {ds}")
        if args.dry_run:
            dataset_sims(spec, grid, mas, args.max_sims)
            continue
        if "classical" in stages:
            classical_stage(spec, args, device, grid, mas, box)
        if "learned" in stages:
            learned_stage(spec, args, cfg, run_dir, tag, device, grid, mas, box, holder)


if __name__ == "__main__":
    main()
