from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pk import PkEstimator
from marks import build_mark
from postprocess import config_fish as CF
from postprocess import fisher_common as FC
from postprocess.simdata import make_loader, moments, pick_device

DEFAULT_LADDER = [0.25, 0.3, 0.35, 0.4, 0.5, 0.69]
LAMBDA_MOMENTS = ["var_l1", "var_l2", "var_l3", "mean_l2sq"]
INVARIANT_MOMENTS = ["var_tr", "var_shear"]
BASE_SCALARS = ["var_d", "skew_d", "var_g", "skew_g", "lnA"]
CACHE_SUBDIR = "fisher_cache"


def parse_overrides(items: Optional[List[str]]) -> dict:
    out: Dict[str, object] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--override expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    cfg = dict(cfg)
    for key, val in overrides.items():
        old = cfg.get(key, "<unset>")
        print(f"  override {key}: {old!r} -> {val!r}")
        cfg[key] = val
    return cfg


def read_run_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing {cfg_path}")
    return json.loads(cfg_path.read_text())


def mas_of(cfg: dict) -> str:
    return str(cfg.get("mas_scheme") or cfg.get("mas") or CF.DEFAULT_MAS)


def build_model(cfg: dict, run_dir: Path, device: torch.device) -> torch.nn.Module:
    model = build_mark(cfg).to(device)
    ckpt_path = run_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing {ckpt_path}")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not (isinstance(blob, dict) and isinstance(blob.get("model"), dict)):
        raise KeyError(
            f"{ckpt_path}: expected a dict with key 'model' (state_dict); got keys {(list(blob) if isinstance(blob, dict) else type(blob))}"
        )
    model.load_state_dict(blob["model"], strict=True)
    model.eval()
    return model


def checkpoint_fingerprint(run_dir: Path) -> Optional[str]:
    import hashlib

    p = run_dir / "best_checkpoint.pt"
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cache_meta(
    cfg: dict,
    dname: str,
    mas: str,
    n_sims: int,
    ids: List[int],
    baselines: bool,
    overrides: Optional[dict] = None,
    ckpt_sha: Optional[str] = None,
) -> dict:
    return {
        "n_sims": int(n_sims),
        "data_name": dname,
        "mas": mas,
        "code_hash": cfg.get("code_hash"),
        "model_class": cfg["_model_class"],
        "ckpt_sha": ckpt_sha,
        "baselines": bool(baselines),
        "mark_normalize": bool(cfg.get("mark_normalize", False)),
        "sim_ids_first_last": [ids[0], ids[-1]] if ids else [],
        "overrides": dict(sorted((overrides or {}).items())),
    }


def meta_matches(cached: dict, wanted: dict) -> bool:
    for key in (
        "n_sims",
        "data_name",
        "mas",
        "code_hash",
        "model_class",
        "mark_normalize",
        "sim_ids_first_last",
        "overrides",
        "ckpt_sha",
    ):
        if cached.get(key) != wanted.get(key):
            return False
    if wanted["baselines"] and (not cached.get("baselines")):
        return False
    return True


def load_cache(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


@torch.no_grad()
def compute_set(
    model,
    est: PkEstimator,
    paths: List[Path],
    mas: str,
    baselines: bool,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    label: str = "",
) -> dict:
    loader = make_loader(paths, batch_size, num_workers, device)
    acc: Dict[str, List[np.ndarray]] = {"Pdd": [], "Pmm": [], "Pdm": []}
    k_vec = None
    t0 = time.time()
    for ib, batch in enumerate(loader):
        d = batch.to(device, non_blocking=True)
        out = model(d)
        pk = est.compute(d, out["final_field"], MAS=(mas, mas)).cpu().numpy()
        if k_vec is None:
            k_vec = pk[0, :, 0].astype(np.float64)
        acc["Pdd"].append(pk[:, :, 1])
        acc["Pmm"].append(pk[:, :, 2])
        acc["Pdm"].append(pk[:, :, 3])
        if baselines:
            g = torch.log1p(d.clamp(min=-0.999))
            g = g - g.mean(dim=(-3, -2, -1), keepdim=True)
            pkg = est.compute(g, MAS=(None, None)).cpu().numpy()
            acc.setdefault("Pgg", []).append(pkg[:, :, 1])
            sc: Dict[str, torch.Tensor] = {}
            _, sc["var_d"], sc["skew_d"] = moments(d)
            _, sc["var_g"], sc["skew_g"] = moments(g)
            sc["lnA"] = torch.log(
                out["mark_amplitude"].reshape(-1).abs().double().clamp(min=1e-300)
            )
            if all((f"lam{i}" in out for i in (1, 2, 3))):
                lam = torch.cat([out["lam1"], out["lam2"], out["lam3"]], dim=1)
                for i in range(3):
                    sc[f"var_l{i + 1}"] = moments(lam[:, i])[1]
                sc["mean_l2sq"] = (
                    (lam.double() ** 2).sum(dim=1).reshape(lam.shape[0], -1).mean(dim=1)
                )
            if "tr" in out and "shear" in out:
                sc["var_tr"] = moments(out["tr"])[1]
                sc["var_shear"] = moments(out["shear"])[1]
            for name, val in sc.items():
                acc.setdefault(f"scalar__{name}", []).append(
                    val.cpu().numpy().astype(np.float64)
                )
        del d, out
        if (ib + 1) % 20 == 0:
            print(
                f"    {label}: {min((ib + 1) * batch_size, len(paths))}/{len(paths)} boxes, {time.time() - t0:.0f}s",
                flush=True,
            )
    res = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    res["k"] = k_vec
    return res


def precompute(args, cfg: dict, run_dir: Path, device: torch.device) -> Dict[str, dict]:
    mas = mas_of(cfg)
    grid = int(cfg.get("grid_dim", CF.GRID_DIM))
    box = float(cfg.get("box_size", CF.BOX_SIZE))
    dname = CF.data_name(grid, mas)
    cache_dir = run_dir / CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    plan = {}
    for name in CF.SETS:
        root = CF.set_root(name)
        n = CF.default_n_sims(name, args.n_fid, args.n_var)
        paths = CF.sim_paths(root, dname, n)
        plan[name] = (root, n, paths)
        ids = CF.sim_ids(paths)
        span = f"ids {ids[0]}..{ids[-1]}" if ids else "NONE FOUND"
        print(f"  {name:10s} {len(paths):6d}/{n} boxes  {span}   ({root}/*/{dname})")
    if args.dry_run:
        return {}
    missing = [n for n, (_, _, p) in plan.items() if not p]
    if missing:
        raise SystemExit(
            f"no boxes found for sets {missing}; check {CF.ENV_FID}/{CF.ENV_DIFF}"
        )
    model = None
    est = PkEstimator(BoxSize=box, device=device)
    caches: Dict[str, dict] = {}
    ckpt_sha = checkpoint_fingerprint(run_dir)
    for name, (root, n, paths) in plan.items():
        meta = cache_meta(
            cfg,
            dname,
            mas,
            len(paths),
            CF.sim_ids(paths),
            args.baselines,
            args.overrides,
            ckpt_sha,
        )
        f = cache_dir / f"{name}.npz"
        cached = None if args.force else load_cache(f)
        if cached is not None and meta_matches(cached["meta"], meta):
            print(f"  {name}: cached ({f})", flush=True)
            caches[name] = cached
            continue
        if cached is not None:
            print(f"  {name}: cache meta differs -> recomputing", flush=True)
        if model is None:
            model = build_model(cfg, run_dir, device)
        t0 = time.time()
        res = compute_set(
            model,
            est,
            paths,
            mas,
            args.baselines,
            args.batch_size,
            args.num_workers,
            device,
            name,
        )
        res["meta"] = meta
        np.savez_compressed(
            f,
            **{
                k: np.array(json.dumps(v)) if k == "meta" else v for k, v in res.items()
            },
        )
        print(
            f"  {name}: {len(paths)} boxes in {time.time() - t0:.0f}s -> {f}",
            flush=True,
        )
        caches[name] = res
    return caches


def load_classical(cdir: Path) -> Dict[str, dict]:
    out = {}
    for name in CF.SETS:
        f = cdir / f"{name}.npz"
        if not f.exists():
            raise FileNotFoundError(f"classical cache missing {f}")
        z = np.load(f, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["meta"] = json.loads(str(d["meta"])) if "meta" in d else {}
        out[name] = d
    return out


def classical_mark_keys(cache: dict) -> List[str]:
    return sorted((k[len("Pmm__") :] for k in cache if k.startswith("Pmm__")))


def fisher_stage(
    args, caches: Dict[str, dict], classical: Optional[Dict[str, dict]]
) -> dict:
    params = CF.PARAMS
    steps = CF.STEPS
    k = np.asarray(caches["fiducial"]["k"], float)
    for name, c in caches.items():
        FC.assert_same_k(k, c["k"], what=f"(learned cache {name} vs fiducial)")
    if classical is not None:
        for name, c in classical.items():
            FC.assert_same_k(
                k, c["k"], what=f"(classical cache {name} vs learned fiducial)"
            )
    scalars = sorted(
        (
            key[len("scalar__") :]
            for key in caches["fiducial"]
            if key.startswith("scalar__")
        )
    )
    has_lambda = all((s in scalars for s in LAMBDA_MOMENTS))
    has_invariant = all((s in scalars for s in INVARIANT_MOMENTS))
    hartlap_opts = [False, True]
    results = {"kmax": {}}
    for kmax in args.kmax:
        m0 = FC.kmask(k, args.kmin, kmax)
        if m0.sum() == 0:
            print(f"kmax={kmax}: no k bins in band -> skipped")
            continue
        specs: Dict[str, List[tuple]] = {
            "unmarked": FC.spectrum_blocks("unmarked", m0),
            "marked": FC.spectrum_blocks("marked", m0),
            "marked_mm_only": FC.spectrum_blocks("marked_mm_only", m0),
        }
        for k2 in args.ladder:
            m2 = FC.kmask(k, args.kmin, k2)
            if m2.sum() == 0:
                continue
            specs[f"unmarked_kmax{k2:g}"] = FC.spectrum_blocks("unmarked", m2)
        if "Pgg" in caches["fiducial"]:
            for k2 in sorted(set([kmax, 0.3, 0.4])):
                m2 = FC.kmask(k, args.kmin, k2)
                if m2.sum():
                    specs[f"unmarked+Pgg(k<={k2:g})"] = specs["unmarked"] + [
                        ("spectrum", "Pgg", m2)
                    ]
        for s in scalars:
            specs[f"unmarked+{s}"] = specs["unmarked"] + FC.scalar_blocks([s])
        if has_lambda:
            specs["unmarked+lambda_moments"] = specs["unmarked"] + FC.scalar_blocks(
                LAMBDA_MOMENTS
            )
            specs["marked+lambda_moments"] = specs["marked"] + FC.scalar_blocks(
                LAMBDA_MOMENTS
            )
        if has_invariant:
            specs["unmarked+invariant_moments"] = specs["unmarked"] + FC.scalar_blocks(
                INVARIANT_MOMENTS
            )
            specs["marked+invariant_moments"] = specs["marked"] + FC.scalar_blocks(
                INVARIANT_MOMENTS
            )
        if scalars:
            specs["unmarked+all_scalars"] = specs["unmarked"] + FC.scalar_blocks(
                scalars
            )
        kres = {}
        for hart in hartlap_opts:
            key = f"hartlap={hart}"
            ref = FC.forecast(
                caches, specs["unmarked"], steps, hartlap=hart, params=params
            )
            recs = {}
            rows = []
            for name, blocks in specs.items():
                r = FC.forecast(caches, blocks, steps, hartlap=hart, params=params)
                recs[name] = FC.gain_record(r, ref, "unmarked")
                rows.append((name, recs[name]["improvement"], recs[name]["dims"]))
            if classical is not None:
                cref_blocks = FC.spectrum_blocks("unmarked", m0)
                cref = FC.forecast(
                    classical, cref_blocks, steps, hartlap=hart, params=params
                )
                recs["classical_unmarked"] = FC.gain_record(cref, ref, "unmarked")
                rows.append(
                    (
                        "classical_unmarked",
                        recs["classical_unmarked"]["improvement"],
                        recs["classical_unmarked"]["dims"],
                    )
                )
                for mk in classical_mark_keys(classical["fiducial"]):
                    blocks = FC.spectrum_blocks(
                        "marked", m0, keys={"Pmm": f"Pmm__{mk}", "Pdm": f"Pdm__{mk}"}
                    )
                    r = FC.forecast(
                        classical, blocks, steps, hartlap=hart, params=params
                    )
                    nm = f"classical__{mk}"
                    recs[nm] = FC.gain_record(r, cref, "classical_unmarked")
                    rows.append((nm, recs[nm]["improvement"], recs[nm]["dims"]))
            kres[key] = {
                "n_bins": int(m0.sum()),
                "n_fid": ref["n_fid"],
                "unmarked_sigma": [float(x) for x in ref["sigma"]],
                "summaries": recs,
            }
            print(
                FC.format_table(
                    rows,
                    params,
                    title=f"\n=== kmax={kmax:g} ({int(m0.sum())} bins, n_fid={ref['n_fid']}) {key} — improvement sigma_unmarked/sigma (classical marks vs their own Pdd) ===",
                )
            )
            print(
                "  unmarked sigma: "
                + "  ".join((f"{p}={s:.4g}" for p, s in zip(params, ref["sigma"])))
            )
        results["kmax"][f"{kmax:g}"] = kres
    return results


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run",
        type=Path,
        required=True,
        help="run directory with config.json + best_checkpoint.pt",
    )
    ap.add_argument("--kmax", type=float, nargs="+", default=[0.2, 0.3])
    ap.add_argument("--kmin", type=float, default=0.0)
    ap.add_argument(
        "--ladder",
        type=float,
        nargs="*",
        default=DEFAULT_LADDER,
        help="unmarked Pdd(k<=kmax') ladder reported relative to Pdd(k<=kmax)",
    )
    ap.add_argument("--n_fid", type=int, default=CF.N_SIMS_FID)
    ap.add_argument("--n_var", type=int, default=CF.N_SIMS_VAR)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument(
        "--hartlap",
        action="store_true",
        help="(both corrected and uncorrected numbers are always reported; this flag selects which one the summary line prints)",
    )
    ap.add_argument(
        "--baselines",
        action="store_true",
        help="also measure Pgg and the one-point scalars",
    )
    ap.add_argument(
        "--classical",
        type=Path,
        default=None,
        help="classical cache dir (preprocess/classical_baseline.py)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON (default <run>/fisher_results.json)",
    )
    ap.add_argument("--force", action="store_true", help="recompute cached sets")
    ap.add_argument(
        "--dry-run", action="store_true", help="only list the simulations found per set"
    )
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="patch config.json before building the model (repeatable; JSON value or string)",
    )
    a = ap.parse_args(argv)
    a.overrides = parse_overrides(a.override)
    return a


def main(argv=None) -> dict:
    args = parse_args(argv)
    run_dir = args.run.resolve()
    cfg = read_run_config(run_dir)
    if args.overrides:
        print("config overrides:")
        cfg = apply_overrides(cfg, args.overrides)
    device = pick_device(args.device)
    print(
        f"run: {run_dir}\nmodel: {cfg['_model_class']}  MAS={mas_of(cfg)}  grid={cfg.get('grid_dim')}  device={device}\nroots: fid={CF.fid_root()}  diff={CF.diff_root()}"
    )
    caches = precompute(args, cfg, run_dir, device)
    if args.dry_run:
        return {}
    classical = load_classical(args.classical.resolve()) if args.classical else None
    results = fisher_stage(args, caches, classical)
    results.update(
        {
            "run": str(run_dir),
            "model_class": cfg["_model_class"],
            "mas": mas_of(cfg),
            "code_hash": cfg.get("code_hash"),
            "params": CF.PARAMS,
            "steps": CF.STEPS,
            "kmin": args.kmin,
            "ladder": args.ladder,
            "baselines": bool(args.baselines),
            "classical_cache": str(args.classical) if args.classical else None,
            "n_sims": {name: int(c["meta"]["n_sims"]) for name, c in caches.items()},
            "k": [float(x) for x in caches["fiducial"]["k"]],
            "hartlap_requested": bool(args.hartlap),
            "config_overrides": args.overrides,
        }
    )
    out = args.out or run_dir / "fisher_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(FC.nan_to_none(results), indent=1))
    print(f"\nwrote {out}")
    hk = f"hartlap={bool(args.hartlap)}"
    for kmax, kres in results["kmax"].items():
        if "marked" in kres[hk]["summaries"]:
            imp = kres[hk]["summaries"]["marked"]["improvement"]
            print(
                f"  kmax={kmax} {hk}: marked improvement "
                + "  ".join(
                    (
                        f"{p}={g:.2f}" if g is not None else f"{p}=nan"
                        for p, g in zip(CF.PARAMS, imp)
                    )
                )
            )
    return results


if __name__ == "__main__":
    main()
