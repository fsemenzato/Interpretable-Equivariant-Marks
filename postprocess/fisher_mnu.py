from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pk import PkEstimator
from postprocess import config_fish as CF
from postprocess import config_mnu as CM
from postprocess import fisher as FI
from postprocess import fisher_common as FC
from postprocess.simdata import pick_device

DEFAULT_KMAX = [0.2, 0.3, 0.4]
DEFAULT_LADDER = [0.3, 0.4, 0.5]
MNU_CACHE_SUBDIR = "fisher_cache_mnu"
REF = "unmarked"


def build_plan(
    field: str, scheme: str, grid: int, mas: str, n_fid: int, n_var: int
) -> Dict[str, dict]:
    plan: Dict[str, dict] = {}
    for name in CM.all_sets(scheme):
        root = CM.set_root(name)
        n = CM.default_n_sims(name, n_fid, n_var)
        dname = CM.data_name(name, field, grid, mas)
        paths = CF.sim_paths(root, dname, n)
        plan[name] = {
            "root": root,
            "data_name": dname,
            "n": n,
            "paths": paths,
            "ids": CF.sim_ids(paths),
        }
    return plan


def print_plan(plan: Dict[str, dict]) -> None:
    for name, sp in plan.items():
        ids = sp["ids"]
        span = f"ids {ids[0]}..{ids[-1]}" if ids else "NONE FOUND"
        short = "   <-- SHORT" if 0 < len(ids) < sp["n"] else ""
        print(
            f"  {name:12s} {len(sp['paths']):6d}/{sp['n']:<6d} {span:18s} ({sp['root']}/*/{sp['data_name']}){short}"
        )


def check_plan(plan: Dict[str, dict]) -> None:
    missing = [n for n, sp in plan.items() if not sp["paths"]]
    if missing:
        hint = ""
        if any((n in CM.MNU_SETS for n in missing)):
            hint = f"\n  the Mnu sets live under {CM.ENV_MNU} (now {CM.mnu_root()}) and the Zeldovich fiducial under {CM.ENV_ZA} (now {CM.za_root()}); --field cb needs df_cb_* in Mnu_p/pp/ppp"
        raise SystemExit(
            f"no boxes found for sets {missing}; check {CF.ENV_FID}/{CF.ENV_DIFF}{hint}"
        )


def cache_dirname(
    ckpt_sha: Optional[str], field: str, n_fid: int, n_var: int, overrides: Mapping
) -> str:
    ov = ""
    if overrides:
        ov = (
            "_ov"
            + hashlib.sha256(
                json.dumps(dict(sorted(overrides.items())), sort_keys=True).encode()
            ).hexdigest()[:8]
        )
    return f"{ckpt_sha or 'nosha'}_{field}{CM.tag_for(n_fid, n_var)}{ov}"


def learned_caches(
    args, cfg: dict, run_dir: Path, device: torch.device, plan: Dict[str, dict]
) -> Dict[str, dict]:
    mas = FI.mas_of(cfg)
    box = float(cfg.get("box_size", CF.BOX_SIZE))
    ckpt_sha = FI.checkpoint_fingerprint(run_dir)
    cdir = (args.cache_dir or run_dir / MNU_CACHE_SUBDIR) / cache_dirname(
        ckpt_sha, args.field, args.n_fid, args.n_var, args.overrides
    )
    cdir.mkdir(parents=True, exist_ok=True)
    print(f"  spectra cache: {cdir}")
    est = PkEstimator(BoxSize=box, device=device)
    model = None
    caches: Dict[str, dict] = {}
    for name, sp in plan.items():
        paths = sp["paths"]
        meta = FI.cache_meta(
            cfg,
            sp["data_name"],
            mas,
            len(paths),
            sp["ids"],
            False,
            args.overrides,
            ckpt_sha,
        )
        f = cdir / f"{name}.npz"
        cached = None if args.force else FI.load_cache(f)
        if cached is not None and FI.meta_matches(cached["meta"], meta):
            print(f"  {name}: cached ({f.name})", flush=True)
            caches[name] = cached
            continue
        if cached is not None:
            print(f"  {name}: cache meta differs -> recomputing", flush=True)
        if name in CF.SETS and (not args.force):
            sources = [run_dir / FI.CACHE_SUBDIR / f"{name}.npz"]
            if args.field != "m":
                sources.append(
                    cdir.parent
                    / cache_dirname(
                        ckpt_sha, "m", args.n_fid, args.n_var, args.overrides
                    )
                    / f"{name}.npz"
                )
            reuse, src = (None, None)
            for g in sources:
                c = FI.load_cache(g)
                if c is not None and FI.meta_matches(c["meta"], meta):
                    reuse, src = (c, g)
                    break
            if reuse is not None:
                print(f"  {name}: reused from {src}", flush=True)
                np.savez_compressed(
                    f,
                    **{
                        kk: np.array(json.dumps(v)) if kk == "meta" else v
                        for kk, v in reuse.items()
                    },
                )
                caches[name] = reuse
                continue
        if model is None:
            model = FI.build_model(cfg, run_dir, device)
        t0 = time.time()
        res = FI.compute_set(
            model,
            est,
            paths,
            mas,
            False,
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


def stencil_derivative(means: Mapping[str, np.ndarray], scheme: str) -> np.ndarray:
    rule = CM.SCHEMES[scheme]
    missing = [k for k in rule["weights"] if k not in means]
    if missing:
        raise KeyError(
            f"Mnu scheme {scheme!r} needs sets {sorted(rule['weights'])}; missing {missing}"
        )
    acc = sum(
        (w * np.asarray(means[k], dtype=np.float64) for k, w in rule["weights"].items())
    )
    return acc / (float(rule["denom"]) * CM.MNU_BASE_STEP)


def summary_sets(
    caches: Mapping[str, Mapping], blocks: Sequence[tuple]
) -> Dict[str, np.ndarray]:
    return {name: FC.build_summary(c, blocks) for name, c in caches.items()}


def jacobian(
    means: Mapping[str, np.ndarray],
    params: Sequence[str],
    scheme: str,
    steps: Mapping[str, float] = CM.STEPS,
) -> np.ndarray:
    cols = []
    for p in params:
        if p == "Mnu":
            cols.append(stencil_derivative(means, scheme))
        else:
            for s in (f"{p}_p", f"{p}_m"):
                if s not in means:
                    raise KeyError(f"missing derivative set {s!r} for {p}")
            cols.append(
                (
                    np.asarray(means[f"{p}_p"], float)
                    - np.asarray(means[f"{p}_m"], float)
                )
                / (2.0 * float(steps[p]))
            )
    return np.stack(cols, axis=1)


def fisher_from_means(
    S_fid: np.ndarray,
    means: Mapping[str, np.ndarray],
    params: Sequence[str] = CM.PARAMS,
    scheme: str = CM.DEFAULT_SCHEME,
    hartlap: bool = False,
    standardize: bool = True,
    steps: Mapping[str, float] = CM.STEPS,
) -> np.ndarray:
    F = FC.fisher_from_jacobian(
        S_fid, jacobian(means, params, scheme, steps), hartlap, standardize
    )
    return 0.5 * (F + F.T)


def fisher_from_caches(
    caches: Mapping[str, Mapping],
    blocks: Sequence[tuple],
    params: Sequence[str] = CM.PARAMS,
    scheme: str = CM.DEFAULT_SCHEME,
    hartlap: bool = False,
    standardize: bool = True,
    n_deriv: Optional[int] = None,
    n_cov: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    S = summary_sets(caches, blocks)
    S_fid = S["fiducial"][:n_cov] if n_cov else S["fiducial"]
    means = {
        n: (v[:n_deriv] if n_deriv else v).mean(axis=0)
        for n, v in S.items()
        if n != "fiducial"
    }
    F = fisher_from_means(S_fid, means, params, scheme, hartlap, standardize)
    return (F, {"dims": int(S_fid.shape[1]), "n_fid": int(S_fid.shape[0])})


def subset_sigmas(
    F: np.ndarray, params: Sequence[str], subset: Sequence[str]
) -> Dict[str, np.ndarray]:
    idx = [list(params).index(p) for p in subset]
    return {
        "fixed": FC.sigma_from_fisher(F[np.ix_(idx, idx)]),
        "marginal": FC.sigma_from_fisher(F)[idx],
    }


def sigma_table(F: np.ndarray, params: Sequence[str] = CM.PARAMS) -> Dict[str, dict]:
    out = {}
    for sname, subset in CM.PARAM_SETS.items():
        if not all((p in params for p in subset)):
            continue
        d = subset_sigmas(F, params, subset)
        out[sname] = {
            "params": list(subset),
            "marginal": d["marginal"],
            "fixed": None if len(subset) == len(params) else d["fixed"],
        }
    return out


def improvement_table(
    sig: Mapping[str, dict], ref: Mapping[str, dict]
) -> Dict[str, dict]:
    out = {}
    for sname, rec in sig.items():
        out[sname] = {"params": rec["params"]}
        for mode in ("marginal", "fixed"):
            out[sname][mode] = (
                None
                if rec[mode] is None
                else FC.improvement(ref[sname][mode], rec[mode])
            )
    return out


def degeneracy(F: np.ndarray, params: Sequence[str] = CM.PARAMS) -> dict:
    five = [p for p in params if p != "Mnu"]
    if not np.all(np.isfinite(F)):
        nan = float("nan")
        return {
            "params": list(params),
            "inflation_marginal_over_fixed": {p: nan for p in five},
            "corr_s8_Mnu": nan,
            "corr_Om_Mnu": nan,
            "corr": np.full((len(params),) * 2, nan),
        }
    d = subset_sigmas(F, params, five)
    C = np.linalg.pinv(F, hermitian=True)
    sd = np.sqrt(np.diag(C))
    corr = C / np.outer(sd, sd)
    i, j = (list(params).index("s8"), list(params).index("Mnu"))
    return {
        "params": list(params),
        "inflation_marginal_over_fixed": {
            p: float(a / b) for p, a, b in zip(five, d["marginal"], d["fixed"])
        },
        "corr_s8_Mnu": float(corr[i, j]),
        "corr_Om_Mnu": float(corr[list(params).index("Om"), j]),
        "corr": corr,
    }


def std5_crosscheck(
    caches: Mapping[str, Mapping],
    blocks: Sequence[tuple],
    F6: np.ndarray,
    hartlap: bool,
    standardize: bool,
    params: Sequence[str] = CM.PARAMS,
) -> float:
    r = FC.forecast(
        caches,
        blocks,
        CF.STEPS,
        hartlap=hartlap,
        standardize=standardize,
        params=CF.PARAMS,
    )
    mine = subset_sigmas(F6, params, CF.PARAMS)["fixed"]
    return float(np.nanmax(np.abs(mine / np.asarray(r["sigma"]) - 1.0)))


def fom_full6(F: np.ndarray) -> float:
    return FC.fom_from_fisher(F)


def summary_specs(
    k: np.ndarray, kmin: float, kmax: float, ladder: Sequence[float]
) -> Dict[str, List[tuple]]:
    m0 = FC.kmask(k, kmin, kmax)
    specs: Dict[str, List[tuple]] = {
        REF: FC.spectrum_blocks("unmarked", m0),
        "marked": FC.spectrum_blocks("marked", m0),
        "marked_mm_only": FC.spectrum_blocks("marked_mm_only", m0),
        "mm_alone": [("spectrum", "Pmm", m0)],
    }
    for k2 in ladder:
        m2 = FC.kmask(k, kmin, k2)
        if m2.sum() and k2 != kmax:
            specs[f"unmarked_kmax{k2:g}"] = FC.spectrum_blocks("unmarked", m2)
    return specs


def classical_mark_keys(cache: Mapping) -> List[str]:
    return sorted((k[len("Pmm__") :] for k in cache if k.startswith("Pmm__")))


def forecast_record(
    F: np.ndarray, dims: int, ref: Optional[dict], ref_name: str
) -> dict:
    sig = sigma_table(F)
    rec = {
        "F": F,
        "dims": int(dims),
        "sigma": sig,
        "ref": ref_name,
        "fom_full6": fom_full6(F),
    }
    if ref is not None:
        rec["improvement"] = improvement_table(sig, ref["sigma"])
        rec["fom_improvement_full6"] = (
            rec["fom_full6"] / ref["fom_full6"]
            if ref["fom_full6"] == ref["fom_full6"] and ref["fom_full6"] > 0
            else float("nan")
        )
    return rec


def pairing_check(
    caches: Mapping[str, Mapping], scheme: str, mask: np.ndarray
) -> List[dict]:
    pairs = [(f"{p}_p", f"{p}_m") for p in CF.PARAMS] + [
        (s, CM.ZA_SET) for s in CM.scheme_sets(scheme) if s != CM.ZA_SET
    ]
    out = []
    for a, b in pairs:
        if a not in caches or b not in caches:
            continue
        A, B = (caches[a]["Pdd"][:, mask], caches[b]["Pdd"][:, mask])
        n = min(len(A), len(B))
        j = A.shape[1] // 2
        r = float(np.corrcoef(A[:n, j], B[:n, j])[0, 1])
        out.append(
            {"a": a, "b": b, "n": int(n), "corr_mid_k": r, "paired": bool(r > 0.5)}
        )
    return out


def spectra_sanity(caches: Mapping[str, Mapping], mask: np.ndarray) -> List[dict]:
    if CM.ZA_SET not in caches:
        return []
    base = caches[CM.ZA_SET]["Pdd"][:, mask].mean(axis=0)
    nk = len(base)
    idx = {"low_k": nk // 10, "mid_k": nk // 2, "high_k": nk - 1}
    rows = []
    for nm in ["Mnu_p", "Mnu_pp", "Mnu_ppp", "fiducial"]:
        if nm in caches:
            m = caches[nm]["Pdd"][:, mask].mean(axis=0)
            rows.append(
                {
                    "set": nm,
                    "Mnu_eV": CM.MNU_VALUE.get(nm, 0.0),
                    **{lab: float(m[i] / base[i] - 1.0) for lab, i in idx.items()},
                }
            )
    return rows


def ic_check(
    caches: Mapping[str, Mapping], blocks: Sequence[tuple], scheme: str
) -> Optional[dict]:
    if (
        "fiducial" not in caches
        or CM.ZA_SET not in caches
        or CM.ZA_SET not in CM.SCHEMES[scheme]["weights"]
    ):
        return None
    S = summary_sets(caches, blocks)
    means = {n: v.mean(axis=0) for n, v in S.items()}
    frac = np.abs(means[CM.ZA_SET] - means["fiducial"]) / np.maximum(
        np.abs(means["fiducial"]), 1e-30
    )
    d_true = stencil_derivative(means, scheme)
    wrong = dict(means)
    wrong[CM.ZA_SET] = means["fiducial"]
    d_wrong = stencil_derivative(wrong, scheme)
    nrm = np.linalg.norm(d_true)
    return {
        "median_frac_diff_ZA_vs_2LPT": float(np.median(frac)),
        "max_frac_diff_ZA_vs_2LPT": float(frac.max()),
        "ic_artefact_over_signal": float(np.linalg.norm(d_wrong - d_true) / nrm)
        if nrm > 0
        else float("nan"),
    }


def stencil_stability(
    caches: Mapping[str, Mapping],
    blocks: Sequence[tuple],
    hartlap: bool,
    standardize: bool,
) -> dict:
    S = summary_sets(caches, blocks)
    means = {n: v.mean(axis=0) for n, v in S.items() if n != "fiducial"}
    out = {}
    for sch in CM.SCHEMES:
        if not set(CM.SCHEMES[sch]["weights"]) <= set(means):
            continue
        F = fisher_from_means(
            S["fiducial"], means, CM.PARAMS, sch, hartlap, standardize
        )
        out[sch] = float(FC.sigma_from_fisher(F)[-1])
    return out


def bootstrap_noise(
    caches: Mapping[str, Mapping],
    blocks: Sequence[tuple],
    scheme: str,
    hartlap: bool,
    standardize: bool,
    n_boot: int = 100,
    seed: int = 0,
) -> dict:
    S = summary_sets(caches, blocks)
    S_fid = S.pop("fiducial")
    n_common = min((len(v) for v in S.values()))
    S = {n: v[:n_common] for n, v in S.items()}
    cinv, sd, _ = FC.inverse_covariance(S_fid, hartlap, standardize)
    rng = np.random.default_rng(seed)
    Js, sigs = ([], [])
    for _ in range(n_boot):
        idx = rng.integers(0, n_common, n_common)
        J = (
            jacobian({n: v[idx].mean(axis=0) for n, v in S.items()}, CM.PARAMS, scheme)
            / sd[:, None]
        )
        Js.append(J)
        F = J.T @ cinv @ J
        sigs.append(FC.sigma_from_fisher(0.5 * (F + F.T)))
    Js = np.array(Js)
    dJ = Js - Js.mean(axis=0, keepdims=True)
    B = np.einsum("bdi,de,bej->ij", dJ, cinv, dJ) / len(Js)
    Fobs = fisher_from_means(
        S_fid,
        {n: v.mean(axis=0) for n, v in S.items()},
        CM.PARAMS,
        scheme,
        hartlap,
        standardize,
    )
    sigs = np.array(sigs)
    p16, p50, p84 = (np.nanpercentile(sigs, q, axis=0) for q in (16, 50, 84))
    return {
        "n_common": int(n_common),
        "n_boot": int(n_boot),
        "boot_rel_width": {
            p: float((b - a) / max(c, 1e-30))
            for p, a, b, c in zip(CM.PARAMS, p16, p84, p50)
        },
        "noise_frac_of_F_diag": {
            p: float(b / max(f, 1e-300))
            for p, b, f in zip(CM.PARAMS, np.diag(B), np.diag(Fobs))
        },
    }


def load_classical(cdir: Path, sets: Sequence[str], field: str) -> Dict[str, dict]:
    out = {}
    for name in sets:
        f = cdir / f"{name}.npz"
        if not f.exists():
            raise FileNotFoundError(
                f"classical cache missing {f} (preprocess/classical_mnu.py)"
            )
        z = np.load(f, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["meta"] = json.loads(str(d["meta"])) if "meta" in d else {}
        cf = d["meta"].get("field")
        if cf is None:
            raise RuntimeError(
                f"{f}: classical cache carries no `field` tag (written by preprocess/classical_baseline.py, which has no Mnu sets?); rebuild with preprocess/classical_mnu.py"
            )
        if cf != field:
            raise RuntimeError(
                f"--field {field} but {f} holds {cf!r} spectra: comparing them would score a {cf}-field classical mark against a {field}-field P(k) baseline"
            )
        out[name] = d
    return out


def check_same_sims(
    learned: Mapping[str, Mapping],
    classical: Mapping[str, Mapping],
    rtol: float = 0.0001,
) -> None:
    for name in classical:
        a, b = (np.asarray(learned[name]["Pdd"]), np.asarray(classical[name]["Pdd"]))
        if a.shape != b.shape:
            raise RuntimeError(
                f"[{name}] learned cache has {a.shape[0]} sims x {a.shape[1]} bins, classical has {b.shape[0]} x {b.shape[1]}: not the same simulations"
            )
        d = float(np.abs(a.mean(0) / np.maximum(b.mean(0), 1e-30) - 1.0).max())
        if d > rtol:
            raise RuntimeError(
                f"[{name}] unmarked P_dd differs between the learned and classical caches by up to {d:.2e}: not the same boxes / field / estimator"
            )


def classical_best(recs: Mapping[str, dict]) -> dict:
    out: dict = {}
    for name, rec in recs.items():
        for sname, d in rec["improvement"].items():
            for mode in ("marginal", "fixed"):
                if d[mode] is None:
                    continue
                for p, g in zip(d["params"], d[mode]):
                    cur = out.setdefault(sname, {}).setdefault(mode, {}).get(p)
                    if g == g and (cur is None or g > cur["gain"]):
                        out[sname][mode][p] = {"mark": name, "gain": float(g)}
    return out


def mode_geometry(grid: int, box: float) -> Tuple[np.ndarray, np.ndarray]:
    _, _, k_center, n_modes, _, _ = PkEstimator(
        BoxSize=box, device="cpu"
    )._get_geometry(int(grid))
    return (k_center.numpy().astype(np.float64), n_modes.numpy().astype(np.float64))


def uniform_rebin_matrix(
    k_shell: np.ndarray, n_modes: np.ndarray, kmax: float, nbins: int
) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, kmax, nbins + 1)
    idx = np.digitize(k_shell, edges) - 1
    ok = (idx >= 0) & (idx < nbins) & (n_modes > 0)
    M = np.zeros((len(k_shell), nbins))
    M[np.where(ok)[0], idx[ok]] = n_modes[ok]
    tot = M.sum(axis=0)
    empty = tot == 0
    if empty.sum() > max(2, nbins // 10):
        raise ValueError(
            f"{int(empty.sum())} of {nbins} uniform bins in (0, {kmax}] contain no shell (grid too coarse for this binning)"
        )
    if empty.any():
        warnings.warn(
            f"uniform bin(s) {np.where(empty)[0].tolist()} contain no floor-binned shell and are dropped ({nbins - int(empty.sum())} bins used)"
        )
        M, tot = (M[:, ~empty], tot[~empty])
    M /= tot
    return (M, k_shell @ M)


def rebin_cache(cache: Mapping, M: np.ndarray, k_new: np.ndarray) -> dict:
    out = {}
    for key, val in cache.items():
        if key == "k":
            out[key] = k_new
        elif (
            isinstance(val, np.ndarray)
            and val.ndim == 2
            and (val.shape[1] == M.shape[0])
        ):
            out[key] = val @ M
        else:
            out[key] = val
    return out


def _ww_eif(rec: dict) -> Optional[dict]:
    d = rec["improvement"]["min3"]["fixed"]
    return (
        None
        if d is None
        else {p: None if g != g else float(g) for p, g in zip(CM.WW_PARAMS, d)}
    )


def weightedwebs_stage(
    args,
    caches: Dict[str, dict],
    classical: Dict[str, Dict[str, dict]],
    grid: int,
    box: float,
) -> dict:
    scheme = args.mnu_deriv
    standardize = not args.no_standardize
    k_shell, n_modes = mode_geometry(grid, box)
    for name, c in caches.items():
        FC.assert_same_k(
            k_shell, c["k"], atol=1e-05, what=f"(learned cache {name} vs pk geometry)"
        )
    variants = {"literal": (args.ww_n_cov, args.ww_n_deriv), "full": (None, args.n_var)}
    res: dict = {"kmax": {}}
    for kmax, nb in zip(args.kmax, args.ww_bins):
        M, k_new = uniform_rebin_matrix(k_shell, n_modes, kmax, nb)
        cw = {n: rebin_cache(c, M, k_new) for n, c in caches.items()}
        clw = {
            lab: {n: rebin_cache(c, M, k_new) for n, c in cs.items()}
            for lab, cs in classical.items()
        }
        for lab, cs in clw.items():
            check_same_sims(cw, cs)
        mask = np.ones(len(k_new), bool)
        kres = {
            "n_bins": int(len(k_new)),
            "n_bins_nominal": int(nb),
            "k": [float(x) for x in k_new],
            "variants": {},
        }
        for vname, (n_cov, n_der) in variants.items():
            vres = {}
            for hart in (False, True):
                recs: Dict[str, dict] = {}
                specs = {
                    REF: FC.spectrum_blocks("unmarked", mask),
                    "marked": FC.spectrum_blocks("marked", mask),
                    "marked_mm_only": FC.spectrum_blocks("marked_mm_only", mask),
                }
                ref = None
                for name, blocks in specs.items():
                    F, info = fisher_from_caches(
                        cw,
                        blocks,
                        CM.WW_PARAMS,
                        scheme,
                        hart,
                        standardize,
                        n_der,
                        n_cov,
                    )
                    rec = _ww_record(F, info, ref, REF)
                    ref = rec if name == REF else ref
                    recs[name] = rec
                crec: Dict[str, dict] = {}
                for lab, cs in clw.items():
                    Fc, ic = fisher_from_caches(
                        cs,
                        FC.spectrum_blocks("unmarked", mask),
                        CM.WW_PARAMS,
                        scheme,
                        hart,
                        standardize,
                        n_der,
                        n_cov,
                    )
                    cref = _ww_record(Fc, ic, ref, REF)
                    crec[f"{lab}_unmarked"] = cref
                    for mk in classical_mark_keys(cs["fiducial"]):
                        blocks = FC.spectrum_blocks(
                            "marked",
                            mask,
                            keys={"Pmm": f"Pmm__{mk}", "Pdm": f"Pdm__{mk}"},
                        )
                        Fm, im = fisher_from_caches(
                            cs,
                            blocks,
                            CM.WW_PARAMS,
                            scheme,
                            hart,
                            standardize,
                            n_der,
                            n_cov,
                        )
                        crec[f"{lab}__{mk}"] = _ww_record(
                            Fm, im, cref, f"{lab}_unmarked"
                        )
                best = {}
                for lab in clw:
                    marks = {n: r for n, r in crec.items() if n.startswith(f"{lab}__")}
                    for j, p in enumerate(CM.WW_PARAMS):
                        n_best = max(
                            marks,
                            key=lambda n: marks[n]["eif_list"][j]
                            if marks[n]["eif_list"][j] == marks[n]["eif_list"][j]
                            else -1,
                        )
                        best.setdefault(lab, {})[p] = {
                            "mark": n_best,
                            "eif": marks[n_best]["eif_list"][j],
                        }
                    wk = f"{lab}__{CM.WW_WHITE_KEY}"
                    if wk in crec:
                        best[lab]["white_p2_ds0.25"] = {
                            "mark": wk,
                            "eif": crec[wk]["eif_list"],
                        }
                vres[f"hartlap={hart}"] = {
                    "n_fid": int(info["n_fid"]),
                    "n_deriv": n_der,
                    "summaries": recs,
                    "classical": crec,
                    "classical_best": best,
                }
                if hart:
                    _print_ww(kmax, nb, vname, vres[f"hartlap={hart}"])
            kres["variants"][vname] = vres
        res["kmax"][f"{kmax:g}"] = kres
    return res


def _ww_record(F: np.ndarray, info: dict, ref: Optional[dict], ref_name: str) -> dict:
    sig = FC.sigma_from_fisher(F)
    rec = {
        "F": F,
        "sigma": [float(x) for x in sig],
        "dims": info["dims"],
        "ref": ref_name,
    }
    if ref is not None:
        rec["eif_list"] = [float(x) for x in FC.improvement(ref["sigma"], sig)]
    else:
        rec["eif_list"] = [1.0] * len(sig)
    rec["eif"] = dict(zip(CM.WW_PARAMS, rec["eif_list"]))
    return rec


def _print_ww(kmax, nb, vname, blk) -> None:
    print(
        f"\n=== Weighted Webs convention: kmax={kmax:g}, {nb} uniform bins, variant {vname} (n_cov={blk['n_fid']}, n_deriv={blk['n_deriv']}), Hartlap: EIF (Om, s8, Mnu) ==="
    )
    rows = [(n, r["eif_list"], r["dims"]) for n, r in blk["summaries"].items()]
    for lab, b in blk["classical_best"].items():
        for p in CM.WW_PARAMS:
            rows.append(
                (
                    f"best {lab} for {p}: {b[p]['mark']}",
                    [b[p]["eif"] if q == p else float("nan") for q in CM.WW_PARAMS],
                    0,
                )
            )
        if "white_p2_ds0.25" in b:
            rows.append((f"{lab} White(p=2,ds=.25)", b["white_p2_ds0.25"]["eif"], 0))
    print(FC.format_table(rows, CM.WW_PARAMS))
    print(
        "  published (their kmax 0.3, R=10): "
        + "; ".join((f"{k}: {v}" for k, v in list(CM.WW_PUBLISHED.items())[:1]))
    )


def fisher_stage(
    args, caches: Dict[str, dict], classical: Optional[Dict[str, dict]]
) -> dict:
    scheme, standardize = (args.mnu_deriv, not args.no_standardize)
    k = np.asarray(caches["fiducial"]["k"], float)
    for name, c in caches.items():
        FC.assert_same_k(k, c["k"], what=f"(learned cache {name} vs fiducial)")
    if classical is not None:
        for name, c in classical.items():
            FC.assert_same_k(
                k, c["k"], what=f"(classical cache {name} vs learned fiducial)"
            )
        check_same_sims(caches, classical)
    cl_marks = classical_mark_keys(classical["fiducial"]) if classical else []
    results: dict = {"kmax": {}}
    for kmax in args.kmax:
        m0 = FC.kmask(k, args.kmin, kmax)
        if m0.sum() == 0:
            print(f"kmax={kmax}: no k bins in band -> skipped")
            continue
        specs = summary_specs(k, args.kmin, kmax, args.ladder)
        kres: dict = {}
        for hart in (False, True):
            key = f"hartlap={hart}"
            recs: Dict[str, dict] = {}
            ref = None
            for name, blocks in specs.items():
                F, info = fisher_from_caches(
                    caches, blocks, CM.PARAMS, scheme, hart, standardize, args.n_var
                )
                if name == REF:
                    ref = forecast_record(F, info["dims"], None, REF)
                    recs[name] = dict(
                        ref, improvement=improvement_table(ref["sigma"], ref["sigma"])
                    )
                    continue
                recs[name] = forecast_record(F, info["dims"], ref, REF)
            block = {
                "n_bins": int(m0.sum()),
                "n_fid": int(caches["fiducial"]["Pdd"].shape[0]),
                "hartlap_factor": {
                    n: _hartlap_of(caches, specs[n]) for n in (REF, "marked")
                },
                "summaries": recs,
            }
            block["degeneracy"] = {n: degeneracy(recs[n]["F"]) for n in (REF, "marked")}
            block["std5_crosscheck_max_rel_diff"] = {
                n: std5_crosscheck(caches, specs[n], recs[n]["F"], hart, standardize)
                for n in (REF, "marked")
            }
            if classical is not None:
                cref_blocks = FC.spectrum_blocks("unmarked", m0)
                Fc, ic = fisher_from_caches(
                    classical,
                    cref_blocks,
                    CM.PARAMS,
                    scheme,
                    hart,
                    standardize,
                    args.n_var,
                )
                cref = forecast_record(Fc, ic["dims"], ref, REF)
                crecs: Dict[str, dict] = {"classical_unmarked": cref}
                for mk in cl_marks:
                    blocks = FC.spectrum_blocks(
                        "marked", m0, keys={"Pmm": f"Pmm__{mk}", "Pdm": f"Pdm__{mk}"}
                    )
                    Fm, im = fisher_from_caches(
                        classical,
                        blocks,
                        CM.PARAMS,
                        scheme,
                        hart,
                        standardize,
                        args.n_var,
                    )
                    crecs[f"classical__{mk}"] = forecast_record(
                        Fm, im["dims"], cref, "classical_unmarked"
                    )
                block["classical"] = crecs
                block["classical_best"] = classical_best(
                    {n: r for n, r in crecs.items() if n != "classical_unmarked"}
                )
                block["learned_vs_best_classical"] = _learned_vs_classical(
                    recs["marked"], block["classical_best"]
                )
            if args.diagnose:
                block["diagnostics"] = {
                    n: {
                        "stencil_sigma_Mnu": stencil_stability(
                            caches, specs[n], hart, standardize
                        ),
                        "bootstrap": bootstrap_noise(
                            caches,
                            specs[n],
                            scheme,
                            hart,
                            standardize,
                            args.n_boot,
                            args.seed,
                        ),
                    }
                    for n in (REF, "marked")
                }
            kres[key] = block
            _print_block(kmax, key, block, cl_marks)
        ic = {n: ic_check(caches, specs[n], scheme) for n in (REF, "marked")}
        kres["ic_check"] = ic
        if args.diagnose:
            kres["pairing"] = pairing_check(caches, scheme, m0)
            kres["spectra_sanity"] = spectra_sanity(caches, m0)
        results["kmax"][f"{kmax:g}"] = kres
    return results


def _hartlap_of(caches, blocks) -> float:
    n = caches["fiducial"]["Pdd"].shape[0]
    return float(FC.hartlap_factor(n, FC.summary_dims(blocks)))


def _learned_vs_classical(learned_rec: dict, best: dict) -> dict:
    out = {}
    for sname, mode, p in (
        ("min3", "marginal", "Mnu"),
        ("min3", "fixed", "Mnu"),
        ("min3", "marginal", "s8"),
        ("full6", "marginal", "Mnu"),
        ("full6", "marginal", "s8"),
        ("std5", "fixed", "s8"),
    ):
        try:
            d = learned_rec["improvement"][sname]
            g = float(d[mode][d["params"].index(p)])
            b = best[sname][mode][p]
        except (KeyError, TypeError, ValueError):
            continue
        out[f"{sname}.{mode}.{p}"] = {
            "learned": g,
            "best_classical": b["gain"],
            "mark": b["mark"],
            "learned_over_best_classical": g / b["gain"]
            if b["gain"] > 0
            else float("nan"),
        }
    return out


def _print_block(kmax: float, key: str, block: dict, cl_marks: Sequence[str]) -> None:
    views = [
        ("full6", "marginal", CM.PARAM_SETS["full6"]),
        ("min3", "fixed", CM.PARAM_SETS["min3"]),
        ("min3", "marginal", CM.PARAM_SETS["min3"]),
        ("std5", "fixed", CM.PARAM_SETS["std5"]),
        ("std5", "marginal", CM.PARAM_SETS["std5"]),
    ]
    recs = dict(block["summaries"])
    for name, rec in block.get("classical", {}).items():
        recs[name] = rec
    print(
        f"\n=== kmax={kmax:g} ({block['n_bins']} bins, n_fid={block['n_fid']}) {key}: improvement sigma_unmarked/sigma  (classical marks vs their own P_dd) ==="
    )
    for sname, mode, params in views:
        rows = []
        for name, rec in recs.items():
            d = rec.get("improvement", {}).get(sname, {}).get(mode)
            if d is None:
                continue
            rows.append((name, list(d), rec["dims"]))
        if rows:
            print(FC.format_table(rows, params, title=f"  -- {sname} [{mode}] --"))
    for n, d in block["degeneracy"].items():
        print(
            f"  degeneracy[{n}]: corr(s8,Mnu)={d['corr_s8_Mnu']:+.3f}  sigma_marg/sigma_fixed: "
            + "  ".join(
                (f"{p}={v:.2f}" for p, v in d["inflation_marginal_over_fixed"].items())
            )
        )
    print(
        "  std5 cross-check vs fisher_common.forecast (max rel diff): "
        + "  ".join(
            (f"{n}={v:.1e}" for n, v in block["std5_crosscheck_max_rel_diff"].items())
        )
    )
    if "learned_vs_best_classical" in block:
        for kname, v in block["learned_vs_best_classical"].items():
            print(
                f"  learned vs best classical {kname:22s}: {v['learned']:.2f}x vs {v['best_classical']:.2f}x ({v['mark']})"
            )


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
    ap.add_argument("--kmax", type=float, nargs="+", default=DEFAULT_KMAX)
    ap.add_argument("--kmin", type=float, default=0.0)
    ap.add_argument(
        "--ladder",
        type=float,
        nargs="*",
        default=DEFAULT_LADDER,
        help="unmarked P_dd(k<=kmax') ladder reported relative to P_dd(k<=kmax)",
    )
    ap.add_argument(
        "--field",
        choices=list(CM.FIELDS),
        default=CM.DEFAULT_FIELD,
        help="m = total matter, cb = CDM+baryons (df_cb for the Mnu>0 sets, df_m = same field for Mnu=0)",
    )
    ap.add_argument(
        "--mnu_deriv",
        choices=sorted(CM.SCHEMES),
        default=CM.DEFAULT_SCHEME,
        help="forward-difference stencil for dS/dMnu (noza2 needs no fiducial_ZA but extrapolates)",
    )
    ap.add_argument("--n_fid", type=int, default=CF.N_SIMS_FID)
    ap.add_argument(
        "--n_var",
        type=int,
        default=CF.N_SIMS_VAR,
        help="sims per derivative set (incl. the Mnu sets)",
    )
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument(
        "--hartlap",
        action="store_true",
        help="(both Hartlap states are always stored; this selects which one the final summary prints)",
    )
    ap.add_argument(
        "--no-standardize",
        dest="no_standardize",
        action="store_true",
        help="disable column standardization before covariance inversion",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="textbook marked field m(1+d)/A - 1: shorthand for --override mark_normalize=true",
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="patch config.json before building the model (repeatable; JSON value or string)",
    )
    ap.add_argument(
        "--classical",
        type=Path,
        nargs="+",
        default=None,
        help="classical-mark cache dir(s) written by preprocess/classical_mnu.py (same --field); several (e.g. top-hat and Gaussian) only with --convention weightedwebs",
    )
    ap.add_argument(
        "--convention",
        choices=["default", "weightedwebs"],
        default="default",
        help="weightedwebs: the arXiv:2607.26021 convention -- cb field, (Om, s8, Mnu) only, kmax 0.3 in 35 uniform bins, EIF = sigma(P_dd)/sigma, literal (2100 cov / 50 deriv sims) and full sample-size variants",
    )
    ap.add_argument(
        "--ww_bins",
        type=int,
        nargs="+",
        default=None,
        help="uniform bins per --kmax in the weightedwebs convention (default 35 for 0.3; 50 for 0.5)",
    )
    ap.add_argument("--ww_n_cov", type=int, default=CM.WW_N_COV)
    ap.add_argument("--ww_n_deriv", type=int, default=CM.WW_N_DERIV)
    ap.add_argument(
        "--diagnose",
        action="store_true",
        help="add pairing, spectra-sanity, stencil-stability and derivative-noise bootstrap diagnostics",
    )
    ap.add_argument("--n_boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON (default results/fisher_mnu_<run>[...].json)",
    )
    ap.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help=f"cache root (default <run>/{MNU_CACHE_SUBDIR})",
    )
    ap.add_argument("--force", action="store_true", help="recompute cached sets")
    ap.add_argument(
        "--dry-run", action="store_true", help="only list the simulations found per set"
    )
    ap.add_argument(
        "--precompute_only",
        action="store_true",
        help="compute/cache the spectra and stop",
    )
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args(argv)
    a.overrides = FI.parse_overrides(a.override)
    if a.convention == "weightedwebs":
        if a.field != "cb":
            if "--field" in (argv if argv is not None else sys.argv):
                ap.error("--convention weightedwebs is defined on the cb field only")
            a.field = "cb"
        if "--kmax" not in (argv if argv is not None else sys.argv):
            a.kmax = [CM.WW_KMAX]
        if a.ww_bins is None:
            a.ww_bins = [
                CM.WW_BINS
                if abs(k - CM.WW_KMAX) < 1e-09
                else int(round(CM.WW_BINS * k / CM.WW_KMAX))
                for k in a.kmax
            ]
        if len(a.ww_bins) != len(a.kmax):
            ap.error("--ww_bins needs one value per --kmax")
    elif a.classical and len(a.classical) > 1:
        ap.error("several --classical caches need --convention weightedwebs")
    if a.normalize:
        a.overrides["mark_normalize"] = True
    return a


def default_out(args, run_dir: Path) -> Path:
    tag = ("_norm" if args.overrides.get("mark_normalize") else "") + (
        "" if args.field == "m" else f"_{args.field}"
    )
    ww = "ww_" if args.convention == "weightedwebs" else ""
    return (
        ROOT
        / "results"
        / f"fisher_mnu_{ww}{run_dir.name}{tag}{CM.tag_for(args.n_fid, args.n_var)}.json"
    )


def main(argv=None) -> dict:
    args = parse_args(argv)
    run_dir = args.run.resolve()
    cfg = FI.read_run_config(run_dir)
    if args.overrides:
        print("config overrides:")
        cfg = FI.apply_overrides(cfg, args.overrides)
    device = pick_device(args.device)
    mas = FI.mas_of(cfg)
    grid = int(cfg.get("grid_dim", CF.GRID_DIM))
    print(
        f"run: {run_dir}\nmodel: {cfg['_model_class']}  MAS={mas}  grid={grid}  field={args.field}  Mnu stencil={args.mnu_deriv}  device={device}"
    )
    print(
        f"roots: fid={CF.fid_root()}  diff={CF.diff_root()}  mnu={CM.mnu_root()}  za={CM.za_root()}"
    )
    plan = build_plan(args.field, args.mnu_deriv, grid, mas, args.n_fid, args.n_var)
    print_plan(plan)
    if args.dry_run:
        return {}
    check_plan(plan)
    caches = learned_caches(args, cfg, run_dir, device, plan)
    if args.precompute_only:
        print("--precompute_only: spectra cached, stopping before the Fisher.")
        return {}
    classical = None
    if args.classical:
        classical = {}
        for d in args.classical:
            c = load_classical(d.resolve(), list(plan), args.field)
            cm = c["fiducial"]["meta"]
            label = (
                "classical"
                if len(args.classical) == 1
                else f"classical_{cm.get('filter')}"
                + ("_norm" if cm.get("normalize") == "mean" else "")
            )
            if label in classical:
                raise SystemExit(
                    f"two --classical caches resolve to the same label {label!r}"
                )
            classical[label] = c
    if args.convention == "weightedwebs":
        results = weightedwebs_stage(
            args, caches, classical or {}, grid, float(cfg.get("box_size", CF.BOX_SIZE))
        )
    else:
        results = fisher_stage(
            args, caches, classical["classical"] if classical else None
        )
    ck = caches["fiducial"]["meta"]
    results.update(
        {
            "run": str(run_dir),
            "model_class": cfg["_model_class"],
            "mas": mas,
            "code_hash": cfg.get("code_hash"),
            "ckpt_sha": ck.get("ckpt_sha"),
            "field": args.field,
            "mnu_deriv": args.mnu_deriv,
            "mnu_stencil": CM.SCHEMES[args.mnu_deriv],
            "mnu_base_step_eV": CM.MNU_BASE_STEP,
            "mnu_zero_point": CM.ZA_SET
            if CM.ZA_SET in CM.SCHEMES[args.mnu_deriv]["weights"]
            else None,
            "covariance_set": "fiducial (2LPT)",
            "standardize": not args.no_standardize,
            "params": CM.PARAMS,
            "steps": CM.STEPS,
            "param_sets": CM.PARAM_SETS,
            "kmin": args.kmin,
            "ladder": args.ladder,
            "convention": args.convention,
            "classical_cache": [str(d) for d in args.classical]
            if args.classical
            else None,
            "classical_meta": {
                lab: {
                    k: cs["fiducial"]["meta"].get(k)
                    for k in (
                        "filter",
                        "normalize",
                        "R",
                        "ds",
                        "p",
                        "field",
                        "convention",
                    )
                }
                for lab, cs in classical.items()
            }
            if classical
            else None,
            "config_overrides": args.overrides,
            "n_sims": {n: int(c["meta"]["n_sims"]) for n, c in caches.items()},
            "k": [float(x) for x in caches["fiducial"]["k"]],
            "hartlap_requested": bool(args.hartlap),
        }
    )
    out = args.out or default_out(args, run_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(FC.nan_to_none(results), indent=1))
    print(f"\nwrote {out}")
    hk = f"hartlap={bool(args.hartlap)}"
    for kmax, kres in results["kmax"].items():
        if args.convention == "weightedwebs":
            for vname, v in kres["variants"].items():
                eif = v["hartlap=True"]["summaries"]["marked"]["eif"]
                print(
                    f"  kmax={kmax} WW[{vname}] learned marked EIF (Om,s8,Mnu) = "
                    + "  ".join((f"{eif[p]:.2f}" for p in CM.WW_PARAMS))
                )
            continue
        imp = kres[hk]["summaries"]["marked"]["improvement"]
        print(
            f"  kmax={kmax} {hk}: marked/Pk  full6-marginal "
            + "  ".join(
                (
                    f"{p}={g:.2f}" if g is not None else f"{p}=nan"
                    for p, g in zip(CM.PARAMS, imp["full6"]["marginal"])
                )
            )
        )
    return results


if __name__ == "__main__":
    main()
