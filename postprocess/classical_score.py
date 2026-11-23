from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from postprocess import config_fish as CF
from postprocess import fisher_common as FC


def load_cache_dir(cdir: Path) -> Dict[str, dict]:
    out = {}
    for name in CF.SETS:
        f = cdir / f"{name}.npz"
        if not f.exists():
            raise FileNotFoundError(f"missing {f}")
        z = np.load(f, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["meta"] = json.loads(str(d["meta"])) if "meta" in d else {}
        out[name] = d
    return out


def score(caches: Dict[str, dict], kmaxes, kmin: float = 0.0) -> dict:
    params, steps = (CF.PARAMS, CF.STEPS)
    k = np.asarray(caches["fiducial"]["k"], float)
    for name, c in caches.items():
        FC.assert_same_k(k, c["k"], what=f"(classical cache {name} vs fiducial)")
    keys = sorted(
        (kk[len("Pmm__") :] for kk in caches["fiducial"] if kk.startswith("Pmm__"))
    )
    s8 = params.index("s8")
    res = {
        "params": params,
        "steps": steps,
        "meta": caches["fiducial"]["meta"],
        "kmax": {},
    }
    for kmax in kmaxes:
        m = FC.kmask(k, kmin, kmax)
        if m.sum() == 0:
            print(f"kmax={kmax}: no bins -> skipped")
            continue
        kres = {}
        for hart in (False, True):
            ref = FC.forecast(
                caches,
                FC.spectrum_blocks("unmarked", m),
                steps,
                hartlap=hart,
                params=params,
            )
            recs, rows = ({}, [])
            for mk in keys:
                blocks = FC.spectrum_blocks(
                    "marked", m, keys={"Pmm": f"Pmm__{mk}", "Pdm": f"Pdm__{mk}"}
                )
                r = FC.forecast(caches, blocks, steps, hartlap=hart, params=params)
                recs[mk] = FC.gain_record(r, ref, "unmarked")
                rows.append((mk, recs[mk]["improvement"], recs[mk]["dims"]))
            finite = {
                mk: v
                for mk, v in recs.items()
                if v["improvement"][s8] is not None
                and np.isfinite(v["improvement"][s8])
            }
            best_s8 = (
                max(finite, key=lambda mk: finite[mk]["improvement"][s8])
                if finite
                else None
            )
            fin_fom = {
                mk: v for mk, v in recs.items() if np.isfinite(v["fom_improvement"])
            }
            best_fom = (
                max(fin_fom, key=lambda mk: fin_fom[mk]["fom_improvement"])
                if fin_fom
                else None
            )
            key = f"hartlap={hart}"
            kres[key] = {
                "n_bins": int(m.sum()),
                "n_fid": ref["n_fid"],
                "unmarked_sigma": [float(x) for x in ref["sigma"]],
                "marks": recs,
                "best_by_s8": best_s8,
                "best_by_fom": best_fom,
            }
            print(
                FC.format_table(
                    rows,
                    params,
                    title=f"\n=== classical grid, kmax={kmax:g} ({int(m.sum())} bins, n_fid={ref['n_fid']}) {key}: sigma_unmarked/sigma ===",
                )
            )
            if best_s8:
                print(
                    f"  best by sigma8: {best_s8}  s8 x{finite[best_s8]['improvement'][s8]:.2f}"
                )
            if best_fom:
                print(
                    f"  best by FoM   : {best_fom}  FoM x{fin_fom[best_fom]['fom_improvement']:.2f}"
                )
        res["kmax"][f"{kmax:g}"] = kres
    return res


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--kmax", type=float, nargs="+", default=[0.2, 0.3])
    ap.add_argument("--kmin", type=float, default=0.0)
    ap.add_argument(
        "--out", type=Path, default=None, help="default <cache>/classical_score.json"
    )
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    caches = load_cache_dir(args.cache.resolve())
    res = score(caches, args.kmax, args.kmin)
    res["cache"] = str(args.cache.resolve())
    out = args.out or args.cache.resolve() / "classical_score.json"
    out.write_text(json.dumps(FC.nan_to_none(res), indent=1))
    print(f"\nwrote {out}")
    return res


if __name__ == "__main__":
    main()
