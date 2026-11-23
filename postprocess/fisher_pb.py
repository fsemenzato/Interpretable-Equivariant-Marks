import argparse
import json
from pathlib import Path
import numpy as np
from . import config_fish as C, fisher_common as F


def precompute(tables, out, n_fid=8000, n_var=500):
    out.mkdir(parents=True, exist_ok=True)
    for name in C.SETS:
        count = n_fid if name == "fiducial" else n_var
        folder = name.replace("Ob_", "Ob2_")
        rows = []
        tri = None
        for i in range(count):
            values = np.loadtxt(tables / folder / str(i) / "Bk_m_z=0.00.txt")
            if values.ndim != 2 or values.shape[1] != 8:
                raise ValueError("Expected eight columns in the Quijote Bk table")
            current = values[:, :3].astype(int)
            if tri is None:
                tri = current
            np.testing.assert_array_equal(current, tri)
            rows.append(values[:, 6])
        np.savez_compressed(
            out / f"{folder}_pb.npz", tri=tri, B=rows, ids=np.arange(count)
        )


def summaries(bispectrum, power, cut, n_fid=8000, n_var=500, box=1000.0):
    result = {}
    triangles = None
    k = None
    for name in C.SETS:
        count = n_fid if name == "fiducial" else n_var
        with np.load(
            bispectrum / f"{name.replace('Ob_', 'Ob2_')}_pb.npz"
        ) as b, np.load(power / f"{name}.npz") as p:
            if len(b["B"]) < count or len(p["Pdd"]) < count:
                raise ValueError(f"Insufficient realizations for {name}")
            np.testing.assert_array_equal(b["ids"][:count], np.arange(count))
            if "meta" in p.files:
                meta = json.loads(str(p["meta"]))
                bounds = meta.get("sim_ids_first_last")
                if bounds and bounds != [0, len(p["Pdd"]) - 1]:
                    raise ValueError(
                        f"{name}: power and bispectrum realization IDs differ"
                    )
            if triangles is None:
                triangles = b["tri"]
                k = p["k"]
            np.testing.assert_array_equal(b["tri"], triangles)
            F.assert_same_k(p["k"], k)
            pmask = F.kmask(k, 0, cut)
            bmask = np.all(triangles * (2 * np.pi / box) <= cut + 1e-09, axis=1)
            result[name] = np.concatenate(
                [p["Pdd"][:count, pmask], b["B"][:count, bmask]], axis=1
            )
    return result


def forecast(bispectrum, power, cut, n_fid=8000, n_var=500, box=1000.0):
    x = summaries(bispectrum, power, cut, n_fid, n_var, box)
    matrix = F.fisher_matrix(
        x["fiducial"],
        {p: x[p + "_p"] for p in C.PARAMS},
        {p: x[p + "_m"] for p in C.PARAMS},
        C.STEPS,
        hartlap=True,
    )
    return {
        "F": matrix.tolist(),
        "sigma": F.sigma_from_fisher(matrix).tolist(),
        "Ncov": len(x["fiducial"]),
        "dimension": x["fiducial"].shape[1],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["precompute", "fisher"], default="fisher")
    p.add_argument("--tables", type=Path)
    p.add_argument("--bispectrum-cache", type=Path, required=True)
    p.add_argument("--power-cache", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--kmax", type=float, nargs="+", default=[0.2, 0.3, np.pi * 128 / 1000]
    )
    p.add_argument("--n-fid", type=int, default=8000)
    p.add_argument("--n-var", type=int, default=500)
    a = p.parse_args()
    if a.stage == "precompute":
        if a.tables is None:
            p.error("--tables is required")
        precompute(a.tables, a.bispectrum_cache, a.n_fid, a.n_var)
    else:
        if a.power_cache is None or a.out is None:
            p.error("--power-cache and --out are required")
        results = {
            str(k): forecast(a.bispectrum_cache, a.power_cache, k, a.n_fid, a.n_var)
            for k in a.kmax
        }
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(
            json.dumps({"params": C.PARAMS, "kmax": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
