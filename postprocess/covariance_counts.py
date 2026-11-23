from pathlib import Path
import argparse, json, hashlib
import numpy as np
from . import fisher_common as F, config_fish as C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--counts", type=int, nargs="+", default=[8000, 10000])
    p.add_argument("--kmax", type=float, nargs="+", default=[0.2, 0.3])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    raw = {}
    sources = {}
    rows = []
    for name in C.SETS:
        path = a.cache / (name + ".npz")
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            raw[name] = {k: z[k].copy() for k in ["k", "Pdd", "Pdm", "Pmm"]}
        np.testing.assert_array_equal(raw[name]["k"], raw["fiducial"]["k"])
    for cut in a.kmax:
        mask = raw["fiducial"]["k"] <= cut + 1e-09
        for count in a.counts:
            if count > len(raw["fiducial"]["Pdd"]):
                raise ValueError("requested count exceeds fiducial cache")
            result = {}
            for label, keys in [("P", ["Pdd"]), ("marked", ["Pdd", "Pdm", "Pmm"])]:
                x = {
                    n: np.concatenate([v[q][:, mask] for q in keys], axis=1)
                    for n, v in raw.items()
                }
                result[label] = F.fisher_sigma(
                    x["fiducial"][:count],
                    {p: x[p + "_p"] for p in C.PARAMS},
                    {p: x[p + "_m"] for p in C.PARAMS},
                    C.STEPS,
                    hartlap=True,
                ).tolist()
            rows.append(
                dict(
                    Ncov=count,
                    kmax=cut,
                    **result,
                    gain=(np.array(result["P"]) / result["marked"]).tolist()
                )
            )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(dict(params=C.PARAMS, sources=sources, rows=rows), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
