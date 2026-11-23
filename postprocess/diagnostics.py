import argparse
from pathlib import Path
import numpy as np
import torch
from .fisher import build_model, read_run_config
from .config_fish import STEPS

from marks._mark_base import ENVIRONMENTS as ENVIRONMENTS


@torch.no_grad()
def fields(run, paths, device="cpu"):
    model = build_model(read_run_config(run), run, torch.device(device))
    result = {
        k: []
        for k in ["delta", "filtered", "mark", "final", "gates", "eigenvalues", "terms"]
    }
    for path in paths:
        delta = torch.from_numpy(np.load(path).astype("float32"))[None, None].to(device)
        out = model(delta)
        values = {
            "delta": delta,
            "filtered": out["delta_f"],
            "mark": out["mark_field"],
            "final": out["final_field"],
            "gates": torch.cat([out["w_" + e] for e in ENVIRONMENTS], 1),
            "eigenvalues": torch.cat([out["lam" + str(i)] for i in [1, 2, 3]], 1),
            "terms": torch.cat([out["expert_" + e] for e in ENVIRONMENTS], 1),
        }
        for key, value in values.items():
            result[key].append(value[0].cpu().numpy())
    return {key: np.stack(value) for key, value in result.items()}


def field_products(web_run, hessian_run, paths, out, device="cpu"):
    out.mkdir(parents=True, exist_ok=True)
    for label, run in [("web", web_run), ("hessian", hessian_run)]:
        x = fields(run, paths, device)
        sl = x["delta"].shape[-1] // 2
        image = lambda a: a[0, ..., sl].swapaxes(-1, -2)
        ld = np.log1p(image(x["delta"])[0])
        mark = image(x["mark"])[0]
        final = image(x["final"])[0]
        w = image(x["gates"])
        terms = image(x["terms"])
        tag = run.name
        if label == "web":
            occ = x["gates"].mean(axis=(0, 2, 3, 4))
            np.savez_compressed(
                out / "web_decomposition.npz", ld=ld, w0=w, prod0=terms, occ0=occ
            )
            np.savez_compressed(
                out / "web_slices.npz", ld=ld, M=mark, D=final, _tag=tag
            )
            np.savez_compressed(out / "tidal.npz", ld=ld, lam=image(x["eigenvalues"]))
        else:
            np.savez_compressed(
                out / "hessian_decomposition.npz", ld=ld, w=w, prod=terms, _tag=tag
            )
            filtered = x["filtered"].ravel().astype(float)
            m = x["mark"].ravel().astype(float)
            edges = np.linspace(*np.quantile(filtered, [0.005, 0.995]), 51)
            idx = np.digitize(filtered, edges) - 1
            curve = []
            for i in range(50):
                mask = idx == i
                if mask.sum() >= 200:
                    curve.append(
                        [
                            filtered[mask].mean(),
                            m[mask].mean(),
                            *np.percentile(m[mask], [16, 84]),
                        ]
                    )
            xc, mu, lo, hi = np.array(curve).T
            hist, he = np.histogram(filtered, bins=edges, density=True)
            np.savez_compressed(
                out / "hessian_profile.npz",
                ld=ld,
                M=mark,
                D=final,
                xc=xc,
                mu=mu,
                lo=lo,
                hi=hi,
                hist=hist,
                he=he,
                _tag=tag,
            )


def responses(cache, out, n_boot=200):
    keys = ["Pdd", "Pmm", "Pdm"]

    def load(name):
        with np.load(cache / f"{name}.npz") as z:
            return ({q: z[q].astype(float) for q in keys}, z["k"])

    fid, k = load("fiducial")
    mean = {q: fid[q].mean(0) for q in keys}
    result = {"k0": k, **{q + "0": mean[q] for q in keys}}
    variations = {
        name: load(name)[0] for p in ["Om", "s8"] for name in [p + "_p", p + "_m"]
    }
    bands = {(p, q): [] for p in ["Om", "s8"] for q in ["Pdd", "Pmm"]}
    for p, q in bands:
        result[f"D_{p}_{q}0"] = (
            variations[p + "_p"][q].mean(0) - variations[p + "_m"][q].mean(0)
        ) / (2 * STEPS[p] * mean[q])
    rng = np.random.RandomState(0)
    n = len(variations["Om_p"]["Pdd"])
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        for p, q in bands:
            bands[p, q].append(
                (
                    variations[p + "_p"][q][idx].mean(0)
                    - variations[p + "_m"][q][idx].mean(0)
                )
                / (2 * STEPS[p] * mean[q])
            )
    for (p, q), values in bands.items():
        result[f"B_{p}_{q}"] = np.percentile(values, [16, 84], axis=0)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "responses.npz", **result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--web-run", type=Path)
    p.add_argument("--hessian-run", type=Path)
    p.add_argument("--fields", type=Path, nargs="+")
    p.add_argument("--spectra-cache", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.fields:
        if not a.web_run or not a.hessian_run:
            p.error("Both run paths are required for field diagnostics")
        field_products(a.web_run, a.hessian_run, a.fields, a.out, a.device)
    if a.spectra_cache:
        responses(a.spectra_cache, a.out)


if __name__ == "__main__":
    main()
