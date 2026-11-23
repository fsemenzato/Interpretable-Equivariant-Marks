import argparse
import json
import os
from pathlib import Path
import numpy as np
from config import expand_path
from preprocess.classical_baseline import mark_key, parse_mark_key
from . import config_fish as C, fisher_common as F, fisher_pb
from .classical_score import load_cache_dir
from .diagnostics import field_products, responses


def load_spectra(path, n_cov=10000, n_deriv=500):
    result = {}
    for name in C.SETS:
        with np.load(path / f"{name}.npz") as z:
            n = n_cov if name == "fiducial" else n_deriv
            result[name] = {
                key: z[key][:n] if z[key].ndim == 2 else z[key]
                for key in z.files
                if key != "meta"
            }
            F.assert_same_k(result[name]["k"], result["fiducial"]["k"])
            if len(result[name]["Pdd"]) != n:
                raise ValueError(f"{path}/{name}: expected {n} realizations")
    return result


def forecast(cache, cut, keys=("Pdd", "Pdm", "Pmm")):
    mask = F.kmask(cache["fiducial"]["k"], 0, cut)
    return F.forecast(
        cache,
        [("spectrum", q, mask) for q in keys],
        C.STEPS,
        hartlap=True,
        params=C.PARAMS,
    )


def family(classical, learned, regression, run_names, cut, out):
    records = classical["kmax"][str(cut)]["hartlap=True"]["marks"]
    sigmas = np.array([forecast(c, cut)["sigma"] for c in learned])
    cells = []
    fish = []
    reg = []
    for key, record in records.items():
        _, ds, p = key.split("_")
        cells.append((round(float(ds[1:]), 2), -float(p[1:])))
        fish.append(np.max(np.asarray(record["sigma"]) / sigmas, axis=0))
        regression_key = mark_key(*parse_mark_key(key))
        row = []
        for param in ["s8", "Om"]:
            mse = np.asarray(regression[regression_key][param]["raw"][0])
            neural = np.array(
                [regression["learned:" + run][param]["raw"][0] for run in run_names]
            )
            row.append(np.max(np.mean(np.sqrt(mse / neural), axis=1)))
        reg.append(row)
    np.savez_compressed(out / "family.npz", cells=cells, fisher=fish, regression=reg)


def constraints(cfg, out):
    cutgrid = np.asarray(cfg["profile_cuts"])
    scan = np.linspace(0.05, 0.4, 50)
    models = {
        name: [load_spectra(Path(p) / "fisher_cache") for p in cfg[name + "_runs"]]
        for name in ["hessian", "web"]
    }
    values = {
        name: np.array([[forecast(c, k)["sigma"] for k in cutgrid] for c in caches])
        for name, caches in models.items()
    }
    references = {
        name: np.array(
            [[forecast(c, k, ("Pdd",))["sigma"] for k in cutgrid] for c in caches]
        )
        for name, caches in models.items()
    }
    classical = load_cache_dir(Path(cfg["classical_cache"]))
    reference = json.loads(Path(cfg["classical_fisher"]).read_text())["kmax"]["0.2"][
        "hartlap=True"
    ]["marks"]
    selected = max(reference, key=lambda key: reference[key]["fom"])
    classical_sigma = np.array(
        [
            forecast(classical, k, ("Pdd", "Pdm__" + selected, "Pmm__" + selected))[
                "sigma"
            ]
            for k in cutgrid
        ]
    )
    hm, web = (cfg["hessian_index"], cfg["web_index"])
    power = Path(cfg["hessian_runs"][hm]) / "fisher_cache"
    bcache = Path(cfg["bispectrum_cache"])
    pb = np.array([fisher_pb.forecast(bcache, power, k)["sigma"] for k in scan])
    common = load_spectra(power, n_cov=8000)
    np.savez_compressed(
        out / "constraints.npz",
        kgrid=cutgrid,
        kscan=scan,
        pb_scan=pb,
        hm_index=hm,
        web_index=web,
        pk=references["hessian"][hm],
        free=values["hessian"],
        web=values["web"],
        classical_fom=classical_sigma,
        F_pk=forecast(common, 0.3, ("Pdd",))["F"],
        F_hm=forecast(common, 0.3)["F"],
        F_pb=fisher_pb.forecast(bcache, power, 0.3)["F"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--stage",
        choices=["diagnostics", "constraints", "family", "render"],
        required=True,
    )
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    for key in [
        "output",
        "figures",
        "field_root",
        "classical_cache",
        "classical_fisher",
        "bispectrum_cache",
        "regression",
    ]:
        cfg[key] = expand_path(cfg[key], a.config.resolve().parent)
    for key in ["hessian_runs", "web_runs"]:
        cfg[key] = [expand_path(value, a.config.resolve().parent) for value in cfg[key]]
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "diagnostics":
        import torch

        torch.set_num_threads(2)
        paths = [
            Path(cfg["field_root"]) / str(i) / C.data_name() for i in cfg["field_ids"]
        ]
        hm = Path(cfg["hessian_runs"][cfg["hessian_index"]])
        web = Path(cfg["web_runs"][cfg["web_index"]])
        field_products(web, hm, paths, out, cfg.get("device", "cpu"))
        responses(hm / "fisher_cache", out)
    elif a.stage == "constraints":
        constraints(cfg, out)
    elif a.stage == "family":
        runs = [Path(r) for r in cfg["hessian_runs"]]
        family(
            json.loads(Path(cfg["classical_fisher"]).read_text()),
            [load_spectra(r / "fisher_cache") for r in runs],
            json.loads(Path(cfg["regression"]).read_text()),
            [r.name for r in runs],
            0.2,
            out,
        )
    else:
        from importlib import import_module
        from .figures import style

        style.FIG = Path(cfg["figures"])
        style.FIG.mkdir(parents=True, exist_ok=True)
        style.LOGS = style.FIG / "logs"
        style.apply()
        for name in [
            "tidal",
            "schematics",
            "slices",
            "responses",
            "constraints",
            "family",
        ]:
            import_module("postprocess.figures." + name).render(out, style.FIG)


if __name__ == "__main__":
    main()
