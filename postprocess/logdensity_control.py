from pathlib import Path
import sys, json, time, hashlib, argparse
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pk import PkEstimator
from postprocess import config_fish as CF, fisher_common as FC
from postprocess.simdata import make_loader


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reference-cache",
        type=Path,
        required=True,
        help="Production baseline cache containing Pdd and Pgg for every cosmology",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    OUT = args.out
    BASE = args.reference_cache
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    est = PkEstimator(BoxSize=1000.0, device=device)
    caches = {}
    checks = {}
    t0 = time.time()
    for name in CF.SETS:
        n = 10000 if name == "fiducial" else 500
        ids = list(range(n))
        paths = [CF.set_root(name) / str(i) / CF.data_name(128, "PCS") for i in ids]
        print("starting", name, n, "boxes", flush=True)
        assert ids == list(range(n)), (name, len(paths), ids[:3], ids[-3:])
        old = load(BASE / f"{name}.npz")
        meta = json.loads(str(old["meta"]))
        assert meta["n_sims"] == n and meta["sim_ids_first_last"] == [0, n - 1]
        assert meta["mas"] == "PCS" and meta["baselines"]
        dst = OUT / f"{name}.npz"
        if dst.exists():
            c = load(dst)
            assert json.loads(str(c["meta"]))["control_version"] == "log_cross_v1"
        else:
            acc = {k: [] for k in ["Pdd", "Pgg", "Pdg"]}
            clamped = 0
            voxels = 0
            k = None
            for ib, d in enumerate(make_loader(paths, 16, 8, device)):
                d = d.to(device, non_blocking=True)
                clamped += int((d < -0.999).sum().item())
                voxels += d.numel()
                g = torch.log1p(d.clamp(min=-0.999))
                g -= g.mean(dim=(-3, -2, -1), keepdim=True)
                pk = est.compute(d, g, MAS=("PCS", None)).cpu().numpy()
                if k is None:
                    k = pk[0, :, 0].astype(np.float64)
                for key, j in [("Pdd", 1), ("Pgg", 2), ("Pdg", 3)]:
                    acc[key].append(pk[:, :, j])
                if (ib + 1) % 20 == 0:
                    print(
                        name,
                        min((ib + 1) * 16, n),
                        "/",
                        n,
                        "elapsed",
                        round(time.time() - t0),
                        flush=True,
                    )
            c = {key: np.concatenate(v) for key, v in acc.items()}
            c["k"] = k
            c["meta"] = json.dumps(
                dict(
                    control_version="log_cross_v1",
                    n_sims=n,
                    ids=ids,
                    grid=128,
                    box=1000.0,
                    MAS=["PCS", None],
                    log_definition="log1p(clamp(delta,min=-0.999)), mean-subtracted",
                    clamped_voxels=clamped,
                    total_voxels=voxels,
                    baseline_meta=meta,
                )
            )
            np.savez_compressed(dst, **c)
        assert c["Pdg"].shape[0] == n
        FC.assert_same_k(c["k"], old["k"])
        validation = {}
        for key in ["Pdd", "Pgg"]:
            assert c[key].shape == old[key].shape
            err = float(
                np.max(np.abs(c[key] - old[key]) / np.maximum(np.abs(old[key]), 1e-20))
            )
            validation[key + "_max_relative_difference"] = err
            assert np.allclose(c[key], old[key], rtol=5e-05, atol=1e-07), (
                name,
                key,
                err,
            )
        bound = np.sqrt(np.maximum(c["Pdd"] * c["Pgg"], 0))
        excess = float(np.max(np.abs(c["Pdg"]) / np.maximum(bound, 1e-20)))
        assert excess <= 1.0001, excess
        validation["max_abs_cross_correlation"] = excess
        checks[name] = validation
        caches[name] = c
        print("verified", name, validation, flush=True)
    results = dict(
        params=CF.PARAMS,
        steps=CF.STEPS,
        n_sims={s: len(c["Pdd"]) for s, c in caches.items()},
        checks=checks,
        kmax={},
        source_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                Path(__file__),
                ROOT / "pk.py",
                ROOT / "postprocess/fisher_common.py",
                ROOT / "postprocess/config_fish.py",
            ]
        },
    )
    for cut in [0.2, 0.3]:
        m = FC.kmask(caches["fiducial"]["k"], 0.0, cut)
        specs = {
            label: [("spectrum", key, m) for key in keys]
            for label, keys in [
                ("Pdd", ["Pdd"]),
                ("Pdd_Pgg", ["Pdd", "Pgg"]),
                ("Pdd_Pgg_Pdg", ["Pdd", "Pgg", "Pdg"]),
            ]
        }
        out = {}
        for hart in [False, True]:
            ref = FC.forecast(
                caches, specs["Pdd"], CF.STEPS, hartlap=hart, params=CF.PARAMS
            )
            out[str(hart)] = {}
            for label, blocks in specs.items():
                r = FC.forecast(
                    caches, blocks, CF.STEPS, hartlap=hart, params=CF.PARAMS
                )
                rec = FC.gain_record(r, ref, "Pdd")
                rec["F"] = r["F"].tolist()
                S = FC.build_summary(caches["fiducial"], blocks)
                sd = S.std(axis=0)
                C = np.cov(S / np.where(sd > 0, sd, 1), rowvar=False)
                ev = np.linalg.eigvalsh(C)
                rec["covariance_rank"] = int(np.sum(ev > ev[-1] * 1e-10))
                rec["covariance_condition"] = float(ev[-1] / ev[0])
                rec["hartlap_factor"] = (
                    FC.hartlap_factor(len(S), S.shape[1]) if hart else 1.0
                )
                sensitivity = {}
                for rc in [1e-08, 1e-10, 1e-12]:
                    FC.RCOND = rc
                    sensitivity[str(rc)] = FC.forecast(
                        caches, blocks, CF.STEPS, hartlap=hart, params=CF.PARAMS
                    )["sigma"].tolist()
                FC.RCOND = 1e-10
                rec["rcond_sigmas"] = sensitivity
                out[str(hart)][label] = rec
                print(
                    "RESULT",
                    cut,
                    hart,
                    label,
                    rec["improvement"],
                    rec["sigma"],
                    flush=True,
                )
        results["kmax"][str(cut)] = out
    (OUT / "results.json").write_text(json.dumps(FC.nan_to_none(results), indent=2))
    print("COMPLETE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
