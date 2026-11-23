from pathlib import Path
import argparse, hashlib, json
import numpy as np, torch
from .fisher import read_run_config, build_model
from pk import PkEstimator


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--fields", type=Path, nargs="+", required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--device", default="cpu")
    a.add_argument("--threads", type=int, default=2)
    a.add_argument("--slice", type=int, default=None)
    a.add_argument("--full-fields", action="store_true")
    args = a.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    cfg = read_run_config(args.run)
    m = build_model(cfg, args.run, device)
    args.out.mkdir(parents=True, exist_ok=True)
    est = PkEstimator(cfg.get("box_size", 1000), device=device)
    manifest = {
        "config": {
            "path": str(args.run / "config.json"),
            "sha256": sha(args.run / "config.json"),
        },
        "checkpoint": {
            "path": str(args.run / "best_checkpoint.pt"),
            "sha256": sha(args.run / "best_checkpoint.pt"),
        },
        "sources": [],
        "outputs": [],
        "configuration": cfg,
        "torch": torch.__version__,
    }
    for i, p in enumerate(args.fields):
        d = torch.from_numpy(np.load(p).astype("float32")).to(device)[None, None]
        n = d.shape[-1]
        sl = args.slice if args.slice is not None else n // 2
        if not 0 <= sl < n:
            raise ValueError("slice outside grid")
        with torch.no_grad():
            out = m(d)
            pk = est.compute(
                d, out["final_field"], MAS=(cfg.get("mas_scheme", "PCS"),) * 2
            )
            values = {"pk": pk.cpu().numpy(), "slice_index": np.array(sl)}
            for key, v in {"delta": d, **out}.items():
                if not isinstance(v, torch.Tensor):
                    continue
                v = v.detach().cpu().numpy()
                if v.ndim == 5:
                    values[key + "_slice"] = v[0, ..., sl]
                    values[key + "_mean"] = v.mean(axis=(-3, -2, -1))
                    values[key + "_std"] = v.std(axis=(-3, -2, -1))
                    if args.full_fields:
                        values[key] = v
                else:
                    values[key] = v
            if hasattr(m, "density_experts"):
                x = torch.linspace(-4, 4, 401, device=device).reshape(1, 1, 401, 1, 1)
                values["expert_x"] = x.flatten().cpu().numpy()
                values["expert_y"] = np.stack(
                    [f(x).flatten().cpu().numpy() for f in m.density_experts]
                )
        dest = args.out / f"field_{i:04d}.npz"
        np.savez_compressed(dest, **values)
        manifest["sources"].append(
            {"path": str(p.resolve()), "sha256": sha(p), "slice": sl}
        )
        manifest["outputs"].append({"path": dest.name, "sha256": sha(dest)})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
