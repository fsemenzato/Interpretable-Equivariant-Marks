import sys
from pathlib import Path
import numpy as np
import torch
from postprocess import config_fish as CF
from postprocess import fisher as FI

VARIANTS = ("none", "box_mean", "fid_const", "box_scale", "roll_half")
CALIB = 16
FID_FILE = "df_m_128_PCS_z=0.npy"


def calibration_boxes(n):
    root = Path(CF.fid_root())
    ids = sorted(
        (
            int(p.name)
            for p in root.iterdir()
            if p.name.isdigit() and (p / FID_FILE).exists()
        )
    )[:n]
    if len(ids) != n:
        raise ValueError(f"Expected {n} calibration fields, found {len(ids)} in {root}")
    return [np.load(root / str(i) / FID_FILE).astype(np.float32) for i in ids]


def gate_wrapped_builder(orig_build):
    def build(cfg, run_dir, device):
        model = orig_build(cfg, run_dir, device)
        variant = cfg.get("gate_diag", "none")
        if variant not in VARIANTS:
            raise SystemExit(f"gate_diag must be one of {VARIANTS}; got {variant!r}")
        if variant == "roll_half" and int(cfg.get("gate_input_roll", 0)) != 64:
            raise SystemExit(
                "gate_diag=roll_half needs --override gate_input_roll=64 as well"
            )
        if variant in ("none", "roll_half"):
            print(
                f"gate_diag: {variant} (model unchanged apart from config overrides)",
                flush=True,
            )
            return model
        gate, orig_gate_fwd, orig_gate_input = (
            model.web_gate,
            model.web_gate.forward,
            model.gate_input,
        )
        boxes = calibration_boxes(CALIB)
        with torch.no_grad():
            stats = []
            for d in boxes:
                eigs, _ = model.eig_extractor(
                    model.hessian(torch.from_numpy(d)[None, None].to(device))
                )
                w = orig_gate_fwd(orig_gate_input(eigs))
                stats.append(
                    (w.mean(dim=(0, 2, 3, 4)).cpu().numpy(), float(eigs.std()))
                )
        w_fid = np.mean([s[0] for s in stats], axis=0)
        s_fid = float(np.mean([s[1] for s in stats]))
        print(
            f"gate_diag: {variant}; calibration on {len(boxes)} fiducial boxes: mean gates {np.round(w_fid, 4).tolist()}, pooled eigenvalue std {s_fid:.4f} (fixed scale s = {model.gate_eig_scale})",
            flush=True,
        )
        if variant == "box_mean":
            gate.forward = (
                lambda x: orig_gate_fwd(x)
                .mean(dim=(2, 3, 4), keepdim=True)
                .expand(-1, -1, *x.shape[2:])
            )
        elif variant == "fid_const":
            c = torch.tensor(w_fid, dtype=torch.float32, device=device).view(
                1, 4, 1, 1, 1
            )
            gate.forward = lambda x: c.expand(x.shape[0], -1, *x.shape[2:]).to(x.dtype)
        elif variant == "box_scale":
            model.gate_input = lambda eigs: orig_gate_input(
                eigs * (s_fid / eigs.std(dim=(1, 2, 3, 4), keepdim=True))
            )
        return model

    return build


if __name__ == "__main__":
    FI.build_model = gate_wrapped_builder(FI.build_model)
    FI.main(sys.argv[1:])
