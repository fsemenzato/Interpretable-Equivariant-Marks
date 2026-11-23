from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Sequence

PARAMS: List[str] = ["Om", "Ob", "h", "ns", "s8"]
PARAM_INFO: Dict[str, Dict[str, object]] = {
    "Om": {"folder": "Om", "step": 0.01, "fiducial": 0.3175, "tex": "\\Omega_m"},
    "Ob": {"folder": "Ob2", "step": 0.002, "fiducial": 0.049, "tex": "\\Omega_b"},
    "h": {"folder": "h", "step": 0.02, "fiducial": 0.6711, "tex": "h"},
    "ns": {"folder": "ns", "step": 0.02, "fiducial": 0.9624, "tex": "n_s"},
    "s8": {"folder": "s8", "step": 0.015, "fiducial": 0.834, "tex": "\\sigma_8"},
}
assert list(PARAM_INFO) == PARAMS
STEPS: Dict[str, float] = {p: float(PARAM_INFO[p]["step"]) for p in PARAMS}
FIDUCIAL: Dict[str, float] = {p: float(PARAM_INFO[p]["fiducial"]) for p in PARAMS}
SIDES = ("p", "m")
SETS: List[str] = ["fiducial"] + [f"{p}_{s}" for p in PARAMS for s in SIDES]
N_SIMS_FID = 10000
N_SIMS_VAR = 500
GRID_DIM = 128
BOX_SIZE = 1000.0
DEFAULT_MAS = "PCS"
DELTA_FID_ROOT = Path("data/quijote_z0_fid")
DELTA_DIFF_ROOT = Path("data/quijote_z0_diffs")
LOCAL_FID_ROOT = Path("data/quijote_z0_fid")
LOCAL_DIFF_ROOT = Path("data/quijote_z0_diffs")
ENV_FID = "QUIJOTE_FID_ROOT"
ENV_DIFF = "QUIJOTE_DIFF_ROOT"


def fid_root() -> Path:
    return Path(os.environ.get(ENV_FID, DELTA_FID_ROOT))


def diff_root() -> Path:
    return Path(os.environ.get(ENV_DIFF, DELTA_DIFF_ROOT))


def data_name(grid: int = GRID_DIM, mas: str = DEFAULT_MAS) -> str:
    return f"df_m_{int(grid)}_{mas}_z=0.npy"


def split_set(name: str):
    if name == "fiducial":
        return (None, None)
    param, side = name.rsplit("_", 1)
    if param not in PARAM_INFO or side not in SIDES:
        raise KeyError(f"unknown set {name!r}; known: {SETS}")
    return (param, side)


def set_root(name: str) -> Path:
    param, side = split_set(name)
    if param is None:
        return fid_root()
    return diff_root() / f"{PARAM_INFO[param]['folder']}_{side}"


def default_n_sims(name: str, n_fid: int = N_SIMS_FID, n_var: int = N_SIMS_VAR) -> int:
    return n_fid if name == "fiducial" else n_var


def sim_paths(root: Path, dname: str, n_sims: int) -> List[Path]:
    root = Path(root)
    ids = (
        sorted((q.name for q in root.iterdir() if q.name.isdigit()), key=int)
        if root.exists()
        else []
    )
    out = []
    for i in ids:
        p = root / i / dname
        if p.exists():
            out.append(p)
            if len(out) == int(n_sims):
                break
    return out


def sim_ids(paths: Sequence[Path]) -> List[int]:
    return [int(Path(p).parent.name) for p in paths]
