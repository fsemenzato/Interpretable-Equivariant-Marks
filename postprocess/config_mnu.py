from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional
from postprocess import config_fish as CF

PARAMS: List[str] = list(CF.PARAMS) + ["Mnu"]
STEPS: Dict[str, float] = dict(CF.STEPS)
FIDUCIAL: Dict[str, float] = dict(CF.FIDUCIAL, Mnu=0.0)
TEX: Dict[str, str] = {
    **{p: str(CF.PARAM_INFO[p]["tex"]) for p in CF.PARAMS},
    "Mnu": "M_\\nu",
}
PARAM_SETS: Dict[str, List[str]] = {
    "full6": list(PARAMS),
    "min3": ["Om", "s8", "Mnu"],
    "std5": list(CF.PARAMS),
}
ZA_SET = "fiducial_ZA"
MNU_SETS: List[str] = [ZA_SET, "Mnu_p", "Mnu_pp", "Mnu_ppp"]
MNU_VALUE: Dict[str, float] = {ZA_SET: 0.0, "Mnu_p": 0.1, "Mnu_pp": 0.2, "Mnu_ppp": 0.4}
MNU_BASE_STEP = 0.1
N_SIMS_MNU = 500
SCHEMES: Dict[str, Dict[str, object]] = {
    "p": {"weights": {ZA_SET: -1.0, "Mnu_p": 1.0}, "denom": 1.0},
    "pp": {"weights": {ZA_SET: -1.0, "Mnu_pp": 1.0}, "denom": 2.0},
    "forward2": {"weights": {ZA_SET: -3.0, "Mnu_p": 4.0, "Mnu_pp": -1.0}, "denom": 2.0},
    "poly3": {
        "weights": {ZA_SET: -21.0, "Mnu_p": 32.0, "Mnu_pp": -12.0, "Mnu_ppp": 1.0},
        "denom": 12.0,
    },
    "noza2": {"weights": {"Mnu_p": -4.0, "Mnu_pp": 5.0, "Mnu_ppp": -1.0}, "denom": 2.0},
}
DEFAULT_SCHEME = "poly3"
FIELDS = ("m", "cb")
DEFAULT_FIELD = "m"
ENV_MNU = "QUIJOTE_MNU_ROOT"
ENV_ZA = "QUIJOTE_ZA_ROOT"


def mnu_root() -> Path:
    return Path(os.environ[ENV_MNU]) if os.environ.get(ENV_MNU) else CF.diff_root()


def za_root() -> Path:
    return Path(os.environ[ENV_ZA]) if os.environ.get(ENV_ZA) else mnu_root() / ZA_SET


def scheme_sets(scheme: str) -> List[str]:
    if scheme not in SCHEMES:
        raise KeyError(f"unknown Mnu scheme {scheme!r}; known: {sorted(SCHEMES)}")
    need = set(SCHEMES[scheme]["weights"])
    return [s for s in MNU_SETS if s in need]


def all_sets(scheme: str = DEFAULT_SCHEME) -> List[str]:
    return list(CF.SETS) + scheme_sets(scheme)


def is_mnu_set(name: str) -> bool:
    return name in MNU_SETS


def set_root(name: str) -> Path:
    if name == ZA_SET:
        return za_root()
    if name in MNU_SETS:
        return mnu_root() / name
    return CF.set_root(name)


def default_n_sims(
    name: str, n_fid: int = CF.N_SIMS_FID, n_var: int = CF.N_SIMS_VAR
) -> int:
    return CF.default_n_sims(name, n_fid, n_var)


def data_name(
    name: str,
    field: str = DEFAULT_FIELD,
    grid: int = CF.GRID_DIM,
    mas: str = CF.DEFAULT_MAS,
) -> str:
    if field not in FIELDS:
        raise ValueError(f"unknown field {field!r}; known: {FIELDS}")
    if field == "cb" and name in MNU_SETS and (name != ZA_SET):
        return f"df_cb_{int(grid)}_{mas}_z=0.npy"
    return f"df_m_{int(grid)}_{mas}_z=0.npy"


def uses_cb(name: str, field: str) -> bool:
    return field == "cb" and name in MNU_SETS and (name != ZA_SET)


def tag_for(n_fid: int, n_var: int) -> str:
    if int(n_fid) == CF.N_SIMS_FID and int(n_var) == CF.N_SIMS_VAR:
        return ""
    return f"_nf{int(n_fid)}_nv{int(n_var)}"


WW_PARAMS: List[str] = ["Om", "s8", "Mnu"]
WW_KMAX = 0.3
WW_BINS = 35
WW_N_COV = 2100
WW_N_DERIV = 50
WW_PUBLISHED: Dict[str, List[float]] = {
    "delta + Wh10 (White, p=2, ds=0.25)": [2.24, 1.87, 1.93],
    "delta + delta_10": [1.43, 1.35, 1.38],
    "delta + shear s_10": [1.27, 1.26, 1.27],
    "delta + delta_10 + s_10": [1.7, 1.62, 1.64],
    "delta + GLFD_10": [1.63, 1.46, 1.51],
    "delta + delta_10 + delta_15": [1.52, 1.43, 1.46],
    "delta + Wh10 + GLFD10": [2.45, 2.07, 2.14],
}
WW_WHITE_KEY = "R10_s0.25_p-2"
