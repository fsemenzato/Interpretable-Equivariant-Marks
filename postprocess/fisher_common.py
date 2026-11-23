from __future__ import annotations
import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np

RCOND = 1e-10
K_TOL = 1e-09


def kmask(k: np.ndarray, kmin: float, kmax: float, tol: float = K_TOL) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    return (k >= kmin - tol) & (k <= kmax + tol)


def assert_same_k(
    k_a: np.ndarray, k_b: np.ndarray, atol: float = 1e-06, what: str = ""
) -> None:
    k_a, k_b = (np.asarray(k_a, float), np.asarray(k_b, float))
    if k_a.shape != k_b.shape or not np.allclose(k_a, k_b, atol=atol, rtol=0):
        raise ValueError(
            f"k vectors disagree {what}: shapes {k_a.shape} vs {k_b.shape}, max |dk| = {(np.abs(k_a - k_b).max() if k_a.shape == k_b.shape else 'n/a')}"
        )


def hartlap_factor(n: int, d: int) -> float:
    if n - d - 2 <= 0:
        warnings.warn(
            f"Hartlap factor undefined: n={n} fiducial sims, D={d} summary dims (need n > D + 2); returning NaN",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    return (n - d - 2) / (n - 1)


def inverse_covariance(samples, hartlap=False, standardize=True):
    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    n, d = samples.shape
    if n < 2:
        raise ValueError("Need at least two covariance realizations")
    scale = samples.std(axis=0) if standardize else np.ones(d)
    scale = np.where(scale > 0, scale, 1.0)
    # Standardize before truncating eigenmodes so units do not set the cutoff.
    cov = np.atleast_2d(np.cov(samples / scale, rowvar=False))
    factor = hartlap_factor(n, d) if hartlap else 1.0
    return (np.linalg.pinv(cov, rcond=RCOND, hermitian=True) * factor, scale, factor)


def fisher_from_jacobian(samples, jacobian, hartlap=False, standardize=True):
    precision, scale, _ = inverse_covariance(samples, hartlap, standardize)
    jacobian = np.asarray(jacobian, dtype=float) / scale[:, None]
    return jacobian.T @ precision @ jacobian


def fisher_matrix(S_fid, S_p, S_m, steps, hartlap=False, standardize=True, params=None):
    params = list(params) if params is not None else list(steps)
    jacobian = np.stack(
        [
            (
                np.asarray(S_p[p], dtype=float).mean(0)
                - np.asarray(S_m[p], dtype=float).mean(0)
            )
            / (2 * steps[p])
            for p in params
        ],
        axis=1,
    )
    return fisher_from_jacobian(S_fid, jacobian, hartlap, standardize)


def sigma_from_fisher(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    if not np.all(np.isfinite(F)):
        return np.full(F.shape[0], np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.diag(np.linalg.pinv(F, hermitian=True)))


def fom_from_fisher(F: np.ndarray) -> float:
    F = np.asarray(F, dtype=np.float64)
    if not np.all(np.isfinite(F)):
        return float("nan")
    det = np.linalg.det(F)
    return float(np.sqrt(det)) if det > 0 else float("nan")


def fisher_sigma(
    S_fid,
    S_p,
    S_m,
    steps,
    hartlap: bool = False,
    standardize: bool = True,
    params: Optional[Sequence[str]] = None,
) -> np.ndarray:
    return sigma_from_fisher(
        fisher_matrix(S_fid, S_p, S_m, steps, hartlap, standardize, params)
    )


def improvement(sig_ref: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return np.asarray(sig_ref, float) / np.asarray(sig, float)


SUMMARY_KINDS = {
    "unmarked": ("Pdd",),
    "marked": ("Pdd", "Pmm", "Pdm"),
    "marked_mm_only": ("Pdd", "Pmm"),
}


def spectrum_blocks(
    kind: str, mask: np.ndarray, keys: Optional[Mapping[str, str]] = None
) -> List[tuple]:
    if kind not in SUMMARY_KINDS:
        raise KeyError(f"unknown summary kind {kind!r}; known: {sorted(SUMMARY_KINDS)}")
    keys = dict(keys or {})
    return [("spectrum", keys.get(name, name), mask) for name in SUMMARY_KINDS[kind]]


def scalar_blocks(names: Iterable[str], prefix: str = "scalar__") -> List[tuple]:
    return [
        ("scalar", f"{prefix}{n}" if not n.startswith(prefix) else n) for n in names
    ]


def build_summary(cache, blocks: Sequence[tuple]) -> np.ndarray:
    cols = []
    for blk in blocks:
        if blk[0] == "spectrum":
            _, key, mask = blk
            arr = np.asarray(cache[key], dtype=np.float64)
            cols.append(arr[:, np.asarray(mask, bool)])
        elif blk[0] == "scalar":
            _, key = blk
            cols.append(np.asarray(cache[key], dtype=np.float64).reshape(-1, 1))
        else:
            raise ValueError(f"bad block {blk!r}")
    if not cols:
        raise ValueError("empty summary")
    return np.concatenate(cols, axis=1)


def summary_dims(blocks: Sequence[tuple]) -> int:
    return sum(
        (int(np.asarray(b[2], bool).sum()) if b[0] == "spectrum" else 1 for b in blocks)
    )


def forecast(
    caches: Mapping[str, object],
    blocks: Sequence[tuple],
    steps: Mapping[str, float],
    hartlap: bool = False,
    standardize: bool = True,
    params: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    params = list(params) if params is not None else list(steps)
    S_fid = build_summary(caches["fiducial"], blocks)
    S_p = {p: build_summary(caches[f"{p}_p"], blocks) for p in params}
    S_m = {p: build_summary(caches[f"{p}_m"], blocks) for p in params}
    F = fisher_matrix(
        S_fid, S_p, S_m, steps, hartlap=hartlap, standardize=standardize, params=params
    )
    return {
        "F": F,
        "sigma": sigma_from_fisher(F),
        "fom": fom_from_fisher(F),
        "dims": int(S_fid.shape[1]),
        "n_fid": int(S_fid.shape[0]),
    }


def gain_record(
    res: Dict[str, object], ref: Dict[str, object], ref_name: str
) -> Dict[str, object]:
    return {
        "sigma": [float(x) for x in res["sigma"]],
        "improvement": [float(x) for x in improvement(ref["sigma"], res["sigma"])],
        "fom": float(res["fom"]),
        "fom_improvement": float(res["fom"] / ref["fom"])
        if ref["fom"] == ref["fom"] and ref["fom"] > 0
        else float("nan"),
        "dims": int(res["dims"]),
        "ref": ref_name,
    }


def nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return nan_to_none(obj.tolist())
    return obj


def format_table(
    rows: Sequence[Tuple[str, Sequence[float], int]],
    params: Sequence[str],
    title: str = "",
) -> str:
    w = max([len(r[0]) for r in rows] + [8])
    lines = []
    if title:
        lines.append(title)
    lines.append(
        f"  {'summary':{w}s}" + "".join((f"{p:>8s}" for p in params)) + "    dims"
    )
    for name, gain, dims in rows:
        cells = "".join(
            (
                f"{g:8.2f}" if g is not None and np.isfinite(g) else f"{'nan':>8s}"
                for g in gain
            )
        )
        lines.append(f"  {name:{w}s}{cells}    {dims:4d}")
    return "\n".join(lines)
