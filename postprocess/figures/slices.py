import numpy as np
from .maps import ENVS
from . import decomposition as D


def render(data, out):
    D.GX = 0.14
    D.BLK_BOT = 0.48
    w = dict(np.load(data / "web_decomposition.npz"))
    full = dict(np.load(data / "web_slices.npz"))
    assert np.array_equal(w["ld"], full["ld"])
    wm = full["M"]
    bias = float(np.mean(wm - w["prod0"].sum(0)))
    assert np.std(wm - w["prod0"].sum(0)) < 1e-05
    log = lambda *args: None
    D.make(
        "fig_web_mark_share",
        w["ld"],
        list(w["w0"]),
        list(w["prod0"]),
        [f"{e.capitalize()} ({100 * w['occ0'][i]:.0f}\\%)" for i, e in enumerate(ENVS)],
        log,
        mark=wm,
        outside=[x < 0.5 for x in w["w0"]],
    )
    log("Web mark bias verified against full-mark slice: " + str(bias))
    h = dict(np.load(data / "hessian_decomposition.npz"))
    sc = dict(np.load(data / "hessian_profile.npz"))
    assert str(h["_tag"]) == str(sc["_tag"])
    assert np.array_equal(h["ld"], sc["ld"])
    assert np.std(sc["M"] - h["prod"].sum(0)) < 1e-05
    log = lambda *args: None
    D.make(
        "fig_hessian_deconstruction_share",
        h["ld"],
        list(h["w"]),
        list(h["prod"]),
        [e.capitalize() for e in ENVS],
        log,
        mark=sc["M"],
    )
