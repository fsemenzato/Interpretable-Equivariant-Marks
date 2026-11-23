import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from postprocess import config_fish as C
from postprocess.paper import constraints, family, load_spectra


class PaperWorkflowTests(unittest.TestCase):
    def test_paper_inputs_from_shared_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "mark"
            raw = run / "fisher_cache"
            raw.mkdir(parents=True)
            classical = root / "classical"
            classical.mkdir()
            bispectrum = root / "bispectrum"
            bispectrum.mkdir()
            rng = np.random.default_rng(8)
            k = np.linspace(0.01, 0.3, 12)
            jac = rng.normal(size=(36, 5))
            key = "R10_s0_p-2.5"
            for name in C.SETS:
                n = 10000 if name == "fiducial" else 500
                x = rng.normal(scale=0.1, size=(n, 36)) + 5
                if name != "fiducial":
                    p, side = name.rsplit("_", 1)
                    x += (
                        (1 if side == "p" else -1)
                        * C.STEPS[p]
                        * jac[:, C.PARAMS.index(p)]
                    )
                dd, dm, mm = np.split(x, 3, axis=1)
                np.savez(raw / f"{name}.npz", k=k, Pdd=dd, Pdm=dm, Pmm=mm)
                np.savez(
                    classical / f"{name}.npz",
                    k=k,
                    Pdd=dd,
                    **{"Pdm__" + key: dm, "Pmm__" + key: mm},
                )
                np.savez(
                    bispectrum / f'{name.replace("Ob_","Ob2_")}_pb.npz',
                    ids=np.arange(n),
                    tri=np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]),
                    B=x[:, :4] + rng.normal(scale=0.1, size=(n, 4)),
                )
            records = {
                "kmax": {
                    "0.2": {
                        "hartlap=True": {
                            "marks": {key: {"fom": 1.0, "sigma": [1.0] * 5}}
                        }
                    }
                }
            }
            report = root / "classical.json"
            report.write_text(json.dumps(records))
            cfg = {
                "profile_cuts": [0.2, 0.3],
                "hessian_runs": [str(run)] * 3,
                "web_runs": [str(run)] * 3,
                "hessian_index": 1,
                "web_index": 2,
                "classical_cache": str(classical),
                "classical_fisher": str(report),
                "bispectrum_cache": str(bispectrum),
            }
            constraints(cfg, root)
            with np.load(root / "constraints.npz") as z:
                self.assertEqual(z["free"].shape, (3, 2, 5))
                self.assertEqual(z["F_pb"].shape, (5, 5))
                self.assertTrue(np.isfinite(z["classical_fom"]).all())
            regression = {
                key: {p: {"raw": [[4.0] * 5]} for p in ["s8", "Om"]},
                "learned:mark": {p: {"raw": [[1.0] * 5]} for p in ["s8", "Om"]},
            }
            family(
                records, [load_spectra(raw)] * 3, regression, ["mark"] * 3, 0.2, root
            )
            with np.load(root / "family.npz") as z:
                np.testing.assert_array_equal(z["regression"], [[2.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
