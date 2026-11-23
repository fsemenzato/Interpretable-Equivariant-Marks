import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/iem_mpl")
import sys, unittest, tempfile, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from config import TrainingConfig, load_config
from losses import build_loss
from train import build_optimizer, _val_distractors, _rng_state, _set_rng_state
from marks import get_mark_class
from pk import PkEstimator

torch.set_num_threads(1)


class PipelineTests(unittest.TestCase):
    def cfg(self):
        c = TrainingConfig()
        c.grid_dim = 8
        c.box_size = 100.0
        c.device = "cpu"
        c.k_max_idx = 3
        c.embedding_dim = 4
        c.projector_hidden_dim = 8
        c.similarity_mode = "cosine"
        c.n_real = 2
        c.n_far = 2
        c.n_close = 1
        c.hessian_nyquist_fix = True
        c.gate_eig_scale = 0.9757
        c.gate_init_tau = 0.6
        c.gate_learnable_tau = False
        return c

    def test_morphology_checkpoint_names(self):
        from marks import build_mark

        cfg = self.cfg()
        cfg.include_gated_experts = False
        cfg.include_ungated_term = True
        model = build_mark(cfg)
        old_config = vars(cfg).copy()
        old_config["_model_class"] = "MarkTIDFourierHWmin"
        old_config["include_cross_term"] = old_config.pop("include_gated_experts")
        old_config["mark_positivity"] = "free"
        old_state = {
            k.replace("offset", "logit_bias").replace("density_experts.", "density_curves."): v
            for k, v in model.state_dict().items()
        }
        restored = build_mark(old_config)
        restored.load_state_dict(old_state, strict=True)
        delta = torch.exp(0.2 * torch.randn(1, 1, 8, 8, 8)) - 1
        torch.testing.assert_close(model(delta)["final_field"], restored(delta)["final_field"], rtol=0, atol=0)

    def test_web_mark_defaults(self):
        model = get_mark_class("WebMark")(self.cfg())
        self.assertEqual(model.hessian_operator, "potential_tweb")
        self.assertEqual(model.gate_mode, "hard")

    def test_mark_gradient_and_equivariance(self):
        torch.manual_seed(12)
        m = get_mark_class("HessianMark")(self.cfg())
        d = torch.exp(0.2 * torch.randn(2, 1, 8, 8, 8)) - 1
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "density_experts" in n:
                    p.add_(0.01 * torch.randn_like(p))
        out = m(d)
        rot = m(torch.rot90(d, 1, (-2, -1)))
        torch.testing.assert_close(
            rot["final_field"],
            torch.rot90(out["final_field"], 1, (-2, -1)),
            atol=1e-05,
            rtol=0.0001,
        )
        shifted = m(torch.roll(d, 1, -1))
        torch.testing.assert_close(
            shifted["final_field"],
            torch.roll(out["final_field"], 1, -1),
            atol=1e-05,
            rtol=0.0001,
        )
        pk = PkEstimator(100.0).compute(d, out["final_field"], MAS=("PCS", "PCS"))
        loss = build_loss(self.cfg(), 5, "main")
        res = loss(pk, torch.randn(2, 5), torch.randn(4, 5))
        res["loss"].backward()
        self.assertTrue(torch.isfinite(res["loss"]))
        self.assertTrue(
            any((p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters()))
        )

    def test_frozen_head_excluded_from_optimizer(self):
        c = self.cfg()
        loss = build_loss(c, 5, "main")
        opt, _ = build_optimizer(None, loss, c)
        ids = {id(p) for g in opt.param_groups for p in g["params"]}
        self.assertTrue(
            all(
                (
                    not p.requires_grad and id(p) not in ids
                    for p in loss.pdd_projector.parameters()
                )
            )
        )

    def test_validation_is_repeatable_and_preserves_rng(self):
        c = self.cfg()
        pool = torch.randn(12, 5)
        theta = pool[:2]
        norm = (torch.zeros(5), torch.ones(5))
        state = torch.get_rng_state().clone()
        a = _val_distractors(c, theta, pool, norm, 0, torch.tensor([0, 1]))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        b = _val_distractors(c, theta, pool, norm, 0, torch.tensor([0, 1]))
        torch.testing.assert_close(a, b)

    def test_fisher_affine_summary_invariance(self):
        from postprocess import fisher_common as F

        rng = np.random.default_rng(3)
        X = rng.normal(size=(1000, 12))
        J = rng.normal(size=(12, 5))
        T = rng.normal(size=(12, 12)) + 6 * np.eye(12)

        def fisher(x, j):
            return F.fisher_from_jacobian(x, j)

        np.testing.assert_allclose(
            fisher(X, J), fisher(X @ T.T, T @ J), rtol=1e-10, atol=1e-10
        )

    def test_original_model_and_config(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "c.json"
            p.write_text(
                json.dumps(
                    {
                        "_model_class": "MarkSH",
                        "data_root": t,
                        "bootstrap_pretrain_from": "",
                        "grid_dim": 8,
                        "sh_basis": "physical",
                    }
                )
            )
            c = load_config(str(p))
            m = get_mark_class("MarkSH")(c)
            self.assertEqual(c.sh_convention, "physics")
            self.assertEqual(
                m(torch.zeros(1, 1, 8, 8, 8))["final_field"].shape, (1, 1, 8, 8, 8)
            )

    def test_config_inheritance_and_cycles(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "base.json").write_text(
                json.dumps({"_model_class": "MarkSH", "data_root": t, "batch_size": 7})
            )
            (root / "child.json").write_text(
                json.dumps({"extends": "base.json", "seed": 43})
            )
            c = load_config(str(root / "child.json"))
            self.assertEqual((c.batch_size, c.seed, c.data_root), (7, 43, t))
            (root / "base.json").write_text(json.dumps({"extends": "child.json"}))
            with self.assertRaises(ValueError):
                load_config(str(root / "child.json"))

    def test_two_stage_training_and_exact_resume(self):
        from train import train_bootstrap, train_main

        for model_name in ["MarkSH", "HessianMark"]:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                rng = np.random.default_rng(7)
                np.savetxt(root / "params.txt", rng.normal(size=(12, 5)))
                for i in range(12):
                    (root / str(i)).mkdir()
                    np.save(
                        root / str(i) / "field.npy",
                        np.expm1(rng.normal(scale=0.2, size=(8, 8, 8))).astype(
                            "float32"
                        ),
                    )
                c = self.cfg()
                c._model_class = model_name
                c.data_root = str(root)
                c.param_file = "params.txt"
                c.data_name = "field.npy"
                c.n_sims = -1
                c.num_workers = 0
                c.batch_size = 3
                c.val_fraction = 0.25
                c.num_epochs = 1
                c.save_dir = str(root / "bootstrap")
                c.bootstrap_pretrain_from = ""
                c.save_loss_curve = False
                c.pdd_projector_lr_multiplier = 1.0
                train_bootstrap(c)
                c.pdd_projector_lr_multiplier = 0.0
                c.bootstrap_pretrain_from = str(root / "bootstrap/best_checkpoint.pt")
                c.save_dir = str(root / "full")
                c.num_epochs = 2
                train_main(c)
                c.save_dir = str(root / "split")
                c.num_epochs = 1
                train_main(c)
                c.num_epochs = 2
                train_main(c, resume=True)
                full = torch.load(
                    root / "full/last_epoch_checkpoint.pt", map_location="cpu"
                )
                split = torch.load(
                    root / "split/last_epoch_checkpoint.pt", map_location="cpu"
                )
                self.assertEqual(full["epoch"], split["epoch"])
                for group in ["model", "loss_fn"]:
                    for k in full[group]:
                        torch.testing.assert_close(
                            full[group][k], split[group][k], rtol=0, atol=0
                        )
                self.assertEqual(full["history"]["train"], split["history"]["train"])
                self.assertEqual(full["history"]["val"], split["history"]["val"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
