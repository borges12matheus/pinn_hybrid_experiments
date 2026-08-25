import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MLP_CONFIG = ROOT / "configs" / "baseline" / "mlp_base.yaml"
PINN_BASE_CONFIG = ROOT / "configs" / "baseline" / "pinn_cont_base.yaml"
PINN_CONT_CONFIG = ROOT / "configs" / "physics_cont" / "pinn_cont_v1.yaml"
PINN_CONT_MOM_CONFIG = ROOT / "configs" / "physics_cont_mom" / "pinn_cont_mom_v1.yaml"
PINN_UTILS = ROOT / "src" / "pinn_utils.py"
TRAIN_PINN = ROOT / "src" / "train_pinn.py"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


class PinnPhysicsTest(unittest.TestCase):
    def test_cont_mom_config_is_defined(self):
        config = load_yaml(PINN_CONT_MOM_CONFIG)

        self.assertEqual(config["pinn"]["physics_mode"], "cont_mom")
        self.assertEqual(config["pinn"]["nut_transform"], "exp")
        self.assertIn("w_mom", config["pinn"])
        self.assertGreater(config["pinn"]["epochs_pre"], 0)
        self.assertGreater(config["pinn"]["epochs_phys"], 0)

    def test_shared_experiment_contract_isometric(self):
        mlp = load_yaml(MLP_CONFIG)
        pinn_base = load_yaml(PINN_BASE_CONFIG)
        pinn_cont = load_yaml(PINN_CONT_CONFIG)
        pinn_cont_mom = load_yaml(PINN_CONT_MOM_CONFIG)

        # "features" varia entre variantes PINN por design: cont_v1/v2/v3 e
        # cont_mom_v1 exigem entradas físicas extras (k, nut_log, wz_log, Re)
        # para o resíduo de PDE, que a MLP baseline não usa.
        shared_sections = [
            "paths",
            "targets",
            "training",
            "early_stopping",
            "scheduler",
            "split",
        ]

        for section in shared_sections:
            self.assertEqual(pinn_base[section], mlp[section])
            self.assertEqual(pinn_cont[section], mlp[section])
            self.assertEqual(pinn_cont_mom[section], mlp[section])

        self.assertEqual(pinn_base["features"], mlp["features"])

        # "name" varia por experimento de propósito; type/seed devem bater.
        for config in (pinn_base, pinn_cont, pinn_cont_mom):
            self.assertEqual(config["experiment"]["type"], mlp["experiment"]["type"])
            self.assertEqual(config["experiment"]["seed"], mlp["experiment"]["seed"])

        # pinn_cont_v1 e pinn_cont_mom_v1 são iterações legadas presas ao
        # dataset antigo (with_wz, Re único); só o par baseline atual
        # (mlp_base / pinn_cont_base) precisa compartilhar o dataset.
        self.assertEqual(pinn_base["dataset"]["parquet"], mlp["dataset"]["parquet"])

    def test_train_pinn_dispatches_physics_mode(self):
        source = TRAIN_PINN.read_text(encoding="utf-8")

        self.assertIn("get_pinn_epoch_runner", source)
        self.assertIn("PHYSICS_MODE", source)
        self.assertIn("NUT_TRANSFORM", source)
        self.assertIn("loss_mom", source)

    def test_pinn_utils_uses_log_nut_transform(self):
        source = PINN_UTILS.read_text(encoding="utf-8")

        self.assertIn("def transform_nut", source)
        self.assertIn("torch.exp(nut_log)", source)
        self.assertIn("pde_residuals_cont_mom", source)
        self.assertIn("nut_transform=\"exp\"", source)
        self.assertIn("nut_phys = transform_nut", source)


if __name__ == "__main__":
    unittest.main()
