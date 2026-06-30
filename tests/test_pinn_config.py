import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MLP_CONFIG = ROOT / "configs" / "mlp_base.yaml"
PINN_CONFIG = ROOT / "configs" / "pinn_base.yaml"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


class PinnConfigTest(unittest.TestCase):
    def test_pinn_config_exists_and_loads(self):
        config = load_yaml(PINN_CONFIG)

        self.assertEqual(config["dataset"]["test_output"], "dataset_test_pinn")
        self.assertEqual(config["model"]["type"], "pinn")
        self.assertGreater(config["pinn"]["epochs_pre"], 0)
        self.assertGreater(config["pinn"]["epochs_phys"], 0)

    def test_pinn_config_isometric_to_mlp_on_shared_fields(self):
        mlp_config = load_yaml(MLP_CONFIG)
        pinn_config = load_yaml(PINN_CONFIG)

        shared_sections = [
            "experiment",
            "paths",
            "features",
            "targets",
            "training",
            "early_stopping",
            "scheduler",
            "split",
        ]

        for section in shared_sections:
            self.assertEqual(pinn_config[section], mlp_config[section])

        self.assertEqual(pinn_config["dataset"]["parquet"], mlp_config["dataset"]["parquet"])
        self.assertEqual(pinn_config["dataset"]["test_output"], "dataset_test_pinn")
        self.assertEqual(pinn_config["model"]["width"], mlp_config["model"]["width"])
        self.assertEqual(pinn_config["model"]["depth"], mlp_config["model"]["depth"])
        self.assertEqual(pinn_config["model"]["activation"], mlp_config["model"]["activation"])


if __name__ == "__main__":
    unittest.main()
