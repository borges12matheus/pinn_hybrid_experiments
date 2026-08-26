#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("run_baseline_pair")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

FAIRNESS_FIELDS: dict[str, list[str]] = {
    "experiment.type": [
        "experiment.type",
    ],
    "dataset.parquet": [
        "dataset.parquet",
    ],
    "targets": [
        "targets",
    ],
    "seed": [
        "experiment.seed",
    ],
    "model.width": [
        "model.width",
    ],
    "model.depth": [
        "model.depth",
    ],
    "model.activation": [
        "model.activation",
    ],
    "training.batch_size": [
        "training.batch_size",
    ],
    "training.lr": [
        "training.lr",
    ],
    "training.weight_decay": [
        "training.weight_decay",
    ],
    "early_stopping.patience": [
        "early_stopping.patience",
    ],
    "scheduler.factor": [
        "scheduler.factor",
    ],
    "scheduler.patience": [
        "scheduler.patience",
    ],
    "split.strategy": [
        "split.strategy",
    ],
    "split.column": [
        "split.column",
    ],
    "split.bins": [
        "split.bins",
    ],
    "split.test_frac": [
        "split.test_frac",
    ],
    "split.version": [
        "split.version",
    ],
}

MLP_EPOCH_PATHS = [
    "training.epochs",
]

PINN_PRE_EPOCH_PATHS = [
    "pinn.epochs_pre",
]

PINN_PHYSICS_FIELDS = [
    "pinn.physics_mode",
    "pinn.epochs_pre",
    "pinn.epochs_phys",
    "pinn.w_data",
    "pinn.w_cont",
    "pinn.w_mom",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Executa MLP e PINN em sequência com validação de isonomia.")
    p.add_argument("--mlp-config", required=True, type=Path)
    p.add_argument("--pinn-config", required=True, type=Path)
    p.add_argument("--mlp-command", required=True, help='Use "{config}" onde entra o YAML.')
    p.add_argument("--pinn-command", required=True, help='Use "{config}" onde entra o YAML.')
    p.add_argument("--mlp-predictions-glob", required=True)
    p.add_argument("--pinn-predictions-glob", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--comparison-script", type=Path, default=Path("src/compare_divergence_models.py"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--allow-missing-fields", action="store_true")
    p.add_argument("--cwd", type=Path, default=Path.cwd())
    p.add_argument("--percentile", type=float, default=99.0)
    return p.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML inválido ou vazio: {path}")
    return data


def dotted_get(data: dict[str, Any], dotted_path: str) -> tuple[bool, Any]:
    current: Any = data
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def first_existing(data: dict[str, Any], paths: list[str]) -> tuple[str | None, Any]:
    for path in paths:
        found, value = dotted_get(data, path)
        if found:
            return path, value
    return None, None


def normalize_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): normalize_value(value[k]) for k in sorted(value)}
    if isinstance(value, float):
        return round(value, 12)
    return value


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_path(
    cfg: dict[str, Any],
    project_root: Path,
) -> Path | None:
    found_dataset, dataset_name = dotted_get(
        cfg,
        "dataset.parquet",
    )

    if not found_dataset or dataset_name is None:
        return None

    found_root, configured_root = dotted_get(
        cfg,
        "paths.root",
    )

    found_process_dir, process_dir = dotted_get(
        cfg,
        "paths.data_process_dir",
    )

    base_path = project_root

    if found_root and configured_root not in (None, "", "."):
        configured_root_path = Path(str(configured_root))

        if configured_root_path.is_absolute():
            base_path = configured_root_path
        else:
            base_path = project_root / configured_root_path

    dataset_path = Path(str(dataset_name))

    if dataset_path.is_absolute():
        return dataset_path.resolve()

    if found_process_dir and process_dir:
        dataset_path = (
            base_path
            / Path(str(process_dir))
            / dataset_path
        )
    else:
        dataset_path = base_path / dataset_path

    return dataset_path.resolve()

def validate_fairness(mlp_cfg: dict[str, Any], pinn_cfg: dict[str, Any], root: Path, allow_missing: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    warnings: list[str] = []

    for label, paths in FAIRNESS_FIELDS.items():
        mlp_path, mlp_value = first_existing(mlp_cfg, paths)
        pinn_path, pinn_value = first_existing(pinn_cfg, paths)
        if mlp_path is None or pinn_path is None:
            msg = f"Campo '{label}' não localizado em ambos os YAMLs (MLP={mlp_path}, PINN={pinn_path})."
            (warnings if allow_missing else failures).append(msg)
            checks.append({"field": label, "status": "warning" if allow_missing else "failure", "mlp_path": mlp_path, "pinn_path": pinn_path, "mlp_value": mlp_value, "pinn_value": pinn_value})
            continue
        equal = normalize_value(mlp_value) == normalize_value(pinn_value)
        checks.append({"field": label, "status": "ok" if equal else "failure", "mlp_path": mlp_path, "pinn_path": pinn_path, "mlp_value": mlp_value, "pinn_value": pinn_value})
        if not equal:
            failures.append(f"Campo incompatível '{label}': MLP={mlp_value!r} | PINN={pinn_value!r}")

    # "features" é reportado à parte, como aviso: variantes PINN legitimamente
    # precisam de campos físicos extras (ex.: Re, nut_log) para o resíduo de
    # PDE, que a MLP não usa — não é uma quebra de isonomia experimental.
    features_mlp_path, features_mlp_value = first_existing(mlp_cfg, ["features"])
    features_pinn_path, features_pinn_value = first_existing(pinn_cfg, ["features"])
    if features_mlp_path is None or features_pinn_path is None:
        msg = "Campo 'features' não localizado em ambos os YAMLs."
        (warnings if allow_missing else failures).append(msg)
        features_status = "warning" if allow_missing else "failure"
    else:
        features_equal = normalize_value(features_mlp_value) == normalize_value(features_pinn_value)
        features_status = "ok" if features_equal else "warning"
        if not features_equal:
            warnings.append(
                f"Campo 'features' difere (esperado para variantes PINN com física extra): "
                f"MLP={features_mlp_value!r} | PINN={features_pinn_value!r}"
            )
    checks.append({
        "field": "features",
        "status": features_status,
        "mlp_path": features_mlp_path,
        "pinn_path": features_pinn_path,
        "mlp_value": features_mlp_value,
        "pinn_value": features_pinn_value,
    })

    mlp_epoch_path, mlp_epochs = first_existing(mlp_cfg, MLP_EPOCH_PATHS)
    pinn_pre_path, pinn_pre_epochs = first_existing(pinn_cfg, PINN_PRE_EPOCH_PATHS)
    if mlp_epoch_path is None or pinn_pre_path is None:
        msg = "Não foi possível comparar epochs da MLP com epochs_pre da PINN."
        (warnings if allow_missing else failures).append(msg)
        epoch_status = "warning" if allow_missing else "failure"
    else:
        equal = normalize_value(mlp_epochs) == normalize_value(pinn_pre_epochs)
        epoch_status = "ok" if equal else "failure"
        if not equal:
            failures.append(f"Épocas supervisionadas diferentes: MLP={mlp_epochs} | PINN pré-treino={pinn_pre_epochs}")
    checks.append({"field": "supervised_epochs", "status": epoch_status, "mlp_path": mlp_epoch_path, "pinn_path": pinn_pre_path, "mlp_value": mlp_epochs, "pinn_value": pinn_pre_epochs})

    mlp_dataset = resolve_dataset_path(mlp_cfg, root)
    pinn_dataset = resolve_dataset_path(pinn_cfg, root)
    if mlp_dataset and pinn_dataset:
        mlp_hash = sha256_file(mlp_dataset)
        pinn_hash = sha256_file(pinn_dataset)
        ok = mlp_dataset == pinn_dataset and mlp_hash is not None and mlp_hash == pinn_hash
        checks.append({"field": "dataset_sha256", "status": "ok" if ok else "failure", "mlp_path": str(mlp_dataset), "pinn_path": str(pinn_dataset), "mlp_value": mlp_hash, "pinn_value": pinn_hash})
        if not ok:
            failures.append("Dataset incompatível por caminho ou SHA-256.")
    else:
        msg = "Não foi possível resolver o caminho do dataset."
        (warnings if allow_missing else failures).append(msg)

    pinn_physics = collect_pinn_physics_config(
        pinn_cfg
    )

    return {
        "valid": not failures,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "pinn_physics": pinn_physics,
    }


def render_command(template: str, config_path: Path) -> list[str]:
    return shlex.split(template.format(config=str(config_path)))


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Executando: %s", shlex.join(command))
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            logger.info(line.rstrip())
            log_file.write(line)
        rc = process.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, command)


def glob_files(root: Path, pattern: str) -> set[Path]:
    return {p.resolve() for p in root.glob(pattern) if p.is_file()}


def newest_new_file(before: set[Path], after: set[Path], started_at: float) -> Path:
    candidates = [p for p in after - before if p.stat().st_mtime >= started_at - 2]
    if not candidates:
        candidates = list(after)
    if not candidates:
        raise FileNotFoundError("Nenhum arquivo de predições físicas foi encontrado.")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def collect_pinn_physics_config(
    pinn_cfg: dict[str, Any],
) -> dict[str, Any]:
    result = {}

    for field in PINN_PHYSICS_FIELDS:
        found, value = dotted_get(
            pinn_cfg,
            field,
        )

        result[field] = value if found else None

    return result

def main() -> int:
    configure_logging()
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.cwd.resolve()
    base_output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir = base_output_dir.parent / f"{base_output_dir.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Diretório de saída: %s", output_dir)

    mlp_config = args.mlp_config if args.mlp_config.is_absolute() else root / args.mlp_config
    pinn_config = args.pinn_config if args.pinn_config.is_absolute() else root / args.pinn_config

    report = validate_fairness(load_yaml(mlp_config), load_yaml(pinn_config), root, args.allow_missing_fields)
    report_path = output_dir / f"fairness_report_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    logger.info("=== Validação de isonomia experimental ===")
    for check in report["checks"]:
        logger.info("[%-7s] %s: MLP=%r | PINN=%r", check["status"].upper(), check["field"], check["mlp_value"], check["pinn_value"])
    for warning in report["warnings"]:
        logger.warning("- %s", warning)

    if not report["valid"]:
        for failure in report["failures"]:
            logger.error("- %s", failure)
        return 2
    if args.validate_only:
        return 0

    mlp_before = glob_files(root, args.mlp_predictions_glob)
    t0 = time.time()
    run_command(render_command(args.mlp_command, mlp_config), root, output_dir / f"train_mlp_{timestamp}.log")
    mlp_predictions = newest_new_file(mlp_before, glob_files(root, args.mlp_predictions_glob), t0)

    pinn_before = glob_files(root, args.pinn_predictions_glob)
    t1 = time.time()
    run_command(render_command(args.pinn_command, pinn_config), root, output_dir / f"train_pinn_{timestamp}.log")
    pinn_predictions = newest_new_file(pinn_before, glob_files(root, args.pinn_predictions_glob), t1)

    comparison_script = args.comparison_script if args.comparison_script.is_absolute() else root / args.comparison_script
    command = [args.python, str(comparison_script), "--mlp-predictions", str(mlp_predictions), "--pinn-predictions", str(pinn_predictions), "--output-dir", str(output_dir / "plots"), "--percentile", str(args.percentile)]
    run_command(command, root, output_dir / f"compare_divergence_{timestamp}.log")

    manifest = {
        "timestamp": timestamp,
        "mlp_config": str(mlp_config),
        "pinn_config": str(pinn_config),
        "mlp_predictions": str(mlp_predictions),
        "pinn_predictions": str(pinn_predictions),
        "fairness_report": str(report_path),
        "comparison_output_dir": str(output_dir / "plots"),
    }
    (output_dir / f"baseline_pair_manifest_{timestamp}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Baseline concluído com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
