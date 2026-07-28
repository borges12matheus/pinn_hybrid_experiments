#!/usr/bin/env python3
"""
Lê logs JSON completos dos experimentos e gera tabelas comparativas.

Estrutura esperada em cada log:
- config
- hardware
- training_history
- final_metrics.evaluation_data.metrics
- final_metrics.evaluation_physics.metrics

Saídas:
- experiments_full.csv
- experiments_full.parquet
- experiments_main_comparison.csv
- experiments_summary_by_seed.csv
- experiments_publication.csv
- comparability_report.csv

Exemplos
--------
python compare_experiment_logs.py

python compare_experiment_logs.py \
    --logs-root logs \
    --output-dir results/comparisons

python compare_experiment_logs.py \
    --logs-root logs \
    --experiment "*cont_base*" \
    --model-type pinn
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# Utilidades
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def nested_get(
    data: dict[str, Any],
    path: str,
    default: Any = np.nan,
) -> Any:
    current: Any = data

    for key in path.split("."):
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def to_number(
    value: Any,
    default: float = np.nan,
) -> float:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_stage_name(stage: str) -> str:
    return (
        str(stage)
        .lower()
        .replace("[", "_")
        .replace("]", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


def safe_join(values: Any) -> str:
    if isinstance(values, list):
        return ", ".join(str(value) for value in values)

    return ""


# ============================================================
# Histórico de treinamento
# ============================================================

def summarize_training_history(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    if not history:
        return {}

    df_history = pd.DataFrame(history)

    if df_history.empty:
        return {}

    if "stage" not in df_history.columns:
        df_history["stage"] = "TRAIN"

    if "epoch" not in df_history.columns:
        df_history["epoch"] = np.arange(
            1,
            len(df_history) + 1,
        )

    summary: dict[str, Any] = {}

    for stage, df_stage in df_history.groupby(
        "stage",
        dropna=False,
    ):
        stage_key = normalize_stage_name(stage)

        df_stage = df_stage.sort_values(
            "epoch"
        ).reset_index(drop=True)

        first_row = df_stage.iloc[0]
        last_row = df_stage.iloc[-1]

        summary[f"{stage_key}_records"] = len(df_stage)
        summary[f"{stage_key}_first_epoch"] = first_row.get(
            "epoch",
            np.nan,
        )
        summary[f"{stage_key}_last_epoch"] = last_row.get(
            "epoch",
            np.nan,
        )
        summary[f"{stage_key}_initial_lr"] = first_row.get(
            "lr",
            np.nan,
        )
        summary[f"{stage_key}_final_lr"] = last_row.get(
            "lr",
            np.nan,
        )

        metrics = [
            "loss_train",
            "loss_val",
            "loss_total",
            "loss_data",
            "loss_cont",
            "loss_mom",
            "val_score",
            "val_data",
            "val_cont",
            "val_mom",
            "best_val",
        ]

        for metric in metrics:
            if metric not in df_stage.columns:
                continue

            values = pd.to_numeric(
                df_stage[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            summary[f"{stage_key}_{metric}_initial"] = float(
                values.iloc[0]
            )
            summary[f"{stage_key}_{metric}_final"] = float(
                values.iloc[-1]
            )
            summary[f"{stage_key}_{metric}_min"] = float(
                values.min()
            )

    return summary


# ============================================================
# Extração de uma execução
# ============================================================

def extract_experiment_row(
    log_data: dict[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    config = log_data.get("config", {})
    experiment = config.get("experiment", {})
    model = config.get("model", {})
    training = config.get("training", {})
    pinn = config.get("pinn", {})
    dataset = config.get("dataset", {})
    split = config.get("split", {})
    hardware = log_data.get("hardware", {})

    accuracy = nested_get(
        log_data,
        "final_metrics.evaluation_data.metrics",
        default={},
    )
    physics = nested_get(
        log_data,
        "final_metrics.evaluation_physics.metrics",
        default={},
    )

    if not isinstance(accuracy, dict):
        accuracy = {}

    if not isinstance(physics, dict):
        physics = {}

    wall_time_sec = to_number(
        log_data.get("wall_time_sec")
    )

    row: dict[str, Any] = {
        # Identificação
        "run_id": log_data.get("run_id"),
        "experiment_name": log_data.get(
            "experiment_name",
            experiment.get("name"),
        ),
        "experiment_config_name": experiment.get("name"),
        "experiment_type": log_data.get(
            "experiment_type",
            experiment.get("type"),
        ),
        "model_type": model.get("type"),
        "seed": experiment.get("seed"),
        "start_time": log_data.get("start_time"),
        "end_time": log_data.get("end_time"),
        "log_path": str(log_path),

        # Tempo
        "wall_time_sec": wall_time_sec,
        "wall_time_min": (
            wall_time_sec / 60.0
            if np.isfinite(wall_time_sec)
            else np.nan
        ),
        "cpu_time_sec": to_number(
            log_data.get("cpu_time_sec")
        ),

        # Dataset e reprodutibilidade
        "dataset": dataset.get("parquet"),
        "dataset_path": log_data.get("dataset_path"),
        "dataset_hash": log_data.get("dataset_hash"),
        "config_path": log_data.get("config_path"),
        "config_hash": log_data.get("config_hash"),
        "split_path": log_data.get("split_path"),
        "split_hash": log_data.get("split_hash"),
        "split_method": log_data.get(
            "split_method",
            split.get("strategy"),
        ),
        "split_bins": log_data.get(
            "split_bins",
            split.get("bins"),
        ),
        "split_test_frac": log_data.get(
            "split_test_frac",
            split.get("test_frac"),
        ),

        # Features e alvos
        "features": safe_join(
            config.get("features", [])
        ),
        "n_features": len(
            config.get("features", [])
        ),
        "targets": safe_join(
            config.get("targets", [])
        ),
        "n_targets": len(
            config.get("targets", [])
        ),

        # Arquitetura
        "width": model.get("width"),
        "depth": model.get("depth"),
        "activation": model.get("activation"),

        # Treinamento
        "batch_size": training.get("batch_size"),
        "epochs": training.get("epochs"),
        "learning_rate": to_number(
            training.get("lr")
        ),
        "weight_decay": to_number(
            training.get("weight_decay")
        ),
        "early_stopping_patience": nested_get(
            config,
            "early_stopping.patience",
        ),
        "scheduler_factor": nested_get(
            config,
            "scheduler.factor",
        ),
        "scheduler_patience": nested_get(
            config,
            "scheduler.patience",
        ),

        # PINN
        "physics_mode": pinn.get("physics_mode"),
        "epochs_pre": pinn.get("epochs_pre"),
        "epochs_phys": pinn.get("epochs_phys"),
        "w_data": to_number(
            pinn.get("w_data")
        ),
        "w_cont": to_number(
            pinn.get("w_cont")
        ),
        "w_mom": to_number(
            pinn.get("w_mom")
        ),
        "best_val": nested_get(
            log_data,
            "final_metrics.best_val",
        ),

        # Hardware
        "platform": hardware.get("platform"),
        "python_version": hardware.get(
            "python_version"
        ),
        "torch_version": hardware.get(
            "torch_version"
        ),
        "cuda_available": hardware.get(
            "cuda_available"
        ),
        "cuda_version": hardware.get(
            "cuda_version"
        ),
        "gpu_name": hardware.get("gpu_name"),
        "gpu_memory_gb": hardware.get(
            "gpu_memory_gb"
        ),
        "ram_total_gb": hardware.get(
            "ram_total_gb"
        ),
        "cpu_count_logical": hardware.get(
            "cpu_count_logical"
        ),
        "cpu_count_physical": hardware.get(
            "cpu_count_physical"
        ),

        # Acurácia coarse
        "mae_uv_coarse": accuracy.get(
            "mae_uv_coarse_to_fine",
            np.nan,
        ),
        "rmse_uv_coarse": accuracy.get(
            "rmse_uv_coarse_to_fine",
            np.nan,
        ),
        "mae_p_coarse": accuracy.get(
            "mae_p_pressure_coarse_to_fine",
            np.nan,
        ),
        "rmse_p_coarse": accuracy.get(
            "rmse_p_coarse_to_fine",
            np.nan,
        ),
        "l2_rel_uv_coarse": accuracy.get(
            "l2_rel_uv_coarse",
            np.nan,
        ),
        "l2_rel_p_coarse": accuracy.get(
            "l2_rel_p_coarse",
            np.nan,
        ),

        # Acurácia corrigida
        "mae_uv_corrected": accuracy.get(
            "mae_uv_corrected_to_fine",
            np.nan,
        ),
        "rmse_uv_corrected": accuracy.get(
            "rmse_uv_corrected_to_fine",
            np.nan,
        ),
        "mae_p_corrected": accuracy.get(
            "mae_p_corrected_to_fine",
            np.nan,
        ),
        "rmse_p_corrected": accuracy.get(
            "rmse_p_corrected_to_fine",
            np.nan,
        ),
        "l2_rel_uv_corrected": accuracy.get(
            "l2_rel_uv_corrected",
            np.nan,
        ),
        "l2_rel_p_corrected": accuracy.get(
            "l2_rel_p_corrected",
            np.nan,
        ),
        "rer_uv": accuracy.get(
            "RER (u,v)",
            np.nan,
        ),
        "rer_p": accuracy.get(
            "RER (p)",
            np.nan,
        ),
        "improvement_mae_uv_pct": accuracy.get(
            "melhora_MAE_pct",
            np.nan,
        ),
        "improvement_mae_p_pct": accuracy.get(
            "melhora_MAE_pct_pressure",
            np.nan,
        ),
        "n_test": accuracy.get(
            "N",
            np.nan,
        ),

        # Física coarse
        "mae_div_coarse": nested_get(
            physics,
            "divergence_coarse.mae",
        ),
        "rmse_div_coarse": nested_get(
            physics,
            "divergence_coarse.rmse",
        ),
        "l2_div_coarse": nested_get(
            physics,
            "divergence_coarse.l2",
        ),
        "linf_div_coarse": nested_get(
            physics,
            "divergence_coarse.linf",
        ),

        # Física fine
        "mae_div_fine": nested_get(
            physics,
            "divergence_fine.mae",
        ),
        "rmse_div_fine": nested_get(
            physics,
            "divergence_fine.rmse",
        ),
        "l2_div_fine": nested_get(
            physics,
            "divergence_fine.l2",
        ),
        "linf_div_fine": nested_get(
            physics,
            "divergence_fine.linf",
        ),

        # Física do delta
        "mae_div_delta": nested_get(
            physics,
            "divergence_delta_pred.mae",
        ),
        "rmse_div_delta": nested_get(
            physics,
            "divergence_delta_pred.rmse",
        ),
        "l2_div_delta": nested_get(
            physics,
            "divergence_delta_pred.l2",
        ),
        "linf_div_delta": nested_get(
            physics,
            "divergence_delta_pred.linf",
        ),

        # Física corrigida
        "mae_div_corrected": nested_get(
            physics,
            "divergence_corrected.mae",
        ),
        "rmse_div_corrected": nested_get(
            physics,
            "divergence_corrected.rmse",
        ),
        "l2_div_corrected": nested_get(
            physics,
            "divergence_corrected.l2",
        ),
        "linf_div_corrected": nested_get(
            physics,
            "divergence_corrected.linf",
        ),
        "mean_div_corrected": nested_get(
            physics,
            "divergence_corrected.mean_signed",
        ),
        "std_div_corrected": nested_get(
            physics,
            "divergence_corrected.std",
        ),

        # Física corrigida versus fine
        "mae_div_error_vs_fine": nested_get(
            physics,
            "divergence_error_corrected_vs_fine.mae",
        ),
        "rmse_div_error_vs_fine": nested_get(
            physics,
            "divergence_error_corrected_vs_fine.rmse",
        ),
        "l2_div_error_vs_fine": nested_get(
            physics,
            "divergence_error_corrected_vs_fine.l2",
        ),
        "linf_div_error_vs_fine": nested_get(
            physics,
            "divergence_error_corrected_vs_fine.linf",
        ),

        # Indicadores físicos
        "improvement_div_rmse_vs_coarse_pct": physics.get(
            "improvement_divergence_rmse_pct",
            np.nan,
        ),
        "improvement_div_mae_vs_coarse_pct": physics.get(
            "improvement_divergence_mae_pct",
            np.nan,
        ),
        "improvement_div_rmse_vs_fine_pct": physics.get(
            "improvement_vs_fine_rmse_pct",
            np.nan,
        ),

        # Artefatos
        "model_path": nested_get(
            log_data,
            "final_metrics.model_path",
        ),
        "accuracy_metrics_path": nested_get(
            log_data,
            "final_metrics.evaluation_data.metrics_path",
        ),
        "physics_metrics_path": nested_get(
            log_data,
            "final_metrics.evaluation_physics.metrics_path",
        ),
        "accuracy_predictions_path": nested_get(
            log_data,
            "final_metrics.evaluation_data.predictions_path",
        ),
        "physics_predictions_path": nested_get(
            log_data,
            "final_metrics.evaluation_physics.predictions_path",
        ),
    }

    row.update(
        summarize_training_history(
            log_data.get(
                "training_history",
                [],
            )
        )
    )

    return row


# ============================================================
# Leitura dos logs
# ============================================================

def build_experiment_log_table(
    logs_root: Path,
    pattern: str,
    experiment_filter: str,
    model_type_filter: str | None,
) -> pd.DataFrame:
    if not logs_root.exists():
        raise FileNotFoundError(
            f"Pasta de logs não encontrada: {logs_root}"
        )

    rows: list[dict[str, Any]] = []

    for log_path in sorted(
        logs_root.rglob(pattern)
    ):
        try:
            log_data = load_json(log_path)

            if "final_metrics" not in log_data:
                print(
                    f"[ignorado] Sem métricas finais: {log_path}"
                )
                continue

            row = extract_experiment_row(
                log_data,
                log_path,
            )

            experiment_name = str(
                row.get("experiment_name") or ""
            )

            if not fnmatch.fnmatch(
                experiment_name,
                experiment_filter,
            ):
                continue

            if (
                model_type_filter is not None
                and row.get("model_type")
                != model_type_filter
            ):
                continue

            rows.append(row)

        except json.JSONDecodeError as exc:
            print(
                f"[erro JSON] {log_path}: {exc}"
            )
        except Exception as exc:
            print(
                f"[erro] {log_path}: {exc}"
            )

    if not rows:
        raise ValueError(
            "Nenhum log completo foi encontrado "
            "com os filtros informados."
        )

    df = pd.DataFrame(rows)

    for column in ["start_time", "end_time"]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    sort_columns = [
        column
        for column in [
            "model_type",
            "experiment_name",
            "seed",
            "start_time",
        ]
        if column in df.columns
    ]

    return (
        df.sort_values(
            sort_columns,
            na_position="last",
        )
        .reset_index(drop=True)
    )


# ============================================================
# Tabelas derivadas
# ============================================================

def build_main_comparison(
    df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "run_id",
        "model_type",
        "experiment_name",
        "experiment_type",
        "seed",
        "width",
        "depth",
        "activation",
        "n_features",
        "features",
        "epochs_pre",
        "epochs_phys",
        "w_data",
        "w_cont",
        "w_mom",
        "wall_time_min",
        "best_val",
        "mae_uv_corrected",
        "rmse_uv_corrected",
        "l2_rel_uv_corrected",
        "mae_p_corrected",
        "rmse_p_corrected",
        "l2_rel_p_corrected",
        "mae_div_corrected",
        "rmse_div_corrected",
        "linf_div_corrected",
        "rmse_div_error_vs_fine",
        "improvement_mae_uv_pct",
        "improvement_mae_p_pct",
        "improvement_div_rmse_vs_coarse_pct",
        "improvement_div_rmse_vs_fine_pct",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[available].copy()


def build_seed_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "wall_time_min",
        "best_val",
        "mae_uv_corrected",
        "rmse_uv_corrected",
        "l2_rel_uv_corrected",
        "mae_p_corrected",
        "rmse_p_corrected",
        "l2_rel_p_corrected",
        "mae_div_corrected",
        "rmse_div_corrected",
        "linf_div_corrected",
        "rmse_div_error_vs_fine",
    ]

    available = [
        metric
        for metric in metrics
        if metric in df.columns
    ]

    group_columns = [
        "model_type",
        "experiment_name",
        "experiment_type",
        "width",
        "depth",
        "activation",
        "features",
        "w_cont",
        "w_mom",
    ]

    group_columns = [
        column
        for column in group_columns
        if column in df.columns
    ]

    summary = (
        df.groupby(
            group_columns,
            dropna=False,
        )[available]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    summary.columns = [
        (
            "_".join(
                str(part)
                for part in column
                if part
            )
            if isinstance(column, tuple)
            else column
        )
        for column in summary.columns
    ]

    return summary


def build_publication_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    id_columns = [
        column
        for column in [
            "model_type",
            "experiment_name",
            "experiment_type",
            "width",
            "depth",
            "activation",
            "features",
            "w_cont",
            "w_mom",
        ]
        if column in summary.columns
    ]

    publication = summary[id_columns].copy()

    metrics = [
        "mae_uv_corrected",
        "rmse_uv_corrected",
        "mae_p_corrected",
        "rmse_p_corrected",
        "mae_div_corrected",
        "rmse_div_corrected",
        "linf_div_corrected",
        "wall_time_min",
    ]

    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in summary.columns:
            continue

        def format_row(row: pd.Series) -> str:
            mean = row.get(mean_col, np.nan)
            std = row.get(std_col, np.nan)

            if pd.isna(mean):
                return "—"

            if pd.isna(std):
                return f"{mean:.6f}"

            return f"{mean:.6f} ± {std:.6f}"

        publication[metric] = summary.apply(
            format_row,
            axis=1,
        )

    return publication


def build_comparability_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "dataset_hash",
        "split_hash",
        "features",
        "split_method",
        "split_test_frac",
        "dataset",
        "experiment_type",
    ]

    rows = []

    for column in columns:
        if column not in df.columns:
            continue

        values = (
            df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        rows.append({
            "field": column,
            "n_unique": len(values),
            "comparable": len(values) <= 1,
            "values": " | ".join(values),
        })

    return pd.DataFrame(rows)


# ============================================================
# Exportação
# ============================================================

def save_tables(
    df_full: pd.DataFrame,
    df_main: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_publication: pd.DataFrame,
    df_comparability: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "full_csv": (
            output_dir / "experiments_full.csv"
        ),
        "full_parquet": (
            output_dir / "experiments_full.parquet"
        ),
        "main_csv": (
            output_dir / "experiments_main_comparison.csv"
        ),
        "summary_csv": (
            output_dir / "experiments_summary_by_seed.csv"
        ),
        "publication_csv": (
            output_dir / "experiments_publication.csv"
        ),
        "comparability_csv": (
            output_dir / "comparability_report.csv"
        ),
    }

    df_full.to_csv(
        paths["full_csv"],
        index=False,
    )
    df_full.to_parquet(
        paths["full_parquet"],
        index=False,
    )
    df_main.to_csv(
        paths["main_csv"],
        index=False,
    )
    df_summary.to_csv(
        paths["summary_csv"],
        index=False,
    )
    df_publication.to_csv(
        paths["publication_csv"],
        index=False,
    )
    df_comparability.to_csv(
        paths["comparability_csv"],
        index=False,
    )

    return paths


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lê logs JSON completos e gera comparativos "
            "de acurácia, física, configuração e custo."
        )
    )

    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Pasta raiz dos logs JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparisons"),
        help="Pasta de saída das tabelas.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Padrão dos arquivos de log.",
    )
    parser.add_argument(
        "--experiment",
        default="*",
        help=(
            "Filtro fnmatch para experiment_name. "
            'Exemplos: "*", "*cont_base*", "pinn_*".'
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=["mlp", "pinn"],
        default=None,
        help="Filtra por tipo de modelo.",
    )
    parser.add_argument(
        "--print-columns",
        action="store_true",
        help="Imprime todas as colunas da tabela completa.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        df_full = build_experiment_log_table(
            logs_root=args.logs_root,
            pattern=args.pattern,
            experiment_filter=args.experiment,
            model_type_filter=args.model_type,
        )

        df_main = build_main_comparison(
            df_full
        )
        df_summary = build_seed_summary(
            df_full
        )
        df_publication = build_publication_table(
            df_summary
        )
        df_comparability = build_comparability_report(
            df_full
        )

        paths = save_tables(
            df_full=df_full,
            df_main=df_main,
            df_summary=df_summary,
            df_publication=df_publication,
            df_comparability=df_comparability,
            output_dir=args.output_dir,
        )

        print("\n" + "=" * 100)
        print("COMPARATIVO DOS EXPERIMENTOS")
        print("=" * 100)

        display_columns = [
            column
            for column in [
                "model_type",
                "experiment_name",
                "seed",
                "w_cont",
                "w_mom",
                "wall_time_min",
                "mae_uv_corrected",
                "rmse_uv_corrected",
                "mae_p_corrected",
                "rmse_p_corrected",
                "mae_div_corrected",
                "rmse_div_corrected",
                "linf_div_corrected",
            ]
            if column in df_main.columns
        ]

        with pd.option_context(
            "display.max_columns",
            None,
            "display.width",
            240,
        ):
            print(
                df_main[display_columns]
                .to_string(
                    index=False,
                    float_format=lambda value: f"{value:.6g}",
                )
            )

        print("\nComparabilidade:")
        print(
            df_comparability.to_string(
                index=False
            )
        )

        print("\nArquivos gerados:")

        for name, path in paths.items():
            print(f"  {name}: {path}")

        print(
            f"\nExecuções lidas: {len(df_full)}"
        )

        if args.print_columns:
            print("\nColunas disponíveis:")

            for column in df_full.columns:
                print(f"  - {column}")

        return 0

    except Exception as exc:
        print(
            f"Erro ao gerar comparativo: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
