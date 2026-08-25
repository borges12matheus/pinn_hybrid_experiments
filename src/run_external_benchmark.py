from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_metrics import load_config, run_metrics_pipeline


TEST_CASES = [
    {
        "name": "extrapolation_lower_Re36000",
        "dataset": (
            "data/data_processed/"
            "dataset_bfs_2d_kepsilon_Re36000_with_sxy_wz.parquet"
        ),
        "category": "extrapolation_lower",
        "reynolds": 36000,
    },
    # Exemplo futuro de interpolação:
    # válido apenas se o treinamento incluir Reynolds abaixo e acima de 30000.
    #
    # {
    #     "name": "interpolation_Re30000",
    #     "dataset": (
    #         "data/data_processed/"
    #         "dataset_bfs_2d_kepsilon_Re30000_with_sxy_wz.parquet"
    #     ),
    #     "category": "interpolation",
    #     "reynolds": 30000,
    # },
    #
    # {
    #     "name": "extrapolation_upper_Re45000",
    #     "dataset": (
    #         "data/data_processed/"
    #         "dataset_bfs_2d_kepsilon_Re45000_with_sxy_wz.parquet"
    #     ),
    #     "category": "extrapolation_upper",
    #     "reynolds": 45000,
    # },
]


DEFAULT_EVALUATION_COLUMNS = [
    "Ux",
    "Uy",
    "p",
    "Ux_f",
    "Uy_f",
    "p_f",
]


def resolve_path(root: Path, path: str | Path) -> Path:
    """
    Resolve caminhos relativos usando o root configurado no YAML.

    Caminhos absolutos são preservados.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    return root / path


def validate_file(path: Path, label: str) -> None:
    """
    Verifica se um arquivo necessário existe.
    """
    if not path.exists():
        raise FileNotFoundError(f"{label} não encontrado: {path}")

    if not path.is_file():
        raise ValueError(f"{label} não é um arquivo válido: {path}")


def validate_external_dataset(
    dataset_path: Path,
    features: list[str],
    targets: list[str],
    evaluation_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Valida o dataset externo antes da inferência.

    Verificações:
    - existência do arquivo;
    - dataset não vazio;
    - presença das colunas obrigatórias;
    - ausência de NaN;
    - ausência de inf e -inf;
    - colunas numéricas utilizadas pelo modelo.
    """
    validate_file(dataset_path, "Dataset externo")

    df = pd.read_parquet(dataset_path)

    if df.empty:
        raise ValueError(f"Dataset externo está vazio: {dataset_path}")

    required_columns = set(features + targets)

    if evaluation_columns:
        required_columns.update(evaluation_columns)

    missing_columns = sorted(required_columns.difference(df.columns))

    if missing_columns:
        raise ValueError(
            "Dataset externo sem as colunas obrigatórias: "
            f"{missing_columns}"
        )

    non_numeric_columns = [
        column
        for column in required_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            "As seguintes colunas obrigatórias não são numéricas: "
            f"{sorted(non_numeric_columns)}"
        )

    invalid_columns = []

    for column in sorted(required_columns):
        values = df[column].to_numpy()

        if not np.isfinite(values).all():
            invalid_columns.append(column)

    if invalid_columns:
        raise ValueError(
            "Dataset contém NaN, inf ou -inf nas colunas: "
            f"{invalid_columns}"
        )

    return df


def calculate_feature_ranges(
    df: pd.DataFrame,
    features: list[str],
) -> dict[str, dict[str, float]]:
    """
    Calcula estatísticas das features no dataset externo.

    Essas informações ajudam a verificar o grau de extrapolação
    em relação ao conjunto de treinamento.
    """
    ranges: dict[str, dict[str, float]] = {}

    for feature in features:
        series = df[feature]

        ranges[feature] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
        }

    return ranges


def save_benchmark_metadata(
    output_path: Path,
    *,
    case: dict[str, Any],
    model_type: str,
    model_path: Path,
    dataset_path: Path,
    xscaler_path: Path,
    yscaler_path: Path,
    num_rows: int,
    features: list[str],
    targets: list[str],
    feature_ranges: dict[str, dict[str, float]],
    training_reynolds: list[int] | None = None,
) -> None:
    """
    Salva os metadados necessários para reprodutibilidade.
    """
    metadata = {
        "case_name": case["name"],
        "category": case["category"],
        "evaluation_reynolds": case.get("reynolds"),
        "training_reynolds": training_reynolds,
        "model_type": model_type,
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "xscaler_path": str(xscaler_path),
        "yscaler_path": str(yscaler_path),
        "num_rows": num_rows,
        "features": features,
        "targets": targets,
        "feature_ranges": feature_ranges,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def evaluate_cases(
    config_path: str | Path,
    model_path: str | Path,
    test_cases: list[dict[str, Any]] | None = None,
    training_reynolds: list[int] | None = None,
    batch_size: int = 4900,
) -> list[dict[str, Any]]:
    """
    Avalia um modelo já treinado em múltiplos datasets externos.

    O modelo e os scalers originais são reutilizados sem retreinamento.
    """
    config_path = Path(config_path)
    validate_file(config_path, "Arquivo de configuração")

    cfg = load_config(config_path)

    root = Path(cfg["paths"]["root"]).resolve()

    model_type = cfg["model"]["type"]
    seed = cfg["experiment"]["seed"]
    features = list(cfg["features"])
    targets = list(cfg["targets"])

    resolved_model_path = resolve_path(root, model_path)

    xscaler_path = (
        root
        / cfg["paths"]["models_dir"]
        / f"{model_type}_scaler_X.pkl"
    )

    yscaler_path = (
        root
        / cfg["paths"]["models_dir"]
        / f"{model_type}_scaler_Y.pkl"
    )

    validate_file(resolved_model_path, "Modelo")
    validate_file(xscaler_path, "Scaler das entradas")
    validate_file(yscaler_path, "Scaler das saídas")

    cases = test_cases if test_cases is not None else TEST_CASES

    if not cases:
        raise ValueError("Nenhum caso de benchmark foi definido.")

    benchmark_results: list[dict[str, Any]] = []

    for case in cases:
        required_case_keys = {
            "name",
            "dataset",
            "category",
            "reynolds",
        }

        missing_case_keys = required_case_keys.difference(case)

        if missing_case_keys:
            raise ValueError(
                f"Caso de teste incompleto. Chaves ausentes: "
                f"{sorted(missing_case_keys)}"
            )

        case_cfg = deepcopy(cfg)

        case_name = str(case["name"])
        category = str(case["category"])
        evaluation_reynolds = int(case["reynolds"])

        dataset_path = resolve_path(root, case["dataset"])

        output_root = (
            root
            / "benchmark"
            / category
            / case_name
            / model_type
        )

        metrics_path = (
            output_root
            / f"metrics_eval_seed{seed}.json"
        )

        predictions_path = (
            output_root
            / f"predictions_seed{seed}.parquet"
        )

        plots_dir = output_root / "plots"
        logs_dir = output_root / "logs"
        metadata_path = output_root / "metadata.json"

        print("=" * 80)
        print(f"Modelo: {model_type}")
        print(f"Caso: {case_name}")
        print(f"Categoria: {category}")
        print(f"Reynolds avaliado: {evaluation_reynolds}")
        print(f"Dataset: {dataset_path}")
        print("=" * 80)

        external_df = validate_external_dataset(
            dataset_path=dataset_path,
            features=features,
            targets=targets,
            evaluation_columns=DEFAULT_EVALUATION_COLUMNS,
        )

        feature_ranges = calculate_feature_ranges(
            external_df,
            features,
        )

        save_benchmark_metadata(
            output_path=metadata_path,
            case=case,
            model_type=model_type,
            model_path=resolved_model_path,
            dataset_path=dataset_path,
            xscaler_path=xscaler_path,
            yscaler_path=yscaler_path,
            num_rows=len(external_df),
            features=features,
            targets=targets,
            feature_ranges=feature_ranges,
            training_reynolds=training_reynolds,
        )

        result = run_metrics_pipeline(
            cfg=case_cfg,
            model_path=resolved_model_path,
            dataset_path=dataset_path,
            xscaler_path=xscaler_path,
            yscaler_path=yscaler_path,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
            plots_dir=plots_dir,
            logs_dir=logs_dir,
            batch_size=batch_size,
        )

        benchmark_result = {
            "case_name": case_name,
            "category": category,
            "reynolds": evaluation_reynolds,
            "model_type": model_type,
            "dataset_path": str(dataset_path),
            "metadata_path": str(metadata_path),
            **result,
        }

        benchmark_results.append(benchmark_result)

        print(f"\nAvaliação concluída: {case_name}")
        print(f"Métricas: {result['metrics_path']}")
        print(f"Predições: {result['predictions_path']}")
        print(f"Gráficos: {result['plots_dir']}\n")

    summary_path = (
        root
        / "benchmark"
        / f"benchmark_summary_{model_type}_seed{seed}.json"
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            benchmark_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Resumo geral salvo em: {summary_path}")

    return benchmark_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia um modelo treinado em datasets externos "
            "para testes de interpolação e extrapolação."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Caminho para o arquivo YAML de configuração.",
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Caminho para os pesos do modelo treinado.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4900,
        help="Tamanho do batch utilizado durante a inferência.",
    )

    parser.add_argument(
        "--training-reynolds",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Lista de Reynolds utilizados no treinamento. "
            "Exemplo: --training-reynolds 20000 36000"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluate_cases(
        config_path=args.config,
        model_path=args.model_path,
        test_cases=TEST_CASES,
        training_reynolds=args.training_reynolds,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()