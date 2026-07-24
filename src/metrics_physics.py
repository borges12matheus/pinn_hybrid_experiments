from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EPS = 1e-12


def _as_float_array(values: np.ndarray | pd.Series, name: str) -> np.ndarray:
    """Converte a entrada para vetor float64 unidimensional."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} deve ser unidimensional. Shape: {array.shape}")
    return array


def spatial_gradient_local_ls(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    n_neighbors: int = 12,
    min_neighbors: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estima df/dx e df/dy por mínimos quadrados locais.

    Para cada ponto i, ajusta:
        f(x, y) ≈ a + b(x-x_i) + c(y-y_i)

    em que b ≈ df/dx e c ≈ df/dy.
    """
    x = _as_float_array(x, "x")
    y = _as_float_array(y, "y")
    field = _as_float_array(field, "field")

    if not (len(x) == len(y) == len(field)):
        raise ValueError("x, y e field devem possuir o mesmo tamanho.")
    if n_neighbors < min_neighbors:
        raise ValueError(f"n_neighbors deve ser >= {min_neighbors}.")

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(field)
    valid_indices = np.flatnonzero(valid)

    dfdx = np.full(len(field), np.nan, dtype=np.float64)
    dfdy = np.full(len(field), np.nan, dtype=np.float64)

    if valid_indices.size < min_neighbors:
        return dfdx, dfdy

    coords = np.column_stack((x[valid], y[valid]))
    values = field[valid]

    # Escala interna para melhorar o condicionamento do ajuste.
    coord_scale = np.std(coords, axis=0)
    coord_scale = np.where(coord_scale > EPS, coord_scale, 1.0)
    coords_scaled = coords / coord_scale

    k = min(max(n_neighbors, min_neighbors), len(coords_scaled))
    tree = cKDTree(coords_scaled)
    _, neighbor_indices = tree.query(coords_scaled, k=k)

    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]

    for local_i, neighbors in enumerate(neighbor_indices):
        center = coords_scaled[local_i]
        local_coords = coords_scaled[neighbors] - center
        local_values = values[neighbors]

        finite_local = np.isfinite(local_coords).all(axis=1) & np.isfinite(local_values)
        local_coords = local_coords[finite_local]
        local_values = local_values[finite_local]

        if len(local_values) < min_neighbors:
            continue

        design = np.column_stack(
            (
                np.ones(len(local_values), dtype=np.float64),
                local_coords[:, 0],
                local_coords[:, 1],
            )
        )

        try:
            coeffs, _, rank, _ = np.linalg.lstsq(design, local_values, rcond=None)
        except np.linalg.LinAlgError:
            continue

        if rank < 3:
            continue

        global_i = valid_indices[local_i]
        dfdx[global_i] = coeffs[1] / coord_scale[0]
        dfdy[global_i] = coeffs[2] / coord_scale[1]

    return dfdx, dfdy


def calculate_divergence(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    n_neighbors: int = 12,
) -> np.ndarray:
    """Calcula div(U) = dUx/dx + dUy/dy."""
    dux_dx, _ = spatial_gradient_local_ls(x, y, ux, n_neighbors=n_neighbors)
    _, duy_dy = spatial_gradient_local_ls(x, y, uy, n_neighbors=n_neighbors)
    return dux_dx + duy_dy


def summarize_divergence(divergence: np.ndarray) -> dict[str, float | int]:
    """Retorna MAE, RMSE, L2, Linf, média assinada e cobertura válida."""
    divergence = _as_float_array(divergence, "divergence")
    valid = np.isfinite(divergence)
    div = divergence[valid]

    if div.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "l2": float("nan"),
            "linf": float("nan"),
            "mean_signed": float("nan"),
            "std": float("nan"),
            "n_valid": 0,
            "valid_fraction": 0.0,
        }

    return {
        "mae": float(np.mean(np.abs(div))),
        "rmse": float(np.sqrt(np.mean(div**2))),
        "l2": float(np.linalg.norm(div)),
        "linf": float(np.max(np.abs(div))),
        "mean_signed": float(np.mean(div)),
        "std": float(np.std(div)),
        "n_valid": int(div.size),
        "valid_fraction": float(div.size / len(divergence)),
    }


def _relative_reduction(reference: float, corrected: float) -> float:
    """Calcula 1 - corrected/reference."""
    if not np.isfinite(reference) or not np.isfinite(corrected):
        return float("nan")
    if abs(reference) <= EPS:
        return float("nan")
    return float(1.0 - corrected / reference)


def print_divergence_metrics(
    metrics: dict[str, Any],
    model_name: str | None = None,
) -> None:
    """Imprime as métricas físicas em formato padronizado."""
    coarse = metrics["divergence_coarse"]
    fine = metrics["divergence_fine"]
    corrected = metrics["divergence_corrected"]
    error = metrics["divergence_error_corrected_vs_fine"]

    title = "MÉTRICAS FÍSICAS DE CONTINUIDADE"
    if model_name:
        title += f" ({model_name})"

    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)

    def print_block(label: str, values: dict[str, float | int]) -> None:
        print(f"\n{label}")
        print(f"  MAE(div U):                 {values['mae']:.6e}")
        print(f"  RMSE(div U):                {values['rmse']:.6e}")
        print(f"  L2(div U):                  {values['l2']:.6e}")
        print(f"  Linf(div U):                {values['linf']:.6e}")
        print(f"  Média assinada(div U):      {values['mean_signed']:.6e}")
        print(f"  Desvio-padrão(div U):       {values['std']:.6e}")
        print(
            f"  Pontos válidos:             "
            f"{values['n_valid']} ({values['valid_fraction'] * 100:.2f}%)"
        )

    print_block("CFD coarse", coarse)
    print_block("CFD fine", fine)
    print_block("Campo corrigido", corrected)

    print("\nCorrigido versus CFD fine")
    print(f"  MAE do erro de divergência: {error['mae']:.6e}")
    print(f"  RMSE do erro de divergência:{error['rmse']:.6e}")
    print(f"  L2 do erro de divergência:  {error['l2']:.6e}")
    print(f"  Linf do erro de divergência:{error['linf']:.6e}")

    print("\nIndicadores relativos")
    print(
        f"  Redução RMSE vs. coarse:    "
        f"{metrics['improvement_divergence_rmse_pct']:.2f}%"
    )
    print(
        f"  Redução MAE vs. coarse:     "
        f"{metrics['improvement_divergence_mae_pct']:.2f}%"
    )
    print(
        f"  Redução RMSE vs. fine:      "
        f"{metrics['improvement_vs_fine_rmse_pct']:.2f}%"
    )
    print("=" * 64)


def evaluate_divergence_metrics(
    pred_df: pd.DataFrame,
    n_neighbors: int = 12,
    model_name: str | None = None,
    print_results: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Avalia a continuidade dos campos coarse, fine e corrigido.

    Colunas obrigatórias:
        x, y,
        Ux, Uy,
        Ux_f, Uy_f,
        Ux_corr, Uy_corr
    """
    required_columns = {
        "x", "y",
        "Ux", "Uy",
        "Ux_f", "Uy_f",
        "Ux_corr", "Uy_corr",
    }

    missing = required_columns - set(pred_df.columns)
    if missing:
        raise ValueError(
            "Colunas ausentes para cálculo das métricas físicas: "
            f"{sorted(missing)}"
        )

    if len(pred_df) < 4:
        raise ValueError("O DataFrame deve conter pelo menos 4 pontos.")

    result_df = pred_df.copy()
    x = result_df["x"].to_numpy(dtype=np.float64)
    y = result_df["y"].to_numpy(dtype=np.float64)

    div_coarse = calculate_divergence(
        x, y,
        result_df["Ux"].to_numpy(dtype=np.float64),
        result_df["Uy"].to_numpy(dtype=np.float64),
        n_neighbors=n_neighbors,
    )
    div_fine = calculate_divergence(
        x, y,
        result_df["Ux_f"].to_numpy(dtype=np.float64),
        result_df["Uy_f"].to_numpy(dtype=np.float64),
        n_neighbors=n_neighbors,
    )
    div_corrected = calculate_divergence(
        x, y,
        result_df["Ux_corr"].to_numpy(dtype=np.float64),
        result_df["Uy_corr"].to_numpy(dtype=np.float64),
        n_neighbors=n_neighbors,
    )

    div_error_vs_fine = div_corrected - div_fine

    result_df["div_coarse"] = div_coarse
    result_df["div_fine"] = div_fine
    result_df["div_corrected"] = div_corrected
    result_df["div_error_vs_fine"] = div_error_vs_fine

    coarse_metrics = summarize_divergence(div_coarse)
    fine_metrics = summarize_divergence(div_fine)
    corrected_metrics = summarize_divergence(div_corrected)
    error_metrics = summarize_divergence(div_error_vs_fine)

    rer_rmse_vs_coarse = _relative_reduction(
        float(coarse_metrics["rmse"]),
        float(corrected_metrics["rmse"]),
    )
    rer_mae_vs_coarse = _relative_reduction(
        float(coarse_metrics["mae"]),
        float(corrected_metrics["mae"]),
    )
    rer_rmse_vs_fine = _relative_reduction(
        float(fine_metrics["rmse"]),
        float(corrected_metrics["rmse"]),
    )

    metrics: dict[str, Any] = {
        "gradient_method": "local_least_squares",
        "n_neighbors": int(n_neighbors),
        "divergence_coarse": coarse_metrics,
        "divergence_fine": fine_metrics,
        "divergence_corrected": corrected_metrics,
        "divergence_error_corrected_vs_fine": error_metrics,
        "RER_divergence_rmse_vs_coarse": rer_rmse_vs_coarse,
        "RER_divergence_mae_vs_coarse": rer_mae_vs_coarse,
        "RER_divergence_rmse_vs_fine": rer_rmse_vs_fine,
        "improvement_divergence_rmse_pct": float(rer_rmse_vs_coarse * 100.0),
        "improvement_divergence_mae_pct": float(rer_mae_vs_coarse * 100.0),
        "improvement_vs_fine_rmse_pct": float(rer_rmse_vs_fine * 100.0),
    }

    if print_results:
        print_divergence_metrics(metrics, model_name=model_name)

    return metrics, result_df