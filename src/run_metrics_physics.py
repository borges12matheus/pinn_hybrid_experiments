import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from metrics_physics import evaluate_divergence_metrics
from run_metrics import build_model


@torch.no_grad()
def predict_full_domain(
    model,
    df: pd.DataFrame,
    feat_cols: list[str],
    xscaler,
    yscaler,
    batch_size: int = 4096,
) -> pd.DataFrame:
    model.eval()
    device = next(model.parameters()).device

    x_mu, x_sd = xscaler
    y_mu, y_sd = yscaler

    X = df[feat_cols].to_numpy(dtype=np.float32)
    x_sd_safe = np.where(np.abs(x_sd) > 1e-12, x_sd, 1.0)
    Xn = (X - x_mu) / x_sd_safe

    predictions = np.zeros((len(df), 3), dtype=np.float32)

    for i0 in range(0, len(df), batch_size):
        i1 = min(i0 + batch_size, len(df))

        xb = torch.as_tensor(
            Xn[i0:i1],
            dtype=torch.float32,
            device=device,
        )

        output_normalized = model(xb)[:, :3].cpu().numpy()
        predictions[i0:i1] = (
            output_normalized * y_sd + y_mu
        )

    result_df = df[
        ["x", "y", "Ux", "Uy", "p", "dUx", "dUy", "dp"]
    ].copy()

    result_df["Ux_f"] = result_df["Ux"] + result_df["dUx"]
    result_df["Uy_f"] = result_df["Uy"] + result_df["dUy"]
    result_df["p_f"] = result_df["p"] + result_df["dp"]

    result_df["dUx_pred"] = predictions[:, 0]
    result_df["dUy_pred"] = predictions[:, 1]
    result_df["dp_pred"] = predictions[:, 2]

    result_df["Ux_corr"] = (
        result_df["Ux"] + result_df["dUx_pred"]
    )
    result_df["Uy_corr"] = (
        result_df["Uy"] + result_df["dUy_pred"]
    )
    result_df["p_corr"] = (
        result_df["p"] + result_df["dp_pred"]
    )

    return result_df

def run_physics_metrics_pipeline(
    cfg,
    model_path,
    dataset_full_path,
    xscaler_path,
    yscaler_path,
    metrics_path,
    predictions_path,
    batch_size=4096,
    n_neighbors=12,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg, model_path).to(device)

    xscaler = joblib.load(xscaler_path)
    yscaler = joblib.load(yscaler_path)

    df_full = pd.read_parquet(dataset_full_path)
    required_columns = {
        *cfg["features"],
        "x",
        "y",
        "Ux",
        "Uy",
        "p",
        "dUx",
        "dUy",
        "dp",
    }

    missing = required_columns - set(df_full.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes no dataset completo: {sorted(missing)}"
        )
    

    pred_df = predict_full_domain(
        model=model,
        df=df_full,
        feat_cols=cfg["features"],
        xscaler=xscaler,
        yscaler=yscaler,
        batch_size=batch_size,
    )

    physics_metrics, physics_df = evaluate_divergence_metrics(
        pred_df=pred_df,
        n_neighbors=n_neighbors,
        model_name=cfg["experiment"]["name"],
        print_results=True,
    )

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            physics_metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    physics_df.to_parquet(
        predictions_path,
        index=False,
    )

    return {
        "metrics": physics_metrics,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
    }