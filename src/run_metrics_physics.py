import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from metrics_physics import evaluate_exported_divergence_metrics
from plots import (
    plot_divergence_compare,
    plot_divergence_error,
)
from run_metrics import build_model

def predict_full_domain_with_divergence(
    model,
    df: pd.DataFrame,
    feat_cols: list[str],
    xscaler,
    yscaler,
    batch_size: int = 4096,
) -> pd.DataFrame:
    model.eval()
    device = next(model.parameters()).device

    x_mu_np, x_sd_np = xscaler
    y_mu_np, y_sd_np = yscaler

    x_mu_np = np.asarray(x_mu_np, dtype=np.float32).reshape(-1)
    x_sd_np = np.asarray(x_sd_np, dtype=np.float32).reshape(-1)

    y_mu_np = np.asarray(y_mu_np, dtype=np.float32).reshape(-1)
    y_sd_np = np.asarray(y_sd_np, dtype=np.float32).reshape(-1)

    x_mu = torch.as_tensor(
        x_mu_np,
        dtype=torch.float32,
        device=device,
    )
    x_sd = torch.as_tensor(
        x_sd_np,
        dtype=torch.float32,
        device=device,
    )
    y_mu = torch.as_tensor(
        y_mu_np,
        dtype=torch.float32,
        device=device,
    )
    y_sd = torch.as_tensor(
        y_sd_np,
        dtype=torch.float32,
        device=device,
    )

    x_sd_safe = torch.where(
        torch.abs(x_sd) > 1e-12,
        x_sd,
        torch.ones_like(x_sd),
    )

    feat_index = {
        name: idx
        for idx, name in enumerate(feat_cols)
    }

    required_features = {"x", "y"}
    missing = required_features - set(feat_index)

    if missing:
        raise ValueError(
            f"Features espaciais ausentes: {sorted(missing)}"
        )

    X_phys_all = df[feat_cols].to_numpy(np.float32)

    n_outputs = int(np.asarray(y_mu_np).reshape(-1).shape[0])

    if n_outputs < 3:
        raise ValueError(
            f"O scaler de saída possui apenas {n_outputs} variável(is), "
            "mas o modelo físico exige pelo menos dUx, dUy e dp."
        )

    predictions = np.zeros(
        (len(df), n_outputs),
        dtype=np.float32,
    )

    div_delta = np.zeros(
        len(df),
        dtype=np.float32,
    )

    for i0 in range(0, len(df), batch_size):
        i1 = min(i0 + batch_size, len(df))

        with torch.enable_grad():
            X_phys = torch.as_tensor(
                X_phys_all[i0:i1],
                dtype=torch.float32,
                device=device,
            )

            x = (
                X_phys[:, [feat_index["x"]]]
                .clone()
                .detach()
                .requires_grad_(True)
            )

            y = (
                X_phys[:, [feat_index["y"]]]
                .clone()
                .detach()
                .requires_grad_(True)
            )

            X_phys_mod = X_phys.clone()

            X_phys_mod[:, [feat_index["x"]]] = x
            X_phys_mod[:, [feat_index["y"]]] = y

            Xn = (
                X_phys_mod - x_mu
            ) / x_sd_safe

            pred_n = model(Xn)

            pred_phys = (
                pred_n * y_sd
                + y_mu
            )

            dUx = pred_phys[:, [0]]
            dUy = pred_phys[:, [1]]

            if not dUx.requires_grad:
                raise RuntimeError(
                    "dUx não possui gradiente. "
                    "Verifique uso de torch.no_grad()."
                )

            dUx_dx = torch.autograd.grad(
                outputs=dUx,
                inputs=x,
                grad_outputs=torch.ones_like(dUx),
                create_graph=False,
                retain_graph=True,
                allow_unused=False,
            )[0]

            dUy_dy = torch.autograd.grad(
                outputs=dUy,
                inputs=y,
                grad_outputs=torch.ones_like(dUy),
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]

        predictions[i0:i1] = (
            pred_phys
            .detach()
            .cpu()
            .numpy()
        )

        div_delta[i0:i1] = (
            dUx_dx + dUy_dy
        ).detach().cpu().numpy().reshape(-1)

    result_df = df[
        [
            "x",
            "y",
            "Ux",
            "Uy",
            "p",
            "dUx",
            "dUy",
            "dp",
            "div_u",
            "div_u_f",
        ]
    ].copy()

    result_df["dUx_pred"] = predictions[:, 0]
    result_df["dUy_pred"] = predictions[:, 1]
    result_df["dp_pred"] = predictions[:, 2]

    result_df["Ux_f"] = (
        result_df["Ux"]
        + result_df["dUx"]
    )

    result_df["Uy_f"] = (
        result_df["Uy"]
        + result_df["dUy"]
    )

    result_df["p_f"] = (
        result_df["p"]
        + result_df["dp"]
    )

    result_df["Ux_corr"] = (
        result_df["Ux"]
        + result_df["dUx_pred"]
    )

    result_df["Uy_corr"] = (
        result_df["Uy"]
        + result_df["dUy_pred"]
    )

    result_df["p_corr"] = (
        result_df["p"]
        + result_df["dp_pred"]
    )

    result_df["div_delta_pred"] = div_delta

    result_df["div_corrected"] = (
        result_df["div_u"]
        + result_df["div_delta_pred"]
    )

    result_df["div_error_vs_fine"] = (
        result_df["div_corrected"]
        - result_df["div_u_f"]
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
        plots_dir=None
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
    

    pred_df = predict_full_domain_with_divergence(
        model=model,
        df=df_full,
        feat_cols=cfg["features"],
        xscaler=xscaler,
        yscaler=yscaler,
        batch_size=batch_size,
    )

    physics_metrics, physics_df = (
        evaluate_exported_divergence_metrics(
            pred_df=pred_df,
            model_name=cfg["experiment"]["name"],
            print_results=True,
        )
    )

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            physics_metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    physics_df.to_parquet(
        predictions_path,
        index=False,
    )

    generated_plots = {}

    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        plot_specs = {
            "divergence_absolute": {
                "path": plots_dir / "divergence_absolute.png",
                "absolute": True,
            },
            "divergence_signed": {
                "path": plots_dir / "divergence_signed.png",
                "absolute": False,
            },
        }

        for plot_name, spec in plot_specs.items():
            plot_divergence_compare(
                df_plot=physics_df,
                field_coarse="div_u",
                field_fine="div_u_f",
                field_corrected="div_corrected",
                model_name=f'{cfg["model"]["type"]}_{cfg["experiment"]["name"]}',
                absolute=spec["absolute"],
                save_path=spec["path"],
            )

            generated_plots[plot_name] = str(
                spec["path"]
            )

        error_path = (
            plots_dir
            / "divergence_error_vs_fine.png"
        )

        plot_divergence_error(
            df_plot=physics_df,
            corrected_field="div_corrected",
            fine_field="div_u_f",
            model_name=f'{cfg["model"]["type"]}_{cfg["experiment"]["name"]}',
            save_path=error_path,
        )

        generated_plots[
            "divergence_error_vs_fine"
        ] = str(error_path)

    return {
        "metrics": physics_metrics,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "plots": generated_plots,
    }