# Inicializa as bibliotecas necessárias
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch

# ----------------------------
# Métricas
# ----------------------------
def mae_vec(u1, v1, u2, v2):
        return float(np.mean(np.sqrt((u1 - u2)**2 + (v1 - v2)**2)))
    
def mae_vec_pressure(p1, p2):
    return float(np.mean(np.abs(p1 - p2)))

def rmse_uv(u1, v1, u2, v2):
    return float(np.sqrt(np.mean((u1-u2)**2 + (v1-v2)**2)))

def rmse_p(p1, p2):
    return float(np.sqrt(np.mean((p1-p2)**2)))

def l2_relative_uv(u_pred, v_pred, u_ref, v_ref):
    num = np.sqrt(np.sum((u_pred - u_ref)**2 + (v_pred - v_ref)**2))
    den = np.sqrt(np.sum(u_ref**2 + v_ref**2))
    return float(num / (den + 1e-12))

def l2_relative_p(p_pred, p_ref):
    num = np.linalg.norm(p_pred - p_ref)
    den = np.linalg.norm(p_ref)
    return float(num / (den + 1e-12))

# -------------------------------
# Função de Avaliação / Métricas
# -------------------------------
@torch.no_grad()
def evaluate_metrics(
    model, parquet_test_path, feat_cols, xscaler, yscaler,
    batch_size=4900, model_metrics = None,
    out_metrics=None,
    out_predictions=None,
    return_predictions=False
):

    model.eval()
    device = next(model.parameters()).device

    df = pd.read_parquet(parquet_test_path)

    Ux_c = df["Ux"].to_numpy(np.float32)
    Uy_c = df["Uy"].to_numpy(np.float32)
    p_c  = df["p"].to_numpy(np.float32)

    dUx_true = df["dUx"].to_numpy(np.float32)
    dUy_true = df["dUy"].to_numpy(np.float32)
    dp_true  = df["dp"].to_numpy(np.float32)

    Ux_f = Ux_c + dUx_true
    Uy_f = Uy_c + dUy_true
    p_f  = p_c  + dp_true

    X = df[feat_cols].to_numpy(np.float32)
    x_mu, x_sd = xscaler
    y_mu, y_sd = yscaler
    Xn = (X - x_mu) / x_sd

    dUx_pred = np.zeros(len(df), dtype=np.float32)
    dUy_pred = np.zeros(len(df), dtype=np.float32)
    dp_pred = np.zeros(len(df), dtype=np.float32)
    
    for i0 in range(0, len(df), batch_size):
        i1 = min(i0 + batch_size, len(df))
        xb = torch.tensor(Xn[i0:i1], dtype=torch.float32, device=device)
        out_n = model(xb)[:, :3].detach().cpu().numpy()
        out = out_n * y_sd + y_mu
        dUx_pred[i0:i1] = out[:, 0]
        dUy_pred[i0:i1] = out[:, 1]
        dp_pred[i0:i1] = out[:, 2]

    Ux_hat = Ux_c + dUx_pred
    Uy_hat = Uy_c + dUy_pred
    p_hat = p_c + dp_pred

    # Plotagem comparativa dos perfis de velocidade e pressão
    pred_df = df[["x", "y", "Ux", "Uy", "p", "dUx", "dUy", "dp"]].copy()

    pred_df["Ux_f"] = pred_df["Ux"] + pred_df["dUx"]
    pred_df["Uy_f"] = pred_df["Uy"] + pred_df["dUy"]
    pred_df["p_f"]  = pred_df["p"]  + pred_df["dp"]

    pred_df["dUx_pred"] = dUx_pred
    pred_df["dUy_pred"] = dUy_pred
    pred_df["dp_pred"]  = dp_pred

    pred_df["Ux_corr"] = pred_df["Ux"] + pred_df["dUx_pred"]
    pred_df["Uy_corr"] = pred_df["Uy"] + pred_df["dUy_pred"]
    pred_df["p_corr"]  = pred_df["p"]  + pred_df["dp_pred"]
    
    #MAE
    mae_coarse = mae_vec(Ux_c, Uy_c, Ux_f, Uy_f)
    mae_p_coarse = mae_vec_pressure(p_c, p_f)
    mae_corr   = mae_vec(Ux_hat, Uy_hat, Ux_f, Uy_f)
    mae_p_corr = mae_vec_pressure(p_hat, p_f)
    mae_deltas = mae_vec(dUx_pred, dUy_pred, dUx_true, dUy_true)
    mae_deltas_p = mae_vec_pressure(dp_pred, dp_true)

    #RMSE
    rmse_uv_coarse = rmse_uv(Ux_c, Uy_c, Ux_f, Uy_f)
    rmse_uv_corr = rmse_uv(Ux_hat, Uy_hat, Ux_f, Uy_f)
    rmse_p_coarse = rmse_p(p_c, p_f)
    rmse_p_corr = rmse_p(p_hat, p_f)

    #RER
    rer_uv = 1.0 - (mae_corr / (mae_coarse + 1e-12))
    melhora_pct_uv = rer_uv * 100.0
    rer_p = 1.0 - (mae_p_corr / (mae_p_coarse + 1e-12))
    melhora_pct_p = rer_p * 100.0

    #L2
    l2_coarse_uv = l2_relative_uv(Ux_c, Uy_c, Ux_f, Uy_f)
    l2_corr_uv   = l2_relative_uv(Ux_hat, Uy_hat, Ux_f, Uy_f)
    l2_coarse_p = l2_relative_p(p_c, p_f)
    l2_corr_p   = l2_relative_p(p_hat, p_f)

    # -------------------------------------------------
    # Métricas de dados e física
    # -------------------------------------------------

    metrics = {
        "mae_uv_coarse_to_fine": mae_coarse,
        "mae_uv_corrected_to_fine": mae_corr,
        "mae_uv_deltas_pred_vs_true": mae_deltas,
        "mae_p_pressure_coarse_to_fine": mae_p_coarse,
        "mae_p_corrected_to_fine": mae_p_corr,
        "mae_p_deltas_pred_vs_true": mae_deltas_p,
        "rmse_uv_coarse_to_fine": rmse_uv_coarse,
        "rmse_uv_corrected_to_fine": rmse_uv_corr,
        "rmse_p_coarse_to_fine": rmse_p_coarse,
        "rmse_p_corrected_to_fine": rmse_p_corr,
        "l2_rel_uv_coarse": l2_coarse_uv,
        "l2_rel_uv_corrected": l2_corr_uv,
        "l2_rel_p_coarse": l2_coarse_p,
        "l2_rel_p_corrected": l2_corr_p,
        "RER (u,v)": float(rer_uv),
        "RER (p)": float(rer_p),
        "melhora_MAE_pct": float(melhora_pct_uv),
        "melhora_MAE_pct_pressure": float(melhora_pct_p),
        "N": int(len(df)),
    }

    if out_metrics is not None:
        out_metrics = Path(out_metrics)
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        with open(out_metrics, "w") as f:
            json.dump(metrics, f, indent=2)

    if out_predictions is not None:
        out_predictions = Path(out_predictions)
        out_predictions.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(out_predictions, index=False)

    print(f"\n===== Métricas ({model_metrics}) =====")
    print(f"MAE vetorial (u,v) (coarse -> fine):      {mae_coarse:.6f}")
    print(f"MAE pressão (p) (coarse -> fine):         {mae_p_coarse:.6f}")
    print(f"MAE vetorial (u,v) (corrigido -> fine):   {mae_corr:.6f}")
    print(f"MAE pressão (p) (corrigido -> fine):      {mae_p_corr:.6f}")
    print(f"MAE vetorial (ΔU_pred -> ΔU_true):        {mae_deltas:.6f}")
    print(f"MAE pressão (Δp_pred -> Δp_true):         {mae_deltas_p:.6f}")
    print(f"RMSE (u,v) (coarse -> fine):              {rmse_uv_coarse:.6f}")
    print(f"RMSE (u,v) (corrigido -> fine):           {rmse_uv_corr:.6f}")
    print(f"RMSE (p)   (coarse -> fine):              {rmse_p_coarse:.6f}")
    print(f"RMSE (p)   (corrigido -> fine):           {rmse_p_corr:.6f}")
    print(f"RER (u,v):                                {rer_uv:.4f}")
    print(f"RER (p):                                  {rer_p:.4f}")
    print(f"Melhora (u,v) (MAE):                      {melhora_pct_uv:.2f}%")
    print(f"Melhora (p) (MAE):                        {melhora_pct_p:.2f}%")

    print(f"\n===== Métricas Físicas ({model_metrics}) =====")
    print(f"L2 relativo (u,v) - Coarse:               {l2_coarse_uv:.6e}")
    print(f"L2 relativo (u,v) - Corrigido:            {l2_corr_uv:.6e}")
    print(f"L2 relativo (p) - Coarse:                 {l2_coarse_p:.6e}")
    print(f"L2 relativo (p) - Corrigido:              {l2_corr_p:.6e}")
    if out_metrics is not None:
        print(f"Salvo: {out_metrics}\n")
    if out_predictions is not None:
        print(f"Predições salvas em: {out_predictions}\n")

    if return_predictions:
        return metrics, pred_df
    return metrics
