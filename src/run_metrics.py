import argparse
from pathlib import Path
from datetime import datetime
import joblib
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import yaml

from logger import ExperimentLogger
from metrics import evaluate_metrics
from plots import (
    plot_error_compare,
    plot_error_histogram,
    plot_field_compare,
)
from train_utils import MLP


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model(cfg, model_path):
    in_dim = len(cfg["features"])
    out_dim = len(cfg["targets"])
    width = cfg["model"]["width"]
    depth = cfg["model"]["depth"]

    model = MLP(in_dim=in_dim, out_dim=out_dim, width=width, depth=depth, act=nn.Tanh)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def run_metrics_pipeline(
    cfg,
    model_path=None,
    dataset_path=None,
    xscaler_path=None,
    yscaler_path=None,
    metrics_path=None,
    predictions_path=None,
    plots_dir=None,
    logs_dir=None,
    batch_size=4900,
    logger=None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = cfg["model"]["type"]
    exp_name = cfg["experiment"]["name"]
    width = cfg["model"]["width"]
    depth = cfg["model"]["depth"]
    seed = cfg["experiment"]["seed"]


    root = Path(cfg["paths"]["root"])
    path_data = root / cfg["paths"]["data_process_dir"]
    path_metric = root / cfg["paths"]["metrics_dir"]
    path_plot = root / cfg["paths"]["plots_dir"]
    path_log = root / cfg["paths"]["logs_dir"]

    if model_path is None:
        model_path = root / cfg["paths"]["models_dir"] / f"{model_type}_{exp_name}_d{depth}_w{width}.pt"
    if dataset_path is None:
        dataset_path = path_data / f"dataset_test_{model_type}_{exp_name}_d{depth}_w{width}.parquet"
    if xscaler_path is None:
        xscaler_path = root / cfg["paths"]["models_dir"] / f"{model_type}_scaler_X.pkl"
    if yscaler_path is None:
        yscaler_path = root / cfg["paths"]["models_dir"] / f"{model_type}_scaler_Y.pkl"
    if metrics_path is None:
        metrics_path = path_metric / f"{model_type}" / f"{model_type}_{exp_name}_d{depth}_w{width}_seed{seed}.json"
    if predictions_path is None:
        predictions_path = path_metric / f"{model_type}_{exp_name}_d{depth}_w{width}_seed{seed}_predictions.parquet"
    if plots_dir is None:
        plots_dir = path_plot / f"{model_type}_{exp_name}_d{depth}_w{width}_seed{seed}"
    if logs_dir is None:
        logs_dir = path_log / f"{model_type}" / f"{model_type}_{cfg['experiment']['name']}_{timestamp}"
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)

    if logger is None:
        logger = ExperimentLogger(
            experiment_name=f"{model_type}_{cfg['experiment']['name']}",
            experiment_type=f"{cfg['experiment']['type']}",
            log_dir= logs_dir,
            config=cfg
        )

    logger.log_message(f"Iniciando avaliacao final para {exp_name}")
    logger.log_message(f"Modelo: {model_path}")
    logger.log_message(f"Dataset: {dataset_path}")

    xscaler = joblib.load(xscaler_path)
    yscaler = joblib.load(yscaler_path)
    model = build_model(cfg, model_path).to("cuda" if torch.cuda.is_available() else "cpu")

    metrics, pred_df = evaluate_metrics(
        model=model,
        parquet_test_path=dataset_path,
        feat_cols=cfg["features"],
        xscaler=xscaler,
        yscaler=yscaler,
        batch_size=batch_size,
        model_metrics=exp_name,
        out_metrics=metrics_path,
        out_predictions=predictions_path,
        return_predictions=True,
    )

    plot_specs = [
        ("Ux", "Ux_f", "Ux_corr", "velocity_u"),
        ("Uy", "Uy_f", "Uy_corr", "velocity_v"),
        ("p", "p_f", "p_corr", "pressure"),
    ]

    for field_c, field_f, field_corr, label in plot_specs:
        plot_field_compare(
            pred_df,
            field_c,
            field_f,
            field_corr,
            title=label,
            save_path=plots_dir / f"{label}_field_compare.png",
        )

        plot_error_compare(
            pred_df,
            field_c,
            field_f,
            field_corr,
            label=label,
            save_path=plots_dir / f"{label}_error_compare.png",
        )

        plot_error_histogram(
            pred_df,
            field_c,
            field_f,
            field_corr,
            label=label,
            save_path=plots_dir / f"{label}_error_hist.png",
        )

    logger.log_message(f"Avaliacao concluida. Metrics: {metrics_path}")
    logger.log_message(f"Predictions: {predictions_path}")
    logger.log_message(f"Plots: {plots_dir}")

    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "plots_dir": str(plots_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--xscaler-path", default=None)
    parser.add_argument("--yscaler-path", default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--plots-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_metrics_pipeline(
        cfg=cfg,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        xscaler_path=args.xscaler_path,
        yscaler_path=args.yscaler_path,
        metrics_path=args.metrics_path,
        predictions_path=args.predictions_path,
        plots_dir=args.plots_dir,
    )
    print(result)


if __name__ == "__main__":
    main()
