#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"x", "y", "div_u", "div_u_f", "div_corrected"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compara a divergência corrigida da MLP e da PINN.")
    p.add_argument("--mlp-predictions", required=True, type=Path)
    p.add_argument("--pinn-predictions", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--h", type=float, default=0.0127)
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--x-min", type=float, default=-130.0)
    p.add_argument("--x-max", type=float, default=50.0)
    p.add_argument("--y-min", type=float, default=0.0)
    p.add_argument("--y-max", type=float, default=9.0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet da {model_name} não encontrado: {path}")
    df = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no arquivo da {model_name}: {sorted(missing)}")
    return df


def align_frames(df_mlp: pd.DataFrame, df_pinn: pd.DataFrame, atol: float = 1e-10) -> tuple[pd.DataFrame, pd.DataFrame]:
    mlp = df_mlp.sort_values(["x", "y"]).reset_index(drop=True)
    pinn = df_pinn.sort_values(["x", "y"]).reset_index(drop=True)
    if len(mlp) != len(pinn):
        raise ValueError(f"Números diferentes de pontos: {len(mlp)} versus {len(pinn)}")
    if not np.allclose(mlp["x"], pinn["x"], rtol=0.0, atol=atol) or not np.allclose(mlp["y"], pinn["y"], rtol=0.0, atol=atol):
        raise ValueError("Os domínios espaciais da MLP e da PINN não estão alinhados.")
    for field in ["div_u", "div_u_f"]:
        if not np.allclose(mlp[field], pinn[field], rtol=1e-7, atol=1e-8, equal_nan=True):
            raise ValueError(f"O campo '{field}' difere entre MLP e PINN.")
    return mlp, pinn


def finite_percentile(arrays: list[np.ndarray], percentile: float) -> float:
    values = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Nenhum valor finito disponível para a escala.")
    vmax = float(np.nanpercentile(np.abs(values), percentile))
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


def finite_log_limits(
    arrays: list[np.ndarray],
    percentile: float,
    lower_percentile: float = 1.0,
    epsilon: float = 1e-8,
) -> tuple[float, float]:
    values = np.concatenate([
        np.asarray(array, dtype=np.float64).reshape(-1)
        for array in arrays
    ])

    values = np.abs(values)
    values = values[
        np.isfinite(values)
        & (values > epsilon)
    ]

    if values.size == 0:
        return epsilon, 1.0

    vmin = float(
        np.nanpercentile(
            values,
            lower_percentile,
        )
    )

    vmax = float(
        np.nanpercentile(
            values,
            percentile,
        )
    )

    vmin = max(vmin, epsilon)

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 10.0

    return vmin, vmax


def style_axis(ax, limits: tuple[float, float, float, float]) -> None:
    ax.set_xlabel(r"$x/H$")
    ax.set_ylabel(r"$y/H$")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("auto")


def save_figure(fig, path: Path, dpi: int, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_divergence_three_fields(
    mlp: pd.DataFrame,
    pinn: pd.DataFrame,
    output_path: Path,
    h: float,
    percentile: float,
    limits: tuple[float, float, float, float],
    dpi: int,
    show: bool,
) -> dict[str, float]:
    x = mlp["x"].to_numpy(np.float64) / h
    y = mlp["y"].to_numpy(np.float64) / h

    values = [
        np.abs(
            mlp["div_u_f"].to_numpy(np.float64)
        ),
        np.abs(
            mlp["div_corrected"].to_numpy(np.float64)
        ),
        np.abs(
            pinn["div_corrected"].to_numpy(np.float64)
        ),
    ]

    titles = [
        "RANS fine",
        "MLP corrigida",
        "PINN corrigida",
    ]

    vmin, vmax = finite_log_limits(
        arrays=values,
        percentile=percentile,
        lower_percentile=1.0,
        epsilon=1e-6,
    )

    norm = LogNorm(
        vmin=vmin,
        vmax=vmax,
        clip=True,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22, 5),
        constrained_layout=True,
    )

    scatter = None

    for ax, field, title in zip(
        axes,
        values,
        titles,
    ):
        scatter = ax.scatter(
            x,
            y,
            c=field,
            s=4,
            cmap="inferno",
            norm=norm,
            rasterized=True,
        )

        ax.set_title(title)

        style_axis(
            ax,
            limits,
        )

    colorbar = fig.colorbar(
        scatter,
        ax=axes,
        label=(
            r"$|\nabla\cdot\mathbf{U}|$ "
            r"[s$^{-1}$]"
        ),
    )

    colorbar.ax.set_yscale("log")

    fig.suptitle(
        "Comparação da continuidade — "
        "escala logarítmica comum",
        fontsize=14,
    )

    save_figure(
        fig=fig,
        path=output_path,
        dpi=dpi,
        show=show,
    )

    return {
        "vmin": float(vmin),
        "vmax": float(vmax),
    }


def plot_four_fields(
    mlp: pd.DataFrame,
    pinn: pd.DataFrame,
    path: Path,
    h: float,
    percentile: float,
    limits,
    dpi: int,
    show: bool,
    absolute: bool,
) -> dict[str, float]:
    x = mlp["x"].to_numpy(np.float64) / h
    y = mlp["y"].to_numpy(np.float64) / h

    values = [
        mlp["div_u"].to_numpy(np.float64),
        mlp["div_u_f"].to_numpy(np.float64),
        mlp["div_corrected"].to_numpy(np.float64),
        pinn["div_corrected"].to_numpy(np.float64),
    ]

    titles = [
        "CFD coarse",
        "CFD fine",
        "MLP corrigida",
        "PINN corrigida",
    ]

    if absolute:
        plot_values = [
            np.abs(value)
            for value in values
        ]

        vmin, vmax = finite_log_limits(
            plot_values,
            percentile=percentile,
            lower_percentile=1.0,
            epsilon=1e-6,
        )

        norm = LogNorm(
            vmin=vmin,
            vmax=vmax,
            clip=True,
        )

        cmap = "inferno"
        label = (
            r"$|\nabla\cdot\mathbf{U}|$ "
            r"[s$^{-1}$]"
        )

        suptitle = (
            "Módulo da divergência — "
            "escala logarítmica comum MLP × PINN"
        )

    else:
        plot_values = values

        vmax = finite_percentile(
            plot_values,
            percentile,
        )

        # Região próxima de zero permanece aproximadamente linear.
        linthresh = max(
            vmax * 1e-3,
            1e-6,
        )

        norm = SymLogNorm(
            linthresh=linthresh,
            linscale=1.0,
            vmin=-vmax,
            vmax=vmax,
            base=10,
            clip=True,
        )

        vmin = -vmax
        cmap = "coolwarm"

        label = (
            r"$\nabla\cdot\mathbf{U}$ "
            r"[s$^{-1}$]"
        )

        suptitle = (
            "Divergência assinada — "
            "escala SymLog comum MLP × PINN"
        )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(28, 4.5),
        constrained_layout=True,
    )

    scatter = None

    for ax, field_values, title in zip(
        axes,
        plot_values,
        titles,
    ):
        scatter = ax.scatter(
            x,
            y,
            c=field_values,
            s=4,
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )

        ax.set_title(title)
        style_axis(ax, limits)

    colorbar = fig.colorbar(
        scatter,
        ax=axes,
        label=label,
    )

    if absolute:
        colorbar.ax.set_yscale("log")

    fig.suptitle(
        suptitle,
        fontsize=14,
    )

    save_figure(
        fig,
        path,
        dpi,
        show,
    )

    return {
        "vmin": float(vmin),
        "vmax": float(vmax),
    }


def plot_error_compare(
        mlp: pd.DataFrame, 
        pinn: pd.DataFrame, 
        path: Path, 
        h: float, 
        percentile: float, 
        limits, 
        dpi: int, 
        show: bool
    ) -> float:
    
    x = mlp["x"].to_numpy(np.float64) / h
    y = mlp["y"].to_numpy(np.float64) / h
    fine = mlp["div_u_f"].to_numpy(np.float64)
    errors = [mlp["div_corrected"].to_numpy(np.float64) - fine, pinn["div_corrected"].to_numpy(np.float64) - fine]
    titles = ["MLP corrigida − CFD fine", "PINN corrigida − CFD fine"]
    vmax = finite_percentile(errors, percentile)

    fig, axes = plt.subplots(1, 2, figsize=(22, 5), constrained_layout=True)
    scatter = None
    for ax, error, title in zip(axes, errors, titles):
        scatter = ax.scatter(x, y, c=error, s=4, cmap="coolwarm", vmin=-vmax, vmax=vmax, rasterized=True)
        ax.set_title(title)
        style_axis(ax, limits)
    fig.colorbar(scatter, ax=axes, label=r"$\nabla\cdot U_{corr}-\nabla\cdot U_{fine}$ [s$^{-1}$]")
    fig.suptitle("Erro de divergência em relação ao CFD fine", fontsize=14)
    save_figure(fig, path, dpi, show)
    return vmax

def plot_absolute_error_compare_log(
    mlp: pd.DataFrame,
    pinn: pd.DataFrame,
    path: Path,
    h: float,
    percentile: float,
    limits,
    dpi: int,
    show: bool,
) -> dict[str, float]:
    x = mlp["x"].to_numpy(np.float64) / h
    y = mlp["y"].to_numpy(np.float64) / h

    fine = mlp["div_u_f"].to_numpy(np.float64)

    errors = [
        np.abs(
            mlp["div_corrected"].to_numpy(np.float64)
            - fine
        ),
        np.abs(
            pinn["div_corrected"].to_numpy(np.float64)
            - fine
        ),
    ]

    titles = [
        r"$|\mathrm{MLP}-\mathrm{CFD}_{fine}|$",
        r"$|\mathrm{PINN}-\mathrm{CFD}_{fine}|$",
    ]

    vmin, vmax = finite_log_limits(
        errors,
        percentile=percentile,
        lower_percentile=1.0,
        epsilon=1e-6,
    )

    norm = LogNorm(
        vmin=vmin,
        vmax=vmax,
        clip=True,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(22, 5),
        constrained_layout=True,
    )

    scatter = None

    for ax, error, title in zip(
        axes,
        errors,
        titles,
    ):
        scatter = ax.scatter(
            x,
            y,
            c=error,
            s=4,
            cmap="inferno",
            norm=norm,
            rasterized=True,
        )

        ax.set_title(title)
        style_axis(ax, limits)

    colorbar = fig.colorbar(
        scatter,
        ax=axes,
        label=(
            r"$|\nabla\cdot U_{corr}"
            r"-\nabla\cdot U_{fine}|$ "
            r"[s$^{-1}$]"
        ),
    )

    colorbar.ax.set_yscale("log")

    fig.suptitle(
        "Módulo do erro de divergência — "
        "escala logarítmica comum",
        fontsize=14,
    )

    save_figure(
        fig,
        path,
        dpi,
        show,
    )

    return {
        "vmin": vmin,
        "vmax": vmax,
    }

def plot_pinn_absolute_gain(
    mlp: pd.DataFrame,
    pinn: pd.DataFrame,
    output_path: Path,
    h: float,
    percentile: float,
    limits: tuple[float, float, float, float],
    dpi: int,
    show: bool,
) -> dict[str, float]:
    
    x = mlp["x"].to_numpy(np.float64) / h
    y = mlp["y"].to_numpy(np.float64) / h

    div_fine = (
        mlp["div_u_f"]
        .to_numpy(np.float64)
    )

    div_mlp = (
        mlp["div_corrected"]
        .to_numpy(np.float64)
    )

    div_pinn = (
        pinn["div_corrected"]
        .to_numpy(np.float64)
    )

    error_mlp = np.abs(
        div_mlp - div_fine
    )

    error_pinn = np.abs(
        div_pinn - div_fine
    )

    gain = error_mlp - error_pinn

    valid_gain = gain[
        np.isfinite(gain)
    ]

    if valid_gain.size == 0:
        raise ValueError(
            "Nenhum valor finito no campo de ganho."
        )

    vmax = float(
        np.nanpercentile(
            np.abs(valid_gain),
            percentile,
        )
    )

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax,
    )

    fig, ax = plt.subplots(
        figsize=(14, 5),
        constrained_layout=True,
    )

    scatter = ax.scatter(
        x,
        y,
        c=gain,
        s=4,
        cmap="coolwarm",
        norm=norm,
        rasterized=True,
    )

    style_axis(
        ax,
        limits,
    )

    ax.set_title(
        "Ganho da PINN sobre a MLP\n"
        "positivo: PINN melhor | negativo: MLP melhor"
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        label=(
            r"$|e_{\mathrm{MLP}}|"
            r"-|e_{\mathrm{PINN}}|$ "
            r"[s$^{-1}$]"
        ),
    )

    save_figure(
        fig=fig,
        path=output_path,
        dpi=dpi,
        show=show,
    )

    improved_fraction = float(
        np.mean(gain > 0)
    )

    worsened_fraction = float(
        np.mean(gain < 0)
    )

    return {
        "vmax": vmax,
        "mean_gain": float(
            np.nanmean(gain)
        ),
        "median_gain": float(
            np.nanmedian(gain)
        ),
        "improved_fraction": improved_fraction,
        "worsened_fraction": worsened_fraction,
    }


def plot_local_scales(
        mlp: pd.DataFrame, 
        pinn: pd.DataFrame, 
        path: Path, 
        h: float, 
        percentile: float, 
        limits, 
        dpi: int, 
        show: bool
    ) -> dict[str, float]:

    x = mlp["x"].to_numpy(np.float64) / h
    y = mlp["y"].to_numpy(np.float64) / h
    values = [np.abs(mlp["div_corrected"].to_numpy(np.float64)), np.abs(pinn["div_corrected"].to_numpy(np.float64))]
    titles = ["MLP corrigida — escala local", "PINN corrigida — escala local"]
    vmaxes = [finite_percentile([values[0]], percentile), finite_percentile([values[1]], percentile)]

    fig, axes = plt.subplots(1, 2, figsize=(22, 5), constrained_layout=True)
    for ax, field_values, vmax, title in zip(axes, values, vmaxes, titles):
        scatter = ax.scatter(x, y, c=field_values, s=4, cmap="inferno", vmin=0.0, vmax=vmax, rasterized=True)
        ax.set_title(title)
        style_axis(ax, limits)
        fig.colorbar(scatter, ax=ax, label=r"$|\nabla\cdot\mathbf{U}|$ [s$^{-1}$]")
    fig.suptitle("Estrutura espacial da divergência — escalas individuais", fontsize=14)
    save_figure(fig, path, dpi, show)
    return {"vmax_mlp": vmaxes[0], "vmax_pinn": vmaxes[1]}


def summary(df: pd.DataFrame, fine: np.ndarray) -> dict[str, float]:
    values = df["div_corrected"].to_numpy(np.float64)
    values = values[np.isfinite(values)]
    error = df["div_corrected"].to_numpy(np.float64) - fine
    error = error[np.isfinite(error)]
    return {
        "mae_abs_div": float(np.mean(np.abs(values))),
        "rmse_div": float(np.sqrt(np.mean(values**2))),
        "linf_div": float(np.max(np.abs(values))),
        "mean_signed_div": float(np.mean(values)),
        "std_div": float(np.std(values)),
        "mae_error_vs_fine": float(np.mean(np.abs(error))),
        "rmse_error_vs_fine": float(np.sqrt(np.mean(error**2))),
        "linf_error_vs_fine": float(np.max(np.abs(error))),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.percentile <= 100.0:
        raise ValueError("--percentile deve estar entre 0 e 100.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mlp, pinn = align_frames(load_predictions(args.mlp_predictions, "MLP"), load_predictions(args.pinn_predictions, "PINN"))
    limits = (args.x_min, args.x_max, args.y_min, args.y_max)

    absolute_path = args.output_dir / "divergence_mlp_vs_pinn_absolute_common_scale.png"
    signed_path = args.output_dir / "divergence_mlp_vs_pinn_signed_common_scale.png"
    error_path = args.output_dir / "divergence_error_mlp_vs_pinn_common_scale.png"
    error_log_path = (
        args.output_dir
        / "divergence_absolute_error_mlp_vs_pinn_log_scale.png"
    )
    local_path = args.output_dir / "divergence_mlp_vs_pinn_local_scales.png"
    gain_path = (
            args.output_dir
            / "pinn_gain_over_mlp_divergence.png"
        )
    

    # absolute_scale = plot_four_fields(
    #     mlp,
    #     pinn,
    #     absolute_path,
    #     args.h,
    #     args.percentile,
    #     limits,
    #     args.dpi,
    #     args.show,
    #     True,
    # )

    absolute_scale = plot_divergence_three_fields(
        mlp=mlp,
        pinn=pinn,
        output_path=absolute_path,
        h=args.h,
        percentile=args.percentile,
        limits=limits,
        dpi=args.dpi,
        show=args.show,
    )

    signed_scale = plot_divergence_three_fields(
        mlp=mlp,
        pinn=pinn,
        output_path=signed_path,
        h=args.h,
        percentile=args.percentile,
        limits=limits,
        dpi=args.dpi,
        show=args.show,
    )

    error_vmax = plot_error_compare(
        mlp, 
        pinn, 
        error_path, 
        args.h, 
        args.percentile, 
        limits, 
        args.dpi, 
        args.show
    )
    
    error_log_scale = plot_absolute_error_compare_log(
        mlp=mlp,
        pinn=pinn,
        path=error_log_path,
        h=args.h,
        percentile=args.percentile,
        limits=limits,
        dpi=args.dpi,
        show=args.show,
    )

    local_scales = plot_local_scales(
        mlp, 
        pinn, 
        local_path, 
        args.h, args.
        percentile, 
        limits, 
        args.dpi, 
        args.show
    )

    gain_metrics = plot_pinn_absolute_gain(
        mlp=mlp,
        pinn=pinn,
        output_path=gain_path,
        h=args.h,
        percentile=args.percentile,
        limits=limits,
        dpi=args.dpi,
        show=args.show,
    )

    fine = mlp["div_u_f"].to_numpy(np.float64)
    report = {
        "mlp_predictions": str(args.mlp_predictions.resolve()),
        "pinn_predictions": str(args.pinn_predictions.resolve()),
        "n_points": int(len(mlp)),
        "h": args.h,
        "percentile": args.percentile,
        "scales": {
            "absolute_common": absolute_scale,
            "signed_common": signed_scale,
            "error_common_abs_vmax": error_vmax,
            **local_scales,
        },
        "summary": {"mlp": summary(mlp, fine), "pinn": summary(pinn, fine)},
        "plots": {"absolute_common": str(absolute_path.resolve()), "signed_common": str(signed_path.resolve()), "error_common": str(error_path.resolve()), "local_scales": str(local_path.resolve())},
    }
    report_path = args.output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Comparação concluída. Relatório: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
