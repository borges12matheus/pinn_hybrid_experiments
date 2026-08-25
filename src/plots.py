import numpy as np
import matplotlib.pyplot as plt

def _finalize_figure(fig, save_path=None, show=False, close=True):
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)


# ----------------------------
# Plots Acurácia
# ----------------------------
def plot_field_compare(
    df_plot,
    field_c,
    field_f,
    field_corr,
    title,
    h=0.0127,
    cmap="viridis",
    save_path=None,
    show=False,
    close=True,
):
    # Coordenadas adimensionais
    x = df_plot["x"].values / h
    y = df_plot["y"].values / h

    vals = np.concatenate([
        df_plot[field_c].values,
        df_plot[field_f].values,
        df_plot[field_corr].values
    ])

    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22, 4),
        constrained_layout=True
    )

    titles = ["Coarse", "Fine", "Corrigido"]
    fields = [field_c, field_f, field_corr]

    for ax, f, t in zip(axes, fields, titles):

        sc = ax.scatter(
            x,
            y,
            c=df_plot[f].values,
            s=4,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(f"{title} — {t}")

        ax.set_xlabel(r"$x/h$")
        ax.set_ylabel(r"$y/h$")

        # Domínio do Driver & Seegmiller
        ax.set_xlim([-130, 50])
        ax.set_ylim([0, 9])

        # Não preservar a razão física extrema
        ax.set_aspect("auto")

    fig.colorbar(sc, ax=axes, label=title)
    _finalize_figure(fig, save_path=save_path, show=show, close=close)

# -------------------------------------------------
# 5.1. Mapas de erro: Coarse - Fine vs Corrigido - Fine
# -------------------------------------------------

def plot_error_compare(
    df_plot,
    field_c,
    field_f,
    field_corr,
    label,
    h=0.0127,
    save_path=None,
    show=False,
    close=True,
):
    x = df_plot["x"].values / h
    y = df_plot["y"].values / h

    err_base = df_plot[field_c].values - df_plot[field_f].values
    err_corr = df_plot[field_corr].values - df_plot[field_f].values

    vmax = max(
        np.nanmax(np.abs(err_base)),
        np.nanmax(np.abs(err_corr))
    )
    vmin = -vmax

    fig, axes = plt.subplots(
        1, 2,
        figsize=(22, 6),
        constrained_layout=True
    )

    plots = [
        (axes[0], err_base, f"Erro {label}: Coarse - Fine"),
        (axes[1], err_corr, f"Erro {label}: Corrigido - Fine"),
    ]

    for ax, err, title in plots:
        sc = ax.scatter(
            x, y,
            c=err,
            s=4,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(title)
        ax.set_xlabel(r"$x/h$")
        ax.set_ylabel(r"$y/h$")
        ax.set_xlim([-130, 50])
        ax.set_ylim([0, 9])
        ax.set_aspect("auto")

    fig.colorbar(sc, ax=axes, label=f"Erro {label}")

    _finalize_figure(fig, save_path=save_path, show=show, close=close)


# -------------------------------------------------
# 5.2. Ganho local: |erro coarse| - |erro corrigido|
# -------------------------------------------------

def plot_local_gain(
    df_plot,
    field_c,
    field_f,
    field_corr,
    label,
    h=0.0127,
    save_path=None,
    show=False,
    close=True,
):
    x = df_plot["x"].values / h
    y = df_plot["y"].values / h

    err_base = df_plot[field_c].values - df_plot[field_f].values
    err_corr = df_plot[field_corr].values - df_plot[field_f].values

    gain = np.abs(err_base) - np.abs(err_corr)

    vmax = np.nanmax(np.abs(gain))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)

    sc = ax.scatter(
        x, y,
        c=gain,
        s=4,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(f"Ganho local — {label}")
    ax.set_xlabel(r"$x/h$")
    ax.set_ylabel(r"$y/h$")
    ax.set_xlim([-130, 50])
    ax.set_ylim([0, 9])
    ax.set_aspect("auto")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(f"|erro coarse| - |erro corrigido| ({label})")

    _finalize_figure(fig, save_path=save_path, show=show, close=close)

def plot_error_histogram(
    df_plot,
    field_c,
    field_f,
    field_corr,
    label,
    bins=100,
    save_path=None,
    show=False,
    close=True,
):
    err_coarse = np.abs(
        df_plot[field_c].values -
        df_plot[field_f].values
    )

    err_corr = np.abs(
        df_plot[field_corr].values -
        df_plot[field_f].values
    )

    fig, ax = plt.subplots(
        figsize=(8, 5),
        constrained_layout=True
    )

    ax.hist(
        err_coarse,
        bins=bins,
        alpha=0.6,
        density=True,
        label="Coarse",
        edgecolor="black"
    )

    ax.hist(
        err_corr,
        bins=bins,
        alpha=0.6,
        density=True,
        label="Corrigido",
        edgecolor="black"
    )

    ax.set_xlabel(f"|Erro| ({label})")
    ax.set_ylabel("Densidade")

    ax.set_title(
        f"Distribuição do erro absoluto - {label}"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    _finalize_figure(fig, save_path=save_path, show=show, close=close)


def plot_scatter_prediction(
    df_plot,
    field_f,
    field_corr,
    label,
    sample_frac=1.0,
    save_path=None,
    show=False,
    close=True,
):
    df_tmp = df_plot

    if sample_frac < 1.0:
        df_tmp = df_plot.sample(
            frac=sample_frac,
            random_state=42
        )

    x = df_tmp[field_f].values
    y = df_tmp[field_corr].values

    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())

    fig, ax = plt.subplots(
        figsize=(6, 6),
        constrained_layout=True
    )

    ax.scatter(
        x,
        y,
        s=3,
        alpha=0.3
    )

    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        'r--',
        linewidth=2,
        label="y=x"
    )

    ax.set_xlabel(f"{label} Fine")
    ax.set_ylabel(f"{label} Corrigido")

    ax.set_title(
        f"{label}: Predito vs Referência"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.set_aspect("equal")

    _finalize_figure(fig, save_path=save_path, show=show, close=close)

##########################################
##### Métricas Físicas
##########################################
def plot_divergence_compare(
    df_plot,
    field_coarse="div_u",
    field_fine="div_u_f",
    field_corrected="div_corrected",
    model_name="Modelo",
    h=0.0127,
    percentile=99.0,
    absolute=True,
    vmax_fixed=None,
    save_path=None,
    show=False,
    close=True,
):
    """
    Compara os campos de divergência do CFD coarse, CFD fine
    e campo corrigido usando a mesma escala de cores.

    Parameters
    ----------
    absolute:
        True  -> plota |div(U)|.
        False -> plota div(U) com escala simétrica em torno de zero.
    percentile:
        Percentil usado para limitar a escala e reduzir o efeito de outliers.
    """
    required = {
        "x",
        "y",
        field_coarse,
        field_fine,
        field_corrected,
    }

    missing = required - set(df_plot.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes para o mapa de divergência: {sorted(missing)}"
        )

    x = df_plot["x"].to_numpy(dtype=np.float64) / h
    y = df_plot["y"].to_numpy(dtype=np.float64) / h

    fields = [
        field_coarse,
        field_fine,
        field_corrected,
    ]

    titles = [
        "CFD coarse",
        "CFD fine",
        f"{model_name.upper()} Corrigido",
    ]

    values = [
        df_plot[field].to_numpy(dtype=np.float64)
        for field in fields
    ]

    all_values = np.concatenate(values)
    finite_values = all_values[np.isfinite(all_values)]

    if finite_values.size == 0:
        raise ValueError(
            "Nenhum valor finito encontrado nos campos de divergência."
        )

    if absolute:
        values = [np.abs(value) for value in values]

        if vmax_fixed is None:
            vmax = np.nanpercentile(
                np.concatenate(values),
                percentile,
            )
        else:
            vmax = float(vmax_fixed)

        vmin = 0.0
        cmap = "inferno"
        colorbar_label = r"$|\nabla\cdot\mathbf{U}|$ [s$^{-1}$]"

    else:
        if vmax_fixed is None:
            vmax = np.nanpercentile(
                np.abs(finite_values),
                percentile,
            )
        else:
            vmax = float(vmax_fixed)

        vmin = -vmax
        cmap = "coolwarm"
        colorbar_label = r"$\nabla\cdot\mathbf{U}$ [s$^{-1}$]"

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22, 4),
        constrained_layout=True,
    )

    scatter = None

    for ax, field_values, title in zip(
        axes,
        values,
        titles,
    ):
        scatter = ax.scatter(
            x,
            y,
            c=field_values,
            s=4,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        ax.set_title(title)
        ax.set_xlabel(r"$x/H$")
        ax.set_ylabel(r"$y/H$")
        ax.set_xlim([-130, 50])
        ax.set_ylim([0, 9])
        ax.set_aspect("auto")

    fig.colorbar(
        scatter,
        ax=axes,
        label=colorbar_label,
    )

    mode = "Módulo" if absolute else "Campo assinado"

    fig.suptitle(
        f"{mode} da divergência — {model_name.upper()}",
        fontsize=14,
    )

    _finalize_figure(
        fig,
        save_path=save_path,
        show=show,
        close=close,
    )

def plot_divergence_mlp_pinn_compare(
    df_mlp,
    df_pinn,
    mlp_field="div_corrected",
    pinn_field="div_corrected",
    coarse_field="div_u",
    fine_field="div_u_f",
    h=0.0127,
    percentile=99.0,
    absolute=True,
    vmax_fixed=None,
    save_path=None,
    show=False,
    close=True,
):
    """
    Compara CFD coarse, CFD fine, MLP e PINN usando uma única escala.

    Espera que os dois DataFrames correspondam ao mesmo domínio espacial.
    """

    required_mlp = {
        "x",
        "y",
        coarse_field,
        fine_field,
        mlp_field,
    }

    required_pinn = {
        "x",
        "y",
        pinn_field,
    }

    missing_mlp = required_mlp - set(df_mlp.columns)
    missing_pinn = required_pinn - set(df_pinn.columns)

    if missing_mlp:
        raise ValueError(
            f"Colunas ausentes no DataFrame MLP: {sorted(missing_mlp)}"
        )

    if missing_pinn:
        raise ValueError(
            f"Colunas ausentes no DataFrame PINN: {sorted(missing_pinn)}"
        )

    if len(df_mlp) != len(df_pinn):
        raise ValueError(
            "MLP e PINN possuem quantidades diferentes de pontos."
        )

    x_mlp = df_mlp["x"].to_numpy(dtype=np.float64)
    y_mlp = df_mlp["y"].to_numpy(dtype=np.float64)

    x_pinn = df_pinn["x"].to_numpy(dtype=np.float64)
    y_pinn = df_pinn["y"].to_numpy(dtype=np.float64)

    if not (
        np.allclose(x_mlp, x_pinn, rtol=0.0, atol=1e-10)
        and np.allclose(y_mlp, y_pinn, rtol=0.0, atol=1e-10)
    ):
        raise ValueError(
            "Os pontos espaciais da MLP e da PINN não estão alinhados."
        )

    x = x_mlp / h
    y = y_mlp / h

    fields = [
        df_mlp[coarse_field].to_numpy(dtype=np.float64),
        df_mlp[fine_field].to_numpy(dtype=np.float64),
        df_mlp[mlp_field].to_numpy(dtype=np.float64),
        df_pinn[pinn_field].to_numpy(dtype=np.float64),
    ]

    titles = [
        "CFD coarse",
        "CFD fine",
        "MLP corrigida",
        "PINN corrigida",
    ]

    finite_values = np.concatenate([
        value[np.isfinite(value)]
        for value in fields
    ])

    if finite_values.size == 0:
        raise ValueError(
            "Nenhum valor finito encontrado nos campos de divergência."
        )

    if absolute:
        plot_values = [
            np.abs(value)
            for value in fields
        ]

        if vmax_fixed is None:
            vmax = np.nanpercentile(
                np.concatenate(plot_values),
                percentile,
            )
        else:
            vmax = float(vmax_fixed)

        vmin = 0.0
        cmap = "inferno"
        colorbar_label = (
            r"$|\nabla\cdot\mathbf{U}|$ [s$^{-1}$]"
        )
        title = "Módulo da divergência — comparação MLP × PINN"

    else:
        plot_values = fields

        if vmax_fixed is None:
            vmax = np.nanpercentile(
                np.abs(finite_values),
                percentile,
            )
        else:
            vmax = float(vmax_fixed)

        vmin = -vmax
        cmap = "coolwarm"
        colorbar_label = (
            r"$\nabla\cdot\mathbf{U}$ [s$^{-1}$]"
        )
        title = "Divergência assinada — comparação MLP × PINN"

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(28, 4.5),
        constrained_layout=True,
    )

    scatter = None

    for ax, values, subtitle in zip(
        axes,
        plot_values,
        titles,
    ):
        scatter = ax.scatter(
            x,
            y,
            c=values,
            s=4,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        ax.set_title(subtitle)
        ax.set_xlabel(r"$x/H$")
        ax.set_ylabel(r"$y/H$")
        ax.set_xlim([-130, 50])
        ax.set_ylim([0, 9])
        ax.set_aspect("auto")

    fig.colorbar(
        scatter,
        ax=axes,
        label=colorbar_label,
    )

    fig.suptitle(
        title,
        fontsize=14,
    )

    _finalize_figure(
        fig,
        save_path=save_path,
        show=show,
        close=close,
    )

def plot_divergence_error(
    df_plot,
    corrected_field="div_corrected",
    fine_field="div_u_f",
    model_name="Modelo",
    h=0.0127,
    percentile=99.0,
    save_path=None,
    show=False,
    close=True,
):
    required = {
        "x",
        "y",
        corrected_field,
        fine_field,
    }

    missing = required - set(df_plot.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes para erro de divergência: {sorted(missing)}"
        )

    x = df_plot["x"].to_numpy(dtype=np.float64) / h
    y = df_plot["y"].to_numpy(dtype=np.float64) / h

    error = (
        df_plot[corrected_field].to_numpy(dtype=np.float64)
        - df_plot[fine_field].to_numpy(dtype=np.float64)
    )

    finite_error = error[np.isfinite(error)]

    if finite_error.size == 0:
        raise ValueError(
            "Nenhum valor finito encontrado no erro de divergência."
        )

    vmax = np.nanpercentile(
        np.abs(finite_error),
        percentile,
    )

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(
        figsize=(18, 5),
        constrained_layout=True,
    )

    scatter = ax.scatter(
        x,
        y,
        c=error,
        s=4,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )

    ax.set_title(
        f"Erro de divergência — {model_name.upper()} menos CFD fine"
    )
    ax.set_xlabel(r"$x/H$")
    ax.set_ylabel(r"$y/H$")
    ax.set_xlim([-130, 50])
    ax.set_ylim([0, 9])
    ax.set_aspect("auto")

    fig.colorbar(
        scatter,
        ax=ax,
        label=r"$\nabla\cdot U_{corr}-\nabla\cdot U_{fine}$ [s$^{-1}$]",
    )

    _finalize_figure(
        fig,
        save_path=save_path,
        show=show,
        close=close,
    )

def plot_divergence_error_mlp_pinn(
    df_mlp,
    df_pinn,
    mlp_field="div_corrected",
    pinn_field="div_corrected",
    fine_field="div_u_f",
    h=0.0127,
    percentile=99.0,
    vmax_fixed=None,
    save_path=None,
    show=False,
    close=True,
):
    """
    Compara o erro de divergência da MLP e da PINN
    em relação ao CFD fine.
    """

    required_mlp = {
        "x",
        "y",
        mlp_field,
        fine_field,
    }

    required_pinn = {
        "x",
        "y",
        pinn_field,
        fine_field,
    }

    missing_mlp = required_mlp - set(df_mlp.columns)
    missing_pinn = required_pinn - set(df_pinn.columns)

    if missing_mlp:
        raise ValueError(
            f"Colunas ausentes na MLP: {sorted(missing_mlp)}"
        )

    if missing_pinn:
        raise ValueError(
            f"Colunas ausentes na PINN: {sorted(missing_pinn)}"
        )

    if len(df_mlp) != len(df_pinn):
        raise ValueError(
            "MLP e PINN possuem quantidades diferentes de pontos."
        )

    x_mlp = df_mlp["x"].to_numpy(dtype=np.float64)
    y_mlp = df_mlp["y"].to_numpy(dtype=np.float64)

    x_pinn = df_pinn["x"].to_numpy(dtype=np.float64)
    y_pinn = df_pinn["y"].to_numpy(dtype=np.float64)

    if not (
        np.allclose(x_mlp, x_pinn, rtol=0.0, atol=1e-10)
        and np.allclose(y_mlp, y_pinn, rtol=0.0, atol=1e-10)
    ):
        raise ValueError(
            "Os pontos espaciais da MLP e da PINN não estão alinhados."
        )

    x = x_mlp / h
    y = y_mlp / h

    error_mlp = (
        df_mlp[mlp_field].to_numpy(dtype=np.float64)
        - df_mlp[fine_field].to_numpy(dtype=np.float64)
    )

    error_pinn = (
        df_pinn[pinn_field].to_numpy(dtype=np.float64)
        - df_pinn[fine_field].to_numpy(dtype=np.float64)
    )

    errors = np.concatenate([
        error_mlp[np.isfinite(error_mlp)],
        error_pinn[np.isfinite(error_pinn)],
    ])

    if errors.size == 0:
        raise ValueError(
            "Nenhum erro de divergência finito foi encontrado."
        )

    if vmax_fixed is None:
        vmax = np.nanpercentile(
            np.abs(errors),
            percentile,
        )
    else:
        vmax = float(vmax_fixed)

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(22, 5),
        constrained_layout=True,
    )

    plots = [
        (
            axes[0],
            error_mlp,
            "Erro de divergência — MLP menos CFD fine",
        ),
        (
            axes[1],
            error_pinn,
            "Erro de divergência — PINN menos CFD fine",
        ),
    ]

    scatter = None

    for ax, error, title in plots:
        scatter = ax.scatter(
            x,
            y,
            c=error,
            s=4,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            rasterized=True,
        )

        ax.set_title(title)
        ax.set_xlabel(r"$x/H$")
        ax.set_ylabel(r"$y/H$")
        ax.set_xlim([-130, 50])
        ax.set_ylim([0, 9])
        ax.set_aspect("auto")

    fig.colorbar(
        scatter,
        ax=axes,
        label=(
            r"$\nabla\cdot U_{corr}"
            r"-\nabla\cdot U_{fine}$ [s$^{-1}$]"
        ),
    )

    _finalize_figure(
        fig,
        save_path=save_path,
        show=show,
        close=close,
    )