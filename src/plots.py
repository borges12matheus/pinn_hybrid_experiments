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
        "Corrigido",
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

        vmax = np.nanpercentile(
            np.concatenate(values),
            percentile,
        )

        vmin = 0.0
        cmap = "inferno"
        colorbar_label = r"$|\nabla\cdot\mathbf{U}|$ [s$^{-1}$]"

    else:
        vmax = np.nanpercentile(
            np.abs(finite_values),
            percentile,
        )

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
        f"{mode} da divergência — {model_name}",
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
        f"Erro de divergência — {model_name} menos CFD fine"
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
