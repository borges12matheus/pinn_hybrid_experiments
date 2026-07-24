# Inicializa as bibliotecas necessárias
import re
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import numpy as np
import logging
from config_experiment import load_config


# Carregando as configurações do experimento
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cfg, CONFIG_PATH = load_config()

# Define os caminhos dos diretórios
ROOT = Path(cfg["paths"]["root"])
PATH_DATA = ROOT / cfg["paths"]["data_process_dir"]
PATH_CFD = ROOT / cfg["paths"]["data_cfd_dir"]

# Define número de Reynolds
path_fine = cfg["cfd_data"]["path_fine"]
path_coarse = cfg["cfd_data"]["path_coarse"]
dataset = cfg["cfd_data"]["dataset"]
RE = cfg["cfd_data"]["Re"]


# Cria as pastas caso não existam
for p in [PATH_DATA]:
    p.mkdir(parents=True, exist_ok=True)

# Helper de logs
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# -------------------------------------------------
# Utilitários Gerais
# -------------------------------------------------
def latest_dir(base: Path) -> Path:
    """Retorna o diretório do último 'tempo'."""
    times = sorted(
        [p for p in base.iterdir() if p.is_dir()],
        key=lambda p: float(p.name)
    )
    return times[-1]


def read_grid_file(path: Path) -> pd.DataFrame:
    """
    Lê o arquivo grid_pinn_epsilon_k_nut_p_U.xy no formato:
    x y z epsilon k nut p Ux Uy Uz
    """
    rows = []
    num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            nums = num_re.findall(line)
            if len(nums) < 10:
                continue

            x, y, z, epsilon, k, nut, p, ux, uy, uz = map(float, nums[:10])
            rows.append((x, y, z, epsilon, k, nut, p, ux, uy, uz))

    return pd.DataFrame(
        rows, columns=["x", "y", "z", "epsilon", "k", "nut", "p", "Ux", "Uy", "Uz"]
    )


def load_case(case_dir: str) -> pd.DataFrame:
    """
    Carrega o grid_pinn_epsilon_k_nut_p_U.xy do último tempo do caso.
    """
    case = Path(case_dir)
    tdir = latest_dir(case)

    fpath = tdir / dataset
    if not fpath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {fpath}")

    return read_grid_file(fpath)

def signed_log1p(series):
    return np.sign(series) * np.log1p(np.abs(series))

#######################################################
# Utilitário para calcular gradientes e vorticidade
#######################################################
def add_velocity_gradients_from_grid(
    df,
    ux_col="Ux",
    uy_col="Uy",
    suffix="",
):
    """
    Calcula gradientes espaciais do campo de velocidade em uma malha 2D:

        du_dx, du_dy, dv_dx, dv_dy

    Também calcula:
        divergência
        vorticidade wz
        taxa de cisalhamento Sxy
    """

    required_cols = {"x", "y", ux_col, uy_col}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")

    result = df.copy()

    xs = np.sort(result["x"].unique())
    ys = np.sort(result["y"].unique())

    if len(xs) < 3 or len(ys) < 3:
        raise ValueError(
            "São necessários pelo menos 3 pontos em x e y "
            "para calcular os gradientes."
        )

    # Verifica se há mais de um valor para a mesma coordenada
    duplicated = result.duplicated(subset=["x", "y"]).any()

    if duplicated:
        raise ValueError(
            "Existem coordenadas (x, y) duplicadas. "
            "Verifique se há múltiplos valores de z ou pontos repetidos."
        )

    Ux_grid = (
        result
        .pivot(index="y", columns="x", values=ux_col)
        .reindex(index=ys, columns=xs)
    )

    Uy_grid = (
        result
        .pivot(index="y", columns="x", values=uy_col)
        .reindex(index=ys, columns=xs)
    )

    # Verifica lacunas da malha, comuns na região sólida do degrau
    valid_mask = (
        np.isfinite(Ux_grid.to_numpy())
        & np.isfinite(Uy_grid.to_numpy())
    )

    if not valid_mask.all():
        logger.info(
            "Aviso: a malha possui pontos ausentes. "
            "Gradientes próximos à região sólida devem ser tratados com cuidado."
        )

    # np.gradient aceita diretamente os vetores de coordenadas.
    # Isso é mais seguro que usar apenas o espaçamento médio.
    du_dy, du_dx = np.gradient(
        Ux_grid.to_numpy(),
        ys,
        xs,
        edge_order=2,
    )

    dv_dy, dv_dx = np.gradient(
        Uy_grid.to_numpy(),
        ys,
        xs,
        edge_order=2,
    )

    divergence = du_dx + dv_dy
    wz = dv_dx - du_dy

    # Componente de cisalhamento do tensor de deformação
    sxy = 0.5 * (du_dy + dv_dx)

    fields = {
        f"du_dx{suffix}": du_dx,
        f"du_dy{suffix}": du_dy,
        f"dv_dx{suffix}": dv_dx,
        f"dv_dy{suffix}": dv_dy,
        f"div_u{suffix}": divergence,
        f"wz{suffix}": wz,
        f"sxy{suffix}": sxy,
    }

    gradient_df = pd.DataFrame({
        "x": np.tile(xs, len(ys)),
        "y": np.repeat(ys, len(xs)),
    })

    for name, values in fields.items():
        gradient_df[name] = values.reshape(-1)

    return result.merge(
        gradient_df,
        on=["x", "y"],
        how="left",
        validate="one_to_one",
    )

# -------------------------------------------------
# Main
# -------------------------------------------------

logger.info("Lendo caso coarse...")
dfc = load_case(PATH_CFD / path_coarse)

logger.info("Lendo caso fine...")
dff = load_case(PATH_CFD / path_fine)

# Merge ponto a ponto (x,y)
df = dfc.merge(
    dff[["x", "y", "epsilon", "k", "nut", "p", "Ux", "Uy"]].rename(
        columns={
            "p": "p_f",
            "Ux": "Ux_f",
            "Uy": "Uy_f",
            "epsilon": "epsilon_f",
            "k": "k_f",
            "nut": "nut_f"
        }
    ),
    on=["x", "y"],
    how="inner"
)

logger.info("Pontos coarse: %d", len(dfc))
logger.info("Pontos fine: %d", len(dff))
logger.info("Pontos após merge: %d", len(df))

if df.empty:
    raise ValueError(
        "O merge coarse-fine não encontrou pontos coincidentes."
    )

coverage_coarse = len(df) / len(dfc)
coverage_fine = len(df) / len(dff)

logger.info("Cobertura coarse: %.2f%%", 100 * coverage_coarse)
logger.info("Cobertura fine: %.2f%%", 100 * coverage_fine)

if coverage_coarse < 0.99 or coverage_fine < 0.99:
    logger.warning(
        "Nem todos os pontos foram pareados. "
        "Coarse: %.2f%% | Fine: %.2f%%",
        100 * coverage_coarse,
        100 * coverage_fine,
    )

# Erro coarse -> fine (targets do ML)
df["dUx"] = df["Ux_f"] - df["Ux"]
df["dUy"] = df["Uy_f"] - df["Uy"]
df["dp"]  = df["p_f"]  - df["p"]
df["depsilon"] = df["epsilon_f"] - df["epsilon"]
df["dk"] = df["k_f"] - df["k"]

# Reynolds (fixo neste experimento)
df["Re"] = RE
# Normalizando valor de Re
df["Re_norm"] = np.log10(df["Re"])

# Calculando gradientes coarse
df = add_velocity_gradients_from_grid(
    df,
    ux_col="Ux",
    uy_col="Uy",
)

# Calculando gradientes fine apenas para targets/análise
df = add_velocity_gradients_from_grid(
    df,
    ux_col="Ux_f",
    uy_col="Uy_f",
    suffix="_f",
)

# Transformação log assinada para grandezas que podem ser negativas
gradient_cols = [
    "du_dx",
    "du_dy",
    "dv_dx",
    "dv_dy",
    "div_u",
    "wz",
    "sxy",
    "du_dx_f",
    "du_dy_f",
    "dv_dx_f",
    "dv_dy_f",
    "div_u_f",
    "wz_f",
    "sxy_f",
]

for col in gradient_cols:
    df[f"{col}_log"] = signed_log1p(df[col])

# nut em log natural, compatível com transformação inversa exp
eps = 1e-12

df["nut_log"] = np.log(df["nut"] + eps)
df["nut_f_log"] = np.log(df["nut_f"] + eps)

# Deltas físicos
df["dnut"] = df["nut_f"] - df["nut"]
df["dwz"] = df["wz_f"] - df["wz"]

# Deltas em escala log
df["dnut_log"] = df["nut_f_log"] - df["nut_log"]
df["dwz_log"] = df["wz_f_log"] - df["wz_log"]

# Remover valores inválidos
required_gradient_cols = [
    "du_dx_log",
    "du_dy_log",
    "dv_dx_log",
    "dv_dy_log",
    "div_u_log",
    "wz_log",
    "sxy_log",
]

df = (
    df
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=required_gradient_cols)
    .reset_index(drop=True)
)

out = df[[
    "x",
    "y",

    # Solução coarse
    "Ux",
    "Uy",
    "p",
    "epsilon",
    "k",
    "nut_log",
    "wz_log",
    "sxy_log",
    "div_u_log",

    # Parâmetro físico
    "Re_norm",

    # Solução fine de referência
    "Ux_f",
    "Uy_f",
    "p_f",
    "epsilon_f",
    "k_f",
    "nut_f_log",
    "wz_f_log",
    "sxy_f_log",
    "div_u_f_log",

    # Targets coarse -> fine
    "dUx",
    "dUy",
    "dp",
    "depsilon",
    "dk",
    "dnut",
    "dwz",
]].copy()

out_name = cfg["cfd_data"]["out_name"]
out.to_parquet(PATH_DATA / out_name, index=False)

logger.info(f"Dataset salvo em: {out_name}")
logger.info("Total de amostras: %d", len(out))