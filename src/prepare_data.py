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

path_fine = cfg["cfd_data"]["path_fine"]
path_coarse = cfg["cfd_data"]["path_coarse"]

dataset_raw = cfg["cfd_data"]["dataset"]

dataset_csv = cfg["cfd_data"].get(
    "dataset_csv",
    Path(dataset_raw).with_suffix(".csv").name,
)

input_format = cfg["cfd_data"].get(
    "input_format",
    "auto",
)

csv_separator = cfg["cfd_data"].get(
    "csv_separator",
    ",",
)

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
# Leitura de arquivos CFD
# -------------------------------------------------

RAW_COLUMNS = [
    "x",
    "y",
    "z",
    "div_u",
    "epsilon",
    "k",
    "nut",
    "p",
    "Ux",
    "Uy",
    "Uz",
]


COLUMN_ALIASES = {
    # Coordenadas
    "Points:0": "x",
    "Points:1": "y",
    "Points:2": "z",
    "coords:0": "x",
    "coords:1": "y",
    "coords:2": "z",

    # Velocidade
    "U:0": "Ux",
    "U:1": "Uy",
    "U:2": "Uz",
    "U_0": "Ux",
    "U_1": "Uy",
    "U_2": "Uz",
    "U.x": "Ux",
    "U.y": "Uy",
    "U.z": "Uz",

    # Divergência
    "div(U)": "div_u",
    "divU": "div_u",
    "div_U": "div_u",

    # Variáveis escalares
    "Epsilon": "epsilon",
    "K": "k",
    "Nut": "nut",
    "P": "p",
}


def normalize_cfd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza os nomes das colunas provenientes de diferentes formatos
    de exportação do OpenFOAM.
    """
    result = df.copy()

    # Remove espaços extras e aspas dos cabeçalhos
    result.columns = [
        str(col).strip().strip('"').strip("'")
        for col in result.columns
    ]

    result = result.rename(columns=COLUMN_ALIASES)

    # Alguns CSVs podem trazer nomes em minúsculo
    lowercase_aliases = {
        "ux": "Ux",
        "uy": "Uy",
        "uz": "Uz",
        "divu": "div_u",
        "div_u": "div_u",
    }

    rename_lower = {}

    for col in result.columns:
        normalized = col.strip().lower()

        if normalized in lowercase_aliases:
            rename_lower[col] = lowercase_aliases[normalized]

    result = result.rename(columns=rename_lower)

    return result

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


def read_raw_grid_file(path: Path) -> pd.DataFrame:
    """
    Lê arquivo raw do sampledSet do OpenFOAM.

    Formato esperado:

        x y z div_u epsilon k nut p Ux Uy Uz
    """
    rows = []

    number_regex = re.compile(
        r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
    )

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            if line.startswith("#") or line.startswith("//"):
                continue

            numbers = number_regex.findall(line)

            if len(numbers) < len(RAW_COLUMNS):
                logger.debug(
                    "Linha %d ignorada em %s: encontrados %d valores.",
                    line_number,
                    path,
                    len(numbers),
                )
                continue

            values = list(
                map(float, numbers[:len(RAW_COLUMNS)])
            )

            rows.append(values)

    if not rows:
        raise ValueError(
            f"Nenhum registro válido encontrado no arquivo raw: {path}"
        )

    return pd.DataFrame(
        rows,
        columns=RAW_COLUMNS,
    )

def read_csv_grid_file(
    path: Path,
    separator: str = ",",
) -> pd.DataFrame:
    """
    Lê arquivo CSV exportado pelo OpenFOAM ou convertido externamente.
    """
    try:
        df = pd.read_csv(
            path,
            sep=separator,
            comment="#",
        )
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Não foi possível interpretar o CSV: {path}"
        ) from exc

    # Caso o separador configurado não funcione e tudo fique em uma coluna
    if len(df.columns) == 1:
        logger.warning(
            "CSV lido com apenas uma coluna. "
            "Tentando detectar o separador automaticamente."
        )

        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            comment="#",
        )

    df = normalize_cfd_columns(df)

    return df

def resolve_case_file(
    case_dir: Path,
    raw_filename: str,
    csv_filename: str | None,
    input_format: str,
) -> tuple[Path, str]:
    """
    Localiza o arquivo CFD no último diretório de tempo.

    Retorna:
        caminho do arquivo
        formato identificado
    """
    if not case_dir.exists():
        raise FileNotFoundError(
            f"Diretório do caso não encontrado: {case_dir}"
        )

    latest_time_dir = latest_dir(case_dir)

    input_format = input_format.lower().strip()

    if input_format not in {"raw", "csv", "auto"}:
        raise ValueError(
            "input_format deve ser 'raw', 'csv' ou 'auto'. "
            f"Recebido: {input_format}"
        )

    candidates: list[tuple[Path, str]] = []

    if input_format in {"raw", "auto"}:
        candidates.append(
            (latest_time_dir / raw_filename, "raw")
        )

    if input_format in {"csv", "auto"}:
        if csv_filename:
            candidates.append(
                (latest_time_dir / csv_filename, "csv")
            )

        # Também tenta trocar automaticamente a extensão
        raw_path = Path(raw_filename)

        candidates.append(
            (
                latest_time_dir / raw_path.with_suffix(".csv").name,
                "csv",
            )
        )

    for candidate_path, detected_format in candidates:
        if candidate_path.exists():
            return candidate_path, detected_format

    searched = "\n".join(
        f"  - {path}"
        for path, _ in candidates
    )

    raise FileNotFoundError(
        "Nenhum arquivo CFD compatível foi encontrado.\n"
        f"Arquivos procurados:\n{searched}"
    )

def load_cfd_case(
    case_dir: str | Path,
    raw_filename: str,
    csv_filename: str | None = None,
    input_format: str = "auto",
    csv_separator: str = ",",
) -> pd.DataFrame:
    """
    Carrega um caso CFD nos formatos raw ou CSV e retorna um
    DataFrame com nomes de colunas padronizados.
    """
    case_path = Path(case_dir)

    file_path, detected_format = resolve_case_file(
        case_dir=case_path,
        raw_filename=raw_filename,
        csv_filename=csv_filename,
        input_format=input_format,
    )

    logger.info(
        "Carregando arquivo CFD: %s | formato=%s",
        file_path,
        detected_format,
    )

    if detected_format == "raw":
        df = read_raw_grid_file(file_path)

    elif detected_format == "csv":
        df = read_csv_grid_file(
            file_path,
            separator=csv_separator,
        )

    else:
        raise RuntimeError(
            f"Formato interno não reconhecido: {detected_format}"
        )

    df = normalize_cfd_columns(df)

    required_columns = {
        "x",
        "y",
        "epsilon",
        "k",
        "nut",
        "p",
        "Ux",
        "Uy",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes em {file_path}: {sorted(missing)}\n"
            f"Colunas disponíveis: {sorted(df.columns.tolist())}"
        )

    # Campos opcionais
    if "z" not in df.columns:
        df["z"] = 0.0

    if "Uz" not in df.columns:
        df["Uz"] = 0.0

    if "div_u" not in df.columns:
        logger.warning(
            "O arquivo %s não contém a divergência exportada pelo OpenFOAM.",
            file_path,
        )

        df["div_u"] = np.nan

    numeric_columns = [
        "x",
        "y",
        "z",
        "div_u",
        "epsilon",
        "k",
        "nut",
        "p",
        "Ux",
        "Uy",
        "Uz",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    before = len(df)

    df = (
        df
        .replace([np.inf, -np.inf], np.nan)
        .dropna(
            subset=[
                "x",
                "y",
                "epsilon",
                "k",
                "nut",
                "p",
                "Ux",
                "Uy",
            ]
        )
        .reset_index(drop=True)
    )

    removed = before - len(df)

    if removed:
        logger.warning(
            "%d linhas inválidas foram removidas de %s.",
            removed,
            file_path,
        )

    if df.empty:
        raise ValueError(
            f"O arquivo não possui linhas válidas após a limpeza: {file_path}"
        )

    return df

#######################################################
# Utilitário para normalização
#######################################################

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
        f"div_u_python{suffix}": divergence,
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

dfc = load_cfd_case(
    case_dir=PATH_CFD / path_coarse,
    raw_filename=dataset_raw,
    csv_filename=dataset_csv,
    input_format=input_format,
    csv_separator=csv_separator,
)

logger.info("Lendo caso fine...")

dff = load_cfd_case(
    case_dir=PATH_CFD / path_fine,
    raw_filename=dataset_raw,
    csv_filename=dataset_csv,
    input_format=input_format,
    csv_separator=csv_separator,
)

# Merge ponto a ponto (x,y)
df = dfc.merge(
    dff[
        [
            "x",
            "y",
            "div_u",
            "epsilon",
            "k",
            "nut",
            "p",
            "Ux",
            "Uy",
        ]
    ].rename(
        columns={
            "div_u": "div_u_f",
            "p": "p_f",
            "Ux": "Ux_f",
            "Uy": "Uy_f",
            "epsilon": "epsilon_f",
            "k": "k_f",
            "nut": "nut_f",
        }
    ),
    on=["x", "y"],
    how="inner",
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
df["depsilon"] = df["epsilon_f"] - df["epsilon"]
df["dk"] = df["k_f"] - df["k"]
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

    "div_u",

    "wz_log",
    "sxy_log",
    "div_u_log",

    "Re",

    # Parâmetro físico
    "Re_norm",

    # Solução fine de referência
    "Ux_f",
    "Uy_f",
    "p_f",
    "epsilon_f",
    "k_f",
    "nut_f_log",

    "div_u_f",
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