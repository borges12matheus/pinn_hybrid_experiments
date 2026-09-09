from pathlib import Path
from datetime import datetime
from logger import ExperimentLogger
from run_metrics import run_metrics_pipeline
from run_metrics_physics import run_physics_metrics_pipeline
from config_experiment import hash_file, load_config

# Carregando as configurações do experimento
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cfg, CONFIG_PATH = load_config()

MODEL_TYPE = cfg["model"]["type"]
EXP_NAME = cfg["experiment"]["name"]
DEPTH = cfg["model"]["depth"]
WIDTH = cfg["model"]["width"]
SEED = cfg["experiment"]["seed"]

# Define os caminhos dos diretórios.
# Saída isolada sob "standalone/" para não se misturar com os artefatos
# gerados automaticamente pelo pipeline de treino (train_mlp.py / train_pinn.py).
ROOT = Path(cfg["paths"]["root"])
PATH_DATA = ROOT / cfg["paths"]["data_process_dir"]
PATH_MODEL = ROOT / cfg["paths"]["models_dir"]
PATH_METRIC_EXP = ROOT / cfg["paths"]["metrics_dir"] / "standalone" / MODEL_TYPE / f"{MODEL_TYPE}_{EXP_NAME}_{timestamp}"
PATH_PLOT_EXP = ROOT / cfg["paths"]["plots_dir"] / "standalone" / MODEL_TYPE / f"{MODEL_TYPE}_{EXP_NAME}_{timestamp}"
PATH_LOG_EXP = ROOT / cfg["paths"]["logs_dir"] / "standalone" / MODEL_TYPE / f"{MODEL_TYPE}_{EXP_NAME}_{timestamp}"

# Cria as pastas caso não existam
for p in [PATH_DATA, PATH_MODEL, PATH_METRIC_EXP, PATH_PLOT_EXP, PATH_LOG_EXP]:
    p.mkdir(parents=True, exist_ok=True)

# Definindo helper de logs
logger = ExperimentLogger(
    experiment_name=f"metrics_standalone_{MODEL_TYPE}_{EXP_NAME}",
    experiment_type=f"{cfg['experiment']['type']}",
    log_dir=PATH_LOG_EXP,
    config=cfg,
)
logger.start()

logger.log_message(f"Avaliacao independente (fora do pipeline de treino) para {MODEL_TYPE}_{EXP_NAME}")
CONFIG_HASH = hash_file(CONFIG_PATH)

MODEL_PATH = PATH_MODEL / f"{MODEL_TYPE}_{EXP_NAME}_d{DEPTH}_w{WIDTH}.pt"
XSCALER_PATH = PATH_MODEL / f"{MODEL_TYPE}_scaler_X.pkl"
YSCALER_PATH = PATH_MODEL / f"{MODEL_TYPE}_scaler_Y.pkl"
DATASET_PATH = PATH_DATA / cfg["dataset"]["parquet"]

# ======================================================
# AVALIAÇÃO DAS MÉTRICAS DE ACURÁCIA NO DATASET CONFIGURADO
# ======================================================

evaluation_result = run_metrics_pipeline(
    cfg=cfg,
    model_path=MODEL_PATH,
    dataset_path=DATASET_PATH,
    xscaler_path=XSCALER_PATH,
    yscaler_path=YSCALER_PATH,
    metrics_path=(
        PATH_METRIC_EXP
        / f"{MODEL_TYPE}_{EXP_NAME}_d{DEPTH}_w{WIDTH}_seed{SEED}_eval.json"
    ),
    predictions_path=(
        PATH_METRIC_EXP
        / f"{MODEL_TYPE}_{EXP_NAME}_d{DEPTH}_w{WIDTH}_seed{SEED}_predictions_eval.parquet"
    ),
    plots_dir=(
        PATH_PLOT_EXP
        / f"{MODEL_TYPE}_{EXP_NAME}_d{DEPTH}_w{WIDTH}_seed{SEED}_eval"
    ),
    logs_dir=PATH_LOG_EXP,
    batch_size=cfg["training"]["batch_size"],
    logger=logger,
)

# ======================================================
# AVALIAÇÃO DAS MÉTRICAS FÍSICAS NO DOMÍNIO COMPLETO
# ======================================================

logger.log_message("Iniciando avaliação física no domínio completo.")

physics_result = run_physics_metrics_pipeline(
    cfg=cfg,
    model_path=MODEL_PATH,
    dataset_full_path=DATASET_PATH,
    xscaler_path=XSCALER_PATH,
    yscaler_path=YSCALER_PATH,
    metrics_path=(
        PATH_METRIC_EXP
        / f"{MODEL_TYPE}_{EXP_NAME}_d{DEPTH}_w{WIDTH}_seed{SEED}_physics.json"
    ),
    predictions_path=(
        PATH_METRIC_EXP
        / f"{MODEL_TYPE}_{EXP_NAME}_d{DEPTH}_w{WIDTH}_seed{SEED}_physics_predictions.parquet"
    ),
    batch_size=cfg["training"]["batch_size"],
    plots_dir=(
        PATH_PLOT_EXP
        / f"{MODEL_TYPE}_{EXP_NAME}_d{DEPTH}_w{WIDTH}_seed{SEED}_physics"
    ),
)

logger.log_message(
    "Avaliação física concluída. "
    f"Metrics: {physics_result['metrics_path']}"
)

# ======================================================
# FINALIZAÇÃO DO EXPERIMENTO
# ======================================================

logger.finish(
    final_metrics={
        "model_path": str(MODEL_PATH),
        "evaluation": evaluation_result,
        "evaluation_physics": physics_result,
    }
)
