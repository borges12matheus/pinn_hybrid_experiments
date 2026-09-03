from pathlib import Path
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib
from logger import ExperimentLogger
from run_metrics import run_metrics_pipeline
from run_metrics_physics import run_physics_metrics_pipeline
from train_utils import DataProcess, MLP, BaseTrainer 
from config_experiment import set_deterministic, hash_file, load_or_create_split
from config_experiment import load_config

# Carregando as configurações do experimento
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cfg, CONFIG_PATH = load_config()
split_cfg = cfg.get("split", {})
SPLIT_STRATEGY = split_cfg.get("strategy", "spatial_x_quantile")
SPLIT_COLUMN = split_cfg.get("column", "x")
SPLIT_BINS = int(split_cfg.get("bins", 8))
SPLIT_VAL_FRAC = float(split_cfg.get("val_frac", 0.1))
SPLIT_TEST_FRAC = float(split_cfg.get("test_frac", 0.2))
SPLIT_VERSION = split_cfg.get("version", "v1")
DATASET_TEST_OUTPUT = cfg["dataset"].get("test_output", "dataset_test_mlp")

# Define os caminhos dos diretórios
ROOT = Path(cfg["paths"]["root"])
PATH_DATA = ROOT / cfg["paths"]["data_process_dir"]
PATH_CFD = ROOT / cfg["paths"]["data_cfd_dir"]
PATH_MODEL = ROOT / cfg["paths"]["models_dir"] / "mlp" / f"mlp_{cfg['experiment']['name']}_{timestamp}"
PATH_METRIC = ROOT / cfg["paths"]["metrics_dir"]
PATH_METRIC_EXP = PATH_METRIC / "mlp" / f"mlp_{cfg['experiment']['name']}_{timestamp}"
PATH_PLOT = ROOT / cfg["paths"]["plots_dir"] / "mlp" / f"mlp_{cfg['experiment']['name']}_{timestamp}"
PATH_LOG = ROOT / cfg["paths"]["logs_dir"]
PATH_LOG_EXP = PATH_LOG / "mlp" / f"mlp_{cfg['experiment']['name']}_{timestamp}"

# Cria as pastas caso não existam
for p in [PATH_DATA, PATH_MODEL, PATH_METRIC_EXP, PATH_PLOT, PATH_LOG_EXP]:
    p.mkdir(parents=True, exist_ok=True)

# Helper de logs
logger = ExperimentLogger(
    experiment_name=f'mlp_{cfg["experiment"]["name"]}',
    experiment_type=f'{cfg["experiment"]["type"]}',
    log_dir= PATH_LOG_EXP,
    config=cfg
)

# Definindo dispositivo para experimento
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.log_message(f"Dispositivo do experimento: {DEVICE}")
CONFIG_HASH = hash_file(CONFIG_PATH)
NUM_WORKERS = int(cfg.get("training", {}).get("num_workers", 4))

# -----------------------------------
# Função de Treino MLP Supervisionada
# -----------------------------------
def train_mlp_epoch(
    parquet_path,
    feat_cols,
    target_cols,
    batch_data=4096,
    epochs=50,
    width=64,
    depth=4,
    lr=1e-3,
    lr_wd=1e-6,
    patience_es=20,
    patience_lr=10,
    factor_lr = 0.5, 
    out_model="model",
    seed=42
):
    set_deterministic(seed)

    model_path = PATH_MODEL / f"mlp_{out_model}_d{depth}_w{width}.pt"

    df = pd.read_parquet(parquet_path)
    DATASET_HASH = hash_file(parquet_path)

    split_path = (
        PATH_DATA
        / "splits"
        / f"{Path(parquet_path).stem}_{SPLIT_STRATEGY}_{SPLIT_COLUMN}"
        / f"b{SPLIT_BINS}_{SPLIT_VERSION}_seed{seed}.json"
    )
    tr_idx, val_idx, te_idx = load_or_create_split(df, seed, split_path, parquet_path,
                                          SPLIT_STRATEGY, 
                                          SPLIT_COLUMN, 
                                          SPLIT_BINS, 
                                          SPLIT_VAL_FRAC,
                                          SPLIT_TEST_FRAC,
                                          SPLIT_VERSION)

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)

    # Dataset supervisionado (dados treino)
    train_ds = DataProcess(df_tr, feat_cols, target_cols)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_data,
        "shuffle": True,
        "drop_last": True,
        "generator": train_generator,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE == "cuda",
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    dl_data = DataLoader(
        train_ds,
        **loader_kwargs,
    )

    # Dataset supervisionado (dados de avaliação)
    # Salvar dataset de teste
    out_test = f"{DATASET_TEST_OUTPUT}_{out_model}_d{depth}_w{width}.parquet"
    df_te.to_parquet(PATH_DATA / out_test, index=False)
    logger.log_message(f"Dataset salvo: {out_test}")
    logger.log_message(f"Split salvo em: {split_path}")
    logger.metadata.update({
        "config_path": str(CONFIG_PATH),
        "config_hash": CONFIG_HASH,
        "dataset_path": str(parquet_path),
        "dataset_hash": DATASET_HASH,
        "split_path": str(split_path),
        "split_hash": hash_file(split_path),
        "split_method": f"{SPLIT_STRATEGY}_{SPLIT_COLUMN}_{SPLIT_VERSION}",
        "split_bins": SPLIT_BINS,
        "split_val_frac": SPLIT_VAL_FRAC,
        "split_test_frac": SPLIT_TEST_FRAC,
    })

    val_ds = DataProcess(
                    df_val,
                    feat_cols, 
                    target_cols, 
                    x_scaler=(train_ds.x_mu, train_ds.x_sd), 
                    y_scaler=(train_ds.y_mu, train_ds.y_sd)
                )
    val_loader_kwargs = {
        "batch_size": batch_data,
        "shuffle": False,
        "drop_last": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE == "cuda",
    }
    if NUM_WORKERS > 0:
        val_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["prefetch_factor"] = 2

    dl_val = DataLoader(val_ds, **val_loader_kwargs)

    in_dim = len(feat_cols)
    out_dim = len(target_cols)
    net = MLP(in_dim=in_dim, out_dim=out_dim, width=width, depth=depth, act=nn.Tanh).to(DEVICE)
    logger.log_message(f"Model device: {next(net.parameters()).device}")

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=lr_wd)

    # treino + early stopping na VAL
    # implementa o scheduler para ajustar o learning rate automaticamente
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        factor=factor_lr,
        patience=patience_lr
    )

    # Instancia a classe de treino
    trainer = BaseTrainer(net, opt, scheduler, model_path, DEVICE, patience_es)
    history = []

    # ---- Treino supervisionado (dados)
    for ep in range(1, epochs + 1):
        
        loss_train = trainer.train_supervised_epoch(dl_data)

        # Faz a avaliação no dataset de teste
        loss_val = trainer.validate_supervised(dl_val)
        trainer.scheduler.step(loss_val)

        trainer.update_best_model(loss_val)
        
        # Armazena histórico de loss
        history.append({
            "epoch": ep,
            "loss_total": loss_train,
            "loss_val": loss_val,
            "lr": opt.param_groups[0]['lr']
        })


        if ep % 10 == 0 or ep == 1:
            # salva os logs por epochs
            logger.log_epoch(
                stage="MLP",
                epoch=ep,
                loss_train=loss_train,
                loss_val=loss_val,
                lr=trainer.optimizer.param_groups[0]["lr"],
                best_val=trainer.best_val
            )
        # Valida Earling Stop
        if trainer.stop_improve():
            logger.log_message(f"Early stopping (val não melhorou). Melhor loss_val: {trainer.best_val:.6e}")
            break
    
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(PATH_LOG_EXP / f'mlp_loss_history_{cfg["experiment"]["name"]}_{timestamp}.csv', index=False)
    return trainer, (train_ds.x_mu, train_ds.x_sd), (train_ds.y_mu, train_ds.y_sd)

# ---------------------------------
# Treinar a MLP e avaliar
# ---------------------------------

# Iniciando a coleta dos logs
logger.start()

trainer, xscaler, yscaler = train_mlp_epoch(
    parquet_path=PATH_DATA / cfg["dataset"]["parquet"],
    feat_cols=cfg["features"],
    target_cols=cfg["targets"],
    batch_data=cfg["training"]["batch_size"],
    epochs=cfg["training"]["epochs"],
    width=cfg["model"]["width"],
    depth=cfg["model"]["depth"],
    lr=float(cfg["training"]["lr"]),
    lr_wd=float(cfg["training"]["weight_decay"]),
    patience_es=cfg["early_stopping"]["patience"],
    patience_lr=cfg["scheduler"]["patience"],
    factor_lr = cfg["scheduler"]["factor"],
    out_model=cfg["experiment"]["name"],
    seed=cfg["experiment"]["seed"],
)

# salvar modelo e scalers
joblib.dump(xscaler, PATH_MODEL / "mlp_scaler_X.pkl")
joblib.dump(yscaler, PATH_MODEL / "mlp_scaler_Y.pkl")
logger.log_message(f"Scalers da MLP salvos em:{PATH_MODEL}")

# ======================================================
# AVALIAÇÃO DAS MÉTRICAS DE ACURÁCIA NO DOMÍNIO DE TESTE
# ======================================================
evaluation_test_result = run_metrics_pipeline(
    cfg=cfg,
    model_path=trainer.model_path,
    dataset_path=(
        PATH_DATA 
        / (
            f"{DATASET_TEST_OUTPUT}"
            f"_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}.parquet"
        )
    ),
    xscaler_path=PATH_MODEL / "mlp_scaler_X.pkl",
    yscaler_path=PATH_MODEL / "mlp_scaler_Y.pkl",
    metrics_path=(
        PATH_METRIC_EXP 
        / (
            f"mlp_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}_test.json"
        )
    ),
    predictions_path=(
        PATH_METRIC_EXP 
        / (
            f"mlp_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}_predictions_test.parquet"
        )
    ),
    plots_dir=(
        PATH_PLOT 
        / (
            f"mlp_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}"
            f"_test"
        )
    ),
    batch_size=cfg["training"]["batch_size"],
    logger=logger,
)

# ======================================================
# AVALIAÇÃO DAS MÉTRICAS DE ACURÁCIA NO DOMÍNIO COMPLETO
# ======================================================
evaluation_full_domain_result = run_metrics_pipeline(
    cfg=cfg,
    model_path=trainer.model_path,
    dataset_path=(PATH_DATA / cfg["dataset"]["parquet"]),
    xscaler_path=PATH_MODEL / "mlp_scaler_X.pkl",
    yscaler_path=PATH_MODEL / "mlp_scaler_Y.pkl",
    metrics_path=(
        PATH_METRIC_EXP 
        / (
            f"mlp_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}_full_domain.json"
        )
    ),
    predictions_path=(
        PATH_METRIC_EXP 
        / (
            f"mlp_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}_predictions_full_domain.parquet"
        )
    ),
    plots_dir=(
        PATH_PLOT 
        / (
            f"mlp_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}"
            f"_full_domain"
        )
    ),
    batch_size=cfg["training"]["batch_size"],
    logger=logger,
)

# ======================================================
# AVALIAÇÃO DAS MÉTRICAS FÍSICAS NO DOMÍNIO COMPLETO
# ======================================================
logger.log_message("Iniciando avaliação física da MLP no domínio completo.")

physics_result = run_physics_metrics_pipeline(
    cfg=cfg,
    model_path=trainer.model_path,
    dataset_full_path=PATH_DATA / cfg["dataset"]["parquet"],
    xscaler_path=PATH_MODEL / "mlp_scaler_X.pkl",
    yscaler_path=PATH_MODEL / "mlp_scaler_Y.pkl",
    metrics_path=(
        PATH_METRIC_EXP
        / (
            f"mlp_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}"
            f"_physics.json"
        )
    ),
    predictions_path=(
        PATH_METRIC_EXP
        / (
            f"mlp_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}"
            f"_physics_predictions.parquet"
        )
    ),
    batch_size=cfg["training"]["batch_size"],
    plots_dir= (
        PATH_PLOT
        /
        (
            f"mlp_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}"
            f"_physics"
        )
    )
)

logger.finish(
    final_metrics={
        "best_val": trainer.best_val,
        "model_path": str(trainer.model_path),
        "evaluation_test": evaluation_test_result,
        "evaluation_full_domain": evaluation_full_domain_result,
        "evaluation_physics": physics_result,
    }
)