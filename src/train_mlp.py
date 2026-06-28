import argparse
import hashlib
import json
import random
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib
from logger import setup_logger, ExperimentLogger
from run_metrics import run_metrics_pipeline
from train_utils import DataProcess, MLP, BaseTrainer 

# Helper para a leitura dos parâmetros e paths
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg, Path(args.config)

cfg, CONFIG_PATH = load_config()
split_cfg = cfg.get("split", {})
SPLIT_STRATEGY = split_cfg.get("strategy", "spatial_x_quantile")
SPLIT_COLUMN = split_cfg.get("column", "x")
SPLIT_BINS = int(split_cfg.get("bins", 8))
SPLIT_TEST_FRAC = float(split_cfg.get("test_frac", 0.2))


def hash_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# Define os caminhos dos diretórios
ROOT = Path(cfg["paths"]["root"])
PATH_DATA = ROOT / cfg["paths"]["data_process_dir"]
PATH_CFD = ROOT / cfg["paths"]["data_cfd_dir"]
PATH_MODEL = ROOT / cfg["paths"]["models_dir"]
PATH_METRIC = ROOT / cfg["paths"]["metrics_dir"]
PATH_PLOT = ROOT / cfg["paths"]["plots_dir"]
PATH_LOG = ROOT / cfg["paths"]["logs_dir"]

for p in [PATH_DATA, PATH_MODEL, PATH_METRIC, PATH_PLOT]:
    p.mkdir(parents=True, exist_ok=True)

# Helper de logs
logger = setup_logger(
    "MLP",
    PATH_LOG / "mlp" / f"mlp_{cfg['experiment']['name']}.log"
)

exp_logger = ExperimentLogger(
    experiment_name=cfg["experiment"]["name"],
    log_dir=PATH_LOG,
    config=cfg
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Dispositivo do experimento: {DEVICE}")
CONFIG_HASH = hash_file(CONFIG_PATH)
NUM_WORKERS = int(cfg.get("training", {}).get("num_workers", 4))


def set_deterministic(seed):
    torch.set_float32_matmul_precision("highest")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def load_or_create_split(
    df,
    seed,
    split_path,
    dataset_path,
    spatial_col=SPLIT_COLUMN,
    n_bins=SPLIT_BINS,
    test_frac=SPLIT_TEST_FRAC,
):
    split_method = f"{SPLIT_STRATEGY}_{spatial_col}_v1"

    if split_path.exists():
        with open(split_path, "r") as f:
            payload = json.load(f)

        if payload.get("dataset_path") != str(dataset_path):
            raise ValueError(
                "O split salvo pertence a outro dataset."
            )
        if payload.get("split_method") != split_method:
            raise ValueError(
                "O split salvo pertence a outra convenção experimental."
            )

        train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
        test_idx = np.asarray(payload["test_idx"], dtype=np.int64)

        if len(train_idx) + len(test_idx) != len(df):
            raise ValueError(
                "Split salvo não corresponde ao tamanho do dataset atual."
            )

        return train_idx, test_idx

    if spatial_col not in df.columns:
        raise KeyError(f"Coluna espacial ausente no dataset: {spatial_col}")

    spatial_values = df[spatial_col]
    if spatial_values.isna().any():
        raise ValueError(f"Coluna espacial com valores ausentes: {spatial_col}")

    bin_labels = pd.qcut(
        spatial_values,
        q=min(n_bins, max(1, spatial_values.nunique())),
        labels=False,
        duplicates="drop",
    )

    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []

    for bin_id in np.unique(bin_labels):
        bin_idx = np.flatnonzero(bin_labels.to_numpy() == bin_id)
        if len(bin_idx) == 0:
            continue

        perm = rng.permutation(bin_idx)
        n_test = int(round(len(bin_idx) * test_frac))
        if len(bin_idx) > 1:
            n_test = max(1, min(len(bin_idx) - 1, n_test))
        else:
            n_test = 0

        test_idx.extend(perm[:n_test].tolist())
        train_idx.extend(perm[n_test:].tolist())

    train_idx = np.asarray(sorted(train_idx), dtype=np.int64)
    test_idx = np.asarray(sorted(test_idx), dtype=np.int64)

    split_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "split_method": split_method,
        "split_strategy": SPLIT_STRATEGY,
        "split_column": spatial_col,
        "split_bins": n_bins,
        "test_frac": test_frac,
        "n_samples": len(df),
        "dataset_path": str(dataset_path),
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }
    with open(split_path, "w") as f:
        json.dump(payload, f, indent=2)

    return train_idx, test_idx

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
        / f"b{SPLIT_BINS}_seed{seed}.json"
    )
    tr_idx, te_idx = load_or_create_split(df, seed, split_path, parquet_path)

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
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
    out_test = f"dataset_test_mlp_{out_model}_d{depth}_w{width}.parquet"
    df_te.to_parquet(PATH_DATA / out_test, index=False)
    logger.info(f"Dataset salvo: {out_test}")
    exp_logger.log_message(f"Split salvo em: {split_path}")
    exp_logger.metadata.update({
        "config_path": str(CONFIG_PATH),
        "config_hash": CONFIG_HASH,
        "dataset_path": str(parquet_path),
        "dataset_hash": DATASET_HASH,
        "split_path": str(split_path),
        "split_hash": hash_file(split_path),
        "split_method": f"{SPLIT_STRATEGY}_{SPLIT_COLUMN}_v1",
        "split_bins": SPLIT_BINS,
        "split_test_frac": SPLIT_TEST_FRAC,
    })

    test_ds = DataProcess(df_te, feat_cols, target_cols, 
                         x_scaler=(train_ds.x_mu, train_ds.x_sd), 
                         y_scaler=(train_ds.y_mu, train_ds.y_sd))
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

    dl_val = DataLoader(test_ds, **val_loader_kwargs)

    in_dim = len(feat_cols)
    out_dim = len(target_cols)
    net = MLP(in_dim=in_dim, out_dim=out_dim, width=width, depth=depth, act=nn.Tanh).to(DEVICE)
    logger.info(f"Model device: {next(net.parameters()).device}")

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

    # ---- Treino supervisionado (dados)
    for ep in range(1, epochs + 1):
        
        loss_train = trainer.train_supervised_epoch(dl_data)

        # Faz a avaliação no dataset de teste
        loss_val = trainer.validate_supervised(dl_val)
        trainer.scheduler.step(loss_val)

        trainer.update_best_model(loss_val)
        
        if trainer.stop_improve():
            logger.info(f"Early stopping (val não melhorou). Melhor loss_val: {trainer.best_val:.6e}")
            break

        if ep % 10 == 0 or ep == 1:
            # salva os logs por epochs
            exp_logger.log_epoch(
                stage="MLP",
                epoch=ep,
                loss_train=loss_train,
                loss_val=loss_val,
                lr=trainer.optimizer.param_groups[0]["lr"],
                best_val=trainer.best_val
            )
            logger.info(
                f"[MLP] ep={ep:03d} "
                f"loss_train={loss_train:.6e} | "
                f"loss_val={loss_val:.6e} | "
                f"lr={opt.param_groups[0]['lr']:.2e}"
            )
    
    return trainer, (train_ds.x_mu, train_ds.x_sd), (train_ds.y_mu, train_ds.y_sd)

# ---------------------------------
# Treinar a MLP e avaliar
# ---------------------------------
logger.info("="*80)
logger.info("Iniciando experimento")
logger.info(cfg["experiment"]["name"])
logger.info("="*80)

# Iniciando a coleta dos logs
exp_logger.start()


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
logger.info(f"Scalers da MLP salvos em:{PATH_MODEL}")

evaluation_result = run_metrics_pipeline(
    cfg=cfg,
    model_path=trainer.model_path,
    dataset_path=PATH_DATA / f"dataset_test_mlp_{cfg['experiment']['name']}_d{cfg['model']['depth']}_w{cfg['model']['width']}.parquet",
    xscaler_path=PATH_MODEL / "mlp_scaler_X.pkl",
    yscaler_path=PATH_MODEL / "mlp_scaler_Y.pkl",
    metrics_path=PATH_METRIC / f"{cfg['experiment']['name']}_d{cfg['model']['depth']}_w{cfg['model']['width']}_seed{cfg['experiment']['seed']}.json",
    predictions_path=PATH_METRIC / f"{cfg['experiment']['name']}_d{cfg['model']['depth']}_w{cfg['model']['width']}_seed{cfg['experiment']['seed']}_predictions.parquet",
    plots_dir=PATH_PLOT / f"{cfg['experiment']['name']}_d{cfg['model']['depth']}_w{cfg['model']['width']}_seed{cfg['experiment']['seed']}",
    batch_size=cfg["training"]["batch_size"],
    logger=logger,
)

# Fechando o log de métricas
exp_logger.finish(final_metrics={
    "best_val": trainer.best_val,
    "model_path": str(trainer.model_path),
    "evaluation": evaluation_result,
})

logger.info("="*80)
logger.info("Treino finalizado")
logger.info(f"Melhor loss: {trainer.best_val:.6e}")
logger.info(f"Modelo salvo: {trainer.model_path}")
logger.info("="*80)
