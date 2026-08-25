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
from pinn_utils import get_pinn_epoch_runner, compute_pinn_losses

# Carregando as configurações do experimento
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cfg, CONFIG_PATH = load_config()
split_cfg = cfg.get("split", {})
pinn_cfg = cfg.get("pinn", {})
SPLIT_STRATEGY = split_cfg.get("strategy", "spatial_x_quantile")
SPLIT_COLUMN = split_cfg.get("column", "x")
SPLIT_BINS = int(split_cfg.get("bins", 8))
SPLIT_TEST_FRAC = float(split_cfg.get("test_frac", 0.2))
SPLIT_VERSION = split_cfg.get("version", "v1")
DATASET_TEST_OUTPUT = cfg["dataset"].get("test_output", "dataset_test_pinn")
PHYSICS_MODE = pinn_cfg.get("physics_mode", "continuity")
MIN_PHYS_EPOCHS = int(pinn_cfg.get("min_phys_epochs", 50))
NUT_TRANSFORM = pinn_cfg.get("nut_transform", "exp")

# Define os caminhos dos diretórios
ROOT = Path(cfg["paths"]["root"])
PATH_DATA = ROOT / cfg["paths"]["data_process_dir"]
PATH_CFD = ROOT / cfg["paths"]["data_cfd_dir"]
PATH_MODEL = ROOT / cfg["paths"]["models_dir"] / "pinn" / f"pinn_{cfg['experiment']['name']}_{timestamp}"
PATH_METRIC = ROOT / cfg["paths"]["metrics_dir"]
PATH_METRIC_EXP = PATH_METRIC / "pinn" / f"pinn_{cfg['experiment']['name']}_{timestamp}"
PATH_PLOT = ROOT / cfg["paths"]["plots_dir"] / "pinn"
PATH_LOG = ROOT / cfg["paths"]["logs_dir"]
PATH_LOG_EXP = PATH_LOG / "pinn" / f"pinn_{cfg['experiment']['name']}_{timestamp}"

# Cria as pastas caso não existam
for p in [PATH_DATA, PATH_MODEL, PATH_METRIC_EXP, PATH_PLOT, PATH_LOG_EXP]:
    p.mkdir(parents=True, exist_ok=True)

# Definindo helper de logs
logger = ExperimentLogger(
    experiment_name=f"pinn_{cfg['experiment']['name']}",
    experiment_type=f"{cfg['experiment']['type']}",
    log_dir= PATH_LOG_EXP ,
    config=cfg
)

# Definindo dispositivo para experimento
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
logger.log_message(f"Dispositivo do experimento: {DEVICE}")
CONFIG_HASH = hash_file(CONFIG_PATH)
NUM_WORKERS = int(cfg.get("training", {}).get("num_workers", 4))

def run_pinn_stage2_epoch(physics_mode, stage2_runner, stage2_kwargs):
    stage2_result = stage2_runner(**stage2_kwargs)

    if physics_mode == "cont_mom":
        loss_total, loss_data, loss_cont, loss_mom = stage2_result
        return loss_total, loss_data, loss_cont, loss_mom

    loss_total, loss_data, loss_cont = stage2_result
    return loss_total, loss_data, loss_cont, None


# Função de Treino PINN
# -----------------------------------
def train_pinn_epoch(
    parquet_path,
    feat_cols,
    target_cols,
    batch_data=4096,
    epochs_pre=50,
    epochs_phys=100,
    width=64,
    depth=4,
    lr=1e-3,
    lr_wd=1e-6,
    patience_es=20,
    patience_lr=10,
    factor_lr = 0.5,
    w_data=1.0,
    w_cont=1.0, 
    w_mom=1.0,
    out_model="model",
    seed=42
):
    set_deterministic(seed)

    model_path = PATH_MODEL / f"pinn_{out_model}_d{depth}_w{width}.pt"

    df = pd.read_parquet(parquet_path)
    DATASET_HASH = hash_file(parquet_path)

    split_path = (
        PATH_DATA
        / "splits"
        / f"{Path(parquet_path).stem}_{SPLIT_STRATEGY}_{SPLIT_COLUMN}"
        / f"b{SPLIT_BINS}_{SPLIT_VERSION}_seed{seed}.json"
    )
    tr_idx, te_idx = load_or_create_split(df, seed, split_path, parquet_path, 
                                          SPLIT_STRATEGY, 
                                          SPLIT_COLUMN, 
                                          SPLIT_BINS, 
                                          SPLIT_TEST_FRAC,
                                          SPLIT_VERSION)

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)

    # Dataset supervisionado (dados treino)
    train_ds = DataProcess(df_tr, feat_cols, target_cols, physics_col="div_u")
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
        "split_test_frac": SPLIT_TEST_FRAC,
    })

    test_ds = DataProcess(df_te, feat_cols, target_cols, 
                         x_scaler=(train_ds.x_mu, train_ds.x_sd), 
                         y_scaler=(train_ds.y_mu, train_ds.y_sd),
                         physics_col="div_u")
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

    # Iniciando vetor de histórico de Loss
    history = []

    # Organizando indice de colunas e normalização
    feat_index = {c: i for i, c in enumerate(feat_cols)}

    x_mu = torch.as_tensor(train_ds.x_mu, dtype=DTYPE, device=DEVICE)
    x_sd = torch.as_tensor(train_ds.x_sd, dtype=DTYPE, device=DEVICE)
    y_mu = torch.as_tensor(train_ds.y_mu, dtype=DTYPE, device=DEVICE)
    y_sd = torch.as_tensor(train_ds.y_sd, dtype=DTYPE, device=DEVICE)
    stage2_runner = get_pinn_epoch_runner(PHYSICS_MODE)

    # ---- STAGE 1: pré-treino supervisionado (dados)
    for ep in range(1, epochs_pre + 1):
        
        loss_pre = trainer.train_supervised_epoch(dl_data)

        # Faz a avaliação no dataset de teste
        loss_val = trainer.validate_supervised(dl_val)
        trainer.scheduler.step(loss_val)

        trainer.update_best_model(loss_val)

        # Armazena histórico de loss
        history.append({
            "epoch": ep,
            "loss_data": loss_pre,
            "loss_val": loss_val,
            "lr": opt.param_groups[0]["lr"],
        })

        if ep % 10 == 0 or ep == 1:
            # salva os logs por epochs
            logger.log_epoch(
                stage="PINN[PRE]",
                epoch=ep,
                loss_train=loss_pre,
                loss_val=loss_val,
                lr=trainer.optimizer.param_groups[0]["lr"],
                best_val=trainer.best_val
            )
        # Valida earling stop
        if trainer.stop_improve():
            logger.log_message(f"Early stopping (val não melhorou). Melhor loss_val: {trainer.best_val:.6e}")
            break
    
    # Restaura o melhor modelo encontrado no pré-treino
    trainer.load_best()

    # Reinicia o scheduler e earling stopping
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
    
    # Salva histórico de stage1 e reinicia vetor de loss
    df_hist_stage_1 = pd.DataFrame(history)
    df_hist_stage_1.to_csv(PATH_LOG_EXP / f'pinn_loss_history_stage1_{cfg["experiment"]["name"]}_{timestamp}.csv', index=False)
    history = []

    ref_data, ref_cont, ref_mom = compute_pinn_losses(
        net=net,
        dl_val=dl_val,
        feat_index=feat_index,
        x_mu=x_mu,
        x_sd=x_sd,
        y_mu=y_mu,
        y_sd=y_sd,
        physics_mode=PHYSICS_MODE,
        nut_transform=NUT_TRANSFORM,
        device=DEVICE,
    )
    logger.log_message(
        "[PINN][REFERÊNCIA] "
        f"ref_data={ref_data:.6e} | "
        f"ref_cont={ref_cont:.6e} | "
        f"ref_mom={ref_mom if ref_mom is not None else 'N/A'}"
    )

    # ---- STAGE 2: dados + física
    for ep in range(1, epochs_phys + 1):
        stage2_kwargs = {
            "net": net,
            "dl_data": dl_data,
            "opt": opt,
            "feat_index": feat_index,
            "x_mu": x_mu,
            "x_sd": x_sd,
            "y_mu": y_mu,
            "y_sd": y_sd,
            "w_data": w_data,
            "w_cont": w_cont,
            "device": DEVICE,
            "nut_transform": NUT_TRANSFORM,
        }
        if PHYSICS_MODE == "cont_mom":
            stage2_kwargs["w_mom"] = w_mom

        loss_total, loss_data, loss_cont, loss_mom = run_pinn_stage2_epoch(
            PHYSICS_MODE,
            stage2_runner,
            stage2_kwargs,
        )

        # Avaliando o erro
        val_data, val_cont, val_mom = compute_pinn_losses(
            net=net,
            dl_val=dl_val,
            feat_index=feat_index,
            x_mu=x_mu,
            x_sd=x_sd,
            y_mu=y_mu,
            y_sd=y_sd,
            physics_mode=PHYSICS_MODE,
            nut_transform=NUT_TRANSFORM,
            device=DEVICE,
        )

        norm_data = val_data / (ref_data + 1e-12)
        norm_cont = val_cont / (ref_cont + 1e-12)

        if PHYSICS_MODE == "cont_mom":
            norm_mom = val_mom / (ref_mom + 1e-12)

            val_score = (
                0.6 * norm_data
                + 0.2 * norm_cont
                + 0.2 * norm_mom
            )
        else:
            norm_mom = None

            val_score = (
                0.7 * norm_data
                + 0.3 * norm_cont
            )
        trainer.scheduler.step(val_score)
        trainer.update_best_model(val_score)
        
        # Armazena histórico de loss
        history.append({
            "epoch": ep,
            "loss_total": loss_total,
            "loss_data": loss_data,
            "loss_cont": loss_cont,
            "loss_mom": loss_mom,

            "val_score": val_score,
            "val_data": val_data,
            "val_cont": val_cont,
            "val_mom": val_mom,

            "norm_data": norm_data,
            "norm_cont": norm_cont,
            "norm_mom": norm_mom,

            "lr": opt.param_groups[0]["lr"],
        })

        if ep % 10 == 0 or ep == 1:
            loss_mom_text = (f"loss_mom={loss_mom:.3e} | " if loss_mom is not None else "")
            val_mom_text = (f"val_mom={val_mom:.3e} | " if val_mom is not None else "")

            logger.log_epoch(
                stage=f"PINN[{PHYSICS_MODE.upper()}]",
                epoch=ep,
                loss_total=loss_total,
                loss_data=loss_data,
                loss_cont=loss_cont,
                **({"loss_mom": loss_mom} if loss_mom is not None else {}),
                val_score=val_score,
                val_data=val_data,
                val_cont=val_cont,
                lr=trainer.optimizer.param_groups[0]["lr"],
                best_val=trainer.best_val,
            )
          
        if ep >= MIN_PHYS_EPOCHS and trainer.stop_improve():
            logger.log_message(
                "Early stopping PINN. "
                f"Melhor val_score: {trainer.best_val:.6e}"
            )
            break
            
    df_hist_stage2 = pd.DataFrame(history)
    df_hist_stage2.to_csv(PATH_LOG_EXP / f'pinn_loss_history_stage2_{cfg["experiment"]["name"]}_{timestamp}.csv', index=False)
    return trainer, (train_ds.x_mu, train_ds.x_sd), (train_ds.y_mu, train_ds.y_sd)

# ---------------------------------
# Treinar a PINN e avaliar
# ---------------------------------

# Iniciando a coleta dos logs
logger.start()

trainer, xscaler, yscaler = train_pinn_epoch(
    parquet_path=PATH_DATA / cfg["dataset"]["parquet"],
    feat_cols=cfg["features"],
    target_cols=cfg["targets"],
    batch_data=cfg["training"]["batch_size"],
    epochs_pre=int(pinn_cfg.get("epochs_pre", 50)),
    epochs_phys=int(pinn_cfg.get("epochs_phys", 100)),
    width=cfg["model"]["width"],
    depth=cfg["model"]["depth"],
    lr=float(cfg["training"]["lr"]),
    lr_wd=float(cfg["training"]["weight_decay"]),
    patience_es=cfg["early_stopping"]["patience"],
    patience_lr=cfg["scheduler"]["patience"],
    factor_lr = cfg["scheduler"]["factor"],
    w_data=float(pinn_cfg.get("w_data", 1.0)),
    w_cont=float(pinn_cfg.get("w_cont", 1.0)),
    w_mom=float(pinn_cfg.get("w_mom", 1.0)),
    out_model=cfg["experiment"]["name"],
    seed=cfg["experiment"]["seed"],
)

# salvar modelo e scalers
joblib.dump(xscaler, PATH_MODEL / "pinn_scaler_X.pkl")
joblib.dump(yscaler, PATH_MODEL / "pinn_scaler_Y.pkl")
logger.log_message(f"Scalers da PINN salvos em:{PATH_MODEL}")

# ======================================================
# AVALIAÇÃO DAS MÉTRICAS DE ACURÁCIA
# ======================================================

evaluation_result = run_metrics_pipeline(
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
    xscaler_path=PATH_MODEL / "pinn_scaler_X.pkl",
    yscaler_path=PATH_MODEL / "pinn_scaler_Y.pkl",
    metrics_path=(
        PATH_METRIC_EXP 
        / (f"pinn_{cfg['experiment']['name']}"
           f"_d{cfg['model']['depth']}"
           f"_w{cfg['model']['width']}"
           f"_seed{cfg['experiment']['seed']}.json"
        )
    ),
    predictions_path=(
        PATH_METRIC_EXP 
        / (
            f"pinn_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}"
            f"_predictions.parquet"
        )
    ),
    plots_dir=(
        PATH_PLOT 
        / (
            f"pinn_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}"
            f"_{timestamp}"
        )
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
    model_path=trainer.model_path,
    dataset_full_path=PATH_DATA / cfg["dataset"]["parquet"],
    xscaler_path=PATH_MODEL / "pinn_scaler_X.pkl",
    yscaler_path=PATH_MODEL / "pinn_scaler_Y.pkl",
    metrics_path=(
        PATH_METRIC_EXP
        / (
            f"pinn_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}"
            f"_physics.json"
        )
    ),
    predictions_path=(
        PATH_METRIC_EXP
        / (
            f"pinn_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}"
            f"_physics_predictions.parquet"
        )
    ),
    batch_size=cfg["training"]["batch_size"],
    plots_dir= (
        PATH_PLOT 
        / (
            f"pinn_{cfg['experiment']['name']}"
            f"_d{cfg['model']['depth']}"
            f"_w{cfg['model']['width']}"
            f"_seed{cfg['experiment']['seed']}"
            f"_{timestamp}"
        )
    )
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
        "best_val": trainer.best_val,
        "model_path": str(trainer.model_path),
        "evaluation_data": evaluation_result,
        "evaluation_physics": physics_result,
    }
)