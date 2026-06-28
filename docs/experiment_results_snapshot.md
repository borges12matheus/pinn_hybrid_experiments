# Snapshot do Primeiro Experimento

Este documento registra o primeiro resultado real obtido no fluxo atual. Ele serve como referência base para comparação justa entre futuras execuções.

## 1. Identificação do run

- `run_id`: `k_epsilon_20260628_040010`
- `experiment_name`: `k_epsilon`
- `seed`: `42`
- `config`: `configs/mlp_base.yaml`
- `dataset`: `data/data_processed/dataset_bfs_2d_kepsilon_with_wz.parquet`
- `split`: `spatial_x_quantile_x_v1`

## 2. Ambiente observado

- `python_version`: `3.11.14`
- `torch_version`: `2.9.1+cu128`
- `cuda_available`: `true`
- `cuda_version`: `12.8`
- `gpu_name`: `NVIDIA GeForce RTX 3060`
- `gpu_memory_gb`: `11.75`
- `cpu_count_logical`: `12`
- `ram_total_gb`: `31.16`

## 3. Configuração usada

- `model`: MLP com `width=64`, `depth=4`, `activation=tanh`
- `training.batch_size`: `4096`
- `training.epochs`: `1000`
- `training.lr`: `1e-3`
- `training.weight_decay`: `1e-6`
- `early_stopping.patience`: `20`
- `scheduler.factor`: `0.5`
- `scheduler.patience`: `10`
- `split.bins`: `8`
- `split.test_frac`: `0.2`

## 4. Artefatos gerados

- modelo: `results/models/mlp_k_epsilon_d4_w64.pt`
- scaler de entrada: `results/models/mlp_scaler_X.pkl`
- scaler de saída: `results/models/mlp_scaler_Y.pkl`
- métricas: `results/metrics/k_epsilon_d4_w64_seed42.json`
- predições: `results/metrics/k_epsilon_d4_w64_seed42_predictions.parquet`
- plots: `results/plots/k_epsilon_d4_w64_seed42/`
- split persistido: `data/data_processed/splits/dataset_bfs_2d_kepsilon_with_wz_spatial_x_quantile_x/b8_seed42.json`
- logs: `logs/k_epsilon_20260628_040010.json` e `logs/k_epsilon_20260628_040010.log`

## 5. Resultado final

- `best_val`: `0.015669597163744577`
- `N`: `20118`

### Métricas principais

| Métrica | Coarse | Corrigido |
|---|---:|---:|
| MAE `u,v` | `6.0168` | `0.7437` |
| MAE `p` | `285.3959` | `17.9232` |
| RMSE `u,v` | `9.7504` | `1.3426` |
| RMSE `p` | `347.7868` | `28.3557` |
| L2 relativo `u,v` | `0.2169` | `0.0299` |
| L2 relativo `p` | `0.8695` | `0.0709` |

### Ganho observado

- `melhora_MAE_pct` em `u,v`: `87.64%`
- `melhora_MAE_pct_pressure`: `93.72%`
- `RER (u,v)`: `0.8764`
- `RER (p)`: `0.9372`

## 6. Leitura prática do resultado

- O corretor reduziu fortemente o erro em relação à solução coarse.
- O ganho foi mais forte em pressão do que em velocidade.
- O resultado funciona como baseline comparativo para novos ajustes de arquitetura, split ou hiperparâmetros.

## 7. Como reproduzir este snapshot

```bash
docker compose up --build train
```

Se o treino já tiver terminado e você quiser apenas refazer avaliação e gráficos:

```bash
docker compose --profile metrics run --rm metrics
```

## 8. Observações importantes

- Este snapshot deve ser tratado como referência experimental, não como limite final de performance.
- A comparação futura só é justa se mantivermos a mesma config, seed, dataset e política de split.
- Se qualquer uma dessas peças mudar, o snapshot deve ser considerado um novo experimento.

