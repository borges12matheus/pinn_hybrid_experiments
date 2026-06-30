# BFS 2D PINN Corrector

Pipeline experimental para treinar uma MLP corretora em um caso BFS 2D, com foco em reprodutibilidade, comparação justa e execução em container com GPU.

Documentação detalhada:
- `docs/experiment_protocol.md`
- `docs/experiment_results_snapshot.md`

## O que este projeto faz
- Treina a MLP a partir de um dataset Parquet processado.
- Usa split espacial estratificado em `x` para evitar vazamento entre treino e teste.
- Salva modelo, scalers, split, métricas, predições e plots.
- Executa o pós-processamento automaticamente ao final do treino.

## Estrutura principal
- `src/train_mlp.py`: treino principal da MLP.
- `src/run_metrics.py`: avaliação final, métricas e geração de plots.
- `src/metrics.py`: cálculo das métricas.
- `src/plots.py`: geração dos gráficos.
- `src/train_utils.py`: dataset, MLP e trainer.
- `configs/mlp_base.yaml`: configuração padrão do experimento MLP.
- `configs/pinn_base.yaml`: configuração espelhada para o experimento PINN.
- `configs/pinn_cont_v1.yaml` e `configs/pinn_cont_mom_v1.yaml`: variantes do PINN por formulação física.
- `docker-compose.yml`: execução com GPU e volumes padronizados.

## Requisitos
- Docker com suporte a GPU.
- `docker compose` instalado.
- Dataset disponível em `data/data_processed/dataset_bfs_2d_kepsilon_with_wz.parquet`.

## Como executar com Docker

### Treino completo
```bash
docker compose up --build train
```

### Apenas métricas e plots
```bash
docker compose --profile metrics run --rm metrics
```

## Como executar localmente
### Treino completo
```bash
python src/train_mlp.py --config configs/mlp_base.yaml
```

### Apenas avaliação final
```bash
python src/run_metrics.py --config configs/mlp_base.yaml
```

## Saídas geradas
- Modelo: `results/models/`
- Scalers: `results/models/mlp_scaler_X.pkl` e `results/models/mlp_scaler_Y.pkl`
- Split persistido: `data/data_processed/splits/`
- Métricas: `results/metrics/`
- Predições: `results/metrics/*_predictions.parquet`
- Plots: `results/plots/`
- Logs: `logs/`

## Reprodutibilidade
O experimento registra:
- `config_hash`
- `dataset_hash`
- `split_hash`
- caminho da config
- caminho do dataset
- política de split usada

Além disso, o container fixa:
- `PYTHONHASHSEED`
- `CUBLAS_WORKSPACE_CONFIG`
- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`

## Validação rápida
```bash
python -m py_compile src/train_mlp.py src/train_utils.py src/metrics.py src/logger.py src/plots.py src/run_metrics.py
```

## Observações
- O treino já chama automaticamente o pós-processamento ao terminar.
- O serviço `metrics` existe no `docker-compose.yml` para rodar a etapa final isoladamente.
- O split atual é espacial estratificado em `x`, com `bins=8` e `test_frac=0.2`.
