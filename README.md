# BFS 2D PINN Corrector

Pipeline experimental para treinar uma PINN Híbrida corretora em um caso canônico BFS 2D baseado no trabalho de **D.M. Driver and H.L. Seegmiller**, disponibilizado na biblioteca do OPENFOAM, com foco em reprodutibilidade, comparação justa e execução em container com GPU.

Documentação detalhada:
- `docs/experiment_protocol.md`
- `docs/experiments_results/experiment_results_snapshot.md`

## O que este projeto faz
- Define um pipeline experimental desde o processamento dos dados de CFD até a construção de modelos comparativos de MLP e PINN híbrida.
- Treina a MLP e a PINN a partir de um dataset Parquet processado.
- Usa split espacial estratificado em `x` para evitar vazamento entre treino e teste.
- Salva modelo, scalers, split, métricas, predições e plots.
- Executa o pós-processamento de métricas comparativas automaticamente ao final do treino.

## Estrutura principal
- `src/train_mlp.py`: treino principal da MLP.
- `src/train_pinn.py`: treino principal da PINN (continuidade e, opcionalmente, momento).
- `src/run_baseline_pair.py`: orquestra o par MLP/PINN baseline e gera a comparação.
- `src/run_metrics.py`: avaliação final, métricas e geração de plots.
- `src/metrics.py`: cálculo das métricas.
- `src/plots.py`: geração dos gráficos.
- `src/train_utils.py`: dataset, MLP e trainer.
- `configs/baseline/mlp_base.yaml`: configuração padrão do experimento MLP.
- `configs/baseline/pinn_cont_base.yaml`: configuração espelhada para o experimento PINN.
- `configs/physics_cont_mom/pinn_cont_mom_v1.yaml` e `configs/physics_cont_mom/pinn_mom_v1.yaml`: variantes do PINN por formulação física (continuidade+momento e momento isolado).
- `docker-compose.yml`: execução com GPU e volumes padronizados.

## Requisitos
- Docker com suporte a GPU.
- `docker compose` instalado.
- Dataset disponível conforme `dataset.parquet` da config usada (baseline atual agosto/2026: `data/data_processed/dataset_bfs_2d_kepsilon_Re36000_full.parquet`).

## Como executar com Docker

### Par baseline completo (MLP + PINN + comparação)
```bash
docker compose up --build run_train_pipeline
```

### Treino isolado
```bash
docker compose up --build train_mlp
docker compose up --build train_pinn
```

Os demais serviços (`prepare_train_data`, `train_pinn_cont_mom`, `benchmark_mlp`, `benchmark_pinn`, `clean_experiments`, `compare_experiments`) estão listados em `docker-compose.yml`.

## Como executar localmente
### Treino completo
```bash
python src/train_mlp.py --config configs/baseline/mlp_base.yaml
python src/train_pinn.py --config configs/baseline/pinn_cont_base.yaml
```

### Apenas avaliação final
```bash
python src/run_metrics.py --config configs/baseline/mlp_base.yaml
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
python -m py_compile src/*.py
pytest tests/
```
Essa mesma validação roda automaticamente via GitHub Actions (`.github/workflows/ci.yml`) a cada push/PR para `dev` e `main`.

## Observações
- O treino já chama automaticamente o pós-processamento ao terminar.
- O serviço `metrics` existe no `docker-compose.yml` para rodar a etapa final isoladamente.
- O split atual é espacial estratificado em `x`, com `bins=8`, `val_frac=0.1` e `test_frac=0.2`.

## Referências
- D.M. Driver and H.L. Seegmiller. Features of a reattaching turbulent shear layer in divergent channel flow. AIAA Journal, 23(2):163–171, 1985.
- OpenFOAM. Disponível em: https://doc.openfoam.com/2606/
