# Protocolo Experimental - BFS 2D PINN Corrector

Este documento descreve o fluxo experimental atual do projeto, com foco em reprodutibilidade, comparação justa e execução em container com GPU.

## 1. Objetivo do experimento

O projeto treina uma MLP corretora para estimar correções em grandezas de um caso BFS 2D:
- `dUx`
- `dUy`
- `dp`

A ideia central é comparar a solução coarse com a solução corrigida, preservando:
- mesma base de dados
- mesma convenção de split
- mesma configuração de treino
- mesmo ambiente de execução

## 2. O que mudou na arquitetura do fluxo

O fluxo foi organizado em etapas claras:

1. `src/train_mlp.py`
   - carrega a configuração
   - define o split espacial
   - treina a MLP
   - salva modelo, scalers e metadata
   - chama automaticamente o pós-processamento

2. `src/run_metrics.py`
   - recarrega modelo e scalers
   - executa as métricas
   - salva predições
   - gera plots

3. `src/metrics.py`
   - calcula MAE, RMSE e métricas relativas

4. `src/plots.py`
   - gera comparações visuais
   - funciona em modo headless

5. `docker-compose.yml`
   - padroniza execução com GPU
   - expõe serviços para treino e métricas

## 3. Estrutura do projeto

Arquivos principais:
- `src/train_mlp.py`: treinamento principal da MLP
- `src/train_pinn.py`: treinamento principal da PINN (continuidade e, opcionalmente, momento)
- `src/run_baseline_pair.py`: orquestra o par MLP/PINN baseline
- `src/run_metrics.py`: avaliação final e gráficos
- `src/train_utils.py`: dataset, MLP e trainer
- `src/metrics.py`: métricas quantitativas
- `src/plots.py`: visualizações
- `src/logger.py`: logs e metadata do experimento
- `configs/baseline/mlp_base.yaml`: configuração-base da MLP
- `configs/baseline/pinn_cont_base.yaml`: configuração-base da PINN
- `docker-compose.yml`: execução com GPU
- `Dockerfile`: imagem base do ambiente

## 4. Configuração experimental

O arquivo `configs/baseline/mlp_base.yaml` é a fonte de verdade do experimento MLP baseline.

Parâmetros principais:
- `experiment.name`: nome do experimento
- `experiment.seed`: seed usada no treino e no split
- `paths.*`: diretórios de entrada e saída
- `dataset.parquet`: dataset de entrada
- `features`: colunas de entrada
- `targets`: colunas supervisionadas
- `model.width`, `model.depth`: arquitetura da MLP
- `training.batch_size`: tamanho do lote
- `training.epochs`: número máximo de épocas
- `training.lr`: learning rate
- `training.weight_decay`: regularização
- `early_stopping.patience`: paciência do early stopping
- `scheduler.factor`, `scheduler.patience`: scheduler
- `split.strategy`: estratégia de split
- `split.column`: coluna usada no split espacial
- `split.bins`: número de bins
- `split.test_frac`: fração de teste por bin

## 5. Convenção de split

Foi adotada a convenção:
- estratégia: `spatial_x_quantile`
- coluna: `x`
- bins: `8`
- fração de teste: `0.2`

Motivo:
- evita vazamento espacial entre treino e teste
- mantém comparação mais honesta
- não é rígido demais
- permite repetição exata do mesmo particionamento

O split é persistido em:
- `data/data_processed/splits/`

Cada split grava:
- `dataset_path`
- `split_method`
- `split_strategy`
- `split_column`
- `split_bins`
- `test_frac`
- índices de treino e teste

## 6. Determinismo e rastreabilidade

O experimento registra:
- `config_hash`
- `dataset_hash`
- `split_hash`
- caminhos dos arquivos usados

Além disso, o container fixa:
- `PYTHONHASHSEED`
- `CUBLAS_WORKSPACE_CONFIG`
- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`

No treino também são fixados:
- seed do Python
- seed do NumPy
- seed do PyTorch
- seed da GPU
- flags de determinismo do cuDNN
- TF32 desligado

## 7. Como executar

### 7.1 Via Docker Compose

Par baseline completo (MLP + PINN + comparação):
```bash
docker compose up --build run_train_pipeline
```

Treino isolado:
```bash
docker compose up --build train_mlp
docker compose up --build train_pinn
```

### 7.2 Localmente

Treino completo:
```bash
python src/train_mlp.py --config configs/baseline/mlp_base.yaml
```

Apenas avaliação:
```bash
python src/run_metrics.py --config configs/baseline/mlp_base.yaml
```

## 8. Entrada esperada

O dataset principal deve existir conforme `dataset.parquet` da config usada. Na config baseline atual:
```text
data/data_processed/dataset_bfs_2d_kepsilon_Re36000_full.parquet
```

Colunas esperadas incluem:
- `x`
- `y`
- `Ux`
- `Uy`
- `p`
- `k`
- `nut_log`
- `Re`
- `dUx`
- `dUy`
- `dp`

## 9. Pipeline de treino

Durante o treino:
1. O dataset é carregado.
2. O split espacial é criado ou reaproveitado.
3. O treino usa `80%` dos dados em cada bin espacial.
4. A validação usa `20%`.
5. O melhor modelo é salvo por `early stopping`.
6. Após o treino, o pós-processamento roda automaticamente.

Artefatos gerados:
- modelo em `results/models/`
- scalers em `results/models/`
- dataset de teste em `data/data_processed/`
- metadata em `logs/`
- métricas em `results/metrics/`
- predições em `results/metrics/`
- plots em `results/plots/`

## 10. Métricas geradas

As métricas calculadas incluem:
- MAE vetorial para `u,v`
- MAE de pressão
- RMSE para `u,v`
- RMSE de pressão
- erro relativo L2 para `u,v`
- erro relativo L2 para pressão
- métricas de comparação entre coarse, corrigido e referência

## 11. Plots gerados

Os gráficos principais incluem:
- comparação de campo
- comparação de erro
- ganho local
- histograma de erro
- dispersão predito vs referência

Os plots são salvos como arquivos `.png` e podem ser gerados sem interface gráfica.

## 12. Estrutura de saídas

Saídas esperadas:
- `results/models/`
- `results/metrics/`
- `results/plots/`
- `logs/`
- `data/data_processed/splits/`

Arquivos importantes:
- `mlp_scaler_X.pkl`
- `mlp_scaler_Y.pkl`
- `*.pt`
- `*_predictions.parquet`
- `*.json`
- `*.png`

## 13. Como validar a execução

### Validação mínima
```bash
python -m py_compile src/*.py
pytest tests/
```
Essa mesma validação roda automaticamente via GitHub Actions (`.github/workflows/ci.yml`) a cada push/PR para `dev` e `main`.

### Validação de GPU no host
```bash
nvidia-smi
```

### Validação do container com GPU
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Validação do compose
```bash
docker compose config
```

## 14. Checklist de replicação

Antes de rodar:
- conferir se o dataset existe
- conferir se a config aponta para os diretórios corretos
- conferir se o Docker tem GPU habilitada
- conferir se o split desejado está definido na config

Depois de rodar:
- verificar se o modelo foi salvo
- verificar se os scalers foram salvos
- verificar se os arquivos de métricas existem
- verificar se os plots foram gerados
- verificar se o `split_hash` ficou registrado no metadata

## 15. Limitações conhecidas

- O modelo e o pipeline estão otimizados para comparação experimental, não para treino massivamente escalável.
- A ocupação da GPU pode ficar baixa se o dataset for pequeno ou se o pipeline de dados for o gargalo.
- O split atual é espacial em `x`; se a distribuição do problema exigir, pode ser necessário evoluir para um split mais rico.

## 16. Como manter o experimento consistente

Para evitar divergências entre execuções:
- não alterar a config sem registrar a mudança
- não trocar o dataset sem regenerar o split
- não reutilizar modelos antigos com configs novas
- manter os artefatos por experimento em diretórios distintos
- preservar a mesma seed quando o objetivo for comparação direta

## 17. Resumo operacional

Fluxo recomendado:
1. preparar dataset
2. ajustar `configs/baseline/mlp_base.yaml` / `configs/baseline/pinn_cont_base.yaml` se necessário
3. rodar `docker compose up --build run_train_pipeline`
4. revisar métricas e plots gerados
5. repetir o experimento apenas com mudanças explícitas

## 18. Snapshot do primeiro resultado

O primeiro run válido do fluxo atual está documentado em:
- `docs/experiments_results/experiment_results_snapshot.md`

Esse arquivo registra:
- ambiente observado
- config usada
- artefatos gerados
- métricas finais
- leitura prática do resultado
- comando de reprodução
