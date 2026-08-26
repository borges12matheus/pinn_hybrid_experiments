# Resumo Experimental – Etapa 2: Validação Física (continuidade vs. momento)

## Objetivo

Avaliar, sob a mesma base experimental da Etapa 1, o efeito de diferentes formulações físicas da PINN sobre a **consistência física** (conservação de massa) e a **acurácia supervisionada**, isolando cada termo do resíduo de PDE:

- `continuity`: apenas continuidade
- `momentum`: apenas momento (nova formulação, implementada e validada nesta etapa)
- `cont_mom`: continuidade + momento juntos (primeira execução bem-sucedida no projeto — havia um bug de desempacotamento do batch no dataloader que impedia essa formulação de treinar até o fim; corrigido antes desta rodada)

Pergunta central:

> Momento agrega valor além do que a continuidade sozinha já entrega — seja em acurácia, seja em consistência física?

## Configuração experimental

Os quatro modelos comparados compartilham, verificado via `run_baseline_pair.py` (checagem de isonomia com hash SHA-256 do dataset):

- Dataset: `dataset_bfs_2d_kepsilon_Re36000_full.parquet` (mesmo arquivo, mesmo hash, para os quatro)
- Split: `spatial_x_quantile`, coluna `x`, bins=8, test_frac=0.2, seed=42
- Arquitetura: width=64, depth=4, tanh
- Treino/scheduler/early stopping idênticos

Features diferem por necessidade física, não por descuido — convenção já adotada no projeto (ver `tests/test_pinn_physics.py`): a MLP e a PINN de continuidade usam só `Re_norm`; as variantes com momento precisam de `Re` em escala física (não normalizada) e `nut_log` para calcular o termo de viscosidade efetiva do resíduo.

| Variante | physics_mode | w_cont | w_mom | run_id |
|---|---|---:|---:|---|
| MLP (baseline) | — | — | — | `mlp_base_20260729_192246` |
| PINN continuidade | continuity | 7.5e-5 | — | `pinn_cont_base_20260729_192724` |
| PINN momento | momentum | — | 1e-5 | `pinn_mom_base_20260825_215647` |
| PINN cont+mom | cont_mom | 7.5e-5 | 5e-5 | `pinn_cont_mom_base_20260826_003555` |

## Resultados — acurácia supervisionada (teste, N=20118)

| Modelo | MAE u,v | RMSE u,v | MAE p | RMSE p |
|---|---:|---:|---:|---:|
| MLP | 0.7040 | 1.3044 | 16.79 | 26.30 |
| PINN continuidade | 0.7915 | 1.4109 | 17.85 | 28.97 |
| PINN momento | 0.7522 | 1.3335 | 20.06 | 32.41 |
| **PINN cont+mom** | **0.8775** | **1.6009** | **24.22** | **40.35** |

## Resultados — consistência física / continuidade (domínio completo, N=100589)

| Modelo | MAE div | RMSE div | L∞ div | viés médio |
|---|---:|---:|---:|---:|
| CFD coarse (referência) | 0.155 | 3.38 | 241.5 | -0.057 |
| CFD fine (referência) | 0.424 | 5.37 | 864.2 | -0.033 |
| MLP | 122.23 | 465.66 | 19221.4 | -5.434 |
| PINN continuidade | 2.546 | 4.340 | 188.2 | +0.0195 |
| PINN momento | 98.17 | 310.12 | 8973.7 | +18.08 |
| **PINN cont+mom** | **2.680** | **5.647** | **235.17** | **-0.1198** |

## Leitura dos resultados

1. **A MLP viola continuidade em 2-3 ordens de grandeza** em relação ao campo coarse bruto — nenhum corretor sem física de continuidade preserva conservação de massa.
2. **Continuidade isolada resolve isso quase por completo**: a divergência cai para a mesma ordem de grandeza do ruído numérico do próprio CFD (RMSE=4.34, abaixo até do RMSE do campo fine=5.37), a um custo moderado e uniforme de acurácia (~6-12% pior que a MLP). Este ponto (`w_cont=7.5e-5`) já havia sido replicado em 3 seeds (mae_div=2.41±0.15, rmse_div=4.47±0.12) em `results/comparisons/experiments_publication.csv` — é o resultado mais sólido desta etapa, mas **esse arquivo foi removido do working tree em 2026-08-26** (ainda recuperável via git, ver nota abaixo).
3. **Momento isolado tem efeito parcial e espacialmente localizado**: reduz a violação de continuidade da MLP em 20-53% (a depender da métrica), mas permanece 2-3 ordens de grandeza acima do coarse/fine — ameniza, não resolve. O ganho se concentra na região de recirculação pós-degrau (visível no mapa espacial de ganho); no resto do domínio, momento e MLP são estatisticamente indistinguíveis. Custo de acurácia maior que o esperado, concentrado em pressão (+19-23% pior que a MLP), coerente com o resíduo de momento penalizar diretamente o gradiente de pressão.
4. **Combinar momento com continuidade piora os dois eixos** em relação à continuidade isolada: RMSE de divergência 30% pior, L∞ 25% pior, e acurácia substancialmente pior (RMSE de pressão 39% pior que a continuidade sozinha, 53% pior que a MLP). Não há evidência, neste ponto do espaço de hiperparâmetros, de que o momento contribua incrementalmente quando a continuidade já está presente — pelo contrário.

**Resposta à pergunta central:** não, momento não agregou valor neste teste, isolado ou combinado com continuidade. A continuidade concentra praticamente todo o ganho de consistência física observado no projeto até aqui.

## Limitações e cautelas

- Os resultados de `momentum` e `cont_mom` são de **seed única (42)** — ainda não replicados como o ponto de continuidade já está. Uma conclusão definitiva sobre o valor (ou ausência de valor) do termo de momento exige repetir em pelo menos +2 seeds (123, 456), seguindo o padrão já usado na Etapa 1.
- Os pesos usados (`w_mom=1e-5` para momento isolado, `w_mom=5e-5` para cont_mom) não passaram por um sweep como `w_cont` passou na Etapa 1. É possível que o termo de momento esteja mal calibrado — grande demais, competindo pelo gradiente sem necessidade — e que um sweep (ex.: 1e-6, 1e-7) mude a leitura.
- `results/comparisons/experiments_publication.csv`, `experiments_main_comparison.csv`, `experiments_full.{csv,parquet}`, `experiments_summary_by_seed.csv`, `comparability_report.csv` e o diretório `seed456/` — todos rastreados no git — foram removidos do working tree em 2026-08-26 (fora desta sessão de trabalho). Ainda recuperáveis via `git checkout` caso sejam necessários; os números citados acima foram extraídos antes da remoção.
- Toda a validação física desta etapa mede apenas o **resíduo de continuidade**. O resíduo de momento em si (equilíbrio de forças, acoplamento pressão-velocidade) nunca foi medido como métrica de validação independente — só usado como termo de treino. Avaliar isso diretamente ajudaria a separar "momento não ajuda a continuidade" (o que já está bem estabelecido aqui) de "momento não ajuda em nada" (ainda em aberto).

## Próximos experimentos

- Sweep de `w_mom` (1e-6, 1e-7, ...) para `momentum` e `cont_mom`, análogo ao sweep de `w_cont` da Etapa 1.
- Replicar `momentum` e `cont_mom` nas seeds 123 e 456 antes de tratar qualquer conclusão como estável.
- Medir o resíduo de momento como métrica de validação (não só como termo de perda) para os quatro modelos.
- Seguindo o resultado desta etapa, a continuidade isolada (`w_cont=7.5e-5`) é a configuração física recomendada para avançar à Etapa 3 (Estudo de Features), a menos que o sweep de `w_mom` mude esse quadro.

## Artefatos

- Comparações: `results/comparisons/baseline_cont_seed42/`, `results/comparisons/mlp_vs_pinn_mom_seed42/`, `results/comparisons/mlp_vs_pinn_cont_mom_seed42_20260826_003110/`
- Configs: `configs/baseline/pinn_cont_base.yaml`, `configs/baseline/pinn_mom_base.yaml`, `configs/baseline/pinn_cont_mom_base.yaml`
- Runs brutos: `logs/pinn/pinn_cont_base_20260729_192724/`, `logs/pinn/pinn_mom_base_20260825_215647/`, `logs/pinn/pinn_cont_mom_base_20260826_003555/`
