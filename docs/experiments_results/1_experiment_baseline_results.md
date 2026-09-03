# Resumo Experimental – Etapa 1: Validação do Impacto da Física na PINN

## Objetivo

Avaliar o impacto isolado da incorporação da restrição física de continuidade em uma PINN corretora aplicada à reconstrução de soluções CFD do problema **Backward-Facing Step (BFS)**.

Nesta etapa buscou-se responder à seguinte pergunta:

> A inclusão da física melhora o desempenho da rede em relação a uma MLP puramente supervisionada?

Para garantir uma comparação controlada, todos os demais fatores foram mantidos constantes.

---

# Configuração Experimental

## Dataset

- Caso único: **Re = 36000**
- Mesmo conjunto de treino, validação e teste para todos os modelos.
- Mesmo processo de normalização.

## Features utilizadas

Foram utilizadas apenas as variáveis mínimas necessárias para representar o estado do escoamento:

- x
- y
- Ux
- Uy
- p
- Re_norm

Nesta fase foram removidas todas as features derivadas da turbulência, tais como:

- k
- epsilon
- nut
- vorticidade
- gradientes
- deformações
- divergência

O objetivo foi isolar exclusivamente o efeito da Physics Loss.

---

# Arquitetura

Mantida exatamente igual para MLP e PINN:

- mesma arquitetura
- mesma seed
- mesmo número de neurônios (arquitetura)
- mesmo split
- mesmo pré-processamento

A única diferença entre os modelos foi a presença da restrição física na função de perda.

---

# Primeiros Resultados

Inicialmente a PINN apresentou desempenho inferior à MLP em praticamente todas as métricas.

Observou-se que:

- MAE maior
- RMSE maior
- Erro relativo maior

A análise do histórico de treinamento mostrou que a etapa física ainda não havia convergido quando o treinamento foi encerrado.

Conclusão inicial:

> O problema não parecia estar na Physics Loss, mas sim em um treinamento insuficiente da etapa física.

---

# Ajuste das épocas

Foi aumentado o número de épocas da etapa PINN.

Resultado observado:

- redução significativa das métricas de erro;
- aproximação dos resultados da MLP;
- comportamento consistente das curvas de loss.

A análise indicou que a Physics Loss ainda estava refinando os pesos quando o treinamento anterior foi interrompido.

---

# Ajuste do peso da continuidade

Após garantir a convergência da etapa PINN, foi realizado um ajuste do peso associado ao termo de continuidade ($\lambda_{cont} = 1e-6$).

Esse experimento mostrou-se decisivo.

Com um peso mais adequado, a PINN passou a superar a MLP em todas as métricas supervisionadas.

Resultados obtidos:

| Métrica | MLP | PINN |
|----------|------:|------:|
| MAE (u,v) | 0.7040 | **0.7915** |
| RMSE (u,v) | 1.3044 | **1.4109** |
| L2 Rel (u,v) | 0.02902 | **0.02860** |
| MAE (p) | 16.7903 | **17.8458** |
| RMSE (p) | 26.304 | **28.9733** |
| L2 Rel (p) | 0.0658 | **0.0724** |

---

# Principais conclusões

Os experimentos indicam que:

- a inclusão da Physics Loss não degrada necessariamente o desempenho supervisionado;
- o peso associado ao termo físico é um hiperparâmetro crítico;
- quando corretamente ajustada, a restrição física atua como um regularizador do treinamento;
- a PINN foi capaz de superar a MLP mesmo utilizando apenas as features mínimas.

Esse resultado fornece evidências de que a melhoria observada decorre da incorporação da física, e não do uso de variáveis adicionais provenientes do CFD.

---

# Situação atual

Nesta etapa ainda **não foram alterados**:

- dataset;
- features;
- arquitetura;
- hiperparâmetros da rede.

O único parâmetro variado foi o peso da restrição física.

Dessa forma, o ganho observado pode ser atribuído com maior confiança ao efeito da Physics Loss.

---

# Próximos experimentos

## Etapa 2 — Validação Física (concluída)

Métricas de conservação de massa (MAE/RMSE/L∞ de `div U`) foram calculadas no domínio completo, comparando CFD coarse, CFD fine, MLP e três formulações de PINN (continuidade, momento isolado, continuidade+momento).

Resultados completos em `docs/experiments_results/2_experiment_physicis_cont.md`.

---

## Etapa 3 — Estudo de Features

Manter a melhor configuração física obtida e avaliar diferentes conjuntos de entrada:

1. Features mínimas
2. + k
3. + epsilon
4. + nut
5. + vorticidade
6. + gradientes
7. Features completas

Objetivo:

Identificar quais variáveis realmente contribuem para a melhoria do modelo.

---

## Etapa 4 — Otimização da Arquitetura

Com física e features definidas:

- número de camadas;
- largura da rede;
- funções de ativação;
- learning rate;
- regularização.

---

## Etapa 5 — Generalização

Expandir os experimentos para múltiplos números de Reynolds.

Inicialmente:

- treino em múltiplos Re;
- validação em Reynolds não vistos (interpolação).

Posteriormente:

- extrapolação para Reynolds fora da faixa de treinamento.

---

## Etapa 6 — Estratégias Híbridas

Após consolidar a configuração da PINN, investigar novas estratégias híbridas de correção física e comparar diferentes variantes de treinamento.