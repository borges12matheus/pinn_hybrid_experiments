# Configurações CFD — BFS 2D fine, Reₕ = 36.000

## Escopo

Este documento mapeia as principais configurações observadas no arquivo
`fine_Re_36000.zip`, que contém o caso OpenFOAM `caso_Re_36000`. A análise foi
baseada nos dicionários do caso, nos campos iniciais, na definição da malha e
nos logs incluídos no arquivo.

O caso representa o escoamento turbulento permanente sobre um backward-facing
step (BFS), baseado no caso de Driver e Seegmiller. A altura do degrau é
`h = 0,0127 m` e o número de Reynolds baseado nessa altura é `Reₕ = 36.000`.

## Tabela-resumo

| Categoria | Configuração mapeada | Valor observado | Arquivo/fonte |
|---|---|---|---|
| Solver | Aplicação | `simpleFoam` (escoamento permanente incompressível) | `system/controlDict` |
| Regime temporal | Esquema temporal | `steadyState` | `system/fvSchemes` |
| Execução | Início/fim | `startTime = 0`, `endTime = 2000` | `system/controlDict` |
| Execução | Passo e gravação | `deltaT = 1`; grava a cada 1000 passos; mantém 5 gravações | `system/controlDict` |
| Modelo de turbulência | Formulação | RAS, turbulência ligada | `constant/turbulenceProperties` |
| Modelo de turbulência | Modelo RAS | `kEpsilon` | `constant/turbulenceProperties` |
| Fluido | Modelo de transporte | Newtoniano | `constant/transportProperties` |
| Fluido | Viscosidade cinemática | `nu = 1,56 × 10⁻⁵ m²/s` | `constant/transportProperties` |
| Entrada | Velocidade | `U = (44,2 0 0) m/s` | `0/U` |
| Entrada | Pressão, `k`, `epsilon` | `p`: gradiente nulo; `k = 1,09 × 10⁻³ m²/s²`; `epsilon = 17,83 m²/s³` | `0/p`, `0/k`, `0/epsilon` |
| Saída | Condições principais | `U`, `k`, `epsilon`: gradiente nulo; `p = 0` | `0/U`, `0/p`, `0/k`, `0/epsilon` |
| Paredes | Velocidade | `noSlip` em `upperWall` e `lowerWall` | `0/U` |
| Paredes | Turbulência | `kqRWallFunction`, `epsilonWallFunction`, `nutUBlendedWallFunction` | `0/k`, `0/epsilon`, `0/nut` |
| Simetria | Regiões de entrada/saída do canal | `lowerWallStartup` e `upperWallStartup` como `symmetryPlane` | `system/blockMeshDict`, campos em `0/` |
| Dimensionalidade | Faces frontal/traseira | `empty` | `system/blockMeshDict`, campos em `0/` |
| Pressão | Referência de pressão | `pInf = 0`, `rhoInf = 1` | `system/pressureCoefficient` |
| Pressão | Velocidade de referência | `UInf = (44,2 0 0) m/s` | `system/pressureCoefficient` |
| Discretização | Gradiente/interpolação | `Gauss linear`; interpolação `linear` | `system/fvSchemes` |
| Discretização | Convecção de `U` | `bounded Gauss LUST grad(U)` | `system/fvSchemes` |
| Discretização | Convecção da turbulência | `bounded Gauss limitedLinear 1` para `k`, `epsilon` e `omega` | `system/fvSchemes` |
| Discretização | Laplaciano/normal | `Gauss linear corrected`; `corrected` | `system/fvSchemes` |
| SIMPLE | Corretores não ortogonais | `nNonOrthogonalCorrectors = 0` | `system/fvSolution` |
| SIMPLE | Formulação | `consistent yes` | `system/fvSolution` |
| Solução | Pressão | `GAMG`, tolerância `1e-10`, `relTol = 0,1`, `DICGaussSeidel` | `system/fvSolution` |
| Solução | `U`, `k`, `epsilon`, `omega` | `smoothSolver`, `symGaussSeidel`, tolerância `1e-10`, `relTol = 0,1` | `system/fvSolution` |
| Relaxação | Equações | `U = 0,3`; demais equações = `0,2` | `system/fvSolution` |
| Pós-processamento | Funções | `devReff`, coeficiente de pressão, amostragem, centros de célula e tensão de cisalhamento | `system/controlDict` |
| Amostragem | Perfis | Perfis de `p`, `U` e `devReff` em `x/h = -4, 1, 4, 6, 10`; 100 pontos em `y` | `system/sample` |
| Amostragem PINN | Campos exportados | `U`, `p`, `k`, `epsilon`, `nut` e `div(U)` em pontos de `gridPoints_driver_seegmiller.xyz` | `system/sample_pinn` |

## Geometria e malha

As coordenadas do `blockMeshDict` são adimensionais antes da aplicação de
`scale = 0,0127`. O domínio físico é o mesmo do caso coarse:

- `x ∈ [-1,651; 0,635] m`, equivalente a `x/h ∈ [-130; 50]`;
- `y ∈ [0; 0,1143] m`, equivalente a `y/h ∈ [0; 9]`;
- espessura computacional `z = 0,0127 m`, tratada como 2D por faces `empty`.

O degrau ocorre em `x = 0`, com altura `h`. A malha fine é composta por 6
blocos hexaédricos. Em comparação com a malha coarse, a resolução foi
multiplicada por 4 nas direções `x` e `y`, mantendo uma célula na direção
computacional `z`.

| Bloco | Resolução `(nₓ, nᵧ, n_z)` | Células |
|---:|---:|---:|
| 0 | `(8, 260, 1)` | 2.080 |
| 1 | `(352, 260, 1)` | 91.520 |
| 2 | `(388, 192, 1)` | 74.496 |
| 3 | `(132, 192, 1)` | 25.344 |
| 4 | `(388, 260, 1)` | 100.880 |
| 5 | `(132, 260, 1)` | 34.320 |
| **Total** |  | **328.640** |

Indicadores registrados pelo `checkMesh`:

| Indicador | Resultado |
|---|---:|
| Pontos | 659.946 |
| Faces | 1.315.892 |
| Faces internas | 655.948 |
| Células | 328.640 hexaédricas |
| Patches | 8 |
| Volume total | 0,00305209 m³ |
| Aspect ratio máximo | 7.869,33 em 2.537 células |
| Não ortogonalidade média/máxima | 3,19° / 60,66° |
| Faces severamente não ortogonais (`> 70°`) | 0 |
| Skewness máxima | 0,276043 |

O `checkMesh` terminou com `Failed 1 mesh checks`, associado ao alto aspect
ratio. A topologia, os volumes e a skewness foram reportados como válidos; a
não ortogonalidade máxima ficou abaixo do limite de 70° indicado no log.
Apesar do refinamento, ainda existem 2.537 células com aspect ratio elevado,
portanto esse indicador continua sendo uma limitação relevante na análise de
gradientes próximos às paredes.

## Condições de contorno e campos iniciais

O campo interno inicial é quase nulo para a velocidade (`U = (1e-8, 0, 0)`)
e nulo para a pressão. Os campos turbulentos são inicializados uniformemente
com `k = 1,09e-3`, `epsilon = 17,83` e `omega = 181.728`; `nut` e `nuTilda`
começam em zero.

Os arquivos `omega` e `nuTilda` estão presentes no diretório inicial e
`omega` aparece na expressão genérica de solvers do `fvSolution`, mas o modelo
declarado é `kEpsilon`. Portanto, a configuração ativa de turbulência deve ser
interpretada como baseada em `k`–`epsilon`; a presença dos demais campos parece
ser compatibilidade/legado do caso e merece conferência antes de reutilizar o
template para outro modelo RAS.

## Saídas e execução observada

O caso inclui:

- campos de solução nos tempos `1000` e `2000`;
- perfis de velocidade, pressão e tensão efetiva;
- coeficiente de pressão `cp` na parede inferior;
- tensão de cisalhamento na parede;
- amostragem adicional para a malha de pontos usada pelo pipeline PINN;
- logs de `blockMesh`, `checkMesh` e `simpleFoam`.

No log incluído, o `simpleFoam` alcança o tempo `2000`, com tempo de execução
registrado de aproximadamente `942 s` em um processo. No último passo, os
resíduos finais foram aproximadamente `5,95e-6` para `Ux`, `6,91e-5` para
`Uy`, `9,33e-4` para `p`, `2,23e-7` para `epsilon` e `5,01e-6` para `k`; o erro
global de continuidade foi `-7,06e-5` e o acumulado foi `-0,06665`. Esses
números são observações do log, não uma confirmação automática de convergência
física ou independência de malha.

## Comparação direta com a malha coarse

| Item | Coarse | Fine | Relação fine/coarse |
|---|---:|---:|---:|
| Células | 20.540 | 328.640 | 16× |
| Pontos | 41.748 | 659.946 | 15,8× |
| Aspect ratio máximo | 7.601,12 | 7.869,33 | maior na fine |
| Não ortogonalidade máxima | 83,34° | 60,66° | reduzida |
| Células com alto aspect ratio | 192 | 2.537 | maior em valor absoluto |
| Tempo de execução registrado | 44,36 s | 942,06 s | aproximadamente 21,2× |

O custo computacional observado cresce mais que o número de células, enquanto
a qualidade geométrica melhora especificamente na não ortogonalidade. Como os
casos usam o mesmo domínio, física, solver e pós-processamento, essa dupla é
adequada para estudos coarse→fine, desde que os estados comparados estejam
alinhados no mesmo tempo e nas mesmas posições de amostragem.

## Pontos de atenção para o uso no PINN

1. Usar `h = 0,0127 m` e `Uref = 44,2 m/s` na normalização das coordenadas e
   das grandezas derivadas.
2. Manter explícita a convenção de pressão: o caso usa pressão cinemática e
   `p = 0` na saída.
3. Tratar `div(U)` como saída de pós-processamento, não como variável primária
   resolvida pelo `simpleFoam`.
4. Ao comparar com o coarse, usar o mesmo domínio e confirmar a interpolação
   fine→pontos de avaliação do pipeline.
5. Registrar os alertas de qualidade de malha, especialmente o alto aspect
   ratio, mesmo que a não ortogonalidade esteja melhor que no coarse.

## Arquivo de origem

`/home/matheus/Documentos/Mestrado/Dissertação/Pesquisa/bfs_cfd/bfs_2d_fine/fine_Re_36000.zip`
