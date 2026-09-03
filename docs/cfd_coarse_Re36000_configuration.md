# Configurações CFD — BFS 2D coarse, Reₕ = 36.000

## Escopo

Este documento mapeia as principais configurações observadas no arquivo
`coarse_Re_36000.zip`, que contém o caso OpenFOAM `caso_Re_36000`. A análise
foi baseada nos dicionários do caso, nos campos iniciais, na definição da
malha e nos logs incluídos no arquivo.

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
| Amostragem PINN | Campos exportados | `U`, `p`, `k`, `epsilon`, `nut` e `div(U)` em pontos do arquivo `gridPoints_driver_seegmiller.xyz` | `system/sample_pinn` |

## Geometria e malha

As coordenadas do `blockMeshDict` são adimensionais antes da aplicação de
`scale = 0,0127`. Assim, o domínio físico verificado no `log.checkMesh` é:

- `x ∈ [-1,651; 0,635] m`, equivalente a `x/h ∈ [-130; 50]`;
- `y ∈ [0; 0,1143] m`, equivalente a `y/h ∈ [0; 9]`;
- espessura computacional `z = 0,0127 m`, tratada como 2D por faces `empty`.

O degrau ocorre em `x = 0`, com altura `h`. A malha coarse é composta por 6
blocos, com as seguintes quantidades de células:

| Bloco | Resolução `(nₓ, nᵧ, n_z)` | Células |
|---:|---:|---:|
| 0 | `(2, 65, 1)` | 130 |
| 1 | `(88, 65, 1)` | 5.720 |
| 2 | `(97, 48, 1)` | 4.656 |
| 3 | `(33, 48, 1)` | 1.584 |
| 4 | `(97, 65, 1)` | 6.305 |
| 5 | `(33, 65, 1)` | 2.145 |
| **Total** |  | **20.540** |

Indicadores registrados pelo `checkMesh`:

| Indicador | Resultado |
|---|---:|
| Pontos | 41.748 |
| Faces | 82.493 |
| Faces internas | 40.747 |
| Células | 20.540 hexaédricas |
| Patches | 8 |
| Volume total | 0,00305209 m³ |
| Aspect ratio máximo | 7.601,12 em 192 células |
| Não ortogonalidade média/máxima | 3,71° / 83,34° |
| Faces severamente não ortogonais | 16 (`> 70°`) |
| Skewness máxima | 0,27582 |

O `checkMesh` terminou com `Failed 1 mesh checks`, associado ao alto aspect
ratio. A topologia, os volumes, a abertura dos contornos e a skewness foram
reportados como válidos. Esse alerta deve ser considerado ao interpretar
gradientes próximos às paredes e ao comparar a solução coarse com uma malha
de maior resolução.

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

No log incluído, o `simpleFoam` alcança o tempo `2000`. No último passo
registrado, os resíduos finais foram aproximadamente `4,49e-6` para `Ux`,
`2,65e-5` para `Uy`, `7,12e-5` para `p`, `3,25e-8` para `epsilon` e `3,86e-6`
para `k`; o erro global de continuidade foi `1,79e-6` e o acumulado foi
`-1,06305`. Esses números são observações do log, não uma confirmação
automática de convergência física ou independência de malha.

## Pontos de atenção para o uso no PINN

1. Usar `h = 0,0127 m` e `Uref = 44,2 m/s` na normalização das coordenadas e
   das grandezas derivadas.
2. Manter explícita a convenção de pressão: o caso usa pressão cinemática e
   `p = 0` na saída.
3. Tratar `div(U)` como saída de pós-processamento, não como variável primária
   resolvida pelo `simpleFoam`.
4. Registrar que a malha possui 192 células com aspect ratio elevado e 16
   faces severamente não ortogonais.
5. Ao comparar com dados fine, confirmar que os campos e as posições de
   amostragem usam a mesma unidade, origem geométrica e convenção de sinal.

## Arquivo de origem

`/home/matheus/Documentos/Mestrado/Dissertação/Pesquisa/bfs_cfd/bfs_2d_coarse/coarse_Re_36000.zip`
