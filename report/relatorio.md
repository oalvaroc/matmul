---
title: "Multiplicação de matrizes: serial, MPI e MPI + OpenBLAS"
author: "Álvaro Costa"
date: "27 de outubro de 2026"
lang: pt-BR
documentclass: article
fontsize: 10pt
geometry: margin=2cm
numbersections: true
colorlinks: true
header-includes: |
  \usepackage{etoolbox}
  \AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{4pt}}
  \usepackage{float}
  \floatplacement{figure}{H}
  \setlength{\parskip}{0.4em}
  \setlength{\abovecaptionskip}{2pt}
---

# Versões e configuração experimental

As três versões calculam $C = A \times B$ em precisão simples, com $A$ e $B$
quadradas de ordem $n$:

* **serial**: laço triplo na ordem linha–$k$–coluna;
* **MPI**: $C$ dividido em $q^2$ blocos $b \times b$ ($q = n/b$). O rank 0
  (mestre) envia a cada trabalhador, sob demanda, a linha de blocos de $A$ e
  a coluna de blocos de $B$ de um $C_{ij}$ ($2qb^2$ floats, `MPI_Send`
  bloqueante) e recebe o bloco calculado. Com $P$ processos há $P-1$
  trabalhadores; o mestre não calcula;
* **MPI + OpenBLAS**: igual, com `cblas_sgemm` no lugar do laço triplo.

Dois limites decorrem do desenho: no máximo $\min(q^2, P-1)$ trabalhadores
ficam ocupados, e o mestre envia sozinho $8n^2q$ bytes, pois cada bloco de
$A$ e $B$ é reenviado $q$ vezes.

**Ambiente.** Cluster ada (OpenPBS), até 2 nós, cada um com 2 AMD EPYC
7443 (48 núcleos, 2,8 GHz, 8 domínios NUMA); Open MPI 5.0.6 sobre UCX,
OpenBLAS 0.3.21. Os programas foram compilados com GCC 12.2 com a
otimização padrão do compilador, que no GCC é **`-O0`, sem otimização**
(`CMAKE_BUILD_TYPE` e `CMAKE_C_FLAGS` vazios), o que penaliza o serial e a
versão MPI, mas não a OpenBLAS, que é pré-compilada. Um processo por núcleo e
`OMP_NUM_THREADS=1`, exceto no teste ranks × threads (seção 5). Matrizes
aleatórias com semente fixa, iguais nas três versões.

**Medição.** No serial, só a multiplicação (`clock_gettime`). Nas versões
MPI, o rank 0 mede com `MPI_Wtime` desde o envio da primeira tarefa até a
recepção do último bloco, incluindo empacotamento, comunicação e cálculo.
Ficam de fora `MPI_Init`, geração das matrizes, escrita do resultado e o
aquecimento das conexões UCX (uma mensagem vazia por trabalhador). Esse custo
é quase todo de conexões entre nós: em um nó fica abaixo de 0,5 ms até 48
processos, mas leva 2,7 s com 65 e 5,0 s (3,9–5,8 s) com 96, de 9 a 470 vezes
a própria multiplicação.

**Repetições.** Cada configuração foi executada 6 vezes (seção 2) ou 5 vezes
(seções 3 a 5). Reportamos a **mediana**, o **coeficiente de variação** (CV)
nas tabelas e o intervalo mínimo–máximo como barras de erro nas figuras.

**Referências.** Speedup $S = T_\text{serial}(n)/T$ e eficiência
$E = S/P$, com $P$ o total de processos. Como a `sgemm` é muito mais rápida
que o laço do serial, $S$ e $E$ da versão OpenBLAS somam o ganho do
paralelismo e o do núcleo de cálculo. Por isso, nas seções 3 e 4 cada
versão é comparada a ela mesma com um trabalhador ($W = 1$, $P = 2$).

# Ordem da matriz e tamanho do bloco (96 processos, 2 nós)

Serial: 0,439 s, 3,51 s, 27,9 s e 224 s para $n$ = 512 a 4096
(CV $\leq$ 0,5 %, 0,61 GFLOP/s): o tempo cresce com $n^3$.

```{.include file="build/tab_bench.md"}
```

![Melhor bloco de cada ordem: tempo e speedup sobre o serial.](build/fig_ordem.pdf)

![Tempo em função do bloco $b$.](build/fig_bloco.pdf)

**Ordem.** A eficiência cresce com $n$ (MPI: 14 % → 52 %), pois o cálculo
cresce com $n^3$ e a comunicação com $n^2q$. A versão OpenBLAS é de 3 a 9
vezes mais rápida que a MPI; seu $E > 100$ % vem do núcleo de cálculo
($\approx$ 17 GFLOP/s por trabalhador contra 0,61 do serial), não do paralelismo.

**Bloco.** Na versão MPI, o cálculo domina e vence o bloco que ocupa mais
trabalhadores: em $n = 2048$, $b = 512$ dá só 16 tarefas (79 trabalhadores
ociosos) e é 2,7× mais lento que $b = 256$. Na versão OpenBLAS o gargalo é o
mestre: o bloco **maior** vence sempre, e o tempo acompanha o volume enviado
(134 MB em 0,064 s com $b = 512$; 268 MB em 0,200 s com $b = 256$), a uma
taxa efetiva de só 0,8–2,1 GB/s, porque o mestre empacota e envia uma tarefa
de cada vez.

**Falhas.** 14 das 30 configurações MPI não terminaram em 2 min, em todas as
repetições: as com $q^2 > 95$ (algum trabalhador recebe uma segunda tarefa)
e todas com $n = 4096$. O mestre aborta em `MPI_Send` com erro de timeout do
transporte UD do UCX entre nós. As seções seguintes usam
`UCX_TLS=rc,sm,self` (conexões confiáveis), e todas as suas 275 execuções
terminaram, inclusive com $q^2$ maior que o número de trabalhadores e com
$n = 4096$.

# Escalabilidade forte

Carga de trabalho fixa e número de processos crescente (lei de Amdahl):
$n$ fixo, com $q^2 = 256$ tarefas ($n = 2048$, $b = 128$; e $n = 4096$,
$b = 256$ só na versão OpenBLAS), e $P$ de 2 a 96 processos (até 48 em um
nó), isto é, $W = P - 1$ de 1 a 95 trabalhadores. Como o mestre não calcula,
o tempo com um único processo calculando é $T(1)$, medido com $W = 1$
($P = 2$), e a eficiência é $E = T(1) / (W\,T(W))$.

![Escalabilidade forte. A linha vertical separa um nó de dois.](build/fig_forte.pdf)

```{.include file="build/tab_forte.md"}
```

**MPI.** Com um trabalhador, a versão MPI (23,8 s) já é 15 % mais rápida que
o serial (28,0 s), porque blocos de 128 cabem na cache L2. A eficiência fica
acima de 97 % até 16 trabalhadores e cai para 56 % com 95, quando o tempo
chega a 0,450 s (62× o serial). Parte da queda é granularidade: com
$\lceil 256/W \rceil$ rodadas de tarefas, a eficiência máxima é 91 % para
$W = 47$ e 90 % para $W = 95$; o restante é o custo crescente de o mestre
servir mais trabalhadores, já que o cálculo por tarefa ($\approx$ 93 ms) fica fixo
enquanto o mestre atende cada vez mais pedidos por unidade de tempo.

**MPI + OpenBLAS.** O melhor tempo é com **4 trabalhadores** (0,0896 s em
$n = 2048$, 312× o serial; 0,478 s em $n = 4096$); a partir daí o tempo
**aumenta** (0,294 s com 47). Um trabalhador faz a `sgemm` a 55 GFLOP/s e
termina uma tarefa em 1,2 ms, menos do que o mestre leva para empacotar e
enviar a seguinte; acima de 4 trabalhadores eles passam a esperar o mestre,
que é a parte sequencial da lei de Amdahl aqui. O tempo subir, em vez de só
estabilizar, e a variabilidade alta nesse trecho (CV até 18 %) indicam
disputa no nó do mestre: trabalhadores ociosos aguardam em `MPI_Recv` com
espera ativa, competindo com o mestre por memória e pelo transporte de
memória compartilhada. A seção 5 confirma: os mesmos processos divididos em
dois nós são até 1,9× mais rápidos.

# Escalabilidade fraca

Carga de trabalho e número de processos crescem juntos (lei de
Gustafson): o trabalho por trabalhador ($2n^3/W$ flops) fica constante com
$n = 1024\sqrt[3]{W}$, $W \in \{1, 8, 27, 64\}$, $b = 256$. Eficiência
$E = T(1)/T(W)$, igual ao speedup escalado $W\,T(1)/T(W)$ dividido por
$W$. A comunicação, porém, cresce com $n^2q$, e com $W = 27$ as 144
tarefas não se dividem igualmente.

![Escalabilidade fraca.](build/fig_fraca.pdf)

```{.include file="build/tab_fraca.md"}
```

**MPI.** O tempo fica quase constante (2,94 s → 3,25 s, eficiência de 85 % a
100 %), o comportamento previsto por Gustafson: com carga proporcional aos
recursos, o custo do mestre fica pequeno diante do cálculo. A queda em
$W = 27$ (85 %) é a divisão desigual de 144 tarefas, com teto de
$144/(27 \cdot 6) = 89$ %.

**MPI + OpenBLAS.** A eficiência cai para 5 % com 64 trabalhadores. O cálculo
por trabalhador é constante, mas o volume enviado pelo mestre ($8n^2q$) vai
de 34 MB a 2,1 GB, e o tempo acompanha esse volume a uma taxa de
3,0–3,3 GB/s a partir de $W = 8$. Como a `sgemm` torna o cálculo desprezível,
o que escala é só a comunicação centralizada.

# Processos, núcleos e nós

Esquerda e centro: 16, 32 e 48 processos em um nó ou divididos em dois.
Direita: os 96 núcleos como ranks × threads OpenMP da OpenBLAS.

![Mesmos processos em 1 ou 2 nós; divisão dos núcleos entre ranks e threads.](build/fig_recursos.pdf)

**Nós.** Na versão MPI, dividir os processos em dois nós não custa nada: o
tempo é igual ou até 7 % menor (0,914 s → 0,846 s com 32 processos), provavelmente porque a
rede não é o gargalo e cada nó tem mais memória e cache por processo. Na
versão OpenBLAS, com 16 processos um nó é melhor (0,154 s contra 0,200 s,
comunicação local mais barata), mas com 32 e 48 dois nós são 1,9× e 1,5× mais
rápidos, o que aponta para a disputa no nó do mestre descrita na seção 3.

**Ranks × threads.** Trocar ranks por threads da OpenBLAS muda pouco
(0,643–0,751 s; melhor 16 × 6), e nenhuma divisão dos 96 núcleos supera os
0,478 s de 5 processos com uma thread. Com $n = 4096$ o mestre envia 2,1 GB,
a 3,3 GB/s no melhor caso: mais núcleos por trabalhador só aceleram um
cálculo que já não é o gargalo.

# Conclusão

A versão MPI escala bem: eficiência forte acima de 80 % até 47
trabalhadores, eficiência fraca de 85 % a 100 % e até 62× o serial. Mas seu
núcleo de cálculo, compilado sem otimização, é lento o bastante para esconder
o custo da distribuição. A versão MPI + OpenBLAS é a mais rápida em termos
absolutos (até 438× o serial com bloco grande) e escala mal: o melhor tempo
de escalabilidade forte usa só 5 processos, e a eficiência fraca cai para
5 %. O gargalo comum é o mestre centralizado, que envia sozinho e em
sequência $8n^2q$ bytes e não calcula; com um núcleo de cálculo rápido, ele
se torna a parte sequencial que, pela lei de Amdahl, limita o speedup. Somam-se
a granularidade ($q^2$ tarefas), a disputa no nó do mestre e o custo de
conectar processos entre nós (5 s com 96). Melhorias: sobrepor empacotamento
e envio (`MPI_Isend`), distribuir cada bloco uma única vez (SUMMA ou Cannon,
sem mestre) e compilar com `-O3`.
