---
title: "Multiplicação de matrizes: serial, MPI e MPI + OpenBLAS"
author: "Álvaro Costa"
date: "28 de outubro de 2026"
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
  bloqueante) e recebe o bloco calculado. Com $P$ processos há $W = P-1$
  trabalhadores; o mestre não calcula;
* **MPI + OpenBLAS**: igual, com `cblas_sgemm` no lugar do laço triplo.

Dois limites decorrem do desenho: no máximo $\min(q^2, W)$ trabalhadores
ficam ocupados, e o mestre envia sozinho e em sequência $8n^2q$ bytes, pois
cada bloco de $A$ e $B$ é reenviado $q$ vezes.

**Ambiente.** Cluster ada (OpenPBS), até 2 nós, cada um com 2 AMD EPYC
7443 (48 núcleos, 2,8 GHz, 8 domínios NUMA); GCC 12.2 com `-O3`
(`CMAKE_BUILD_TYPE=Release`), Open MPI 5.0.6 sobre UCX
(`UCX_TLS=rc,sm,self`), OpenBLAS 0.3.21 (OpenMP). Um processo por núcleo e
`OMP_NUM_THREADS=1`, exceto onde indicado. Matrizes aleatórias com semente
fixa, iguais nas três versões. A versão MPI reproduz o resultado do serial
bit a bit; a OpenBLAS difere em até $5 \cdot 10^{-6}$ (relativo), efeito da
ordem das somas.

**Medição.** No serial, só a multiplicação (`clock_gettime`). Nas versões
MPI, o rank 0 mede com `MPI_Wtime` desde o envio da primeira tarefa até a
recepção do último bloco, incluindo empacotamento, comunicação e cálculo.
Ficam de fora `MPI_Init`, geração das matrizes, escrita do resultado e o
aquecimento das conexões UCX (uma mensagem vazia por trabalhador). Esse
aquecimento é desprezível em um nó (mediana 0,1 ms), mas entre nós leva
1,7 s com 48 processos e 5,1 s com 96, mais que a própria multiplicação.

**Repetições.** Cada configuração foi executada 5 vezes (job 1034793, 420
execuções, todas concluídas). Reportamos a **mediana**, o **coeficiente de
variação** (CV) nas tabelas e o intervalo mínimo–máximo como barras de erro
nas figuras.

**Referências.** Na seção 2, speedup $S = T_\text{serial}(n)/T$ e eficiência
$E = S/96$, os núcleos alocados. Esse $S$ soma o ganho do paralelismo e o do
núcleo de cálculo: com um trabalhador, o laço em blocos da versão MPI faz
12,6 GFLOP/s e a `sgemm` 55–75 GFLOP/s, contra 3,8 do serial. Por isso, nas
seções 3 e 4 cada versão é comparada a ela mesma com um trabalhador
($W = 1$, $P = 2$).

# Ordem da matriz e tamanho do bloco

48 processos em 2 nós (24 por nó). A versão OpenBLAS usa 2 threads por rank
e ocupa os 96 núcleos; a MPI usa metade deles. Serial: 0,0715 s, 0,563 s,
4,48 s, 15,3 s e 36,5 s para $n$ = 512 a 4096 (CV $\leq$ 0,5 %,
3,8 GFLOP/s): o tempo cresce com $n^3$.

```{.include file="build/tab_bench.md"}
```

![Melhor bloco de cada ordem: tempo e speedup sobre o serial.](build/fig_ordem.pdf)

![Tempo em função do bloco $b$.](build/fig_bloco.pdf)

**Ordem.** O speedup cresce com $n$ (MPI: 6,7 → 66; OpenBLAS: 3,6 → 79),
pois o cálculo cresce com $n^3$ e a comunicação com $n^2q$. Até $n = 1024$ a
versão MPI é a mais rápida; a `sgemm`, 4 vezes mais rápida por trabalhador,
só compensa a partir de $n = 2048$ (0,101 s contra 0,151 s; 0,462 s contra
0,550 s em $n = 4096$). O cálculo deixou de ser o gargalo.

**Bloco.** Blocos pequenos são sempre ruins: $b = 64$ em $n = 4096$ faz o
mestre enviar 8,6 GB (2,2–2,3 s nas duas versões, contra 0,46–0,55 s com
$b = 512$). Na versão MPI, o bloco ótimo é o maior que ainda ocupa os 47
trabalhadores: em $n = 2048$, $b = 512$ dá só 16 tarefas e perde para
$b = 256$. Na OpenBLAS, o maior bloco vence sempre, e o tempo acompanha o
volume enviado (1,07 GB em 0,462 s e 2,15 GB em 0,935 s com $n = 4096$:
2,3 GB/s). A versão OpenBLAS tem variabilidade alta (CV até 41 %), e $b = 128$
é mais lento que $b = 64$, sinal de threads disputando núcleos (seção 5).

# Escalabilidade forte

Carga de trabalho fixa e número de processos crescente (lei de Amdahl):
$n$ fixo, com $q^2 = 256$ tarefas ($n = 2048$, $b = 128$; e $n = 4096$,
$b = 256$ só na versão OpenBLAS), e $P$ de 2 a 96 (até 48 em um nó).
Eficiência $E = T(1) / (W\,T(W))$, com $T(1)$ medido com $W = 1$.

![Escalabilidade forte. A linha vertical separa um nó de dois.](build/fig_forte.pdf)

```{.include file="build/tab_forte.md"}
```

**A parte sequencial é o mestre.** Cada tarefa de $n = 2048$, $b = 128$
leva 2 MB do mestre ao trabalhador. Na versão MPI o cálculo de uma tarefa
leva 5,3 ms e o envio $\approx$ 0,57 ms, de modo que o mestre só alimenta
cerca de 9 trabalhadores. A eficiência fica em 100 % até $W = 4$ e cai a
58 % com 16, quando o tempo (0,146 s, 31× o serial) chega ao limite de
enviar os 537 MB a 3,7 GB/s; 256 tarefas dividem-se exatamente entre 16
trabalhadores, então a queda não é de granularidade. Pela lei de Amdahl, o
speedup sobre $T(1)$ fica limitado a $1{,}36/0{,}146 \approx 9$. Com a
`sgemm` (1,2 ms por tarefa) o limite chega antes: o melhor tempo é com
**4 trabalhadores** nos dois tamanhos (0,0740 s em $n = 2048$; 0,453 s em
$n = 4096$), e $E$ cai para 34 % e 48 % com 8.

**Mais processos pioram.** Além do ótimo, o tempo **aumenta**: em um nó, a
versão MPI passa de 0,146 s (17 processos) a 0,308 s (33) e 0,318 s (48).
Os trabalhadores ociosos esperam em `MPI_Recv` com espera ativa, e o mestre
disputa com eles a memória e o transporte de memória compartilhada do nó.
Com 65 processos em dois nós o tempo volta a cair (0,249 s); a seção 5
isola esse efeito.

# Escalabilidade fraca

Carga de trabalho e número de processos crescem juntos (lei de
Gustafson): o trabalho por trabalhador ($2n^3/W$ flops) fica constante com
$n = 1024\sqrt[3]{W}$, $W \in \{1, 8, 27, 64\}$, $b = 256$. Eficiência
$E = T(1)/T(W)$, igual ao speedup escalado $W\,T(1)/T(W)$ dividido por $W$.
A comunicação, porém, cresce com $n^2q$: de 34 MB a 2,15 GB.

![Escalabilidade fraca.](build/fig_fraca.pdf)

```{.include file="build/tab_fraca.md"}
```

A versão MPI mantém o tempo até $W = 8$ (95 %) e cai para 47 % e 25 %; a
OpenBLAS cai já em $W = 8$ (46 %) e chega a 5 %. Com $W \geq 27$ as duas
versões levam praticamente o mesmo tempo (0,30–0,34 s e 0,61–0,64 s),
qualquer que seja o núcleo de cálculo: o tempo é o do mestre enviando
$8n^2q$ bytes a 2,7–3,5 GB/s. O cenário de Gustafson pressupõe que a parte
sequencial não cresce com o problema; aqui ela cresce mais rápido que o
cálculo por trabalhador, que é constante.

# Processos, núcleos e nós

Esquerda e centro: 16, 32 e 48 processos ($n = 2048$, $b = 128$) em um nó ou
divididos em dois. Direita: os 96 núcleos dos dois nós como ranks × threads
OpenMP da OpenBLAS ($n = 4096$, $b = 256$).

![Mesmos processos em 1 ou 2 nós; divisão dos núcleos entre ranks e threads.](build/fig_recursos.pdf)

**Nós.** Com 16 processos um nó é melhor (MPI: 0,145 s contra 0,163 s;
OpenBLAS: 0,143 s contra 0,196 s), pois a comunicação local é mais barata.
Com 32 e 48, dois nós são 2,0× e 1,6× mais rápidos nas duas versões: o tempo
de um nó com 32 processos (0,29–0,30 s) é o dobro do de 16, o que confirma a
disputa no nó do mestre da seção 3. Mas dois nós custam 0,4–1,7 s de
aquecimento das conexões, fora da medição.

**Ranks × threads.** 96 ranks com 1 thread é o melhor arranjo (0,691 s), e
ainda é mais lento que 5 processos em um nó (0,453 s). Os arranjos com
threads são irregulares (16 × 6: 10,7 s; 4 × 24: 1,70 s) e muito
desbalanceados: em 16 × 6, 4 ranks fizeram 34–38 tarefas e 11 fizeram
9–11. Como os ranks não foram fixados a núcleos, threads de ranks
diferentes dividem núcleos e se atrasam; o resultado mede essa disputa, não o
paralelismo híbrido. O mesmo efeito explica a variabilidade com 2 threads
da seção 2.

# Conclusão

Com `-O3`, o núcleo de cálculo deixou de ser o gargalo das duas versões
paralelas. Ambas escalam bem só enquanto o mestre dá conta de alimentar os
trabalhadores: eficiência forte de 100 % até 4 trabalhadores, melhor tempo
com 5 (OpenBLAS) a 17 (MPI) processos e speedup máximo de 79× sobre o serial
com 96 núcleos. A partir daí, o mestre centralizado, que envia sozinho e em
sequência $8n^2q$ bytes e não calcula, é a parte sequencial de Amdahl e faz a
eficiência fraca cair para 25 % (MPI) e 5 % (OpenBLAS). Com um núcleo de
cálculo mais rápido, esse limite chega antes, e a OpenBLAS ganha pouco da
MPI. Somam-se a disputa no nó do mestre, que faz mais processos aumentarem o
tempo, a falta de afinidade entre threads e núcleos, e o custo de conectar
processos entre nós. Melhorias: sobrepor empacotamento e envio
(`MPI_Isend`), distribuir cada bloco uma única vez (SUMMA ou Cannon, sem
mestre), esperar sem ocupar a CPU e fixar ranks e threads a núcleos.
