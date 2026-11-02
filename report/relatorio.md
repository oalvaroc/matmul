---
title: "Multiplicação de matrizes distribuída: serial, MPI e MPI + OpenBLAS"
author: |
  Álvaro Costa - RA 246909\
  Instituto de Computação\
  UNICAMP
lang: pt-BR
documentclass: article
fontsize: 10pt
geometry: margin=2cm
numbersections: true
colorlinks: true
header-includes: |
  \usepackage{etoolbox}
  \AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{3.5pt}}
  \usepackage{float}
  \floatplacement{figure}{H}
  \setlength{\parskip}{0.4em}
  \setlength{\abovecaptionskip}{2pt}
---

# Versões e configuração

As três versões calculam $C = A \times B$ em precisão simples (`float`), com
$A$ e $B$ quadradas de ordem $n$, geradas com a mesma semente (42).

* **Serial**: laço triplo na ordem linha–$k$–coluna.
* **MPI**: mestre–trabalhadores. $C$ é dividida em $q^2$ blocos $b \times b$
  ($q = n/b$); cada bloco é uma tarefa. O mestre (rank 0) envia a cada
  trabalhador livre a linha de blocos de $A$ e a coluna de blocos de $B$
  necessárias e recebe o bloco de $C$ calculado. Com $P$ processos há
  $W = P - 1$ trabalhadores; o mestre não calcula.
* **MPI + OpenBLAS**: igual à MPI, com `cblas_sgemm` no trabalhador.

![Esquema mestre–trabalhadores. Para calcular o bloco $C_{ij}$, o mestre
empacota a linha $i$ de blocos de $A$ e a coluna $j$ de blocos de $B$ em uma
única mensagem e a envia a um trabalhador livre, que devolve $C_{ij}$.](build/fig_esquema.pdf)

Como cada bloco de $A$ e $B$ é enviado $q$ vezes, o mestre movimenta sozinho
$V = 8n^2q + 4n^2$ bytes. Esse volume cresce quando $b$ diminui, e é o que
limita o desempenho, como mostram as seções seguintes.

**Ambiente.** Cluster Lovelace, até 2 nós com 48 núcleos cada (2 AMD EPYC
7443). GCC 12.2 `-O3`, Open MPI 5.0.6 (UCX), OpenBLAS 0.3.21. Processos sem
fixação a núcleos; 1 thread por processo, exceto onde indicado.

**O que é medido.** O tempo da multiplicação: no serial, `mat_mult`; nas
versões MPI, a função distribuída inteira (envio das tarefas, cálculo,
recepção de $C$), medida no rank 0 com `MPI_Wtime`. Ficam de fora a geração
das matrizes, a verificação (seção 2) e o estabelecimento das conexões MPI,
feito antes do cronômetro (até 5 s com dois nós).

**Repetições.** Cada uma das 84 configurações rodou 5 vezes (420 execuções,
todas concluídas). A medida representativa é a mediana; a variabilidade
é dada pelo coeficiente de variação (CV = desvio-padrão / média) nas
tabelas e pelo intervalo mínimo–máximo na figura. O CV fica em geral abaixo
de 5 %; os maiores (até 17 %) ocorrem com muitos processos ou threads por nó.

**Speedup e eficiência.** Usamos duas referências:

* $S_\text{ser} = T_\text{serial}(n) / T$: ganho sobre o serial com o mesmo
  $n$, e $E_\text{ser} = S_\text{ser} / c$, com $c$ os núcleos ocupados
  (processos × threads, mestre incluído). Inclui o ganho do núcleo de
  cálculo: um trabalhador sozinho faz 12,6 GFLOP/s (MPI) e 55–75 GFLOP/s
  (OpenBLAS), contra 3,8 GFLOP/s do serial; por isso $E_\text{ser}$ pode
  passar de 100 %.
* $S_1 = T(W{=}1) / T(W)$ e $E_1 = S_1 / W$: ganho sobre a mesma versão com
  um trabalhador. Isola o efeito do paralelismo.

# Validação dos resultados

Com a opção `-c`, as versões MPI verificam o próprio resultado depois que o
cronômetro para: o rank 0 calcula o produto com a função da versão serial (a
referência) e mede o maior erro relativo entre os elementos,
$e = \max_{ij} |c_{ij} - r_{ij}| / |r_{ij}|$. O programa imprime se passou, o
erro e o tempo da verificação, que não entra no tempo medido. Como a
verificação custa uma multiplicação serial, ela é feita na 1ª repetição de
cada configuração.

**Tolerância:** $e \leq 10\sqrt{n}\,\varepsilon$, com
$\varepsilon = 2^{-23}$ a precisão do `float`. Cada elemento de $C$ é uma
soma de $n$ produtos, e cada adição em `float` arredonda o resultado, com um
erro da ordem de $\varepsilon$. Se todos os arredondamentos fossem para o
mesmo lado, o erro acumulado seria $n\varepsilon$, mas esse é um pior caso
raro: na prática eles vão para cima ou para baixo quase ao acaso e se
cancelam em parte. É o comportamento de um passeio aleatório: depois de $n$
passos de tamanho $\varepsilon$ em sentidos aleatórios, a distância típica
do ponto de partida é $\sqrt{n}\,\varepsilon$, e não $n\varepsilon$. Essa é
a estimativa usual do erro de somas longas em ponto flutuante; o fator 10 é
margem de segurança. Um erro de programação (um bloco
fora do lugar, um produto esquecido) gera erros de $10^{-4}$ ou mais, bem
acima da tolerância.

```{.include file="build/tab_validacao.md"}
Versão serial (referência) e verificação. "Erro" é o maior $e$ entre as
configurações com essa ordem; entre parênteses, aprovadas/verificadas.
```

Todas as 79 verificações passaram. A MPI reproduz o serial bit a bit
($e = 0$), pois soma na mesma ordem. A OpenBLAS soma em outra ordem e erra
no máximo $5{,}1 \cdot 10^{-6}$, 15 vezes abaixo da tolerância; seu erro
cresce cerca de 1,4× a cada duplicação de $n$, como o modelo prevê
($\sqrt{2}$).

# Ordem da matriz e tamanho do bloco

O serial cresce com $n^3$ (3,8 GFLOP/s constantes). A tabela 2 compara as
versões paralelas com 48 processos em 2 nós (a OpenBLAS com 2 threads por
processo) para cada combinação de $n$ e $b$.

```{.include file="build/tab_blocos.md"}
48 processos, 2 nós. "Volume" é $V$, o que passa pelo mestre.
$E_\text{ser}$ divide por 48 núcleos (MPI) ou 96 (BLAS, 2 threads). Em
negrito, o melhor bloco de cada $n$.
```

O speedup cresce com $n$: de 7 para 66 (MPI) e de 3 para 76 (OpenBLAS);
a eficiência por núcleo, de 14 % para 137 % e de 4 % para 80 %.
O cálculo cresce com $n^3$, e a comunicação mais devagar; com $n = 512$ o
tempo (10–20 ms) é quase todo distribuição. A OpenBLAS só vence a partir de
$n = 2048$, e por pouco: com 47 trabalhadores o cálculo já não é o gargalo.

![Tempo em função do bloco $b$ para cada ordem $n$ (48 processos, 2 nós).
Barras: mínimo e máximo das 5 repetições.](build/fig_bloco.pdf)

O bloco equilibra dois efeitos. Blocos pequenos multiplicam o volume:
com $n = 4096$ e $b = 64$ o mestre movimenta 8,7 GB e as duas versões levam
os mesmos 2,2 s, qualquer que seja o núcleo de cálculo. Blocos grandes
deixam poucas tarefas: com $n = 2048$, $b = 512$ dá só 16 tarefas para 47
trabalhadores. O melhor bloco é o maior que ainda ocupa os
trabalhadores. Na OpenBLAS, $b = 128$ é sempre pior que $b = 64$,
provavelmente pelas threads sem fixação a núcleos (seção 6).

# Escalabilidade forte

Problema fixo ($n = 2048$, $b = 128$; e $n = 4096$, $b = 256$ só na
OpenBLAS), com $P$ de 2 a 96. Acima de 48 processos, dois nós.

![Escalabilidade forte. Linha vertical: passagem para dois nós. $S_1$ e
$E_1$ são relativos à mesma versão com um trabalhador. Barras: mínimo e
máximo das 5 repetições.](build/fig_forte.pdf)

```{.include file="build/tab_forte.md"}
Tempo mediano em segundos (CV), speedup sobre o serial e eficiência sobre um
trabalhador ($E_1 = S_1/W$).
```

Até 4 trabalhadores a eficiência se mantém próxima de 100 %. Depois, o mestre
limita. O melhor tempo da MPI é com 17 processos ($S_\text{ser} = 30$); o da
OpenBLAS, com apenas 5 ($S_\text{ser} = 61$ e 82). Quanto mais rápido o
cálculo, mais cedo o mestre não consegue alimentar os trabalhadores: uma tarefa
leva 5,3 ms na MPI e 1,2 ms na OpenBLAS, e o mestre leva até 1 ms para
enviá-la.

Além do ótimo, mais processos pioram o tempo: a MPI vai de 0,150 s (17
processos) a 0,324 s (48); a OpenBLAS com 33 a 96 processos é tão lenta
quanto com um trabalhador. A vazão do mestre cai de 7,5 GB/s para cerca de
2 GB/s quando 33 a 48 processos dividem seu nó: os trabalhadores ociosos
esperam ativamente e disputam núcleos e memória com ele. Não é
desbalanceamento: com $W = 32$ cada trabalhador fez exatamente 8 tarefas.

# Escalabilidade fraca

O problema cresce com os trabalhadores: $n = 1024\sqrt[3]{W}$, de modo que
cada trabalhador faz sempre 2,15 GFLOP ($b = 256$).

![Escalabilidade fraca. Ideal: tempo constante e eficiência de 100 %.
Barras: mínimo e máximo das 5 repetições.](build/fig_fraca.pdf)

```{.include file="build/tab_fraca.md"}
Tempo mediano (CV) e eficiência $E_W = T(W{=}1)/T(W)$; ideal: 100 %.
```

A eficiência cai para 25 % (MPI) e 5 % (OpenBLAS) com 64 trabalhadores. O
cálculo por trabalhador é constante, mas o volume no mestre cresce com
$W$ (de 38 MB a 2,2 GB). Com $W \geq 27$ as duas versões levam o mesmo
tempo: o do mestre enviando a 3–3,7 GB/s. A OpenBLAS cai antes porque seu
cálculo é 5 vezes mais rápido.

# Processos, núcleos e nós

```{.include file="build/tab_nos.md"}
Mesmos processos em 1 ou 2 nós ($n = 2048$, $b = 128$): tempo mediano (CV) e
tempo de conexão com 2 nós (s, fora da medição).
```

**Nós.** Com 16 processos, um nó é melhor (comunicação local é mais barata).
Com 32 e 48, dois nós são 1,6× a 2× mais rápidos, porque o nó do mestre fica
menos cheio. O custo é a conexão entre nós, fora da medição: 0,45 s a 1,8 s.

```{.include file="build/tab_hibrido.md"}
Os 96 núcleos de dois nós como processos × threads (OpenBLAS, $n = 4096$,
$b = 256$). "Máx/média tarefas": 1 = carga equilibrada.
```

**Processos × threads.** 96 processos com 1 thread é o melhor arranjo
(0,694 s), mas ainda mais lento que 5 processos em um nó (0,448 s). Os
arranjos híbridos são até 15 vezes mais lentos e desbalanceados: sem
fixação a núcleos, threads de processos diferentes disputam os mesmos
núcleos.

# Gargalos e conclusão

As duas versões paralelas estão corretas (79 de 79 verificações). O
maior speedup sobre o serial foi 82×, com a OpenBLAS e só 5 processos
($n = 4096$: 0,448 s contra 36,5 s). Mas o ganho vem mais do núcleo de
cálculo (3,3× do laço em blocos, 14–20× da `sgemm`) do que da escala.

Os gargalos, em ordem de impacto:

1. **Mestre centralizado.** Um só processo envia $V = 8n^2q$ bytes, em
   sequência e sem calcular. É a parte sequencial (lei de Amdahl) e cresce
   com o problema, o que derruba também a escalabilidade fraca.
2. **Disputa no nó do mestre.** Processos ociosos no mesmo nó reduzem a
   vazão do mestre de 7,5 para cerca de 2 GB/s.
3. **Granularidade.** Blocos pequenos multiplicam o volume; blocos grandes
   deixam trabalhadores sem tarefa.
4. **Sem fixação a núcleos.** Threads e processos disputam núcleos,
   causando lentidão e variabilidade.

**Melhorias sugeridas:** (i) trocar o mestre central por um algoritmo em
que os processos trocam blocos entre si (SUMMA ou Cannon); (ii) sobrepor
envio e cálculo com `MPI_Isend` e fazer o mestre calcular também; (iii)
fixar processos e threads a núcleos (`--bind-to core`, `OMP_PROC_BIND`);
(iv) usar o maior bloco $b$ que ainda dê ao menos uma tarefa por
trabalhador.

# Uso de Inteligência Artificial

Este relatório foi elaborado com o apoio do assistente Claude Code (modelo Claude Opus 5.5, Anthropic). A ferramenta foi usada para:

* criar o modelo do relatório;
* revisar e formatar o texto escrito;
* formatar tabelas a partir dos dados obtidos;
* criar script de geração dos gráficos;

O planejamento e a execução dos experimentos, assim como a coleta dos dados, foram feitos pelo autor. Todo o conteúdo gerado foi revisado pelo autor, que é responsável pelo texto final.
