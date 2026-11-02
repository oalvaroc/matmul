#!/usr/bin/env python3

"""Figures and tables for report-new/relatorio.md.

Reads the results.csv of one ada-bench.sh job (and its logs/, for how many
tasks each rank computed) and writes vector figures and Markdown tables to
the output directory. Every statistic is over the 5 repetitions of one case:
the median is the representative time; min-max and the coefficient of
variation (CV = standard deviation / mean) are its spread.

References, also stated in the report:
  S_ser = T_serial(n) / T            speedup over the serial program
  S_1   = T(W = 1) / T(W)            speedup over the same program, 1 worker
  E     = S / resources              resources: cores (P x threads) for S_ser,
                                     workers W for S_1
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("job", help="results/<job> directory of ada-bench.sh.")
parser.add_argument("-o", "--output-dir", default="build")

LABELS = {"serial": "serial", "mpi": "MPI", "mpiblas": "MPI + OpenBLAS"}
SHORT = {"serial": "Serial", "mpi": "MPI", "mpiblas": "BLAS"}
COLORS = {"serial": "#2a78d6", "mpi": "#eb6834", "mpiblas": "#1baf7a"}
ORDER_COLORS = {512: "#86b6ef", 1024: "#5598e7", 2048: "#1c5cab", 4096: "#0d366b"}
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
KEYS = ["experiment", "program", "order", "bs", "procs", "nodes", "threads"]

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


# ---------------------------------------------------------------- data

def summarize(runs):
    ok = runs[runs["status"] == "ok"]
    g = ok.groupby(KEYS)["elapsed_s"]
    s = g.agg(median="median", min="min", max="max", mean="mean", std="std",
              reps="count").reset_index()
    s["cv"] = s["std"] / s["mean"]
    s = s.merge(ok.groupby(KEYS)["warmup_s"].median().rename("warmup").reset_index(),
                on=KEYS, how="left")
    s["gflops"] = 2 * s["order"].astype(float) ** 3 / s["median"] / 1e9
    s["cores"] = s["procs"] * s["threads"]
    s["workers"] = np.where(s["program"] == "serial", 1, s["procs"] - 1)
    q = np.where(s["bs"] > 0, s["order"] // s["bs"].replace(0, 1), 1)
    s["q"] = q
    s["tasks"] = q ** 2
    # bytes the master sends (A and B blocks, each q times) and receives (C)
    s["volume"] = np.where(s["program"] == "serial", np.nan,
                           (8.0 * q + 4.0) * s["order"].astype(float) ** 2)
    return s


def task_counts(job):
    """Tasks per rank of every run, from the 'rank w computed k of t blocks'
    lines: {(case name, rep): [k for each worker]}."""
    line = re.compile(r"rank \d+ computed (\d+) of \d+ blocks")
    counts = {}
    for path in glob.glob(os.path.join(job, "logs", "*.log")):
        name, rep = os.path.basename(path)[:-4].rsplit("-r", 1)
        with open(path) as f:
            ks = [int(m.group(1)) for m in line.finditer(f.read())]
        if ks:
            counts[(name, int(rep))] = ks
    return counts


# ---------------------------------------------------------------- format

def num(x, digits=3):
    """Number with a decimal comma and `digits` significant digits."""
    if pd.isna(x):
        return "--"
    if x == 0:
        return "0"
    mag = int(np.floor(np.log10(abs(x))))
    decimals = max(digits - 1 - mag, 0)
    return f"{x:.{decimals}f}".replace(".", ",")


def pct(x, decimals=0):
    return "--" if pd.isna(x) else f"{x * 100:.{decimals}f}".replace(".", ",") + " %"


def sci(x):
    if x == 0:
        return "0"
    mant, exp = f"{x:.1e}".split("e")
    return f"${mant.replace('.', '{,}')} \\cdot 10^{{{int(exp)}}}$"


def size(b):
    for unit, f in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= f:
            return num(b / f) + " " + unit
    return f"{b:.0f} B"


def table(rows, heads, align=None):
    align = align or "r" * len(heads)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(heads) + " |",
             "| " + " | ".join(sep[a] for a in align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def write(out, name, text):
    path = os.path.join(out, name)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")


# ---------------------------------------------------------------- plots

def style(ax, title, xlabel, ylabel, logx=True, logy=True):
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    if logx:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}".replace(".", ",")))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, which="both")


def errorline(ax, d, x, color, label=None, y="median", ls="-", marker="o"):
    """Median as a line, min-max of the repetitions as error bars."""
    ax.errorbar(d[x], d[y], yerr=[d[y] - d["min"], d["max"] - d[y]], fmt=marker,
                ls=ls, color=color, linewidth=1.8, markersize=3.5, capsize=2.5,
                elinewidth=0.9, label=label)


def bars(ax, labels, d, color, offset=0.0, width=0.6, label=None):
    x = np.arange(len(labels)) + offset
    ax.bar(x, d["median"], width, color=color, label=label,
           yerr=[d["median"] - d["min"], d["max"] - d["median"]],
           capsize=2, ecolor=INK_MUTED, error_kw={"elinewidth": 0.9})
    return x


def save(fig, out, name):
    for ax in fig.axes:
        lo, hi = ax.get_ylim()
        if ax.get_yscale() == "log" and hi / lo > 1000:
            ax.yaxis.set_major_locator(LogLocator(subs=(1,)))
    path = os.path.join(out, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def t_cv(r):
    return f"{num(r['median'])} ({pct(r['cv'], 1)})"


# ---------------------------------------------------------------- sections

def blocks(s, out, tser):
    bl = s[s["experiment"] == "blocks"].copy()
    bl["s_ser"] = bl["order"].map(tser) / bl["median"]
    bl["e_ser"] = bl["s_ser"] / bl["cores"]
    best = bl.loc[bl.groupby(["program", "order"])["median"].idxmin()]

    rows = []
    for (order, bs), g in bl.groupby(["order", "bs"]):
        m, b = (g[g["program"] == p].iloc[0] for p in ("mpi", "mpiblas"))
        bold = {p: best[(best["program"] == p) & (best["order"] == order)]["bs"].iloc[0] == bs
                for p in ("mpi", "mpiblas")}
        cell = lambda r, p: (f"**{num(r['median'])}**" if bold[p] else num(r["median"])) \
            + f" ({pct(r['cv'], 1)})"
        rows.append([str(order), str(bs), str(m["tasks"]), size(m["volume"]),
                     cell(m, "mpi"), num(m["s_ser"]), pct(m["e_ser"]),
                     cell(b, "mpiblas"), num(b["s_ser"]), pct(b["e_ser"])])
    write(out, "tab_blocos.md", table(
        rows, ["n", "b", "tarefas", "volume", "MPI: T (CV)", "$S_\\text{ser}$",
               "$E_\\text{ser}$", "BLAS: T (CV)", "$S_\\text{ser}$", "$E_\\text{ser}$"]))

    # time against the block size, one panel per program, same y range
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), layout="constrained", sharey=True)
    for ax, prog in zip(axes, ("mpi", "mpiblas")):
        for order, color in ORDER_COLORS.items():
            d = bl[(bl["program"] == prog) & (bl["order"] == order)].sort_values("bs")
            errorline(ax, d, "bs", color, f"n = {order}")
        style(ax, LABELS[prog], "bloco b", "tempo (s)" if prog == "mpi" else "")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right", ncols=2)
    save(fig, out, "fig_bloco.pdf")
    return bl


def master_bound(s, out):
    """Effective master throughput V / T of the runs with 15+ workers."""
    # hybrid runs with many threads are slowed down by something else (sec. 6)
    m = s[s["experiment"].isin(["blocks", "strong", "weak", "nodes"]) & (s["workers"] >= 15)]
    bw = m["volume"] / m["median"]
    return bw


def strong(s, out, tser, counts):
    st = s[s["experiment"] == "strong"].copy()
    st["case"] = st["program"] + "-" + st["order"].astype(str)
    one = st[st["procs"] == 2].set_index("case")["median"]
    st["s_ser"] = st["order"].map(tser) / st["median"]
    st["s_1"] = st["case"].map(one) / st["median"]
    st["e_1"] = st["s_1"] / st["workers"]
    st["e_ser"] = st["s_ser"] / st["cores"]
    cases = ["mpi-2048", "mpiblas-2048", "mpiblas-4096"]
    dash = {"mpi-2048": "-", "mpiblas-2048": "-", "mpiblas-4096": "--"}
    name = {"mpi-2048": "MPI, n = 2048", "mpiblas-2048": "BLAS, n = 2048",
            "mpiblas-4096": "BLAS, n = 4096"}

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(6.6, 2.4), layout="constrained")
    for c in cases:
        d = st[st["case"] == c].sort_values("procs")
        color = COLORS[d["program"].iloc[0]]
        errorline(a1, d, "procs", color, name[c], ls=dash[c])
        a2.plot(d["workers"], d["s_1"], color=color, ls=dash[c], lw=1.8, marker="o", ms=3)
        a3.plot(d["workers"], d["e_1"] * 100, color=color, ls=dash[c], lw=1.8, marker="o", ms=3)
    w = np.array([1, 95])
    a2.plot(w, w, ":", color=INK_MUTED, lw=0.9)
    a2.annotate("ideal", (12, 12), fontsize=7, color=INK_MUTED, rotation=38,
                ha="right", va="bottom")
    style(a1, "Tempo", "processos P", "tempo (s)")
    style(a2, "Speedup S₁ = T(1) / T(W)", "trabalhadores W", "", logy=True)
    style(a3, "Eficiência S₁ / W", "trabalhadores W", "%", logy=False)
    a2.set_ylim(0.8, 100)
    a3.set_ylim(0, 110)
    a1.axvline(48.5, color=GRID, lw=1)
    for a in (a2, a3):
        a.axvline(47.5, color=GRID, lw=1)
    a3.legend(*a1.get_legend_handles_labels(), frameon=False, fontsize=6.5, loc="lower left")
    save(fig, out, "fig_forte.pdf")

    rows = []
    for (procs, nodes), g in st.groupby(["procs", "nodes"]):
        row = [str(procs), str(nodes)]
        for c in cases:
            r = g[g["case"] == c].iloc[0]
            row += [t_cv(r), num(r["s_ser"]), pct(r["e_1"])]
        rows.append(row)
    heads = ["P", "Nós"]
    for c in cases:
        short = {"mpi-2048": "MPI 2048", "mpiblas-2048": "BLAS 2048",
                 "mpiblas-4096": "BLAS 4096"}[c]
        heads += [f"{short}: T (CV)", "$S_\\text{ser}$", "$E_1$"]
    write(out, "tab_forte.md", table(rows, heads))

    # load balance: largest over smallest number of tasks of a busy worker
    imb = []
    for _, r in st.iterrows():
        n = f"strong-{r['program']}-n{r['order']}-b{r['bs']}-p{r['procs']}-N{r['nodes']}-t{r['threads']}"
        ks = counts.get((n, 1))
        if ks:
            imb.append((r["case"], r["procs"], max(ks), min(ks)))
    return st, pd.DataFrame(imb, columns=["case", "procs", "kmax", "kmin"])


def weak(s, out):
    wk = s[s["experiment"] == "weak"].copy()
    one = wk[wk["procs"] == 2].set_index("program")["median"]
    wk["e"] = wk["program"].map(one) / wk["median"]
    wk["work"] = 2 * wk["order"].astype(float) ** 3 / wk["workers"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.3), layout="constrained")
    for prog in ("mpi", "mpiblas"):
        d = wk[wk["program"] == prog].sort_values("workers")
        errorline(a1, d, "workers", COLORS[prog], LABELS[prog])
        a2.plot(d["workers"], d["e"] * 100, "-o", color=COLORS[prog], lw=1.8, ms=3.5)
    a2.axhline(100, color=INK_MUTED, lw=0.9, ls=":")
    style(a1, "Tempo (ideal: constante)", "trabalhadores W", "tempo (s)")
    style(a2, "Eficiência fraca T(1) / T(W)", "trabalhadores W", "%", logy=False)
    a2.set_ylim(0, 115)
    a1.legend(frameon=False, fontsize=7)
    save(fig, out, "fig_fraca.pdf")

    rows = []
    for (w, order), g in wk.groupby(["workers", "order"]):
        m, b = (g[g["program"] == p].iloc[0] for p in ("mpi", "mpiblas"))
        rows.append([str(w), str(order), str(m["procs"]), str(m["nodes"]), size(m["volume"]),
                     t_cv(m), pct(m["e"]), t_cv(b), pct(b["e"])])
    write(out, "tab_fraca.md", table(
        rows, ["W", "n", "P", "Nós", "volume",
               "MPI: T (CV)", "$E_W$", "BLAS: T (CV)", "$E_W$"]))
    return wk


def resources(s, out, counts):
    nd = s[s["experiment"] == "nodes"]
    hy = s[s["experiment"] == "hybrid"].sort_values("procs", ascending=False).copy()

    rows = []
    for p in sorted(nd["procs"].unique()):
        row = [str(p)]
        for prog in ("mpi", "mpiblas"):
            for nodes in (1, 2):
                r = nd[(nd["program"] == prog) & (nd["procs"] == p) & (nd["nodes"] == nodes)].iloc[0]
                row.append(t_cv(r))
            r2 = nd[(nd["program"] == prog) & (nd["procs"] == p) & (nd["nodes"] == 2)].iloc[0]
            row.append(num(r2["warmup"]))
        rows.append(row)
    write(out, "tab_nos.md", table(
        rows, ["P", "MPI: 1 nó", "MPI: 2 nós", "conexão 2 nós",
               "BLAS: 1 nó", "BLAS: 2 nós", "conexão 2 nós"]))

    rows = []
    for _, r in hy.iterrows():
        n = f"hybrid-mpiblas-n{r['order']}-b{r['bs']}-p{r['procs']}-N{r['nodes']}-t{r['threads']}"
        ks = [counts[(n, rep)] for rep in range(1, 6) if (n, rep) in counts]
        spread = [max(k) / (sum(k) / len(k)) for k in ks]
        rows.append([f"{r['procs']} × {r['threads']}", t_cv(r), num(r["gflops"]),
                     num(np.median(spread))])
    write(out, "tab_hibrido.md", table(
        rows, ["processos × threads", "T (CV)", "GFLOP/s", "máx/média tarefas"]))


def validation(runs, s, out):
    """The serial reference and the check of the mpi and mpiblas runs against
    it (-c, first repetition of each case), one row per order."""
    ser = s[s["experiment"] == "serial"].set_index("order")
    v = runs[runs["check"].notna()]
    rows = []
    for order, r in ser.iterrows():
        row = [str(order), f"{num(r['median'])} ({pct(r['cv'], 1)})", num(r["gflops"]),
               sci(10 * np.sqrt(order) * 2.0 ** -23)]
        for prog in ("mpi", "mpiblas"):
            g = v[(v["program"] == prog) & (v["order"] == order)]
            err = "0" if (g["max_rel_err"] == 0).all() else sci(g["max_rel_err"].max())
            row.append(f"{err} ({(g['check'] == 'ok').sum()}/{len(g)})")
        rows.append(row)
    write(out, "tab_validacao.md", table(
        rows, ["n", "serial: T (CV)", "GFLOP/s", "tolerância", "MPI: erro (ok)",
               "BLAS: erro (ok)"]))
    print(f"validation: {(v['check'] == 'ok').sum()}/{len(v)} ok")


def main():
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    runs = pd.read_csv(os.path.join(args.job, "results.csv"))
    print(f"{len(runs)} runs, {(runs['status'] == 'ok').sum()} ok")
    s = summarize(runs)
    tser = s[s["program"] == "serial"].set_index("order")["median"]
    counts = task_counts(args.job)

    bl = blocks(s, args.output_dir, tser)
    bw = master_bound(s, args.output_dir)
    st, imb = strong(s, args.output_dir, tser, counts)
    wk = weak(s, args.output_dir)
    resources(s, args.output_dir, counts)
    validation(runs, s, args.output_dir)

    # numbers quoted in the text
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
    cols = ["experiment", "program", "order", "bs", "procs", "nodes", "threads",
            "median", "cv", "gflops", "warmup"]
    print(bl[cols + ["s_ser", "e_ser", "volume"]].to_string())
    print(st[cols + ["s_ser", "s_1", "e_1", "e_ser"]].to_string())
    print(wk[cols + ["e", "volume"]].to_string())
    print(imb.to_string())
    print("master throughput (GB/s):", (bw / 1e9).describe().to_string())
    print("CV:", s.groupby("experiment")["cv"].describe().to_string())


if __name__ == "__main__":
    main()
