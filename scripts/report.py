#!/usr/bin/env python3

"""Figures and tables for report/relatorio.md.

Reads the results.csv files of ada-bench.sh (jobs are pooled) and writes
vector figures and Markdown tables to the output directory. Every
statistic is over the completed repetitions of one case: the median is the
representative time, min/max and the coefficient of variation (CV) its spread.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("csv", nargs="+", help="results.csv files of ada-bench.sh.")
parser.add_argument("-o", "--output-dir", default="report/build")

PROGRAMS = ["serial", "mpi", "mpiblas"]
LABELS = {"serial": "serial", "mpi": "MPI", "mpiblas": "MPI + OpenBLAS"}
COLORS = {"serial": "#2a78d6", "mpi": "#eb6834", "mpiblas": "#1baf7a"}
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
KEYS = ["experiment", "program", "order", "bs", "procs", "nodes", "threads"]
CORES_PER_NODE = 48

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


# ---------------------------------------------------------------- data

def read(paths):
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def summarize(runs, keys):
    """One row per case: statistics of its completed repetitions, and how
    many repetitions were attempted and how they ended."""
    ok = runs[(runs["status"] == "ok") & (runs["elapsed_s"] > 0)]
    g = ok.groupby(keys)["elapsed_s"]
    s = g.agg(median="median", min="min", max="max", mean="mean", std="std",
              reps="count").reset_index()
    s["cv"] = s["std"] / s["mean"]
    w = ok.groupby(keys)["warmup_s"].median().rename("warmup")
    s = s.merge(w, on=keys, how="left")
    fail = runs[runs["status"] != "ok"].groupby(keys)["status"].first().rename("failed")
    s = s.merge(fail, on=keys, how="outer")
    s["gflops"] = 2 * s["order"].astype(float) ** 3 / s["median"] / 1e9
    return s


# ---------------------------------------------------------------- format

def num(x, digits=3):
    """Number with a decimal comma, `digits` significant digits."""
    if pd.isna(x):
        return "--"
    if x == 0:
        return "0"
    mag = int(np.floor(np.log10(abs(x))))
    decimals = max(digits - 1 - mag, 0)
    return f"{x:.{decimals}f}".replace(".", ",")


def pct(x):
    return "--" if pd.isna(x) else f"{x * 100:.1f}".replace(".", ",") + " %"


def md_table(df, cols, align=None, caption=None):
    """A pandoc pipe table. cols: {header: function(row) -> str}."""
    heads = list(cols)
    align = align or ["r"] * len(heads)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(heads) + " |",
             "| " + " | ".join(sep[a] for a in align) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(f(row) for f in cols.values()) + " |")
    if caption:
        lines += ["", f": {caption}"]
    return "\n".join(lines) + "\n"


def write(out_dir, name, text):
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")


# ---------------------------------------------------------------- plots

def style(ax, title, xlabel, ylabel, logx=True, logy=True, base=2):
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    if logx:
        ax.set_xscale("log", base=base)
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


def line(ax, d, x, color, label, y="median", spread=True):
    """Median as a line; min-max of the repetitions as error bars."""
    yerr = None
    if spread and "min" in d:
        yerr = [d[y] - d["min"], d["max"] - d[y]]
    ax.errorbar(d[x], d[y], yerr=yerr, fmt="-o", color=color, linewidth=2,
                markersize=4, capsize=3, elinewidth=1, label=label)


def new_figure(w=6.2, h=3.4):
    return plt.subplots(figsize=(w, h), layout="constrained")


def save(fig, out_dir, name):
    # 1-2-5 ticks crowd an axis that spans many decades
    for ax in fig.axes:
        lo, hi = ax.get_ylim()
        if ax.get_yscale() == "log" and hi / lo > 1000:
            ax.yaxis.set_major_locator(LogLocator(subs=(1,)))
    path = os.path.join(out_dir, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def cell(r, col="median"):
    """Median and CV in one cell, or why the case has no time."""
    if pd.isna(r.get(col)):
        return "travou" if pd.notna(r.get("failed")) else "--"
    return f"{num(r[col])} ({pct(r['cv'])})"


def pivot(s, index, programs):
    """One row per `index` value, the rows of each program side by side as
    {program: row} (missing programs map to an empty Series)."""
    rows = []
    for key, g in s.groupby(index, sort=True):
        by = {p: g[g["program_key"] == p].iloc[0] if (g["program_key"] == p).any()
              else pd.Series(dtype=object) for p in programs}
        rows.append((key if isinstance(key, tuple) else (key,), by))
    return rows


def pivot_table(rows, index_heads, groups, align_index="r"):
    """groups: [(program, [(header, function(row) -> str)])]."""
    heads = list(index_heads) + [f"{h}" for _, cols in groups for h, _ in cols]
    lines = ["| " + " | ".join(heads) + " |",
             "| " + " | ".join(["---:"] * len(heads)) + " |"]
    for key, by in rows:
        cells = [str(k) for k in key]
        for prog, cols in groups:
            r = by[prog]
            cells += ["--" if r.empty else f(r) for _, f in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- bench

def bench(runs, out, serial_ref):
    """The serial and blocks experiments."""
    s = summarize(runs, KEYS)
    s["speedup"] = s["order"].map(serial_ref) / s["median"]
    # over every core of the nodes, used or not
    s["eff"] = s["speedup"] / np.where(s["program"] == "serial", 1,
                                       s["nodes"] * CORES_PER_NODE)
    ok = s[s["median"].notna()]
    best = ok.loc[ok.groupby(["program", "order"])["median"].idxmin()]

    # time and speedup vs order, best block size of each program
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.4), layout="constrained")
    for prog in PROGRAMS:
        d = best[best["program"] == prog].sort_values("order")
        line(a1, d, "order", COLORS[prog], LABELS[prog])
        if prog != "serial":
            a2.plot(d["order"], d["speedup"], "-o", color=COLORS[prog], lw=2, ms=4)
    a2.axhline(96, color=INK_MUTED, lw=1, ls="--")
    a2.annotate("P = 96", (512, 96), xytext=(0, 3), textcoords="offset points",
                fontsize=8, color=INK_MUTED)
    style(a1, "Tempo (melhor bloco)", "ordem n", "tempo (s)")
    style(a2, "Speedup sobre o serial", "ordem n", "speedup")
    a1.legend(frameon=False, fontsize=7.5)
    save(fig, out, "fig_ordem.pdf")

    # time vs block size, one panel per program, same y range
    mpi = ok[ok["program"] != "serial"]
    orders = sorted(mpi["order"].unique())
    colors = dict(zip(orders, RAMP[1::max(1, len(RAMP) // len(orders))]))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), layout="constrained",
                             sharey=True)
    for ax, prog in zip(axes, ("mpi", "mpiblas")):
        for order in orders:
            d = mpi[(mpi["program"] == prog) & (mpi["order"] == order)].sort_values("bs")
            if not d.empty:
                line(ax, d, "bs", colors[order], f"n={order}")
        style(ax, LABELS[prog], "bloco b", "tempo (s)" if prog == "mpi" else "")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, out, "fig_bloco.pdf")

    # MPI cases that completed in at least one program, side by side
    m = s[s["program"] != "serial"].copy()
    done = m.groupby(["order", "bs"])["median"].transform(lambda t: t.notna().any())
    m = m[done].assign(program_key=m["program"])
    m["q2"] = (m["order"] // m["bs"]) ** 2
    cols = [("T (CV)", cell), ("S", lambda r: num(r["speedup"])),
            ("E", lambda r: pct(r["eff"]))]
    rows = pivot(m, ["order", "bs", "q2"], ["mpi", "mpiblas"])
    write(out, "tab_bench.md", pivot_table(
        rows, ["n", "b", "q²"],
        [("mpi", [(f"MPI: {h}", f) for h, f in cols]),
         ("mpiblas", [(f"BLAS: {h}", f) for h, f in cols])]))
    return s


# ---------------------------------------------------------------- scaling

def scaling(runs, out, serial_ref):
    """The strong, weak, nodes and hybrid experiments."""
    s = summarize(runs, KEYS)
    s["speedup"] = s["order"].map(serial_ref) / s["median"]
    s["workers"] = s["procs"] - 1

    # strong: speedup over the same program with 1 worker, per worker
    st = s[s["experiment"] == "strong"].copy()
    if not st.empty:
        one = st[st["procs"] == 2].set_index(["program", "order"])["median"]
        st["rel"] = [one.get((p, o), np.nan) / t for p, o, t in
                     zip(st["program"], st["order"], st["median"])]
        st["rel_eff"] = st["rel"] / st["workers"]
        st["program_key"] = st["program"] + "-" + st["order"].astype(str)
        cases = sorted(st["program_key"].unique(), key=lambda k: (k.startswith("mpiblas"), k))
        dash = {2048: "-", 4096: "--"}
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5), layout="constrained")
        for key in cases:
            d = st[(st["program_key"] == key) & st["median"].notna()].sort_values("procs")
            prog, order = d["program"].iloc[0], d["order"].iloc[0]
            label = f"{LABELS[prog]}, n={order}"
            line(a1, d, "procs", COLORS[prog], label)
            a1.lines[-1].set_linestyle(dash[order])
            a2.plot(d["procs"], d["rel_eff"] * 100, color=COLORS[prog], ls=dash[order],
                    lw=2, marker="o", ms=3.5)
        style(a1, "Tempo", "processos P", "tempo (s)")
        style(a2, "Eficiência T(1) / (W·T(W))", "processos P", "%", logy=False)
        a2.set_ylim(0, 110)
        for a in (a1, a2):
            a.axvline(48.5, color=GRID, lw=1)
        a1.legend(frameon=False, fontsize=7)
        save(fig, out, "fig_forte.pdf")
        cols = [("T (CV)", cell), ("E", lambda r: pct(r["rel_eff"]))]
        names = {"mpi-2048": "MPI 2048", "mpiblas-2048": "BLAS 2048", "mpiblas-4096": "BLAS 4096"}
        write(out, "tab_forte.md", pivot_table(
            pivot(st, ["procs", "nodes"], cases), ["P", "Nós"],
            [(k, [(f"{names.get(k, k)}: {h}", f) for h, f in cols]) for k in cases]))

    # weak: T(1 worker) / T(W workers)
    wk = s[s["experiment"] == "weak"].copy()
    if not wk.empty:
        one = wk[wk["procs"] == 2].set_index("program")["median"]
        wk["weak_eff"] = wk["program"].map(one) / wk["median"]
        wk["program_key"] = wk["program"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.3), layout="constrained")
        for prog in ("mpi", "mpiblas"):
            d = wk[(wk["program"] == prog) & wk["median"].notna()].sort_values("procs")
            line(a1, d, "workers", COLORS[prog], LABELS[prog])
            a2.plot(d["workers"], d["weak_eff"] * 100, "-o", color=COLORS[prog], lw=2, ms=4)
        a2.axhline(100, color=INK_MUTED, lw=1, ls=":")
        style(a1, "Tempo (ideal: constante)", "workers W", "tempo (s)")
        style(a2, "Eficiência T(1) / T(W)", "workers W", "%", logy=False)
        a2.set_ylim(0, 115)
        a1.legend(frameon=False, fontsize=7.5)
        save(fig, out, "fig_fraca.pdf")
        cols = [("T (CV)", cell), ("E", lambda r: pct(r["weak_eff"]))]
        write(out, "tab_fraca.md", pivot_table(
            pivot(wk, ["workers", "order"], ["mpi", "mpiblas"]), ["W", "n"],
            [("mpi", [(f"MPI: {h}", f) for h, f in cols]),
             ("mpiblas", [(f"BLAS: {h}", f) for h, f in cols])]))

    # resources: 1 vs 2 nodes for both programs, and ranks x threads
    nd = s[s["experiment"] == "nodes"]
    hy = s[s["experiment"] == "hybrid"].sort_values("procs", ascending=False)
    if nd.empty and hy.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.3), layout="constrained")
    for ax, prog in zip(axes, ("mpi", "mpiblas")):
        d = nd[nd["program"] == prog]
        procs = sorted(d["procs"].unique())
        x = np.arange(len(procs))
        for i, (nodes, color) in enumerate(((1, RAMP[1]), (2, RAMP[4]))):
            dd = d[d["nodes"] == nodes].set_index("procs").reindex(procs)
            ax.bar(x + (i - 0.5) * 0.38, dd["median"], 0.36, color=color,
                   yerr=[dd["median"] - dd["min"], dd["max"] - dd["median"]],
                   capsize=2, ecolor=INK_MUTED, label=f"{nodes} nó" + ("s" if nodes > 1 else ""))
        ax.set_xticks(x, [str(p) for p in procs])
        style(ax, f"{'MPI' if prog == 'mpi' else 'BLAS'}, n=2048", "processos P", "tempo (s)" if prog == "mpi" else "",
              logx=False, logy=False)
    axes[0].legend(frameon=False, fontsize=7)
    ax = axes[2]
    if not hy.empty:
        ax.bar([f"{p}×{t}" for p, t in zip(hy["procs"], hy["threads"])], hy["median"], 0.6,
               color=COLORS["mpiblas"], capsize=2, ecolor=INK_MUTED,
               yerr=[hy["median"] - hy["min"], hy["max"] - hy["median"]])
        ax.tick_params(axis="x", labelsize=7)
    style(ax, "BLAS, n=4096", "ranks × threads", "", logx=False, logy=False)
    save(fig, out, "fig_recursos.pdf")


def main():
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    runs = read(args.csv)
    serial = runs[(runs["program"] == "serial") & (runs["status"] == "ok")]
    serial_ref = serial.groupby("order")["elapsed_s"].median()
    exp = runs["experiment"]
    bench(runs[exp.isin(["serial", "blocks"])], args.output_dir, serial_ref)
    scaling(runs[~exp.isin(["serial", "blocks"])], args.output_dir, serial_ref)


if __name__ == "__main__":
    main()
