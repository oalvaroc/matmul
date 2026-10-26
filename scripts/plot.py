#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


parser = argparse.ArgumentParser(description="Plot matmul results from CSV file.")
parser.add_argument("csv", type=argparse.FileType(), help="CSV file with results.")
parser.add_argument("-o", "--output-dir", default=".",
                    help="Directory for the plot images (default: current directory).")
parser.add_argument("--min-order", type=int, default=64,
                    help="Smallest order shown in the block size panels (default: 64).")

PROGRAMS = ["serial", "mpi", "mpiblas"]
LABELS = {"serial": "serial", "mpi": "mpi", "mpiblas": "mpi + OpenBLAS"}
# one fixed color per program, in categorical slot order
COLORS = {"serial": "#2a78d6", "mpi": "#eb6834", "mpiblas": "#1baf7a"}
# ordinal blue ramp for matrix orders, light (small) to dark (large)
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def load(csv):
    """All runs, and the median time of the completed repetitions per case."""
    runs = pd.read_csv(csv)
    ok = runs[(runs["status"] == "ok") & (runs["elapsed_s"] > 0)]
    ok = ok.groupby(["program", "order", "bs", "procs"], as_index=False)["elapsed_s"].median()
    return runs, ok


def best_per_order(ok):
    """Fastest run of each program at each order, with its speedup over serial."""
    best = ok.loc[ok.groupby(["program", "order"])["elapsed_s"].idxmin()]
    serial = ok[ok["program"] == "serial"].set_index("order")["elapsed_s"]
    best = best.assign(speedup=best["order"].map(serial) / best["elapsed_s"])
    return best.sort_values(["program", "order"])


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, which="both")


def label_end(ax, x, y, text, color):
    """Direct label just right of a line's last point."""
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
    ax.plot(x, y, "o", color=color, markersize=5)


def plot_time_vs_order(ax, best):
    for prog in PROGRAMS:
        d = best[best["program"] == prog]
        if d.empty:
            continue
        ax.plot(d["order"], d["elapsed_s"], "-o", color=COLORS[prog],
                linewidth=2, markersize=5, label=LABELS[prog])
        label_end(ax, d["order"].iloc[-1], d["elapsed_s"].iloc[-1],
                  LABELS[prog], COLORS[prog])
    style(ax, "Time vs matrix order (MPI: best block size)",
          "matrix order n", "elapsed time (s)")
    ax.legend(frameon=False, loc="upper left")


def plot_speedup(ax, best):
    ax.axhline(1, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.annotate("serial", (1, 1), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points", ha="right",
                fontsize=9, color=INK_MUTED)
    for prog in PROGRAMS[1:]:
        d = best[(best["program"] == prog) & best["speedup"].notna()]
        if d.empty:
            continue
        ax.plot(d["order"], d["speedup"], "-o", color=COLORS[prog],
                linewidth=2, markersize=5, label=LABELS[prog])
    style(ax, "Speedup over serial (best block size)",
          "matrix order n", "speedup (serial time / time)")
    ax.legend(frameon=False, loc="upper left")


def plot_time_vs_block(ax, runs, ok, prog, orders, colors):
    d = ok[ok["program"] == prog]
    for order in orders:
        # every block size that was attempted: the ones that did not complete
        # become NaN, which breaks the line instead of bridging the gap
        tried = runs[(runs["program"] == prog) & (runs["order"] == order)]["bs"]
        o = d[d["order"] == order].set_index("bs")["elapsed_s"]
        o = o.reindex(sorted(tried.unique()))
        if o.isna().all():
            continue
        ax.plot(o.index, o.values, "-o", color=colors[order],
                linewidth=2, markersize=4, label=f"n={order}")
    procs = d["procs"].max() if not d.empty else 0
    style(ax, f"{LABELS[prog]}: time vs block size ({procs:g} processes)",
          "block size bs", "elapsed time (s)")


def new_figure():
    return plt.subplots(figsize=(7.5, 5), layout="constrained")


def save(fig, out_dir, name, footnote):
    fig.text(0.01, -0.01, footnote, fontsize=8, color=INK_MUTED,
             ha="left", va="top")
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"plot written to {path}")


def main():
    args = parser.parse_args()

    runs, ok = load(args.csv)
    best = best_per_order(ok)

    mpi_ok = ok[ok["program"] != "serial"]
    orders = sorted(o for o in mpi_ok["order"].unique() if o >= args.min_order)
    steps = np.linspace(0, len(RAMP) - 1, max(len(orders), 1)).round().astype(int)
    colors = {order: RAMP[s] for order, s in zip(orders, steps)}

    os.makedirs(args.output_dir, exist_ok=True)
    missing = runs[runs["status"] != "ok"]["status"].str.replace(r"\(.*\)", "", regex=True)
    counts = ", ".join(f"{n} {s}" for s, n in missing.value_counts().items())
    footnote = f"Only completed runs are shown. Omitted: {counts or 'none'}."

    fig, ax = new_figure()
    plot_time_vs_order(ax, best)
    save(fig, args.output_dir, "time_vs_order.png", footnote)

    fig, ax = new_figure()
    plot_speedup(ax, best)
    save(fig, args.output_dir, "speedup.png", footnote)

    # separate files, but the same y range so mpi and mpiblas compare directly
    block_figs = []
    for prog in ("mpi", "mpiblas"):
        fig, ax = new_figure()
        plot_time_vs_block(ax, runs, ok, prog, orders, colors)
        ax.legend(title="matrix order", frameon=False, loc="upper left",
                  bbox_to_anchor=(1.01, 1))
        block_figs.append((prog, fig, ax))
    lows, highs = zip(*(ax.get_ylim() for _, _, ax in block_figs))
    for prog, fig, ax in block_figs:
        ax.set_ylim(min(lows), max(highs))
        save(fig, args.output_dir, f"block_size_{prog}.png", footnote)
    print()

    table = best[best["program"] != "serial"].pivot(
        index="order", columns="program", values=["bs", "elapsed_s", "speedup"])
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.width", 120):
        print("best block size per order:")
        print(table)


if __name__ == "__main__":
    main()
