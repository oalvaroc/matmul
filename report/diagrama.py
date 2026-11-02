#!/usr/bin/env python3

"""Diagram of the master-worker scheme of src/mpi.c for the report.

Left: the task for block C[i][j] is the row of blocks A[i][0..q-1] and the
column B[0..q-1][j], packed into one message. Right: the master sends these
messages to the W workers and gets each C[i][j] back.
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

A_COLOR = "#2a78d6"
B_COLOR = "#1baf7a"
C_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#c9c8c3"

plt.rcParams.update({"font.size": 8.5, "pdf.fonttype": 42,
                     "font.family": "DejaVu Sans"})

Q = 4        # blocks per row and column in the drawing
I, J = 1, 2  # the task drawn


def cell(ax, x, y, w, h, color=None, label=None, fontsize=7):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color or "white",
                           edgecolor=GRID if color is None else "white", lw=0.8))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color="white")


def matrix(ax, x, y, s, name, highlight):
    """q x q grid of blocks of side s with its lower-left corner at (x, y);
    highlight(r, c) gives the label of a highlighted block, or None."""
    for r in range(Q):
        for c in range(Q):
            label = highlight(r, c)
            cell(ax, x + c * s, y + (Q - 1 - r) * s, s, s,
                 color=name_color(name) if label else None, label=label)
    ax.add_patch(Rectangle((x, y), Q * s, Q * s, fill=False, edgecolor=INK_MUTED, lw=1))
    ax.text(x + Q * s / 2, y + Q * s + 1, name, ha="center", va="bottom",
            fontsize=10, fontweight="bold")


def name_color(name):
    return {"A": A_COLOR, "B": B_COLOR, "C": C_COLOR}[name]


def message(ax, x, y, w, h, labels=True):
    """Packed task: q blocks of A then q blocks of B, each of width w."""
    for k in range(Q):
        cell(ax, x + k * w, y, w, h, A_COLOR, f"$A_{{i{k}}}$" if labels else None)
        cell(ax, x + (Q + k) * w, y, w, h, B_COLOR, f"$B_{{{k}j}}$" if labels else None)
    ax.add_patch(Rectangle((x, y), 2 * Q * w, h, fill=False, edgecolor=INK_MUTED, lw=0.8))


def arrow(ax, start, end, color=INK_MUTED, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=1, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1",
                                facecolor=color, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8,
            color="white", linespacing=1.3)


def main(path):
    fig, ax = plt.subplots(figsize=(6.6, 2.75))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 41.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # left: the blocks of task (i, j) and the message they are packed into
    s = 3.2
    ax_, bx_, my_ = 3, 20, 22
    matrix(ax, ax_, my_, s, "A", lambda r, c: f"$A_{{i{c}}}$" if r == I else None)
    matrix(ax, bx_, my_, s, "B", lambda r, c: f"$B_{{{r}j}}$" if c == J else None)
    ax.text(ax_ - 0.6, my_ + (Q - 1 - I) * s + s / 2, "i", ha="right", va="center",
            fontsize=8, color=INK_MUTED, style="italic")
    ax.text(bx_ + J * s + s / 2, my_ - 0.6, "j", ha="center", va="top",
            fontsize=8, color=INK_MUTED, style="italic")

    mx, my, mw, mh = 2, 8, 3.6, 3.4
    message(ax, mx, my, mw, mh)
    ax.text(mx + Q * mw, my - 1.2, "mensagem da tarefa (i, j):\n2q blocos b×b = 2qb² floats",
            ha="center", va="top", fontsize=7.5, color=INK, linespacing=1.3)
    arrow(ax, (ax_ + Q * s / 2, my_ - 0.8), (mx + Q * mw / 2, my + mh + 0.6))
    arrow(ax, (bx_ + Q * s / 2, my_ - 0.8), (mx + Q * mw * 1.5, my + mh + 0.6))
    ax.text(mx + Q * mw, 17, "empacota", ha="center", va="center", fontsize=7.5,
            color=INK_MUTED, style="italic")

    # right: the master hands out tasks and collects the C blocks
    bx, bw = 42, 11
    box(ax, bx, 1.5, bw, 38.5, "mestre\n(rank 0)\n\ntem A, B\ne monta C", INK_MUTED)

    wx, ww, wh = 82, 16, 6
    workers = [(36, "trabalhador 1"), (27, "trabalhador 2"), (18, "trabalhador 3"),
               (5, "trabalhador W")]
    ax.text(wx + ww / 2, 11.8, "⋮", ha="center", va="center", fontsize=14,
            color=INK_MUTED)
    ax.text(67.5, 11.8, "⋮", ha="center", va="center", fontsize=14, color=INK_MUTED)
    sw, sh = 1.4, 1.8  # mini message on the send arrows
    for k, (y, name) in enumerate(workers):
        box(ax, wx, y - wh / 2, ww, wh, name, "#1c5cab")
        arrow(ax, (bx + bw, y + 1.3), (wx, y + 1.3))
        message(ax, 67.5 - Q * sw, y + 1.3 - sh / 2, sw, sh, labels=False)
        arrow(ax, (wx, y - 1.3), (bx + bw, y - 1.3), color=C_COLOR, ls="--")
        cell(ax, 66.6, y - 1.3 - 0.9, 1.8, 1.8, C_COLOR)
        if k == 0:
            ax.text(67.5, y + 1.3 + sh / 2 + 0.4, "tarefa (MPI_Send)", ha="center",
                    va="bottom", fontsize=7, color=INK)
            ax.text(67.5, y - 1.3 - 1.3, "bloco $C_{ij}$ (b² floats)", ha="center",
                    va="top", fontsize=7, color=C_COLOR)

    fig.savefig(path, facecolor="white")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/fig_esquema.pdf")
