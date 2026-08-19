from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import matplotlib as mpl


mpl.rcParams["font.family"] = "DejaVu Serif"
mpl.rcParams["mathtext.fontset"] = "stix"


OUT_BASENAME = "arch_direct"


COLORS = {
    "ink": "#2b2b2b",
    "muted": "#6f6f6f",
    "light_border": "#d4d4d4",
    "panel_fill": "#fbfbfc",
    "blue": "#2749d8",
    "blue_fill": "#eef2ff",
    "teal": "#1f7a82",
    "teal_fill": "#eef8f8",
    "green": "#1f9d55",
    "green_fill": "#eefaf1",
    "orange": "#d97706",
    "orange_fill": "#fff6ea",
    "gray_fill": "#f7f7f7",
    "gray_arrow": "#707070",
}


def rounded_box(ax, x, y, w, h, edge, fill, lw=1.8, r=0.18, z=2):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def panel(ax, x, y, w, h, title):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.24",
        linewidth=1.2,
        edgecolor=COLORS["light_border"],
        facecolor=COLORS["panel_fill"],
        zorder=0,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h + 0.08,
        title,
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="semibold",
        color=COLORS["muted"],
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
    )
    return box


def subgroup(ax, x, y, w, h, title):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.18",
        linewidth=1.2,
        edgecolor="#999999",
        linestyle=(0, (4, 4)),
        facecolor="none",
        zorder=1,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.28,
        y + h + 0.08,
        title,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="semibold",
        color=COLORS["muted"],
        bbox=dict(facecolor="white", edgecolor="none", pad=0.9),
    )
    return box


def text_block(ax, x, y, lines, fontsize=12, color=None, weight=None, align="center", z=3):
    color = color or COLORS["ink"]
    ax.text(
        x,
        y,
        "\n".join(lines),
        ha=align,
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        zorder=z,
    )


def token_row(ax, x, y, n, color):
    for i in range(n):
        rounded_box(ax, x + i * 0.21, y, 0.13, 0.10, "#9a9a9a", color, lw=0.8, r=0.04, z=3)


def arrow(ax, start, end, color, lw=2.0, style="-|>", rad=0.0, z=4, mutation=18):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(patch)
    return patch


def ortho_arrow(ax, pts, color, lw=2.0, style="-|>", z=4, ls="-", mutation=18):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=z)
    arrow(ax, pts[-2], pts[-1], color, lw=lw, style=style, z=z, mutation=mutation)


def op_circle(ax, x, y, label, edge="#9b9b9b"):
    circ = Circle((x, y), radius=0.17, facecolor="white", edgecolor=edge, linewidth=1.4, zorder=4)
    ax.add_patch(circ)
    ax.text(x, y, label, ha="center", va="center", fontsize=13, color=COLORS["ink"], zorder=5)
    return circ


def build():
    fig = plt.figure(figsize=(16, 7.1), dpi=180)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.1)
    ax.axis("off")

    panel(ax, 0.15, 1.02, 4.35, 5.72, "(1) Encoders")
    panel(ax, 4.65, 1.02, 8.25, 5.72, "(2) Conditional Fusion  [proposed]")
    panel(ax, 13.05, 1.02, 2.95, 5.72, "(3) Prediction")

    # Input cards
    input_x, input_w, input_h = 0.22, 1.48, 0.84
    encoder_x, encoder_w, encoder_h = 1.80, 2.60, 1.02
    y_positions = [5.52, 4.18, 2.84, 1.50]

    inputs = [
        ["OHLCV", "window", r"$[t-L+1,\; t]$"],
        ["Aligned", "text context", r"$t^\ast \leq t$"],
        ["Causal", "web scalars", r"available at $t$"],
        ["Freshness", r"/ lag", r"$\tau = t - t^\ast$"],
    ]
    for y, lines in zip(y_positions, inputs):
        rounded_box(ax, input_x, y, input_w, input_h, "#a7a7a7", "white", lw=1.4, r=0.12, z=2)
        text_block(ax, input_x + input_w / 2, y + input_h / 2, lines, fontsize=12)

    # Encoder cards
    rounded_box(ax, encoder_x, y_positions[0] - 0.08, encoder_w, encoder_h, COLORS["blue"], COLORS["blue_fill"], z=2)
    text_block(ax, encoder_x + encoder_w / 2, y_positions[0] + 0.64, ["Price Encoder"], fontsize=16, weight="semibold")
    ax.text(encoder_x + 0.18, y_positions[0] + 0.42, r"$\mathbf{H}^p \in \mathbb{R}^{L\times d}$", ha="left", va="center", fontsize=12.6, zorder=3)
    ax.text(encoder_x + 0.18, y_positions[0] + 0.21, "price tokens", ha="left", va="center", fontsize=10.8, color=COLORS["muted"], zorder=3)
    ax.text(encoder_x + 1.18, y_positions[0] + 0.42, r"$\mathbf{h}_p \in \mathbb{R}^{d}$", ha="left", va="center", fontsize=12.6, zorder=3)
    ax.text(encoder_x + 1.18, y_positions[0] + 0.21, "CLS summary", ha="left", va="center", fontsize=10.8, color=COLORS["muted"], zorder=3)
    token_row(ax, encoder_x + 1.90, y_positions[0] - 0.45, 4, "#c9cdf9")

    rounded_box(ax, encoder_x, y_positions[1] - 0.02, encoder_w, 0.82, COLORS["teal"], COLORS["teal_fill"], z=2)
    text_block(ax, encoder_x + encoder_w / 2, y_positions[1] + 0.50, ["Text Encoder"], fontsize=15.5, weight="semibold")
    ax.text(encoder_x + encoder_w / 2, y_positions[1] + 0.18, r"$\mathbf{h}_t \in \mathbb{R}^{d}$", ha="center", va="center", fontsize=14.2, zorder=3)
    token_row(ax, encoder_x + 1.90, y_positions[1] - 0.34, 4, "#cdeceb")

    rounded_box(ax, encoder_x, y_positions[2] - 0.02, encoder_w, 0.82, COLORS["green"], COLORS["green_fill"], z=2)
    text_block(ax, encoder_x + encoder_w / 2, y_positions[2] + 0.50, ["Web Scalar Encoder"], fontsize=14.6, weight="semibold")
    ax.text(encoder_x + encoder_w / 2, y_positions[2] + 0.18, r"$\mathbf{h}_w \in \mathbb{R}^{d}$", ha="center", va="center", fontsize=14.2, zorder=3)
    token_row(ax, encoder_x + 1.90, y_positions[2] - 0.34, 4, "#cef3d3")

    # Fusion groups
    subgroup(ax, 5.00, 4.72, 6.95, 1.60, "Grounding")
    subgroup(ax, 5.20, 1.62, 6.95, 3.08, "Trust-Aware Fusion")

    rounded_box(ax, 6.0, 4.78, 3.35, 1.58, "#1f4a86", "#f1f5fa", z=2)
    text_block(ax, 7.68, 5.90, ["Text-Conditioned", "Price Attention"], fontsize=15.5, weight="semibold")
    ax.text(7.68, 5.52, "retrieval: text queries price memory", ha="center", va="center", fontsize=11.2, color=COLORS["muted"], zorder=3)
    ax.text(7.68, 5.15, r"$Q=\mathbf{h}_t,\;\; K,V=\mathbf{H}^p$", ha="center", va="center", fontsize=16, zorder=3)
    ax.text(7.68, 4.84, r"$\mathbf{h}_c\;=\;\mathrm{MHA}(\mathbf{h}_t,\mathbf{H}^p)$", ha="center", va="center", fontsize=16, zorder=3)
    ax.text(7.68, 4.60, "event-conditioned market context", ha="center", va="center", fontsize=10.6, color=COLORS["muted"], zorder=3)

    rounded_box(ax, 5.82, 2.34, 4.42, 1.48, COLORS["orange"], COLORS["orange_fill"], z=2)
    text_block(ax, 8.03, 3.65, ["Conditional Gate"], fontsize=16, weight="semibold")
    ax.text(8.03, 3.28, "trust control from context shift, web context, freshness", ha="center", va="center", fontsize=10.9, color=COLORS["muted"], zorder=3)
    ax.text(8.03, 2.95, r"$\Delta_{pc}\;\;=\;\;\mathbf{h}_c\;-\;\mathbf{h}_p$", ha="center", va="center", fontsize=16, zorder=3)
    ax.text(8.03, 2.62, r"$\mathbf{g}\;=\;\sigma(W_g[\mathbf{h}_p,\mathbf{h}_c,\Delta_{pc},\mathbf{h}_w,\tau])$", ha="center", va="center", fontsize=14.2, zorder=3)
    ax.text(8.03, 2.30, r"$\mathbf{g}\in(0,1)^d$", ha="center", va="center", fontsize=10.6, color=COLORS["muted"], zorder=3)
    token_row(ax, 9.48, 3.60, 5, "#fde6ca")

    op_circle(ax, 10.42, 5.38, r"$\odot$")
    op_circle(ax, 11.08, 5.38, r"$+$")

    rounded_box(ax, 9.45, 4.08, 2.45, 0.96, COLORS["blue"], COLORS["blue_fill"], z=2)
    text_block(ax, 10.68, 4.64, ["Fused State"], fontsize=15, weight="semibold")
    ax.text(10.68, 4.34, r"$\mathbf{h}_f \;=\; \mathbf{h}_p + \mathbf{g}\odot \mathbf{h}_c$", ha="center", va="center", fontsize=15.5, zorder=3)
    ax.text(10.68, 4.07, r"price-only fallback when $\mathbf{g}\rightarrow 0$", ha="center", va="top", fontsize=10.2, color=COLORS["muted"], zorder=3)

    # Prediction cards
    rounded_box(ax, 13.68, 4.92, 1.78, 0.72, "#ababab", "white", lw=1.4, r=0.12, z=2)
    text_block(ax, 14.57, 5.36, ["Concat"], fontsize=15, weight="semibold")
    ax.text(14.57, 5.05, r"$[\mathbf{h}_f;\mathbf{h}_w]$", ha="center", va="center", fontsize=15, zorder=3)
    ax.text(14.57, 4.81, "direct predictive cue", ha="center", va="center", fontsize=10.0, color=COLORS["green"], zorder=3, bbox=dict(facecolor="white", edgecolor="none", pad=0.2))

    rounded_box(ax, 13.64, 3.82, 1.88, 0.98, "#ababab", "white", lw=1.4, r=0.12, z=2)
    text_block(ax, 14.58, 4.42, ["Task Head"], fontsize=16, weight="semibold")
    ax.text(14.58, 4.07, r"$\hat{p}(y{=}1)$", ha="center", va="center", fontsize=16, zorder=3)
    ax.text(14.58, 3.80, "task-agnostic fusion, binary head shown", ha="center", va="top", fontsize=10.1, color=COLORS["muted"], zorder=3)

    # Paths
    blue = COLORS["blue"]
    teal = COLORS["teal"]
    green = COLORS["green"]
    orange = COLORS["orange"]

    # Inputs to encoders
    arrow(ax, (1.74, 5.94), (1.75, 5.94), blue, lw=2.0)
    arrow(ax, (1.74, 4.63), (1.75, 4.63), teal, lw=2.0)
    arrow(ax, (1.74, 3.29), (1.75, 3.29), green, lw=2.0)

    # Encoder to attention
    ortho_arrow(ax, [(4.37, 5.94), (4.80, 5.94), (6.0, 5.66)], blue, lw=2.2)
    ortho_arrow(ax, [(4.37, 4.63), (5.00, 4.63), (5.00, 5.32), (6.0, 5.32)], teal, lw=2.0)

    # Price skip to add
    ortho_arrow(ax, [(4.40, 5.94), (5.74, 5.94), (5.74, 6.36), (11.08, 6.36), (11.08, 5.55)], blue, lw=2.0)

    # Attention to multiply
    ortho_arrow(ax, [(9.35, 5.38), (9.92, 5.38), (10.42, 5.38)], COLORS["gray_arrow"], lw=2.2)

    # Gate inputs
    ortho_arrow(ax, [(3.25, 5.44), (3.25, 3.08), (5.82, 3.08)], orange, lw=1.9, ls=(0, (3, 3)))
    ortho_arrow(ax, [(9.35, 4.74), (9.35, 3.82), (8.65, 3.82)], orange, lw=1.9, ls=(0, (3, 3)))
    ortho_arrow(ax, [(4.37, 3.29), (5.20, 3.29), (5.82, 3.29)], orange, lw=1.9, ls=(0, (3, 3)))
    ortho_arrow(ax, [(1.74, 1.92), (4.78, 1.92), (4.78, 2.55), (5.82, 2.55)], orange, lw=1.9, ls=(0, (3, 3)))

    # Gate to multiply
    ortho_arrow(ax, [(10.24, 3.08), (10.24, 5.20), (10.25, 5.20), (10.42, 5.20)], orange, lw=2.0, ls=(0, (3, 3)))

    # Multiply to add and add to fused
    arrow(ax, (10.59, 5.38), (10.91, 5.38), orange, lw=2.0)
    ortho_arrow(ax, [(11.08, 5.21), (11.08, 4.92), (10.98, 4.92)], COLORS["gray_arrow"], lw=2.1)

    # Fused to concat
    ortho_arrow(ax, [(11.90, 4.56), (12.60, 4.56), (12.60, 5.28), (13.68, 5.28)], COLORS["gray_arrow"], lw=2.2)

    # Web direct cue
    ortho_arrow(ax, [(3.05, 2.80), (3.05, 1.72), (14.22, 1.72), (14.22, 4.92)], green, lw=1.8, ls=(0, (1, 2)))

    # Concat to task head
    ortho_arrow(ax, [(14.57, 4.92), (14.57, 4.80), (14.57, 4.80), (14.57, 4.80)], COLORS["gray_arrow"], lw=0.0)
    ortho_arrow(ax, [(14.57, 4.92), (14.57, 4.66), (14.58, 4.66), (14.58, 4.80)], COLORS["gray_arrow"], lw=2.0)

    # visual legend
    subgroup(ax, 3.90, 0.18, 8.55, 0.68, "Visual Legend")
    legend_y = 0.52
    ax.plot([4.12, 4.38], [legend_y, legend_y], color=blue, lw=2.2)
    arrow(ax, (4.38, legend_y), (4.46, legend_y), blue, lw=2.2, mutation=16)
    ax.text(4.58, legend_y, "price path", ha="left", va="center", fontsize=11.3, color=COLORS["muted"])

    ax.plot([5.68, 5.94], [legend_y, legend_y], color=teal, lw=2.0)
    arrow(ax, (5.94, legend_y), (6.02, legend_y), teal, lw=2.0, mutation=16)
    ax.text(6.16, legend_y, "text path", ha="left", va="center", fontsize=11.3, color=COLORS["muted"])

    ax.plot([7.08, 7.34], [legend_y, legend_y], color=green, lw=2.0)
    arrow(ax, (7.34, legend_y), (7.42, legend_y), green, lw=2.0, mutation=16)
    ax.text(7.56, legend_y, "web path", ha="left", va="center", fontsize=11.3, color=COLORS["muted"])

    ax.plot([8.38, 8.64], [legend_y, legend_y], color=orange, lw=1.9, ls=(0, (3, 3)))
    arrow(ax, (8.64, legend_y), (8.72, legend_y), orange, lw=1.9, mutation=16)
    ax.text(8.86, legend_y, "gate control", ha="left", va="center", fontsize=11.3, color=COLORS["muted"])

    ax.plot([10.16, 10.42], [legend_y, legend_y], color=green, lw=1.8, ls=(0, (1, 2)))
    arrow(ax, (10.42, legend_y), (10.50, legend_y), green, lw=1.8, mutation=16)
    ax.text(10.64, legend_y, "direct predictive cue", ha="left", va="center", fontsize=11.3, color=COLORS["muted"])

    op_circle(ax, 12.00, legend_y, r"$+$", edge="#888888")
    ax.text(12.24, legend_y, "residual add", ha="left", va="center", fontsize=11.3, color=COLORS["muted"])

    return fig


def main():
    out_dir = Path(__file__).resolve().parent
    fig = build()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{OUT_BASENAME}.{ext}", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
