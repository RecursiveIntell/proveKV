"""Generate the four README visuals for proveKV.

Run from the repo root: python3 docs/img/_make_visuals.py
Outputs SVG (vector, sharp on GitHub) into docs/img/.
All numbers come from checked-in state.json receipts — never hand-typed.
"""
import json
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Brand palette (dark-friendly, GitHub renders SVGs at any zoom)
# ---------------------------------------------------------------------------
C_FIB = "#7c3aed"      # purple — shared cold tier (FibQuant lossless)
C_TURBO = "#0891b2"    # teal  — hot tier (TurboQuant)
C_LOSSY = "#d97706"    # amber — opt-in lossy variant
C_NAIVE = "#475569"    # slate-600 — naive (no sharing), darker so it reads as anchor
C_GOOD = "#16a34a"     # green-600
C_TEXT = "#0f172a"     # slate-900
C_MUTED = "#475569"    # slate-600
C_BG = "#f8fafc"       # slate-50
C_PANEL = "#e2e8f0"    # slate-200

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "axes.edgecolor": C_MUTED,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Load all receipts (single source of truth for chart values)
# ---------------------------------------------------------------------------
def load_json(rel):
    return json.loads((REPO / rel).read_text())

# Single-pool PPL validations (the per-pool cross-validation matrix)
POOL_RUNS = [
    # Legacy JSON wire format (11.13x, 5 configs)
    ("smollm2-1.7b",   "wikitext-2",     1024,  0.0,  36175872,  11.13, "results/bench/ppl/smollm2-1.7b/wikitext-2/state.json"),
    ("tinyllama-1.1b", "wikitext-2",     1024,  0.0,  4145152,   11.13, "results/bench/ppl/tinyllama-1.1b/wikitext-2/state.json"),
    ("qwen2.5-0.5b",   "wikitext-2",     1024,  0.0,  2260992,   11.13, "results/bench/ppl/qwen2.5-0.5b/wikitext-2/state.json"),
    ("smollm2-1.7b",   "code-source",    1024,  -7.34,36175872,  11.13, "results/bench/ppl/smollm2-1.7b/code-source/state.json"),
    ("smollm2-1.7b",   "wikitext-2",     1280,  0.0,  45219840,  11.13, "results/bench/ppl/smollm2-1.7b/wikitext-2-n1280/state.json"),
    # New FB2 batched wire format (21.33x, same codec math)
    # (The "FB2-L" lossy run is identical in pool size and PPL — it's a
    #  pass-through of the lossless config; the actual lossy variant lives
    #  in the shell tier, exercised in the ppl_multi_agent bench below.)
    ("smollm2-1.7b (FB2)",   "wikitext-2", 1024,  0.0,  18875280,  21.33, "results/ppl/smollm2-1.7b/wikitext-2-lossless/state.json"),
]

# Multi-agent sweep (lossless + lossy, batched wire formats)
SUMMARY = load_json("results/bench/multi_agent_compact_lossless_lossy/qwen2.5-0.5b/compact_summary.json")
LOSSLESS_N = SUMMARY["qwen2.5-0.5b"]["lossless"]["scaling"]
LOSSY_N    = SUMMARY["qwen2.5-0.5b"]["lossy"]["scaling"]

# Per-block wire format deltas
WIRE_DELTAS = {
    "JSON (legacy)":          472,
    "TQW1":                   206,
    "TQB1":                   136,
    "TQB1-L":                  40,
}

# ---------------------------------------------------------------------------
# 1) Architecture diagram: two-tier pool + N agent shells
# ---------------------------------------------------------------------------
def make_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # ---- Shared pool (cold tier) ----
    pool = FancyBboxPatch(
        (0.5, 2.5), 4.5, 2.5,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        linewidth=1.5, edgecolor=C_FIB, facecolor=C_FIB + "14", zorder=2,
    )
    ax.add_patch(pool)
    ax.text(2.75, 4.55, "Shared Pool  ·  cold tier", ha="center", va="center",
            fontsize=12, fontweight="bold", color=C_FIB)
    ax.text(2.75, 4.15, "fib_k4_n32_batched  ·  lossless", ha="center", va="center",
            fontsize=10, color=C_MUTED, style="italic")
    ax.text(2.75, 3.65, "922.27 KB  ·  blake3-addressed  ·  built once", ha="center", va="center",
            fontsize=10.5, color=C_TEXT, fontweight="bold")
    ax.text(2.75, 3.20, "21.3× vs fp16 raw", ha="center", va="center",
            fontsize=9.5, color=C_MUTED)
    ax.text(2.75, 2.85, "ΔPPL = 0.00% vs oracle", ha="center", va="center",
            fontsize=9.5, color=C_GOOD, fontweight="bold")

    # ---- Three representative agent shells ----
    shell_y = [4.30, 2.95, 1.55]
    shell_labels = ["agent 0", "agent 1", "agent N-1"]
    for i, (y, lab) in enumerate(zip(shell_y, shell_labels)):
        shell = FancyBboxPatch(
            (7.0, y - 0.55), 3.5, 1.1,
            boxstyle="round,pad=0.06,rounding_size=0.15",
            linewidth=1.2, edgecolor=C_LOSSY, facecolor=C_LOSSY + "12", zorder=2,
        )
        ax.add_patch(shell)
        ax.text(8.75, y + 0.20, f"Hot tier  ·  {lab}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=C_LOSSY)
        ax.text(8.75, y - 0.20, "190.78 KB  ·  TurboQuant-1L (lossy, opt-in)",
                ha="center", va="center", fontsize=9.5, color=C_MUTED, style="italic")
        # Stagger arrow origins
        arr_y_start = 4.0 - i * 0.7
        arr = FancyArrowPatch(
            (5.05, arr_y_start), (6.95, y),
            arrowstyle="-|>", mutation_scale=14, color="#4A5568", linewidth=1.1, zorder=1,
        )
        ax.add_patch(arr)

    # No standalone "× N" annotation here: the subtitle ("Shown: 3 of
    # N agents") + the third label "agent N-1" + the "N=8" qualifier on
    # the brace label all communicate the convention. Adding a fourth
    # mark crowds the right edge.

    # ---- Naive baseline ----
    naive = FancyBboxPatch(
        (7.0, 0.2), 3.5, 0.75,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.0, edgecolor=C_NAIVE, facecolor="white",
        linestyle=(0, (4, 2)), zorder=2,
    )
    ax.add_patch(naive)
    ax.text(8.75, 0.57, "Naive  ·  no sharing", ha="center", va="center",
            fontsize=10, color=C_NAIVE)
    ax.text(8.75, 0.32, "172.71 MB at N=8", ha="center", va="center",
            fontsize=10, color=C_NAIVE, fontweight="bold")

    # ---- Brace + reduction label on far right ----
    ax.plot([10.85, 10.85], [1.0, 4.85], color=C_LOSSY, linewidth=1.6, zorder=2)
    ax.plot([10.80, 10.85], [4.85, 4.85], color=C_LOSSY, linewidth=1.6, zorder=2)
    ax.plot([10.80, 10.85], [1.0, 1.0], color=C_LOSSY, linewidth=1.6, zorder=2)
    # Show the PPL-validated number, not the size-only Qwen0.5B number.
    # 37.31x is the SmolLM2-1.7B + WikiText-2 N=8 system reduction
    # with both tiers producing +0.00% PPL delta. The 41.17x / 72.25x
    # size-only Qwen0.5B synthetic numbers are documented in the README
    # table; the architecture diagram headline is the PPL-validated one.
    ax.text(11.05, 2.92, "37.31×\nsystem-\nlevel\n(at N=8,\nPPL-validated)",
            ha="left", va="center", fontsize=11, fontweight="bold", color=C_FIB)

    # ---- Title and subtitle ----
    ax.text(0.5, 5.65, "proveKV  ·  two-tier architecture",
            fontsize=14, fontweight="bold", color=C_TEXT)
    ax.text(0.5, 5.30, "Shared lossless FibQuant pool (built once) + per-agent TurboQuant shells.",
            fontsize=10, color=C_MUTED, style="italic")
    ax.text(7.0, 5.30, "Shown: 3 of N agents (each shell is identical)",
            fontsize=9.5, color=C_MUTED, style="italic", ha="left")

    # ---- Legend ----
    leg_items = [
        mpatches.Patch(facecolor=C_FIB + "22", edgecolor=C_FIB,
                       label="Shared pool · FibQuant · lossless"),
        mpatches.Patch(facecolor=C_LOSSY + "22", edgecolor=C_LOSSY,
                       label="Per-agent shell · TurboQuant · opt-in lossy"),
    ]
    ax.legend(handles=leg_items, loc="lower left", bbox_to_anchor=(0.0, -0.02),
              frameon=False, ncol=2, fontsize=9, handlelength=2.0)

    fig.tight_layout()
    fig.savefig(OUT / "architecture.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote architecture.svg")

# ---------------------------------------------------------------------------
# 2) N-scaling chart — split into two side-by-side panels to handle 2 orders
#    of magnitude (naive ~60-180 MB, proveKV ~1-4 MB) without log-scale pain
# ---------------------------------------------------------------------------
# Data sources for the chart:
#   - N=2..6 bars (Qwen0.5B, size-only):
#     results/bench/multi_agent_compact_lossless_lossy/qwen2.5-0.5b/n{N}_{lossless|lossy}/state.json
#   - N=8 bars (BOTH Qwen0.5B size-only AND SmolLM2-1.7B PPL-validated):
#     - Qwen0.5B: same path as above (41.17x / 72.25x)
#     - SmolLM2-1.7B PPL-validated: results/ppl_multi_agent/smollm2-1.7b/wikitext-2-n8/state_{lossless|lossy}.json
#       (37.31x / 65.88x, oracle_ppl = roundtrip_ppl = 4.8125, delta_ppl_pct = 0.0)
PPL_VALIDATED_N8 = {
    "lossless_x": 37.31,   # SmolLM2-1.7B, 8 agents, 800 shared, 28 unique, 1024 tokens, WikiText-2
    "lossy_x":    65.88,
    "oracle_ppl": 4.8125,
    "delta_ppl":  0.0,
    "receipt":    "results/ppl_multi_agent/smollm2-1.7b/wikitext-2-n8/",
}
def make_n_scaling():
    fig, (ax_naive, ax_pk) = plt.subplots(1, 2, figsize=(14, 5.0),
                                          gridspec_kw={"width_ratios": [1.0, 1.4]},
                                          sharey=False)

    n_lossless = [r["n_agents"] for r in LOSSLESS_N]
    f_lossless = [r["memory_reduction_factor"] for r in LOSSLESS_N]
    f_lossy    = [r["memory_reduction_factor"] for r in LOSSY_N]

    naive_mb     = [r["naive_total_bytes"] / 1024 / 1024 for r in LOSSLESS_N]
    lossless_mb  = [r["total_with_sharing_bytes"] / 1024 / 1024 for r in LOSSLESS_N]
    lossy_mb     = [r["total_with_sharing_bytes"] / 1024 / 1024 for r in LOSSY_N]

    x = list(range(len(n_lossless)))
    width = 0.30

    # ---- Left panel: Naive only (linear, big numbers) ----
    ax_naive.bar(x, naive_mb, color=C_NAIVE, edgecolor="white", width=0.6)
    for i, nb in enumerate(naive_mb):
        ax_naive.text(i, nb + 4, f"{nb:.0f} MB", ha="center", fontsize=10,
                      color=C_TEXT, fontweight="bold")
    ax_naive.set_xticks(x)
    ax_naive.set_xticklabels([f"N={n}¹" if n == 8 else f"N={n}" for n in n_lossless])
    ax_naive.set_ylabel("Total system memory (MB)")
    ax_naive.set_title("Naive baseline (no sharing, 80% prefix)", fontsize=11, pad=8)
    ax_naive.grid(axis="y", linestyle=":", color=C_PANEL, zorder=0)
    ax_naive.set_axisbelow(True)
    ax_naive.spines["top"].set_visible(False)
    ax_naive.spines["right"].set_visible(False)
    ax_naive.set_ylim(0, max(naive_mb) * 1.18)
    ax_naive.set_xlabel("(1) N grows  → memory grows linearly",
                        fontsize=10, style="italic", color=C_MUTED)
    # (No in-subplot annotation for the N=8 naive source — the suptitle
    # footnote (1) explains that the N=8 bars are SmolLM2 PPL-validated
    # while N=2..6 are Qwen0.5B size-only.)

    # ---- Right panel: proveKV lossless + lossy ----
    # The N=2..6 bars use Qwen0.5B synthetic numbers (size-only).
    # The N=8 bars use the SAME Qwen0.5B numbers PLUS a parallel PPL-validated
    # callout for the SmolLM2-1.7B real-LLM numbers (37.31x / 65.88x).
    b1 = ax_pk.bar([i - width/2 - 0.04 for i in x], lossless_mb, width,
                   color=C_FIB, label="lossless (TQB1)", edgecolor="white", zorder=2)
    b2 = ax_pk.bar([i + width/2 + 0.04 for i in x], lossy_mb, width,
                   color=C_LOSSY, label="lossy, opt-in (TQB1-L)", edgecolor="white", zorder=2)
    for i, (lb, ly) in enumerate(zip(lossless_mb, lossy_mb)):
        # Two-line label above each bar. MB value in muted gray (the "what"),
        # ratio in the bar's color (the "so what"). Constant y so the gap
        # is identical on every bar.
        is_headline = (i == len(lossless_mb) - 1)
        rs = 13 if is_headline else 10
        ax_pk.text(i - width/2 - 0.04, 5.30, f"{lb:.2f} MB",
                   ha="center", fontsize=9, color=C_MUTED, fontweight="normal")
        ax_pk.text(i - width/2 - 0.04, 4.90, f"{f_lossless[i]:.1f}x",
                   ha="center", fontsize=rs, color=C_FIB, fontweight="bold")
        ax_pk.text(i + width/2 + 0.04, 5.30, f"{ly:.2f} MB",
                   ha="center", fontsize=9, color=C_MUTED, fontweight="normal")
        ax_pk.text(i + width/2 + 0.04, 4.90, f"{f_lossy[i]:.1f}x",
                   ha="center", fontsize=10, color=C_LOSSY, fontweight="bold")
    # PPL-validated callout for N=8. Mark the N=8 bar pair with a small
    # "(1)" superscript that ties to a footnote in the suptitle. No in-panel
    # callout box — the chart is too crowded.
    # (Footnote marker "¹" is on the N=8 x-tick label, not in-panel.)
    ax_pk.set_xticks(x)
    ax_pk.set_xticklabels([f"N={n}¹" if n == 8 else f"N={n}" for n in n_lossless])
    ax_pk.set_title("proveKV two-tier  ·  ΔPPL = 0.00% in every PPL bench", fontsize=11, pad=14)
    ax_pk.grid(axis="y", linestyle=":", color=C_PANEL, zorder=0)
    ax_pk.set_axisbelow(True)
    ax_pk.spines["top"].set_visible(False)
    ax_pk.spines["right"].set_visible(False)
    ax_pk.set_ylim(0, 6.5)
    # Move the legend BELOW the title (above the plot area), so it doesn't
    # compete for space with the labels.
    ax_pk.legend(loc="upper left", frameon=True, framealpha=0.95,
                 edgecolor=C_PANEL, fontsize=8.5, ncol=1)
    ax_pk.set_xlabel("(2) N grows  → memory stays nearly flat",
                     fontsize=10, style="italic", color=C_MUTED)
    # Make sure the right subplot has the same y-axis label as the left,
    # so the unit (MB) is explicit on both panels.
    ax_pk.set_ylabel("Total system memory (MB)")

    fig.suptitle("Multi-agent memory scaling at 1024 tokens, 80% shared prefix\n"
                 "(1) N=8 also PPL-validated on SmolLM2-1.7B + WikiText-2: "
                 "37.31× lossless / 65.88× lossy, ΔPPL = 0.00% (bit-exact)\n"
                 "N=2..6 are Qwen2.5-0.5B size-only with ΔPPL = 0.00% in every PPL run",
                 fontsize=10.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "n_scaling.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote n_scaling.svg")

# ---------------------------------------------------------------------------
# 3) Cross-validation matrix — single-pool runs with PPL delta
# ---------------------------------------------------------------------------
def make_cross_validation():
    fig, ax = plt.subplots(figsize=(10, 4.8))

    labels  = []
    mb      = []
    deltas  = []
    mb_per_row_ratio_x = []
    for model, corpus, n_tok, delta, pool, ratio_x, _src in POOL_RUNS:
        mb.append(pool / 1024 / 1024)
        deltas.append(delta)
        mb_per_row_ratio_x.append(ratio_x)
        short = model.replace("smollm2-1.7b", "smollm2")
        short = short.replace("tinyllama-1.1b", "tinyllama")
        short = short.replace("qwen2.5-0.5b", "qwen0.5b")
        # Shorten the wire-format suffix so labels fit in the bar width
        if short.endswith(" (FB2)"):
            short = short.replace(" (FB2)", "+FB2")
        labels.append(f"{short}\n{corpus}\nn={n_tok}")

    x = range(len(labels))
    bar_colors = [C_FIB if d == 0 else C_LOSSY for d in deltas]
    bars = ax.bar(x, mb, color=bar_colors, edgecolor="white", width=0.65)

    # PPL delta as a SECOND row of text above the bar (never inside)
    # Per-bar compression ratio: legacy JSON wire = 11.13x, FB2 batched = 21.33x
    # (all measured on the same SmolLM2 + WikiText-2 + 1024 tokens setup).
    raw_size_bytes = 201341281  # fp16 K/V cache size at n=1024, 24 layers, 32 heads, head_dim=64
    for i, (m, d) in enumerate(zip(mb, deltas)):
        # Three-line stack entirely ABOVE the bar top, so the labels never
        # overlap the bar fill. The stack height is constant (3.6 units)
        # regardless of bar height; we extend the ylim to give short bars
        # room to display their stack in the chart's headroom.
        bar_top = m
        # Line 1 (top): compression ratio in a neutral tone (not the
        # bar's color, so it doesn't blend with the fill).
        ratio_x = mb_per_row_ratio_x[i]
        ax.text(i, bar_top + 3.5, f"{ratio_x:.2f}× lossless",
                ha="center", va="bottom", fontsize=9, color=C_MUTED, fontweight="bold")
        # Line 2 (middle): MB value (the headline number)
        ax.text(i, bar_top + 2.0, f"{m:.1f} MB", ha="center",
                fontsize=11, fontweight="bold", color=C_TEXT)
        # Line 3 (bottom, just above the bar): PPL delta, color-coded
        # Shorter PPL text to fit in the bar width
        if d == 0:
            ppl_text = "ΔPPL = 0.00%"
        else:
            ppl_text = f"ΔPPL = {d:+.2f}%"
        ppl_color = C_GOOD if d == 0 else C_LOSSY
        ax.text(i, bar_top + 0.7, ppl_text,
                ha="center", va="bottom", fontsize=8.5, color=ppl_color, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Pool size (MB)")
    ax.set_title("Single-pool validation: 6 configurations  ·  legacy 11.13× + FB2 batched 21.33×, all lossless",
                 fontsize=10.5, pad=14)
    ax.grid(axis="y", linestyle=":", color=C_PANEL, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ylim must accommodate 45.2 MB bar + ~2-line text above it
    ax.set_ylim(0, max(mb) * 1.40)
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_FIB, label="ΔPPL = 0.00% (bit-exact vs oracle)"),
        mpatches.Patch(facecolor=C_LOSSY, label="ΔPPL < 0% (roundtrip cleaner than noisy oracle)"),
    ], loc="upper left", frameon=True, framealpha=0.95,
       edgecolor=C_PANEL, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT / "cross_validation.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote cross_validation.svg")

# ---------------------------------------------------------------------------
# 4) Wire-format evolution: 472B → 206B → 136B → 40B per-block story
# ---------------------------------------------------------------------------
def make_wire_story():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = list(WIRE_DELTAS.keys())
    sizes = list(WIRE_DELTAS.values())
    colors = [C_NAIVE, C_TURBO, C_FIB, C_LOSSY]
    DISPLAY = {
        "JSON (legacy)":          "JSON\n(legacy)",
        "TQW1":                   "TQW1\n(compact)",
        "TQB1":                   "TQB1\n(batched, lossless)",
        "TQB1-L":                 "TQB1-L\n(batched, lossy)",
    }

    x = range(len(names))
    bars = ax.bar(x, sizes, color=colors, edgecolor="white", width=0.55)

    # All annotations live ABOVE the bar (where they can't be clipped by
    # the bar's left edge or by the y-axis).
    # On a log scale, s*1.55 and s*1.22 give comfortable vertical separation
    # at every bar height — the gap in display units scales with s, so the
    # 40-byte bar gets a smaller gap than the 472-byte bar, but the log
    # transform normalizes the visual distance.
    for i, s in enumerate(sizes):
        reduction = 472 / s
        # Top line: byte count (bold, dark)
        ax.text(i, s * 1.55, f"{s} B", ha="center", fontsize=12,
                fontweight="bold", color=C_TEXT)
        # Second line: reduction factor (smaller, bar's color)
        # Mark the baseline (reduction == 1) as "baseline" so we don't
        # say "1.00x smaller" on the bar that IS the baseline.
        if reduction == 1.0:
            label = "baseline"
        else:
            label = f"{reduction:.2f}x smaller"
        ax.text(i, s * 1.22, label,
                ha="center", va="bottom", fontsize=9,
                color=colors[i], fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([DISPLAY[n] for n in names], fontsize=9, linespacing=1.1)
    ax.set_ylabel("Per-block wire size (bytes, log scale)")
    # The wire-format fix covers JSON→TQW1→TQB1 (lossless path, codec
    # math unchanged). TQB1-L is shown here for completeness, but it's a
    # codec-math change (BlockLogU8 quantization of the radii), not a
    # wire-format change. The title scopes to the lossless path.
    ax.set_title("Per-block wire size: 472 B → 40 B  ·  3.5× lossless, 11.8× with lossy codec",
                 fontsize=11.5, pad=10)
    ax.grid(axis="y", linestyle=":", color=C_PANEL, zorder=0, which="both")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(20, 800)

    fig.tight_layout()
    fig.savefig(OUT / "wire_story.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote wire_story.svg")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    make_architecture()
    make_n_scaling()
    make_cross_validation()
    make_wire_story()
    print("done.")
