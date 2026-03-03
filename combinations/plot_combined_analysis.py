"""
Combined analysis: Full accuracy trajectory
For each distilled model: Baseline → Distilled → PTQ → Progressive → QAT+KD
Produces 3 plots:
  1. combined_trajectory.png   – slope chart, all 8 models on one canvas
  2. per_model_grid.png        – 2x4 grid, one subplot per model (bars + size)
  3. delta_heatmap.png         – accuracy delta vs Baseline for all models × stages
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"]   = 11

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
PTQ_DIR   = BASE_DIR / "Post-Training Quantization (PTQ)"  / "results"
PROG_DIR  = BASE_DIR / "Progressive Quantization"          / "results"
QAT_DIR   = BASE_DIR / "Quantization-Aware Distillation (QAT + KD)" / "results"
RES_DIR   = BASE_DIR.parent / "results"   # distillation experiment results
OUT_DIR   = BASE_DIR / "../plots/combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Mapping: (Distill, Size) -> (baseline_json, distilled_json) ─────────────
# baseline  = student trained WITHOUT knowledge distillation
# distilled = student trained WITH knowledge distillation (best config per paper)
DISTILL_FILES = {
    ("Logit",       "Small"): (
        RES_DIR / "logit_distillation_S"     / "baseline_no_distill_small.json",
        RES_DIR / "logit_distillation_S"     / "logit_T3_alpha0.5_small.json",
    ),
    ("Logit",       "Tiny"): (
        RES_DIR / "logit_distillation_T"     / "baseline_no_distill_tiny.json",
        RES_DIR / "logit_distillation_T"     / "logit_T5_alpha0.5_tiny.json",
    ),
    ("Attention",   "Small"): (
        RES_DIR / "attention_distillation_S" / "attention_baseline_no_distill_small.json",
        RES_DIR / "attention_distillation_S" / "attention_lr5e-5_small.json",
    ),
    ("Attention",   "Tiny"): (
        RES_DIR / "attention_distillation_t" / "attention_baseline_no_distill_tiny.json",
        RES_DIR / "attention_distillation_t" / "attention_lr5e-5_tiny.json",
    ),
    ("Contrastive", "Small"): (
        RES_DIR / "contrastive_distillation_S" / "contrastive_baseline_no_contrast_small.json",
        RES_DIR / "contrastive_distillation_S" / "contrastive_alpha0.3_small.json",
    ),
    ("Contrastive", "Tiny"): (
        RES_DIR / "contrastive_distillation_T" / "contrastive_baseline_no_contrast_tiny.json",
        RES_DIR / "contrastive_distillation_T" / "contrastive_T0.10_tiny.json",
    ),
    ("Feature",     "Small"): (
        RES_DIR / "feature_distillation_S"  / "feature_baseline_no_distill_small.json",
        RES_DIR / "feature_distillation_S"  / "feature_alpha0.7_proj_small.json",
    ),
    ("Feature",     "Tiny"): (
        RES_DIR / "feature_distillation_T"  / "feature_baseline_no_distill_tiny.json",
        RES_DIR / "feature_distillation_T"  / "feature_alpha0.9_proj_tiny.json",
    ),
}

# ── Colors & markers ───────────────────────────────────────────────────────
DISTILL_COLORS = {
    "Logit":       "#2E5090",   # deep blue
    "Attention":   "#D35400",   # burnt orange
    "Contrastive": "#8E44AD",   # deep purple
    "Feature":     "#27AE60",   # forest green
}
SIZE_STYLE = {"Small": "-",  "Tiny": "--"}
SIZE_MARK  = {"Small": "o",  "Tiny": "s"}

STAGE_COLORS = {
    "Baseline":    "#BDC3C7",   # light grey  – no distillation
    "Distilled":   "#F39C12",   # amber       – after KD, before quant
    "PTQ":         "#E74C3C",   # red
    "Progressive": "#3498DB",   # blue
    "QAT + KD":    "#2ECC71",   # green
}
STAGES = ["Baseline", "Distilled", "PTQ", "Progressive", "QAT + KD"]


# ── Data loading ────────────────────────────────────────────────────────────
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_model_info(name):
    if   "attention"    in name: distill = "Attention"
    elif "contrastive"  in name: distill = "Contrastive"
    elif "feature"      in name: distill = "Feature"
    elif "logit"        in name: distill = "Logit"
    else:                         distill = "Unknown"
    size = "Small" if "small" in name else ("Tiny" if "tiny" in name else "?")
    return distill, size

def collect_data():
    """Return DataFrame with one row per (model, stage).
    Stages: Baseline → Distilled → PTQ → Progressive → QAT + KD
    """
    ptq_raw  = load_json(PTQ_DIR  / "ptq_summary.json")
    prog_raw = load_json(PROG_DIR / "progressive_summary.json")
    qat_raw  = load_json(QAT_DIR  / "qat_kd_summary.json")

    # fp32_ref: distilled FP32 accuracy (= PTQ fp32 column)
    fp32_ref = {}
    for item in ptq_raw:
        d, s = parse_model_info(item["model_name"])
        fp32_ref[f"{d}_{s}"] = {
            "acc":  item["fp32"]["test_accuracy"],
            "size": item["fp32"]["model_size_mb"],
        }

    # baseline_ref: student accuracy WITHOUT distillation
    baseline_ref = {}
    for (d, s), (baseline_path, _) in DISTILL_FILES.items():
        if baseline_path.exists():
            bdata = load_json(baseline_path)
            baseline_ref[f"{d}_{s}"] = {
                "acc":  bdata["test_accuracy"],
                "size": fp32_ref[f"{d}_{s}"]["size"],  # same arch → same size
            }

    rows = []

    def add_row(distill, size, stage, acc, size_mb, base_acc):
        rows.append({
            "Distill":      distill,
            "Size":         size,
            "Model":        f"{distill}\n({size})",
            "Stage":        stage,
            "Acc":          acc,
            "SizeMB":       size_mb,
            "BaseAcc":      base_acc,           # no-distillation baseline
            "DeltaBase":    acc - base_acc,     # Δ vs raw student
            "DeltaDistill": acc - fp32_ref.get(f"{distill}_{size}", {}).get("acc", acc),
        })

    for (d, s), (baseline_path, _) in DISTILL_FILES.items():
        key      = f"{d}_{s}"
        base_acc = baseline_ref.get(key, {}).get("acc", 0)
        size_mb  = fp32_ref.get(key, {}).get("size", 0)

        # Stage 1: Baseline (no distillation)
        add_row(d, s, "Baseline", base_acc, size_mb, base_acc)

        # Stage 2: Distilled FP32
        dist_acc = fp32_ref.get(key, {}).get("acc", base_acc)
        add_row(d, s, "Distilled", dist_acc, size_mb, base_acc)

    # Stage 3: PTQ
    for item in ptq_raw:
        d, s    = parse_model_info(item["model_name"])
        base    = baseline_ref.get(f"{d}_{s}", {}).get("acc", 0)
        add_row(d, s, "PTQ",
                item["int8"]["test_accuracy"], item["int8"]["model_size_mb"], base)

    # Stage 4: Progressive
    for item in prog_raw:
        d, s = parse_model_info(item.get("model_name", ""))
        base = baseline_ref.get(f"{d}_{s}", {}).get("acc", 0)
        add_row(d, s, "Progressive",
                item["int8"]["test_accuracy"], item["int8"]["model_size_mb"], base)

    # Stage 5: QAT + KD
    for item in qat_raw:
        d, s = parse_model_info(item["model_name"])
        base = baseline_ref.get(f"{d}_{s}", {}).get("acc", 0)
        add_row(d, s, "QAT + KD",
                item["final_metrics"]["test_accuracy"],
                item["final_metrics"]["model_size_mb"], base)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1 – Three separate panels, one per quantization method
#   Each panel: Baseline ──► Distilled ──► [PTQ | Progressive | QAT+KD]
# ═══════════════════════════════════════════════════════════════════════════
def plot_combined_trajectory(df):
    QUANT_METHODS = ["PTQ", "Progressive", "QAT + KD"]
    # x positions inside each panel
    X = {"Baseline": 0, "Distilled": 1, "Quant": 2}

    fig, axes = plt.subplots(1, 3, figsize=(20, 9), sharey=True)
    fig.suptitle(
        "Accuracy Trajectory: Baseline  →  After Distillation  →  Quantization\n"
        "Three panels = three independent quantization methods applied to the same distilled models",
        fontsize=13, fontweight="bold"
    )

    models = df[["Distill", "Size"]].drop_duplicates().values

    for ax, q_method in zip(axes, QUANT_METHODS):
        q_color = STAGE_COLORS[q_method]

        # ── Shaded phase bands ──────────────────────────────────────
        ax.axvspan(-0.35, 0.5,  alpha=0.07, color="#BDC3C7", zorder=0)
        ax.axvspan(0.5,   1.5,  alpha=0.07, color="#F39C12", zorder=0)
        ax.axvspan(1.5,   2.35, alpha=0.07, color=q_color,   zorder=0)

        # ── Phase header labels ─────────────────────────────────────
        for xv, label, col in [
            (0,   "Baseline",   STAGE_COLORS["Baseline"]),
            (1,   "Distilled",  STAGE_COLORS["Distilled"]),
            (2,   q_method,     q_color),
        ]:
            ax.text(xv, 1.013, label,
                    ha="center", va="bottom",
                    transform=ax.get_xaxis_transform(),
                    fontsize=10, fontweight="bold", color=col)

        ax.set_title(f"Quantization: {q_method}",
                     fontsize=13, fontweight="bold", color=q_color, pad=28)

        for distill, size in models:
            sub   = df[(df["Distill"] == distill) & (df["Size"] == size)]
            color = DISTILL_COLORS.get(distill, "#888")
            mark  = SIZE_MARK[size]
            ls    = SIZE_STYLE[size]

            b_acc    = sub[sub["Stage"] == "Baseline" ]["Acc"].values[0]
            dist_acc = sub[sub["Stage"] == "Distilled"]["Acc"].values[0]
            q_row    = sub[sub["Stage"] == q_method]
            if q_row.empty:
                continue
            q_acc = q_row["Acc"].values[0]

            ys = [b_acc, dist_acc, q_acc]
            xs = [X["Baseline"], X["Distilled"], X["Quant"]]

            # Full line Baseline → Distilled → Quant
            ax.plot(xs, ys,
                    color=color, linewidth=2.5, linestyle=ls,
                    marker=mark, markersize=9,
                    markeredgecolor="white", markeredgewidth=1.5,
                    alpha=0.88, zorder=3)

            # ── Value labels at each point ──────────────────────────
            # Baseline (left)
            ax.text(X["Baseline"] - 0.1, b_acc,
                    f"{b_acc:.1f}%",
                    ha="right", va="center", fontsize=8.5, color=color)

            # KD gain label (above midpoint of first segment)
            kd_delta = dist_acc - b_acc
            sign = "+" if kd_delta >= 0 else ""
            ax.text(0.5, max(b_acc, dist_acc) + 0.6,
                    f"{sign}{kd_delta:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                    color="#27AE60" if kd_delta >= 0 else "#E74C3C",
                    fontweight="bold")

            # Distilled value (above marker)
            ax.text(X["Distilled"], dist_acc + 0.3,
                    f"{dist_acc:.1f}%",
                    ha="center", va="bottom", fontsize=8, color=color)

            # Quant delta vs distilled (right of endpoint)
            delta_q = q_acc - dist_acc
            sign_q  = "+" if delta_q >= 0 else ""
            dcol    = "#27AE60" if delta_q >= 0 else "#E74C3C"
            ax.text(X["Quant"] + 0.08, q_acc,
                    f"{q_acc:.1f}%\n({sign_q}{delta_q:.2f}%)",
                    ha="left", va="center", fontsize=8,
                    color=dcol, fontweight="bold")

        ax.set_xticks([X["Baseline"], X["Distilled"], X["Quant"]])
        ax.set_xticklabels(["Baseline", "Distilled", q_method],
                           fontsize=10, fontweight="bold")
        for tick, col in zip(ax.get_xticklabels(),
                             [STAGE_COLORS["Baseline"],
                              STAGE_COLORS["Distilled"], q_color]):
            tick.set_color(col)

        ax.set_xlabel("")
        ax.set_xlim(-0.45, 2.6)
        ax.grid(True, axis="y", alpha=0.22, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)

    axes[0].set_ylabel("Test Accuracy (%)", fontsize=13,
                       fontweight="bold", labelpad=10)

    # ── Shared legend below all panels ──────────────────────────────
    legend_items = []
    for distill, color in DISTILL_COLORS.items():
        legend_items.append(
            mlines.Line2D([], [], color=color, linewidth=2.5, marker="o",
                          markersize=8, label=f"{distill} — Small")
        )
        legend_items.append(
            mlines.Line2D([], [], color=color, linewidth=2.5, marker="s",
                          markersize=8, linestyle="--", label=f"{distill} — Tiny")
        )

    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.92, edgecolor="#ccc",
               bbox_to_anchor=(0.5, -0.03),
               title="Distillation method  (─ = Small,  -- = Tiny)",
               title_fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    path = OUT_DIR / "combined_trajectory.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2 – Per-model grid (2 × 4), bars + size reduction
# ═══════════════════════════════════════════════════════════════════════════
def plot_per_model_grid(df):
    order = [("Logit", "Small"), ("Logit", "Tiny"),
             ("Attention", "Small"), ("Attention", "Tiny"),
             ("Contrastive", "Small"), ("Contrastive", "Tiny"),
             ("Feature", "Small"), ("Feature", "Tiny")]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(
        "Per-Model Accuracy: Baseline  →  Distilled  →  PTQ  →  Progressive  →  QAT + KD",
        fontsize=15, fontweight="bold"
    )

    stage_colors = [STAGE_COLORS[s] for s in STAGES]

    for ax, (distill, size) in zip(axes.flat, order):
        sub = df[(df["Distill"] == distill) & (df["Size"] == size)].copy()
        sub["x"] = sub["Stage"].map({s: i for i, s in enumerate(STAGES)})
        sub = sub.sort_values("x")

        if sub.empty:
            ax.axis("off")
            continue

        x      = np.arange(len(STAGES))
        w      = 0.55
        accs   = sub["Acc"].tolist()
        sizes  = sub["SizeMB"].tolist()
        color  = DISTILL_COLORS.get(distill, "#888")

        # Accuracy bars
        bars = ax.bar(x, accs, w, color=stage_colors, alpha=0.82,
                      edgecolor="white", linewidth=1.5)
        base_val = sub[sub["Stage"] == "Baseline"]["Acc"].values[0]
        dist_val = sub[sub["Stage"] == "Distilled"]["Acc"].values[0]

        for bar, val, stage in zip(bars, accs, STAGES):
            if stage == "Baseline":
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.15,
                        f"{val:.1f}%", ha="center", va="bottom",
                        fontsize=8, color="#555")
            elif stage == "Distilled":
                delta     = val - base_val
                sign      = "+" if delta >= 0 else ""
                delta_col = "#27AE60" if delta >= 0 else "#E74C3C"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"{val:.1f}%", ha="center", va="bottom",
                        fontsize=8, color="#555")
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.85,
                        f"{sign}{delta:.1f}%", ha="center", va="bottom",
                        fontsize=7.5, color=delta_col, fontweight="bold")
            else:
                # quantization stages: delta vs distilled FP32
                delta     = val - dist_val
                sign      = "+" if delta >= 0 else ""
                delta_col = "#27AE60" if delta >= 0 else "#E74C3C"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"{val:.1f}%", ha="center", va="bottom",
                        fontsize=8, color="#555")
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.85,
                        f"{sign}{delta:.1f}%", ha="center", va="bottom",
                        fontsize=7.5, color=delta_col, fontweight="bold")

        # Model size line on secondary axis
        ax2 = ax.twinx()
        ax2.plot(x, sizes, "D--", color=color, linewidth=1.8,
                 markersize=6, alpha=0.7, label="Size (MB)")
        for xi, si in zip(x, sizes):
            ax2.text(xi + 0.22, si + 0.5, f"{si:.0f}MB",
                     fontsize=7.5, color=color, alpha=0.85)
        ax2.set_ylabel("Size (MB)", color=color, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=color, labelsize=8)

        base_line = sub[sub["Stage"] == "Baseline"]["Acc"].values[0]
        dist_line = sub[sub["Stage"] == "Distilled"]["Acc"].values[0]
        ax.axhline(base_line, color=STAGE_COLORS["Baseline"],
                   linewidth=1.2, linestyle=":", alpha=0.7, label="Baseline")
        ax.axhline(dist_line, color=STAGE_COLORS["Distilled"],
                   linewidth=1.2, linestyle=":", alpha=0.7, label="Distilled")

        ymin = min(accs) - 3
        ymax = max(accs) + 3
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(STAGES, fontsize=8.5)
        ax.set_ylabel("Test Accuracy (%)", fontsize=9)
        ax.set_title(f"{distill}  ({size})", fontsize=11,
                     fontweight="bold", color=color, pad=5)
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")
        ax.spines[["top"]].set_visible(False)

    plt.tight_layout()
    path = OUT_DIR / "per_model_grid.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 – Delta heatmap  (models × quantization methods)
# ═══════════════════════════════════════════════════════════════════════════
def plot_delta_heatmap(df):
    """Two heatmaps: Δ vs Baseline and absolute accuracy — full 5-stage picture."""
    all_stages   = ["Distilled", "PTQ", "Progressive", "QAT + KD"]
    order = [
        ("Logit",       "Small"), ("Logit",       "Tiny"),
        ("Attention",   "Small"), ("Attention",   "Tiny"),
        ("Contrastive", "Small"), ("Contrastive", "Tiny"),
        ("Feature",     "Small"), ("Feature",     "Tiny"),
    ]
    model_labels = [f"{d} ({s})" for d, s in order]

    # Build matrices  (rows=models, cols=stages)
    delta_matrix = np.zeros((len(order), len(all_stages)))
    acc_matrix   = np.zeros_like(delta_matrix)

    for i, (distill, size) in enumerate(order):
        for j, stage in enumerate(all_stages):
            row = df[(df["Distill"] == distill) &
                     (df["Size"]   == size)    &
                     (df["Stage"]  == stage)]
            if not row.empty:
                delta_matrix[i, j] = row["DeltaBase"].values[0]  # Δ vs Baseline
                acc_matrix  [i, j] = row["Acc"].values[0]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             gridspec_kw={"width_ratios": [1, 1]})
    fig.suptitle(
        "Accuracy Change vs No-Distillation Baseline  "
        "(Δ = stage accuracy − baseline accuracy)",
        fontsize=14, fontweight="bold"
    )

    # ── Left: delta heatmap ──────────────────────────────────────────────
    ax = axes[0]
    absmax = max(abs(delta_matrix.min()), abs(delta_matrix.max()), 1.0)
    im = ax.imshow(delta_matrix, cmap="RdYlGn", aspect="auto",
                   vmin=-absmax, vmax=absmax)

    for i in range(len(order)):
        for j in range(len(all_stages)):
            val    = delta_matrix[i, j]
            sign   = "+" if val >= 0 else ""
            tcolor = "black" if abs(val) < absmax * 0.6 else "white"
            ax.text(j, i, f"{sign}{val:.1f}%",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color=tcolor)

    plt.colorbar(im, ax=ax, label="Δ Accuracy vs Baseline (%)", shrink=0.85)
    ax.set_xticks(range(len(all_stages)))
    ax.set_xticklabels(all_stages, fontsize=10, fontweight="bold")
    for tick, stage in zip(ax.get_xticklabels(), all_stages):
        tick.set_color(STAGE_COLORS.get(stage, "#333"))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(model_labels, fontsize=10)
    ax.set_title("Δ Accuracy vs Baseline (no KD)", fontsize=12, fontweight="bold")
    for tick, (d, _) in zip(ax.get_yticklabels(), order):
        tick.set_color(DISTILL_COLORS.get(d, "#333"))

    # ── Right: absolute accuracy heatmap ─────────────────────────────────
    ax2 = axes[1]
    all_stages_full = ["Baseline"] + all_stages
    acc_full = np.zeros((len(order), len(all_stages_full)))
    for i, (distill, size) in enumerate(order):
        for j, stage in enumerate(all_stages_full):
            row = df[(df["Distill"] == distill) &
                     (df["Size"]   == size)    &
                     (df["Stage"]  == stage)]
            if not row.empty:
                acc_full[i, j] = row["Acc"].values[0]

    im2 = ax2.imshow(acc_full, cmap="Blues", aspect="auto",
                     vmin=acc_full.min() - 2, vmax=acc_full.max() + 1)

    for i in range(len(order)):
        for j in range(len(all_stages_full)):
            val = acc_full[i, j]
            ax2.text(j, i, f"{val:.1f}%",
                     ha="center", va="center", fontsize=9.5,
                     fontweight="bold",
                     color="white" if val < acc_full.mean() else "#1a1a2e")

    plt.colorbar(im2, ax=ax2, label="Test Accuracy (%)", shrink=0.85)
    ax2.set_xticks(range(len(all_stages_full)))
    ax2.set_xticklabels(all_stages_full, fontsize=10, fontweight="bold")
    for tick, stage in zip(ax2.get_xticklabels(), all_stages_full):
        tick.set_color(STAGE_COLORS.get(stage, "#333"))
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(model_labels, fontsize=10)
    ax2.set_title("Absolute Accuracy — All Stages", fontsize=12, fontweight="bold")
    for tick, (d, _) in zip(ax2.get_yticklabels(), order):
        tick.set_color(DISTILL_COLORS.get(d, "#333"))

    plt.tight_layout()
    path = OUT_DIR / "delta_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  COMBINED QUANTIZATION ANALYSIS  (all methods × all models)")
    print("=" * 65)

    df = collect_data()

    print("\n[1/3] Accuracy trajectory slope chart...")
    plot_combined_trajectory(df)

    print("\n[2/3] Per-model accuracy + size grid...")
    plot_per_model_grid(df)

    print("\n[3/3] Delta heatmap (change vs FP32)...")
    plot_delta_heatmap(df)

    print(f"\n{'=' * 65}")
    print(f"  ALL PLOTS SAVED TO: {OUT_DIR.absolute()}")
    print(f"{'=' * 65}")

    # Quick summary table
    print("\n  Model                  Baseline  Distilled   PTQ    Prog   QAT+KD   Best")
    print("  " + "-" * 73)
    order = [("Logit",      "Small"), ("Logit",      "Tiny"),
             ("Attention",  "Small"), ("Attention",  "Tiny"),
             ("Contrastive","Small"), ("Contrastive","Tiny"),
             ("Feature",    "Small"), ("Feature",    "Tiny")]

    for distill, size in order:
        accs = {}
        for stage in STAGES:
            row = df[(df["Distill"] == distill) & (df["Size"] == size) &
                     (df["Stage"]  == stage)]
            accs[stage] = row["Acc"].values[0] if not row.empty else float("nan")

        best_q = max(accs["PTQ"], accs["Progressive"], accs["QAT + KD"])
        best_m = [s for s in ["PTQ", "Progressive", "QAT + KD"]
                  if abs(accs[s] - best_q) < 0.001][0]
        print(f"  {distill+' '+size:<22} "
              f"{accs['Baseline']:7.2f}%  "
              f"{accs['Distilled']:7.2f}%  "
              f"{accs['PTQ']:6.2f}%  "
              f"{accs['Progressive']:6.2f}%  "
              f"{accs['QAT + KD']:6.2f}%   "
              f"{best_m}")


if __name__ == "__main__":
    main()
