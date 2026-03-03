"""
Quantization experiments analysis: Before vs After comparison
Methods: PTQ, Progressive Quantization, QAT + KD
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

BASE_DIR = Path(__file__).parent
PTQ_DIR  = BASE_DIR / "Post-Training Quantization (PTQ)" / "results"
PROG_DIR = BASE_DIR / "Progressive Quantization" / "results"
QAT_DIR  = BASE_DIR / "Quantization-Aware Distillation (QAT + KD)" / "results"

OUTPUT_DIR = BASE_DIR / "../plots/quantization_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

COLOR_FP32    = "#E8B4B8"   # pink  - FP32
COLOR_QUANT   = "#6BB1E1"   # blue  - quantized
COLOR_BETTER  = "#2ECC71"   # green - improvement
COLOR_WORSE   = "#E74C3C"   # red   - degradation

METHOD_COLORS = {
    "PTQ":         "#E74C3C",
    "Progressive": "#3498DB",
    "QAT + KD":    "#2ECC71",
}


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_model_info(model_name):
    if "attention"    in model_name: distill = "Attention"
    elif "contrastive" in model_name: distill = "Contrastive"
    elif "feature"    in model_name: distill = "Feature"
    elif "logit"      in model_name: distill = "Logit"
    else:                             distill = "Unknown"
    size = "Small" if "small" in model_name else ("Tiny" if "tiny" in model_name else "?")
    return distill, size


def collect_data():
    """Load all results into a single DataFrame."""
    ptq_data  = load_json(PTQ_DIR  / "ptq_summary.json")
    prog_data = load_json(PROG_DIR / "progressive_summary.json")
    qat_data  = load_json(QAT_DIR  / "qat_kd_summary.json")

    rows = []

    # --- PTQ ---
    for item in ptq_data:
        distill, model_sz = parse_model_info(item["model_name"])
        rows.append({
            "Method":      "PTQ",
            "Distillation": distill,
            "Model":       model_sz,
            "Label":       f"{distill}\n({model_sz})",
            "fp32_acc":    item["fp32"]["test_accuracy"],
            "q_acc":       item["int8"]["test_accuracy"],
            "fp32_size":   item["fp32"]["model_size_mb"],
            "q_size":      item["int8"]["model_size_mb"],
            "fp32_ms":     item["fp32"]["inference_time_ms"],
            "q_ms":        item["int8"]["inference_time_ms"],
        })

    # FP32 reference cache from PTQ
    fp32_ref = {}
    for item in ptq_data:
        d, s = parse_model_info(item["model_name"])
        fp32_ref[f"{d}_{s}"] = (
            item["fp32"]["test_accuracy"],
            item["fp32"]["model_size_mb"],
            item["fp32"]["inference_time_ms"],
        )

    # --- Progressive ---
    for item in prog_data:
        distill, model_sz = parse_model_info(item.get("model_name", ""))
        key                = f"{distill}_{model_sz}"
        fp32_acc_r, fp32_sz_r, fp32_ms_r = fp32_ref.get(key, (None, None, None))
        rows.append({
            "Method":      "Progressive",
            "Distillation": distill,
            "Model":       model_sz,
            "Label":       f"{distill}\n({model_sz})",
            "fp32_acc":    item["fp32"]["test_accuracy"],
            "q_acc":       item["int8"]["test_accuracy"],
            "fp32_size":   item["fp32"]["model_size_mb"],
            "q_size":      item["int8"]["model_size_mb"],
            "fp32_ms":     fp32_ms_r,
            "q_ms":        None,
        })

    # --- QAT + KD ---
    for item in qat_data:
        distill, model_sz = parse_model_info(item["model_name"])
        key                = f"{distill}_{model_sz}"
        fp32_acc_r, fp32_sz_r, fp32_ms_r = fp32_ref.get(key, (None, None, None))
        rows.append({
            "Method":      "QAT + KD",
            "Distillation": distill,
            "Model":       model_sz,
            "Label":       f"{distill}\n({model_sz})",
            "fp32_acc":    fp32_acc_r,
            "q_acc":       item["final_metrics"]["test_accuracy"],
            "fp32_size":   fp32_sz_r,
            "q_size":      item["final_metrics"]["model_size_mb"],
            "fp32_ms":     fp32_ms_r,
            "q_ms":        None,
        })

    df = pd.DataFrame(rows)
    df["delta_acc"]          = df["q_acc"]  - df["fp32_acc"]
    df["delta_size"]         = df["q_size"] - df["fp32_size"]
    df["size_reduction_pct"] = (1 - df["q_size"] / df["fp32_size"]) * 100
    return df


# ---------------------------------------------------------------------------
# PLOT 1: Accuracy arrow chart  FP32 -> INT8
# ---------------------------------------------------------------------------
def plot_before_after_accuracy(df):
    methods = ["PTQ", "Progressive", "QAT + KD"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=False)
    fig.suptitle("Test Accuracy: FP32  ->  INT8  (before vs after quantization)",
                 fontsize=16, fontweight="bold", y=1.01)

    for ax, method in zip(axes, methods):
        sub    = df[df["Method"] == method].copy().reset_index(drop=True)
        labels = sub["Label"].tolist()
        y      = np.arange(len(labels))

        ax.set_title(method, fontsize=14, fontweight="bold",
                     color=METHOD_COLORS[method], pad=10)

        for i, row in sub.iterrows():
            fp    = row["fp32_acc"]
            q     = row["q_acc"]
            delta = row["delta_acc"]
            color = COLOR_BETTER if delta >= 0 else COLOR_WORSE

            # Arrow from FP32 to quantized
            ax.annotate("", xy=(q, i), xytext=(fp, i),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=2.5, mutation_scale=18))
            # Dots
            ax.scatter(fp, i, color=COLOR_FP32,  s=120, zorder=5,
                       edgecolors="white", linewidths=1.5)
            ax.scatter(q,  i, color=COLOR_QUANT, s=120, zorder=5,
                       edgecolors="white", linewidths=1.5)
            # Value labels
            ax.text(fp - 0.3, i + 0.28, f"{fp:.1f}%",
                    ha="right", va="bottom", fontsize=9, color="#555")
            ax.text(q  + 0.3, i + 0.28, f"{q:.1f}%",
                    ha="left",  va="bottom", fontsize=9, color="#555")
            # Delta annotation
            sign = "+" if delta >= 0 else ""
            ax.text((fp + q) / 2, i - 0.28, f"{sign}{delta:.2f}%",
                    ha="center", va="top", fontsize=9,
                    color=color, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Test Accuracy (%)", fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        ax.set_xlim(sub[["fp32_acc", "q_acc"]].min().min() - 3,
                    sub[["fp32_acc", "q_acc"]].max().max() + 3)
        ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=COLOR_FP32,   label="FP32 (before)"),
        mpatches.Patch(color=COLOR_QUANT,  label="INT8 (after)"),
        mpatches.Patch(color=COLOR_BETTER, label="Improvement"),
        mpatches.Patch(color=COLOR_WORSE,  label="Degradation"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

    plt.tight_layout()
    path = OUTPUT_DIR / "before_after_accuracy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close()


# ---------------------------------------------------------------------------
# PLOT 2: Model size arrow chart  FP32 -> INT8
# ---------------------------------------------------------------------------
def plot_before_after_size(df):
    methods = ["PTQ", "Progressive", "QAT + KD"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=False)
    fig.suptitle("Model Size: FP32  ->  INT8  (before vs after quantization)",
                 fontsize=16, fontweight="bold", y=1.01)

    for ax, method in zip(axes, methods):
        sub    = df[df["Method"] == method].copy().reset_index(drop=True)
        labels = sub["Label"].tolist()
        y      = np.arange(len(labels))

        ax.set_title(method, fontsize=14, fontweight="bold",
                     color=METHOD_COLORS[method], pad=10)

        for i, row in sub.iterrows():
            fp  = row["fp32_size"]
            q   = row["q_size"]
            pct = row["size_reduction_pct"]

            # Arrow (always green - size always decreases)
            ax.annotate("", xy=(q, i), xytext=(fp, i),
                        arrowprops=dict(arrowstyle="-|>", color=COLOR_BETTER,
                                        lw=2.5, mutation_scale=18))
            ax.scatter(fp, i, color=COLOR_FP32,  s=120, zorder=5,
                       edgecolors="white", linewidths=1.5)
            ax.scatter(q,  i, color=COLOR_QUANT, s=120, zorder=5,
                       edgecolors="white", linewidths=1.5)

            ax.text(fp + 2, i + 0.28, f"{fp:.0f} MB",
                    ha="left",  va="bottom", fontsize=9, color="#555")
            ax.text(q  - 2, i + 0.28, f"{q:.0f} MB",
                    ha="right", va="bottom", fontsize=9, color="#555")
            ax.text((fp + q) / 2, i - 0.28, f"-{pct:.1f}%",
                    ha="center", va="top", fontsize=9,
                    color=COLOR_BETTER, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Model Size (MB)", fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        ax.set_xlim(0, sub["fp32_size"].max() * 1.15)
        ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=COLOR_FP32,   label="FP32 (before)"),
        mpatches.Patch(color=COLOR_QUANT,  label="INT8 (after)"),
        mpatches.Patch(color=COLOR_BETTER, label="Size reduction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

    plt.tight_layout()
    path = OUTPUT_DIR / "before_after_size.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close()


# ---------------------------------------------------------------------------
# PLOT 3: Summary dashboard - accuracy delta + size reduction per method
# ---------------------------------------------------------------------------
def plot_summary_dashboard(df):
    methods = ["PTQ", "Progressive", "QAT + KD"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.suptitle("Summary Dashboard: Accuracy Change vs Size Reduction",
                 fontsize=16, fontweight="bold")

    for ax, method in zip(axes, methods):
        sub    = df[df["Method"] == method].copy()
        labels = sub["Label"].tolist()
        x      = np.arange(len(labels))
        w      = 0.35

        # --- Accuracy delta bars ---
        deltas     = sub["delta_acc"].values
        bar_colors = [COLOR_BETTER if v >= 0 else COLOR_WORSE for v in deltas]
        b1 = ax.bar(x - w/2, deltas, w, color=bar_colors,
                    alpha=0.85, edgecolor="white", linewidth=1.5)
        for bar, val in zip(b1, deltas):
            sign = "+" if val >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.05 if val >= 0 else -0.05),
                    f"{sign}{val:.2f}%",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=8.5, fontweight="bold")

        # --- Size reduction bars (secondary axis) ---
        ax2   = ax.twinx()
        sizes = sub["size_reduction_pct"].values
        b2    = ax2.bar(x + w/2, sizes, w, color=COLOR_QUANT,
                        alpha=0.7, edgecolor="white", linewidth=1.5)
        for bar, val in zip(b2, sizes):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     f"-{val:.1f}%",
                     ha="center", va="bottom", fontsize=8.5,
                     color=COLOR_QUANT, fontweight="bold")

        ax.set_title(method, fontsize=14, fontweight="bold",
                     color=METHOD_COLORS[method])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Delta Accuracy (%)", color=COLOR_WORSE, fontweight="bold")
        ax2.set_ylabel("Size Reduction (%)",  color=COLOR_QUANT, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.tick_params(axis="y", labelcolor=COLOR_WORSE)
        ax2.tick_params(axis="y", labelcolor=COLOR_QUANT)
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")
        ax.spines[["top"]].set_visible(False)

    h1 = mpatches.Patch(color=COLOR_BETTER, label="Accuracy delta (+)")
    h2 = mpatches.Patch(color=COLOR_WORSE,  label="Accuracy delta (-)")
    h3 = mpatches.Patch(color=COLOR_QUANT,  label="Size reduction")
    fig.legend(handles=[h1, h2, h3], loc="lower center",
               ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    path = OUTPUT_DIR / "summary_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close()


# ---------------------------------------------------------------------------
# PLOT 4: Progressive quantization - step-by-step details
# ---------------------------------------------------------------------------
def plot_progressive_steps(df_all):
    prog_data = load_json(PROG_DIR / "progressive_summary.json")

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Progressive Quantization: Accuracy and Compression per Step",
                 fontsize=15, fontweight="bold")

    for idx, item in enumerate(prog_data):
        if idx >= 8:
            break
        ax = axes[idx // 4][idx % 4]

        if "history" not in item:
            ax.axis("off")
            continue

        history  = item["history"]
        steps    = [h["step"] for h in history]
        acc_aft  = [h["val_acc_after_ft"] for h in history]
        comp     = [h["compression_ratio"] for h in history]
        fp32_acc = item["fp32"]["val_accuracy"]

        ax.axhline(fp32_acc, color=COLOR_FP32, linewidth=2,
                   linestyle="--", label=f"FP32 ({fp32_acc:.1f}%)")
        ax.plot(steps, acc_aft, "o-", color=COLOR_QUANT,
                linewidth=2.5, markersize=8, label="INT8 (val)")
        ax.fill_between(steps, fp32_acc, acc_aft,
                        where=[a < fp32_acc for a in acc_aft],
                        alpha=0.15, color=COLOR_WORSE,  label="Loss vs FP32")
        ax.fill_between(steps, fp32_acc, acc_aft,
                        where=[a >= fp32_acc for a in acc_aft],
                        alpha=0.15, color=COLOR_BETTER, label="Gain vs FP32")

        ax2 = ax.twinx()
        ax2.plot(steps, comp, "s--", color="#E67E22",
                 linewidth=1.5, markersize=6, alpha=0.8)
        ax2.set_ylabel("Compression", color="#E67E22", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#E67E22", labelsize=8)

        distill, sz = parse_model_info(item.get("model_name", ""))
        ax.set_title(f"{distill} - {sz}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Quantization Step", fontsize=9)
        ax.set_ylabel("Val Accuracy (%)",  fontsize=9)
        ax.set_xticks(steps)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=7, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "progressive_steps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close()


# ---------------------------------------------------------------------------
# PLOT 5: QAT + KD learning curves
# ---------------------------------------------------------------------------
def plot_qat_curves():
    qat_data = load_json(QAT_DIR / "qat_kd_summary.json")

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("QAT + KD: Learning Curves (Accuracy per Epoch)",
                 fontsize=15, fontweight="bold")

    for idx, item in enumerate(qat_data[:8]):
        ax     = axes[idx // 4][idx % 4]
        h      = item["training_history"]
        epochs = range(1, len(h["val_acc"]) + 1)

        ax.plot(epochs, h["val_acc"],  "o-", color=COLOR_QUANT,
                linewidth=2, markersize=5, label="Val")
        ax.plot(epochs, h["test_acc"], "s-", color=METHOD_COLORS["PTQ"],
                linewidth=2, markersize=5, label="Test", alpha=0.8)

        best_val  = h["best_val_acc"]
        best_test = h["best_test_acc"]
        ax.axhline(best_val,  color=COLOR_QUANT,          linewidth=1, linestyle="--", alpha=0.5)
        ax.axhline(best_test, color=METHOD_COLORS["PTQ"], linewidth=1, linestyle="--", alpha=0.5)
        ax.text(len(epochs), best_val  + 0.3, f"max {best_val:.1f}%",
                ha="right", fontsize=8, color=COLOR_QUANT,          fontweight="bold")
        ax.text(len(epochs), best_test - 0.8, f"max {best_test:.1f}%",
                ha="right", fontsize=8, color=METHOD_COLORS["PTQ"], fontweight="bold")

        distill, sz = parse_model_info(item["model_name"])
        ax.set_title(f"{distill} - {sz}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Epoch",         fontsize=9)
        ax.set_ylabel("Accuracy (%)",  fontsize=9)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "qat_learning_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  QUANTIZATION ANALYSIS - BEFORE vs AFTER PLOTS")
    print("=" * 60)

    df = collect_data()

    print("\n[1/5] Accuracy arrow chart (before -> after)...")
    plot_before_after_accuracy(df)

    print("\n[2/5] Model size arrow chart (before -> after)...")
    plot_before_after_size(df)

    print("\n[3/5] Summary dashboard (accuracy delta + size reduction)...")
    plot_summary_dashboard(df)

    print("\n[4/5] Progressive quantization step details...")
    plot_progressive_steps(df)

    print("\n[5/5] QAT + KD learning curves...")
    plot_qat_curves()

    print(f"\n{'=' * 60}")
    print(f"  ALL PLOTS SAVED TO: {OUTPUT_DIR.absolute()}")
    print(f"{'=' * 60}")

    print("\nSUMMARY (averages across models):")
    for method in ["PTQ", "Progressive", "QAT + KD"]:
        sub        = df[df["Method"] == method]
        avg_delta  = sub["delta_acc"].mean()
        avg_shrink = sub["size_reduction_pct"].mean()
        sign       = "+" if avg_delta >= 0 else ""
        print(f"  {method:15s}  accuracy {sign}{avg_delta:.2f}%   model size -{avg_shrink:.1f}%")


if __name__ == "__main__":
    main()
