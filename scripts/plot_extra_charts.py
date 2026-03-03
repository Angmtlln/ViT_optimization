import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── НАСТРОЙКИ ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR  = PROJECT_ROOT / "results"
PLOTS_DIR    = PROJECT_ROOT / "plots"

METHOD_COLORS = {
    "feature":     "#FF6B35",
    "contrastive": "#A020F0",
    "attention":   "#1E88FF",
    "logit":       "#16C60C",
}
METHOD_NAMES = {
    "feature":     "Feature KD",
    "contrastive": "Contrastive KD",
    "attention":   "Attention KD",
    "logit":       "Logit KD",
}
MODEL_COLORS = {"S": "#1E88FF", "T": "#FF6B35"}

FOLDERS = {
    "S": {
        "feature":     "feature_distillation_S",
        "contrastive": "contrastive_distillation_S",
        "attention":   "attention_distillation_S",
        "logit":       "logit_distillation_S",
    },
    "T": {
        "feature":     "feature_distillation_T",
        "contrastive": "contrastive_distillation_T",
        "attention":   "attention_distillation_t",
        "logit":       "logit_distillation_T",
    },
}

BASELINES = {"S": 76.25, "T": 71.88}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#AAAAAA",
    "axes.linewidth": 0.9,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
})


# ── ЗАГРУЗКА ──────────────────────────────────────────────────────────────────
def load_all():
    data = {}
    for mk in ["S", "T"]:
        data[mk] = {}
        for method, folder in FOLDERS[mk].items():
            fp = RESULTS_DIR / folder
            rows = []
            if fp.exists():
                for jf in sorted(fp.glob("*.json")):
                    with open(jf, encoding="utf-8") as f:
                        d = json.load(f)
                    s = jf.stem.lower()
                    rows.append({
                        "stem":        jf.stem,
                        "acc":         d.get("test_accuracy"),
                        "is_baseline": any(x in s for x in ["baseline","no_distill","no_contrast"]),
                        "hp":          d.get("hyperparameters", {}),
                    })
            data[mk][method] = rows
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 1. BAR CHART — Best per method, S vs T
# ══════════════════════════════════════════════════════════════════════════════
def plot_best_bar(data):
    methods = ["feature", "contrastive", "attention", "logit"]
    out_dir = PLOTS_DIR / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {mk: {} for mk in ["S", "T"]}
    for mk in ["S", "T"]:
        for m in methods:
            accs = [e["acc"] for e in data[mk][m]
                    if e["acc"] is not None and not e["is_baseline"]]
            best[mk][m] = max(accs) if accs else 0

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    x = np.arange(len(methods))
    w = 0.30

    for i, mk in enumerate(["S", "T"]):
        vals = [best[mk][m] for m in methods]
        bars = ax.bar(x + i * w, vals, w,
                      color=[METHOD_COLORS[m] for m in methods],
                      alpha=0.75 if mk == "T" else 0.95,
                      edgecolor="white", linewidth=1.2,
                      label=f"Student-{mk}")
        # Baseline пунктир
        bl = BASELINES[mk]
        ax.axhline(bl, color=MODEL_COLORS[mk], ls="--", lw=1.2, alpha=0.6)
        ax.text(x[-1] + w + 0.05, bl + 0.15,
                f"Baseline-{mk} {bl:.1f}%",
                fontsize=7.5, color=MODEL_COLORS[mk], va="bottom")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="#333")

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax.set_ylim(60, 100)
    ax.set_title("Best Test Accuracy per KD Method\nStudent-S vs Student-T",
                 fontsize=12, fontweight="bold", pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(color=MODEL_COLORS["S"], label="Student-S (87.2M)"),
        mpatches.Patch(color=MODEL_COLORS["T"], label="Student-T (69.5M)"),
    ]
    ax.legend(handles=handles, fontsize=9, frameon=True)

    plt.tight_layout()
    out = out_dir / "best_per_method_bar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. IMPROVEMENT OVER BASELINE  (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
def plot_improvement(data):
    methods = ["feature", "contrastive", "attention", "logit"]
    out_dir = PLOTS_DIR / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
    fig.suptitle("Improvement over Baseline for Every Experiment",
                 fontsize=12, fontweight="bold")

    for ax, mk in zip(axes, ["S", "T"]):
        bl = BASELINES[mk]
        all_labels, all_deltas, all_colors = [], [], []

        for m in methods:
            for e in data[mk][m]:
                if e["acc"] is None or e["is_baseline"]:
                    continue
                all_labels.append(e["stem"].replace(f"_{mk.lower()}_best", "")
                                           .replace(f"_{mk.lower()}", "")
                                           .replace("_small", "").replace("_tiny", ""))
                all_deltas.append(e["acc"] - bl)
                all_colors.append(METHOD_COLORS[m])

        # Сортировка по улучшению
        order = np.argsort(all_deltas)[::-1]
        labels  = [all_labels[i]  for i in order]
        deltas  = [all_deltas[i]  for i in order]
        colors  = [all_colors[i]  for i in order]

        y = np.arange(len(labels))
        bars = ax.barh(y, deltas, color=colors, alpha=0.85,
                       edgecolor="white", linewidth=0.8)

        ax.axvline(0, color="#555", lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.2)
        ax.invert_yaxis()
        ax.set_xlabel("Δ Accuracy vs Baseline (%)", fontsize=9)
        ax.set_title(f"Student-{mk}", fontsize=10, fontweight="bold",
                     color=MODEL_COLORS[mk])
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        for bar, val in zip(bars, deltas):
            ax.text(val + (0.1 if val >= 0 else -0.1),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.1f}%",
                    va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=6, color="#222")

    # Легенда
    handles = [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_NAMES[m])
               for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = out_dir / "improvement_over_baseline.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BOX / VIOLIN PLOT — разброс accuracy по методу
# ══════════════════════════════════════════════════════════════════════════════
def plot_violin(data):
    methods = ["feature", "contrastive", "attention", "logit"]
    out_dir = PLOTS_DIR / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="white")
    fig.suptitle("Accuracy Distribution per KD Method\n(all experiments, excl. baseline)",
                 fontsize=12, fontweight="bold")

    for ax, mk in zip(axes, ["S", "T"]):
        plot_data  = []
        positions  = []
        for i, m in enumerate(methods):
            accs = [e["acc"] for e in data[mk][m]
                    if e["acc"] is not None and not e["is_baseline"]]
            if accs:
                plot_data.append(accs)
                positions.append(i)

        if not plot_data:
            continue

        vp = ax.violinplot(plot_data, positions=positions,
                           showmedians=True, showextrema=True,
                           widths=0.65)

        for i, (body, pos) in enumerate(zip(vp["bodies"], positions)):
            m = methods[pos]
            body.set_facecolor(METHOD_COLORS[m])
            body.set_edgecolor(METHOD_COLORS[m])
            body.set_alpha(0.55)

        vp["cmedians"].set_color("#222")
        vp["cmedians"].set_linewidth(2)
        vp["cmins"].set_color("#888")
        vp["cmaxes"].set_color("#888")
        vp["cbars"].set_color("#888")

        # Scatter поверх
        for i, (accs, pos) in enumerate(zip(plot_data, positions)):
            jitter = np.random.uniform(-0.08, 0.08, len(accs))
            ax.scatter([pos + j for j in jitter], accs,
                       color=METHOD_COLORS[methods[pos]],
                       s=28, zorder=3, edgecolors="white", linewidth=0.5,
                       alpha=0.9)

        # Baseline
        bl = BASELINES[mk]
        ax.axhline(bl, color="#EE3333", ls="--", lw=1.3, alpha=0.7)
        ax.text(len(methods) - 0.4, bl + 0.2,
                f"Baseline {bl:.1f}%",
                fontsize=7.5, color="#EE3333", va="bottom", ha="right")

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=9)
        ax.set_ylabel("Test Accuracy (%)", fontsize=9)
        ax.set_title(f"Student-{mk}", fontsize=10, fontweight="bold",
                     color=MODEL_COLORS[mk])
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    out = out_dir / "violin_distribution.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. RADAR CHART — лучшие результаты на одном полотне
# ══════════════════════════════════════════════════════════════════════════════
def plot_radar(data):
    methods = ["feature", "contrastive", "attention", "logit"]
    out_dir = PLOTS_DIR / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [METHOD_NAMES[m] for m in methods]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # замкнуть

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True),
                           facecolor="white")

    for mk in ["S", "T"]:
        vals = []
        for m in methods:
            accs = [e["acc"] for e in data[mk][m]
                    if e["acc"] is not None and not e["is_baseline"]]
            vals.append(max(accs) if accs else 0)
        vals += vals[:1]

        ax.plot(angles, vals, "o-", lw=2, color=MODEL_COLORS[mk],
                label=f"Student-{mk}", markersize=7)
        ax.fill(angles, vals, alpha=0.12, color=MODEL_COLORS[mk])

        for ang, val, lbl in zip(angles[:-1], vals[:-1], labels):
            ax.annotate(f"{val:.1f}%",
                        xy=(ang, val),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color=MODEL_COLORS[mk])

    # Baseline
    bl_avg = (BASELINES["S"] + BASELINES["T"]) / 2
    bl_vals = [bl_avg] * N + [bl_avg]
    ax.plot(angles, bl_vals, "--", lw=1.2, color="#999", alpha=0.6,
            label=f"Avg Baseline ({bl_avg:.1f}%)")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(60, 100)
    ax.set_yticks([65, 70, 75, 80, 85, 90, 95, 100])
    ax.set_yticklabels(["65%","70%","75%","80%","85%","90%","95%","100%"],
                       fontsize=7, color="#888")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="-", alpha=0.25)
    ax.set_title("Best Accuracy Radar\nStudent-S vs Student-T",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), fontsize=9)

    plt.tight_layout()
    out = out_dir / "radar_best.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. HEATMAP — alpha × Temperature для Logit и Feature
# ══════════════════════════════════════════════════════════════════════════════
def plot_heatmap(data):
    out_dir = PLOTS_DIR / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    fig.suptitle("Hyperparameter Heatmap (Test Accuracy %)",
                 fontsize=13, fontweight="bold")

    configs = [
        ("S", "logit",   "temperature", "kd_alpha",  "T",    "α"),
        ("T", "logit",   "temperature", "kd_alpha",  "T",    "α"),
        ("S", "feature", "temperature", "alpha",     "T",    "α"),
        ("T", "feature", "temperature", "alpha",     "T",    "α"),
    ]

    for ax, (mk, method, xkey, ykey, xlabel, ylabel) in zip(axes.flat, configs):
        rows = [e for e in data[mk][method]
                if e["acc"] is not None and not e["is_baseline"]]

        xs = sorted(set(e["hp"].get(xkey) for e in rows if e["hp"].get(xkey) is not None))
        ys = sorted(set(e["hp"].get(ykey) for e in rows if e["hp"].get(ykey) is not None))

        if not xs or not ys:
            ax.set_visible(False)
            continue

        grid = np.full((len(ys), len(xs)), np.nan)
        for e in rows:
            x_val = e["hp"].get(xkey)
            y_val = e["hp"].get(ykey)
            if x_val in xs and y_val in ys:
                xi = xs.index(x_val)
                yi = ys.index(y_val)
                grid[yi, xi] = e["acc"]

        color = METHOD_COLORS[method]
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "custom", ["#FFFFFF", color], N=256)

        im = ax.imshow(grid, aspect="auto", cmap=cmap,
                       vmin=max(60, np.nanmin(grid) - 2),
                       vmax=min(100, np.nanmax(grid) + 1))

        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(v) for v in xs], fontsize=9)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels([str(v) for v in ys], fontsize=9)
        ax.set_xlabel(f"{xlabel} ({xkey})", fontsize=9)
        ax.set_ylabel(f"{ylabel} ({ykey})", fontsize=9)
        ax.set_title(f"Student-{mk} · {METHOD_NAMES[method]}",
                     fontsize=10, fontweight="bold", color=color)

        for yi in range(len(ys)):
            for xi in range(len(xs)):
                v = grid[yi, xi]
                if not np.isnan(v):
                    ax.text(xi, yi, f"{v:.1f}", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if v > (np.nanmin(grid) + np.nanmax(grid)) / 2 else "#333")

        plt.colorbar(im, ax=ax, shrink=0.8, label="Test Acc (%)")

    plt.tight_layout()
    out = out_dir / "heatmap_hyperparams.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. S vs T SCATTER — каждая точка = один эксперимент
# ══════════════════════════════════════════════════════════════════════════════
def plot_s_vs_t_scatter(data):
    out_dir = PLOTS_DIR / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = ["feature", "contrastive", "attention", "logit"]

    # Сопоставляем общие stem-имена между S и T
    pairs = {}
    for m in methods:
        stems_s = {e["stem"].replace("_small", "").replace("_S", ""):
                   e["acc"] for e in data["S"][m] if e["acc"] is not None}
        stems_t = {e["stem"].replace("_tiny",  "").replace("_T", "").replace("_t", ""):
                   e["acc"] for e in data["T"][m] if e["acc"] is not None}
        common = set(stems_s) & set(stems_t)
        for key in common:
            pairs.setdefault(m, []).append((stems_s[key], stems_t[key]))

    fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")

    lo, hi = 60, 100
    ax.plot([lo, hi], [lo, hi], "--", color="#AAAAAA", lw=1.2, label="S = T")
    ax.fill_between([lo, hi], [lo, hi], [hi, hi],
                    alpha=0.04, color="#1E88FF", label="S wins")
    ax.fill_between([lo, hi], [lo, lo], [lo, hi],
                    alpha=0.04, color="#FF6B35", label="T wins")

    for m in methods:
        pts = pairs.get(m, [])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys,
                   color=METHOD_COLORS[m], s=70, label=METHOD_NAMES[m],
                   edgecolors="white", linewidth=0.8, zorder=3, alpha=0.9)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Student-S Test Accuracy (%)", fontsize=10)
    ax.set_ylabel("Student-T Test Accuracy (%)", fontsize=10)
    ax.set_title("Student-S vs Student-T\n(each point = one experiment)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, frameon=True)

    plt.tight_layout()
    out = out_dir / "scatter_S_vs_T.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    (PLOTS_DIR / "combined").mkdir(parents=True, exist_ok=True)

    print("📂 Загрузка данных...")
    data = load_all()

    print("\n📊 1. Bar chart (best per method)...")
    plot_best_bar(data)

    print("📊 2. Improvement over baseline...")
    plot_improvement(data)

    print("📊 3. Violin distribution...")
    plot_violin(data)

    print("📊 4. Radar chart...")
    plot_radar(data)

    print("📊 5. Heatmap hyperparameters...")
    plot_heatmap(data)

    print("📊 6. S vs T scatter...")
    plot_s_vs_t_scatter(data)

    print(f"\n✅ Все графики сохранены в: {PLOTS_DIR / 'combined'}")


if __name__ == "__main__":
    main()