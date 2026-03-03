import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

# ── НАСТРОЙКИ ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(r"c:\Users\amirn\OneDrive\Рабочий стол\ViT_opti\ViT_optimization\results")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "plots" / "kd_comparison"

METHOD_COLORS = {
    "feature":     "#FF6600",
    "contrastive": "#9B30FF",
    "attention":   "#1565C0",
    "logit":       "#2ECC40",
}

METHOD_MARKERS = {
    "feature":     "^",
    "contrastive": "D",
    "attention":   "o",
    "logit":       "s",
}

METHOD_NAMES = {
    "feature":     "Feature KD",
    "contrastive": "Contrastive KD",
    "attention":   "Attention KD",
    "logit":       "Logit KD",
}

# Чередующийся фон секций
SECTION_BG = ["#F0F4FF", "#F5F5F5", "#EFF8F0", "#FFF8F0"]

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

# ── МЕТКИ ─────────────────────────────────────────────────────────────────────
def build_label(method, hp, stem):
    stem_l = stem.lower()
    if any(x in stem_l for x in ["baseline", "no_distill", "no_contrast"]):
        return "Baseline (no KD)"

    if method == "logit":
        T     = hp.get("temperature", "?")
        alpha = hp.get("kd_alpha", "?")
        return f"T={T}, α={alpha}"

    if method == "feature":
        alpha = hp.get("alpha", "?")
        T     = hp.get("temperature", "?")
        proj  = ", proj" if hp.get("use_projection", False) else ""
        return f"α={alpha}, T={T}{proj}"

    if method == "attention":
        if "lr2e-4" in stem_l: return "lr=2e-4"
        if "lr5e-5" in stem_l: return "lr=5e-5"
        alpha = hp.get("alpha", "?")
        loss  = hp.get("attention_loss_type", "mse")
        return f"α={alpha}, {loss}"

    if method == "contrastive":
        alpha = hp.get("alpha_contrast", "?")
        if "tkd2"  in stem_l: return f"α={alpha}, T_kd=2"
        if "tkd6"  in stem_l: return f"α={alpha}, T_kd=6"
        if "t0.05" in stem_l: return f"α={alpha}, T_c=0.05"
        if "t0.10" in stem_l: return f"α={alpha}, T_c=0.10"
        T_kd = hp.get("temperature_kd", "?")
        T_c  = hp.get("temperature_contrast", "?")
        return f"α={alpha}, T_kd={T_kd}, T_c={T_c}"

    return stem


# ── ЗАГРУЗКА ──────────────────────────────────────────────────────────────────
def load_method(model_key, method):
    folder_path = RESULTS_DIR / FOLDERS[model_key][method]
    if not folder_path.exists():
        print(f"⚠️  Не найдена: {folder_path}")
        return []
    entries = []
    for jf in sorted(folder_path.glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            d = json.load(f)
        is_bl = any(x in jf.stem.lower() for x in ["baseline", "no_distill", "no_contrast"])
        hp    = d.get("hyperparameters", {})
        entries.append({
            "label":       build_label(method, hp, jf.stem),
            "test_acc":    d.get("test_accuracy"),
            "is_baseline": is_bl,
        })
    baselines = [e for e in entries if     e["is_baseline"]]
    others    = sorted(
        [e for e in entries if not e["is_baseline"]],
        key=lambda x: x["test_acc"] or 0,
        reverse=True
    )
    return others + baselines


# ── РИСОВАНИЕ ОДНОЙ БОЛЬШОЙ КОЛОНКИ (как в оригинале) ─────────────────────────
def draw_column(ax, all_entries, x_min=65, x_max=100):
    """
    all_entries: dict {method: [entries]}
    Все методы рисуются в одном ax, разделённые секциями с фоном.
    Название метода — справа за пределами графика.
    """
    method_order = ["feature", "contrastive", "attention", "logit"]

    # Собираем все строки подряд, запоминаем диапазоны секций
    rows        = []   # [{label, test_acc, is_baseline, method}, ...]
    section_map = {}   # method -> (y_start, y_end)

    for method in method_order:
        entries = all_entries.get(method, [])
        y_start = len(rows)
        for e in entries:
            if e["test_acc"] is not None:
                rows.append({**e, "method": method})
        y_end = len(rows) - 1
        section_map[method] = (y_start, y_end)

    total = len(rows)
    if total == 0:
        return

    # ── Фоновые полосы секций ─────────────────────────────────────────────────
    for idx, method in enumerate(method_order):
        y0, y1 = section_map[method]
        if y0 > y1:
            continue
        ax.axhspan(y0 - 0.5, y1 + 0.5,
                   facecolor=SECTION_BG[idx],
                   alpha=0.6, zorder=0)

    # ── Точки и метки ─────────────────────────────────────────────────────────
    # Узнаём лучший в каждой секции
    best_in_section = {}
    for method in method_order:
        accs = [r["test_acc"] for r in rows
                if r["method"] == method and not r["is_baseline"]]
        best_in_section[method] = max(accs) if accs else None

    y_tick_positions = list(range(total))
    y_tick_labels    = [r["label"] for r in rows]

    for y_i, row in enumerate(rows):
        method   = row["method"]
        acc      = row["test_acc"]
        color    = METHOD_COLORS[method]
        marker   = METHOD_MARKERS[method]
        is_best  = (not row["is_baseline"] and acc == best_in_section[method])

        # Жёлтый фон для лучшего
        if is_best:
            ax.axhspan(y_i - 0.42, y_i + 0.42,
                       facecolor="#FFFACD", alpha=1.0, zorder=1)

        if row["is_baseline"]:
            # Пунктирная вертикаль + пустой маркер
            ax.axvline(x=acc, color=color, linestyle="--",
                       linewidth=1.0, alpha=0.65, zorder=2)
            ax.plot(acc, y_i,
                    marker=marker, color=color,
                    markersize=8, markerfacecolor="white",
                    markeredgewidth=1.8, zorder=5)
        else:
            ax.plot(acc, y_i,
                    marker=marker, color=color,
                    markersize=7, zorder=5)

            # Позиция метки: справа или влево от точки
            # Проверяем, не налезает ли на соседние точки в той же секции
            ha       = "left"
            offset_x = 5
            if acc > x_max - 5:
                ha       = "right"
                offset_x = -5

            label_text = f"{acc:.2f}%"
            ax.annotate(label_text,
                        xy=(acc, y_i),
                        xytext=(offset_x, 0),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                        fontweight="bold" if is_best else "normal",
                        va="center", ha=ha,
                        zorder=6)

            if is_best:
                # "★ BEST" — чуть правее/левее метки
                sign = 1 if ha == "left" else -1
                ax.annotate("★ BEST",
                            xy=(acc, y_i),
                            xytext=(offset_x + sign * 42, 0),
                            textcoords="offset points",
                            fontsize=6.5,
                            color="#B8860B",
                            fontweight="bold",
                            va="center", ha=ha,
                            zorder=6)

    # ── Названия методов справа ───────────────────────────────────────────────
    for idx, method in enumerate(method_order):
        y0, y1 = section_map[method]
        if y0 > y1:
            continue
        mid_y = (y0 + y1) / 2.0
        ax.text(x_max + 0.4, mid_y,
                METHOD_NAMES[method],
                fontsize=8, fontweight="bold",
                color=METHOD_COLORS[method],
                va="center", ha="left",
                clip_on=False)

    # ── Оси ───────────────────────────────────────────────────────────────────
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, fontsize=6.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.7, total - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Test Accuracy (%)", fontsize=8.5)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", pad=3)
    ax.set_facecolor("white")

    # Тонкая серая рамка
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(0.8)


def save_single_method_plot(model_key, method, entries, model_title):
    """Сохраняет отдельный график для (модель, метод)."""
    # Высота под число экспериментов, чтобы подписи не слипались
    n = max(1, len([e for e in entries if e.get("test_acc") is not None]))
    fig_h = max(5.5, n * 0.55 + 1.8)

    fig, ax = plt.subplots(1, 1, figsize=(10.5, fig_h), facecolor="white")
    fig.subplots_adjust(left=0.20, right=0.86, top=0.88, bottom=0.12)

    # Используем существующую отрисовку, но только для одного метода
    draw_column(ax, {method: entries}, x_min=65, x_max=100)

    ax.set_title(f"{model_title}\n{METHOD_NAMES[method]}", fontsize=10, fontweight="bold", pad=10)

    out_file = OUT_DIR / f"{model_key}_{method}_kd.png"
    plt.savefig(out_file, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅  Сохранено: {out_file}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    method_order = ["feature", "contrastive", "attention", "logit"]
    model_keys   = ["S", "T"]

    model_titles = {
        "S": "Student-S · PE-Core-S16-384\n(87.2M параметров)",
        "T": "Student-T · PE-Core-T16-384\n(69.5M параметров)",
    }

    # Создаём папку outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Загрузка данных
    all_data = {}
    for mk in model_keys:
        all_data[mk] = {}
        for method in method_order:
            all_data[mk][method] = load_method(mk, method)

    # Считаем общее число строк на колонку
    def count_rows(mk):
        return sum(
            sum(1 for e in all_data[mk][m] if e["test_acc"] is not None)
            for m in method_order
        )

    n_rows_S = count_rows("S")
    n_rows_T = count_rows("T")
    n_rows   = max(n_rows_S, n_rows_T)

    ROW_H   = 0.32
    fig_h   = n_rows * ROW_H + 3.0
    fig_w   = 24

    # 1) Текущий общий график
    fig, (ax_s, ax_t) = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        facecolor="white"
    )
    fig.subplots_adjust(
        left=0.10, right=0.88,
        top=0.93,  bottom=0.07,
        wspace=0.60
    )

    fig.suptitle(
        "Полная сравнительная оценка методов Knowledge Distillation\n"
        "Датасет: Kvasir-v2 (8 классов)  ·  Учитель: PE-Core-L14-336 (671M params)",
        fontsize=12, fontweight="bold", y=0.975
    )

    ax_s.set_title(model_titles["S"], fontsize=10, fontweight="bold", pad=10)
    ax_t.set_title(model_titles["T"], fontsize=10, fontweight="bold", pad=10)

    draw_column(ax_s, all_data["S"])
    draw_column(ax_t, all_data["T"])

    legend_elems = []
    for m in method_order:
        legend_elems.append(
            mlines.Line2D([], [],
                          color=METHOD_COLORS[m],
                          marker=METHOD_MARKERS[m],
                          markersize=8, linestyle="None",
                          label=METHOD_NAMES[m])
        )
    legend_elems.append(
        mlines.Line2D([], [],
                      color="gray", marker="o",
                      markersize=8, linestyle="--",
                      markerfacecolor="white", markeredgewidth=1.6,
                      label="Baseline (no KD) — вертикаль")
    )
    legend_elems.append(
        mpatches.Patch(facecolor="#FFFACD", edgecolor="#B8860B",
                       label="Лучший результат в группе")
    )

    fig.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=len(legend_elems),
        fontsize=8,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.005)
    )

    full_out = OUT_DIR / "kd_comparison_chart.png"
    plt.savefig(full_out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅  Сохранено: {full_out}")

    # 2) 8 отдельных графиков (S/T × 4 метода)
    for mk in model_keys:
        for method in method_order:
            save_single_method_plot(
                model_key=mk,
                method=method,
                entries=all_data[mk][method],
                model_title=model_titles[mk].replace("\n", " ")
            )

    print(f"\nГотово. Все графики в: {OUT_DIR}")


if __name__ == "__main__":
    main()