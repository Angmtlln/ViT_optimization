import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# ==============================
# STYLE
# ==============================

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11

# ==============================
# LOADING
# ==============================

def load_results(results_dir='../results/contrastive_distillation_S'):
    results_dir = Path(results_dir)
    results = {}

    json_files = sorted(results_dir.glob('*.json'))
    print(f'📂 Найдено {len(json_files)} экспериментов\n')

    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            results[jf.stem] = data

            hp = data['hyperparameters']
            print(f'✓ {jf.stem}')
            print(
                f'  α={hp["alpha_contrast"]}, '
                f'Tc={hp["temperature_contrast"]}, '
                f'Tkd={hp["temperature_kd"]}'
            )
    print()
    return results

# ==============================
# MAIN LEARNING CURVE
# ==============================

def plot_learning_curves(results, save_path):
    fig, ax = plt.subplots(figsize=(14, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, data), color in zip(sorted(results.items()), colors):
        m = data['metrics']
        epochs = m['epochs']
        val_acc = m['val_accuracy']
        budget = [ep * data['train_samples'] for ep in epochs]

        if 'baseline' in name:
            style = dict(color='red', linestyle='--', linewidth=3, marker='X')
            label = 'Baseline (no contrast)'
        else:
            hp = data['hyperparameters']
            label = (
                f'α={hp["alpha_contrast"]}, '
                f'Tc={hp["temperature_contrast"]}, '
                f'Tkd={hp["temperature_kd"]}'
            )
            style = dict(color=color, linewidth=2.5, marker='o')

        ax.plot(budget, val_acc, label=label, **style)

        ax.annotate(
            f'{val_acc[-1]:.1f}%',
            (budget[-1], val_acc[-1]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9
        )

    ax.set_xlabel('Training Budget (iterations)', fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
    ax.set_title(
        'Contrastive Distillation: Performance vs Training Budget',
        fontweight='bold',
        pad=20
    )

    ax.legend(ncol=2, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {save_path}')

# ==============================
# ALPHA EFFECT
# ==============================

def plot_alpha_effect(results, save_path):
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, data in sorted(results.items()):
        hp = data['hyperparameters']
        if hp['temperature_contrast'] != 0.07:
            continue

        m = data['metrics']
        budget = [ep * data['train_samples'] for ep in m['epochs']]
        acc = m['val_accuracy']

        if 'baseline' in name:
            label = 'α = 0.0'
            ax.plot(budget, acc, '--X', linewidth=3)
        else:
            label = f'α = {hp["alpha_contrast"]}'
            ax.plot(budget, acc, '-o', linewidth=2.5)

    ax.set_title(
        'Effect of Contrastive Weight α\n(Tc=0.07)',
        fontweight='bold'
    )
    ax.set_xlabel('Training Budget')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {save_path}')

# ==============================
# TEMPERATURE EFFECT
# ==============================

def plot_temperature_effect(results, save_path):
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, data in sorted(results.items()):
        hp = data['hyperparameters']
        if hp['alpha_contrast'] != 0.5:
            continue

        m = data['metrics']
        budget = [ep * data['train_samples'] for ep in m['epochs']]
        acc = m['val_accuracy']

        label = f'Tc={hp["temperature_contrast"]}, Tkd={hp["temperature_kd"]}'
        ax.plot(budget, acc, '-o', linewidth=2.5, label=label)

    ax.set_title(
        'Effect of Temperature on Contrastive Distillation\n(α=0.5)',
        fontweight='bold'
    )
    ax.set_xlabel('Training Budget')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {save_path}')

# ==============================
# LOSS CURVES
# ==============================

def plot_loss_curves(results, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))

    for name, data in sorted(results.items()):
        m = data['metrics']
        budget = [ep * data['train_samples'] for ep in m['epochs']]
        label = name.replace('contrastive_', '')

        axs[0, 0].plot(budget, m['train_loss'], label=label)
        axs[0, 1].plot(budget, m['val_loss'], label=label)
        axs[1, 0].plot(budget, m['train_contrast_loss'], label=label)
        axs[1, 1].plot(budget, m['val_contrast_loss'], label=label)

    axs[0, 0].set_title('Train Total Loss')
    axs[0, 1].set_title('Val Total Loss')
    axs[1, 0].set_title('Train Contrastive Loss')
    axs[1, 1].set_title('Val Contrastive Loss')

    for ax in axs.flat:
        ax.set_xlabel('Training Budget')
        ax.grid(alpha=0.3)

    axs[0, 0].legend(ncol=2, fontsize=9)

    plt.suptitle('Loss Curves: Contrastive Distillation', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {save_path}')

# ==============================
# COSINE SIMILARITY
# ==============================

def plot_cosine_similarity(results, save_path):
    names, values = [], []

    for name, data in sorted(results.items()):
        names.append(name.replace('contrastive_', ''))
        values.append(data['avg_cosine_similarity'])

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, values)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v, f'{v:.3f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Average Cosine Similarity')
    ax.set_title(
        'Feature Alignment between Teacher and Student',
        fontweight='bold'
    )
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {save_path}')

# ==============================
# MAIN
# ==============================

def main():
    results = load_results()
    plots_dir = Path('../plots/contrastive_distillation_S')
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(results, plots_dir / 'learning_curve.png')
    plot_alpha_effect(results, plots_dir / 'alpha_effect.png')
    plot_temperature_effect(results, plots_dir / 'temperature_effect.png')
    plot_loss_curves(results, plots_dir / 'loss_curves.png')
    plot_cosine_similarity(results, plots_dir / 'cosine_similarity.png')

    print('\n🎉 Все графики для contrastive distillation готовы!')

if __name__ == '__main__':
    main()
