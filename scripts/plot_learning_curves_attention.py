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

def load_results(results_dir='../results/attention_distillation_T'):
    results_dir = Path(results_dir)
    results = {}

    json_files = sorted(results_dir.glob('*.json'))
    print(f'📂 Найдено {len(json_files)} экспериментов\n')

    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            results[jf.stem] = data

            # Extract parameters from filename if not in hyperparameters
            filename = jf.stem
            hp = data['hyperparameters']
            
            # Parse alpha from filename
            if 'alpha' in filename and 'baseline' not in filename:
                alpha_str = filename.split('alpha')[1].split('_')[0]
                alpha = float(alpha_str)
            else:
                alpha = hp.get('alpha_attention', 0.0)
            
            # Parse loss type from filename
            if 'cosine' in filename:
                loss_type = 'cosine'
            elif 'l1' in filename:
                loss_type = 'l1'
            else:
                loss_type = 'mse'
            
            # Parse learning rate from filename
            if 'lr' in filename:
                lr_str = filename.split('lr')[1].split('_')[0].split(' ')[0]
                lr = lr_str
            else:
                lr = hp.get('learning_rate', '1e-4')
            
            print(f'✓ {jf.stem}')
            print(f'  α={alpha}, loss={loss_type}, lr={lr}')
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
            label = 'Baseline (no attention distill)'
        else:
            # Parse from filename
            if 'alpha' in name:
                alpha_str = name.split('alpha')[1].split('_')[0]
                alpha = alpha_str
            else:
                alpha = 'N/A'
            
            if 'cosine' in name:
                loss_type = 'cosine'
            elif 'l1' in name:
                loss_type = 'l1'
            else:
                loss_type = 'mse'
            
            if 'lr' in name:
                lr = name.split('lr')[1].split('_')[0].split(' ')[0]
                label = f'lr={lr}'
            else:
                label = f'α={alpha}, loss={loss_type}'
            
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
        'Attention Distillation: Performance vs Training Budget',
        fontweight='bold',
        pad=20
    )

    ax.legend(ncol=2, framealpha=0.95, fontsize=9)
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
        # Only MSE loss, no lr experiments
        if 'lr' in name or ('cosine' in name) or ('l1' in name):
            continue

        m = data['metrics']
        budget = [ep * data['train_samples'] for ep in m['epochs']]
        acc = m['val_accuracy']

        if 'baseline' in name:
            label = 'α = 0.0 (baseline)'
            ax.plot(budget, acc, '--', color='red', linewidth=3, marker='X', label=label)
        elif 'alpha' in name:
            alpha_str = name.split('alpha')[1].split('_')[0]
            label = f'α = {alpha_str}'
            ax.plot(budget, acc, '-o', linewidth=2.5, label=label)

    ax.set_title(
        'Effect of Attention Weight α\n(loss=MSE)',
        fontweight='bold'
    )
    ax.set_xlabel('Training Budget (iterations)')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {save_path}')

# ==============================
# LOSS TYPE EFFECT
# ==============================

def plot_loss_type_effect(results, save_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    found_any = False
    for name, data in sorted(results.items()):
        # Only alpha=0.5 experiments
        if 'alpha0.5' not in name or 'lr' in name or 'baseline' in name:
            continue
        
        m = data['metrics']
        budget = [ep * data['train_samples'] for ep in m['epochs']]
        acc = m['val_accuracy']
        
        if 'cosine' in name:
            loss_type = 'cosine'
        elif 'l1' in name:
            loss_type = 'l1'
        else:
            loss_type = 'mse'
        
        label = f'loss={loss_type}'
        ax.plot(budget, acc, '-o', linewidth=2.5, label=label, markersize=6)
        found_any = True

    if not found_any:
        print('⚠️ No experiments with α=0.5 and different loss types found')
        plt.close()
        return

    ax.set_title(
        'Effect of Attention Loss Type\n(α=0.5)',
        fontweight='bold'
    )
    ax.set_xlabel('Training Budget (iterations)')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {save_path}')

# ==============================
# LEARNING RATE EFFECT
# ==============================

def plot_lr_effect(results, save_path):
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, data in sorted(results.items()):
        if 'lr' not in name and 'baseline' not in name:
            continue
            
        m = data['metrics']
        budget = [ep * data['train_samples'] for ep in m['epochs']]
        acc = m['val_accuracy']

        if 'baseline' in name:
            label = 'baseline (lr=1e-4)'
        else:
            lr = name.split('lr')[1].split('_')[0].split(' ')[0]
            label = f'lr={lr}'
        
        ax.plot(budget, acc, '-o', linewidth=2.5, label=label, markersize=6)

    ax.set_title(
        'Effect of Learning Rate on Attention Distillation',
        fontweight='bold'
    )
    ax.set_xlabel('Training Budget (iterations)')
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
        display_name = name.replace('attention_', '').replace(' (1)', '')

        axs[0, 0].plot(budget, m['train_loss'], label=display_name, linewidth=1.5)
        axs[0, 1].plot(budget, m['val_loss'], label=display_name, linewidth=1.5)
        
        if 'train_attention_loss' in m:
            axs[1, 0].plot(budget, m['train_attention_loss'], label=display_name, linewidth=1.5)
        if 'val_attention_loss' in m:
            axs[1, 1].plot(budget, m['val_attention_loss'], label=display_name, linewidth=1.5)

    axs[0, 0].set_title('Train Total Loss', fontweight='bold')
    axs[0, 1].set_title('Val Total Loss', fontweight='bold')
    axs[1, 0].set_title('Train Attention Loss', fontweight='bold')
    axs[1, 1].set_title('Val Attention Loss', fontweight='bold')

    for ax in axs.flat:
        ax.set_xlabel('Training Budget (iterations)')
        ax.grid(alpha=0.3)

    axs[0, 0].legend(ncol=2, fontsize=8)

    plt.suptitle('Loss Curves: Attention Distillation', fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {save_path}')

# ==============================
# ATTENTION MSE
# ==============================

def plot_attention_mse(results, save_path):
    names, values = [], []

    for name, data in sorted(results.items()):
        if 'avg_attention_mse' in data:
            display_name = name.replace('attention_', '').replace(' (1)', '')
            names.append(display_name)
            values.append(data['avg_attention_mse'])

    if not names:
        print('⚠️ No attention MSE data found')
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(names, values, color='steelblue', alpha=0.7)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v, f'{v:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Average Attention MSE', fontweight='bold')
    ax.set_title(
        'Attention Map Distance (MSE) between Teacher and Student',
        fontweight='bold',
        pad=20
    )
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
    plots_dir = Path('../plots/attention_distillation_T')
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(results, plots_dir / 'learning_curve.png')
    plot_alpha_effect(results, plots_dir / 'alpha_effect.png')
    plot_loss_type_effect(results, plots_dir / 'loss_type_effect.png')
    plot_lr_effect(results, plots_dir / 'lr_effect.png')
    plot_loss_curves(results, plots_dir / 'loss_curves.png')
    plot_attention_mse(results, plots_dir / 'attention_mse.png')

    print('\n🎉 Все графики для attention distillation готовы!')

if __name__ == '__main__':
    main()