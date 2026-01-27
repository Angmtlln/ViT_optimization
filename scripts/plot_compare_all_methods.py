import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# ==============================
# PROFESSIONAL STYLE
# ==============================

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linewidth'] = 0.8

# Professional color palette (muted, corporate)
COLORS = {
    'Attention': '#2E5090',      # Deep blue
    'Feature': '#D35400',        # Burnt orange
    'Logit': '#27AE60',          # Forest green
    'Contrastive': '#8E44AD',    # Deep purple
    'Baseline': '#7F8C8D'        # Steel grey
}

MARKERS = {
    'Attention': 'o',
    'Feature': 's',
    'Logit': '^',
    'Contrastive': 'D',
    'Baseline': 'X'
}

# ==============================
# LOADING BEST FROM EACH METHOD
# ==============================

def load_best_results():
    results_dir = Path('../results')
    methods = {
        'Attention': 'attention_t',
        'Feature': 'feature_t',
        'Logit': 'logit_t',
        'Contrastive': 'contrastive_t'
    }
    
    best_results = {}
    
    for method_name, folder in methods.items():
        method_dir = results_dir / folder
        if not method_dir.exists():
            print(f'⚠️ Папка {folder} не найдена')
            continue
            
        json_files = list(method_dir.glob('*.json'))
        if not json_files:
            print(f'⚠️ В {folder} нет JSON файлов')
            continue
        
        best_test_acc = -1
        best_file = None
        best_data = None
        
        for jf in json_files:
            if 'baseline' in jf.stem:
                continue
                
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_acc = data.get('test_accuracy', 0)
                
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_file = jf.stem
                    best_data = data
        
        if best_data:
            best_results[method_name] = {
                'data': best_data,
                'file': best_file,
                'test_acc': best_test_acc
            }
            print(f'✓ {method_name}: {best_file} (test={best_test_acc:.2f}%)')
    
    baseline_file = results_dir / 'attention_t' / 'attention_baseline_no_distill.json'
    if baseline_file.exists():
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline_data = json.load(f)
            best_results['Baseline'] = {
                'data': baseline_data,
                'file': 'baseline',
                'test_acc': baseline_data.get('test_accuracy', 0)
            }
    
    print()
    return best_results

# ==============================
# COMPARISON PLOT (REFINED)
# ==============================

def plot_comparison(results, save_path):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
    ax.set_facecolor('#FAFAFA')
    
    for method_name, result in sorted(results.items()):
        data = result['data']
        m = data['metrics']
        epochs = m['epochs']
        val_acc = m['val_accuracy']
        budget = [ep * data['train_samples'] for ep in epochs]
        
        if method_name == 'Baseline':
            style = dict(
                color=COLORS[method_name],
                linestyle='--',
                linewidth=3.5,
                marker=MARKERS[method_name],
                markersize=11,
                markeredgewidth=2,
                markerfacecolor='white',
                label=f'{method_name} (no distillation)',
                alpha=0.85
            )
        else:
            style = dict(
                color=COLORS[method_name],
                linewidth=3,
                marker=MARKERS[method_name],
                markersize=9,
                markeredgewidth=1.5,
                markerfacecolor=COLORS[method_name],
                markeredgecolor='white',
                label=f'{method_name} (test={result["test_acc"]:.2f}%)',
                alpha=0.9
            )
        
        line = ax.plot(budget, val_acc, **style)[0]
        
        # Annotate final value with background
        bbox_props = dict(
            boxstyle='round,pad=0.4',
            facecolor=COLORS[method_name],
            edgecolor='white',
            alpha=0.85,
            linewidth=1.5
        )
        ax.annotate(
            f'{val_acc[-1]:.1f}%',
            (budget[-1], val_acc[-1]),
            textcoords="offset points",
            xytext=(12, 6),
            fontsize=11,
            fontweight='bold',
            color='white',
            bbox=bbox_props
        )
    
    ax.set_xlabel('Training Budget (iterations)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Validation Accuracy (%)', fontweight='bold', fontsize=16)
    ax.set_title(
        'Сравнение методов дистилляции знаний для Vision Transformers\n' +
        'Датасет: Kvasir-v2 (медицинская классификация, 8 классов)',
        fontweight='bold',
        pad=25,
        fontsize=20
    )
    
    ax.legend(
        loc='lower right',
        framealpha=0.97,
        fontsize=13,
        edgecolor='#CCCCCC',
        fancybox=True,
        shadow=True
    )
    ax.grid(alpha=0.25, linewidth=0.8, color='#BBBBBB')
    ax.set_xlim(left=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f'✅ Saved: {save_path}')

# ==============================
# BAR CHART (REFINED)
# ==============================

def plot_test_accuracy_comparison(results, save_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('#FAFAFA')
    
    methods = []
    test_accs = []
    colors_list = []
    
    for method_name, result in sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
        methods.append(method_name)
        test_accs.append(result['test_acc'])
        colors_list.append(COLORS.get(method_name, '#7F8C8D'))
    
    bars = ax.barh(
        methods,
        test_accs,
        color=colors_list,
        alpha=0.85,
        edgecolor='white',
        linewidth=2.5,
        height=0.65
    )
    
    # Annotate bars with shadow effect
    for bar, acc in zip(bars, test_accs):
        # Shadow
        ax.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height()/2 + 0.02,
            f'{acc:.2f}%',
            va='center',
            fontsize=14,
            fontweight='bold',
            color='#CCCCCC',
            alpha=0.6
        )
        # Main text
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height()/2,
            f'{acc:.2f}%',
            va='center',
            fontsize=14,
            fontweight='bold',
            color='#2C3E50'
        )
    
    ax.set_xlabel('Test Accuracy (%)', fontweight='bold', fontsize=16)
    ax.set_title(
        'Финальная точность на тестовой выборке\nСравнение методов дистилляции',
        fontweight='bold',
        pad=25,
        fontsize=19
    )
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.25, linewidth=0.8, color='#BBBBBB')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f'✅ Saved: {save_path}')

# ==============================
# TABLE: SUMMARY
# ==============================

def print_summary_table(results):
    print('\n' + '='*80)
    print('СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТОДОВ')
    print('='*80)
    print(f'{"Метод":<15} {"Файл":<40} {"Test Acc":<12} {"Best Val":<12}')
    print('-'*80)
    
    for method_name, result in sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
        data = result['data']
        test_acc = result['test_acc']
        best_val = data.get('best_val_acc', 0)
        filename = result['file'][:38]
        
        print(f'{method_name:<15} {filename:<40} {test_acc:>10.2f}% {best_val:>10.2f}%')
    
    print('='*80 + '\n')

# ==============================
# MAIN
# ==============================

def main():
    results = load_best_results()
    
    if not results:
        print('❌ Не удалось загрузить результаты')
        return
    
    plots_dir = Path('../plots/comparison_all_methods')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(results, plots_dir / 'comparison_learning_curves.png')
    plot_test_accuracy_comparison(results, plots_dir / 'comparison_test_accuracy.png')
    print_summary_table(results)
    
    print('🎉 Графики сравнения всех методов готовы!')
    print(f'📂 Сохранено в: {plots_dir.absolute()}')

if __name__ == '__main__':
    main()