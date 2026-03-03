import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11

def load_results(results_dir='../results/logit_distillation_S'):
    """Загрузка всех результатов из папки"""
    results_dir = Path(results_dir)
    results = {}
    
    json_files = sorted(results_dir.glob('*.json'))
    
    print(f'📂 Найдено {len(json_files)} файлов результатов\n')
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            exp_name = json_file.stem
            results[exp_name] = data
            
            print(f'  ✓ {exp_name}')
            if 'hyperparameters' in data:
                T = data['hyperparameters'].get('temperature', 'N/A')
                alpha = data['hyperparameters'].get('kd_alpha', 'N/A')
                print(f'    Temperature={T}, Alpha={alpha}')
    
    print()
    return results

def plot_main_learning_curve(results, save_path='../plots/logit_distillation_S/learning_curve_main.png'):
    """
    ОСНОВНОЙ ГРАФИК: Performance vs Training Budget
    
    X-axis: Training Budget (epochs × dataset_size)
    Y-axis: Validation Accuracy (%)
    """
    # Create output directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Цвета для разных экспериментов
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Для baseline используем особый стиль
    baseline_style = {
        'color': 'red',
        'linestyle': '--',
        'linewidth': 3.5,
        'marker': 'X',
        'markersize': 12,
        'alpha': 0.9,
        'zorder': 10
    }
    
    # Стиль для других экспериментов
    default_style = {
        'linestyle': '-',
        'linewidth': 2.5,
        'marker': 'o',
        'markersize': 8,
        'alpha': 0.85
    }
    
    # Отрисовка линий
    for (name, data), color in zip(sorted(results.items()), colors):
        metrics = data['metrics']
        epochs = metrics['epochs']
        val_acc = metrics['val_accuracy']
        
        # Вычисляем training budget (epochs × train_samples)
        train_samples = data['train_samples']
        training_budget = [ep * train_samples for ep in epochs]
        
        # Определяем стиль и label
        if name == 'baseline_no_distill':
            label = '❌ Baseline (No Distillation)'
            style = baseline_style
        else:
            T = data['hyperparameters']['temperature']
            alpha = data['hyperparameters']['kd_alpha']
            
            # Красивые labels
            if 'alpha0.5' in name:
                label = f'T={T:.1f}, α=0.5'
            else:
                label = f'T={T:.1f}, α={alpha:.1f}'
            
            style = {**default_style, 'color': color}
        
        # Рисуем линию
        ax.plot(training_budget, val_acc, label=label, **style)
        
        # Добавляем аннотацию к последней точке
        final_acc = val_acc[-1]
        final_budget = training_budget[-1]
        ax.annotate(f'{final_acc:.1f}%', 
                   xy=(final_budget, final_acc),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=9, alpha=0.7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=style['color'], alpha=0.2))
    
    # Настройка осей
    ax.set_xlabel('Training Budget (iterations)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=15, fontweight='bold')
    ax.set_title('Learning Curves: Logit Distillation\nPerformance vs Training Budget', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Легенда
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95, ncol=2,
             edgecolor='black', fancybox=True, shadow=True,
             title='Distillation Configuration', title_fontsize=12)
    
    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(left=0)
    
    # Форматирование оси X (показываем в тысячах)
    ax.ticklabel_format(style='plain', axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Сохранен: {save_path}')
    plt.close()

def plot_temperature_comparison(results, save_path='../plots/logit_distillation_S/temperature_effect.png'):
    """
    График: Влияние температуры (при α=0.5)
    """
    # Create output directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Фильтруем эксперименты с alpha=0.5
    temp_experiments = {name: data for name, data in results.items() 
                       if 'alpha0.5' in name}
    
    # Добавляем baseline для сравнения
    if 'baseline_no_distill' in results:
        temp_experiments['baseline_no_distill'] = results['baseline_no_distill']
    
    # Цвета для разных температур
    temp_colors = {
        1.0: '#1f77b4',  # синий
        3.0: '#ff7f0e',  # оранжевый
        5.0: '#2ca02c',  # зеленый
        10.0: '#d62728', # красный
        'baseline': '#7f7f7f'  # серый
    }
    
    for name, data in sorted(temp_experiments.items()):
        metrics = data['metrics']
        epochs = metrics['epochs']
        val_acc = metrics['val_accuracy']
        
        train_samples = data['train_samples']
        training_budget = [ep * train_samples for ep in epochs]
        
        if name == 'baseline_no_distill':
            label = 'Baseline (No KD)'
            color = temp_colors['baseline']
            linestyle = '--'
            linewidth = 3.5
            marker = 'X'
            markersize = 11
        else:
            T = data['hyperparameters']['temperature']
            label = f'Temperature = {T:.1f}'
            color = temp_colors.get(T, '#000000')
            linestyle = '-'
            linewidth = 3
            marker = 'o'
            markersize = 9
        
        ax.plot(training_budget, val_acc, 
               label=label, color=color, linestyle=linestyle,
               linewidth=linewidth, marker=marker, markersize=markersize,
               alpha=0.9)
    
    ax.set_xlabel('Training Budget (iterations)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=15, fontweight='bold')
    ax.set_title('Effect of Temperature on Learning\n(Fixed α=0.5)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95,
             edgecolor='black', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Сохранен: {save_path}')
    plt.close()

def plot_alpha_comparison(results, save_path='../plots/logit_distillation_S/alpha_effect.png'):
    """
    График: Влияние веса дистилляции α (при T=4.0)
    """
    # Create output directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Фильтруем эксперименты с T=4.0
    alpha_experiments = {name: data for name, data in results.items() 
                        if 'T4_' in name}
    
    # Добавляем baseline
    if 'baseline_no_distill' in results:
        alpha_experiments['baseline_no_distill'] = results['baseline_no_distill']
    
    # Цвета для разных alpha
    alpha_colors = {
        0.0: '#7f7f7f',  # серый (baseline)
        0.3: '#8c564b',  # коричневый
        0.7: '#e377c2',  # розовый
        0.9: '#9467bd',  # фиолетовый
    }
    
    for name, data in sorted(alpha_experiments.items()):
        metrics = data['metrics']
        epochs = metrics['epochs']
        val_acc = metrics['val_accuracy']
        
        train_samples = data['train_samples']
        training_budget = [ep * train_samples for ep in epochs]
        
        if name == 'baseline_no_distill':
            label = 'α = 0.0 (No KD)'
            alpha_val = 0.0
            linestyle = '--'
            linewidth = 3.5
            marker = 'X'
            markersize = 11
        else:
            alpha_val = data['hyperparameters']['kd_alpha']
            label = f'α = {alpha_val:.1f}'
            linestyle = '-'
            linewidth = 3
            marker = 's'
            markersize = 9
        
        color = alpha_colors.get(alpha_val, '#000000')
        
        ax.plot(training_budget, val_acc, 
               label=label, color=color, linestyle=linestyle,
               linewidth=linewidth, marker=marker, markersize=markersize,
               alpha=0.9)
    
    ax.set_xlabel('Training Budget (iterations)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=15, fontweight='bold')
    ax.set_title('Effect of Distillation Weight (α) on Learning\n(Fixed Temperature=4.0)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95,
             edgecolor='black', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Сохранен: {save_path}')
    plt.close()

def plot_loss_curves(results, save_path='../plots/logit_distillation_S/loss_curves.png'):
    """
    График: Training & Validation Loss vs Budget
    """
    # Create output directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, data), color in zip(sorted(results.items()), colors):
        metrics = data['metrics']
        epochs = metrics['epochs']
        
        train_samples = data['train_samples']
        training_budget = [ep * train_samples for ep in epochs]
        
        if name == 'baseline_no_distill':
            label = 'Baseline'
            linestyle = '--'
            linewidth = 3
            marker = 'X'
        else:
            T = data['hyperparameters']['temperature']
            alpha = data['hyperparameters']['kd_alpha']
            label = f'T={T:.1f}, α={alpha:.1f}'
            linestyle = '-'
            linewidth = 2.5
            marker = 'o'
        
        # Training Loss
        ax1.plot(training_budget, metrics['train_loss'], 
                label=label, color=color, linestyle=linestyle,
                linewidth=linewidth, marker=marker, markersize=7, alpha=0.85)
        
        # Validation Loss
        ax2.plot(training_budget, metrics['val_loss'], 
                label=label, color=color, linestyle=linestyle,
                linewidth=linewidth, marker=marker, markersize=7, alpha=0.85)
    
    # Training Loss
    ax1.set_xlabel('Training Budget', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Training Loss', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Validation Loss
    ax2.set_xlabel('Training Budget', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_title('Validation Loss', fontsize=15, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    fig.suptitle('Loss Curves: Logit Distillation', fontsize=17, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Сохранен: {save_path}')
    plt.close()

def plot_summary_table(results, save_path='../plots/logit_distillation_S/summary_table.png'):
    """
    Таблица: Сводка финальных результатов
    """
    # Create output directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Собираем данные для таблицы
    table_data = []
    
    for name, data in sorted(results.items()):
        metrics = data['metrics']
        
        if name == 'baseline_no_distill':
            config = 'Baseline (No KD)'
        else:
            T = data['hyperparameters']['temperature']
            alpha = data['hyperparameters']['kd_alpha']
            config = f'T={T:.1f}, α={alpha:.1f}'
        
        final_val_acc = metrics['val_accuracy'][-1]
        test_acc = data['test_accuracy']
        final_loss = metrics['val_loss'][-1]
        total_epochs = metrics['epochs'][-1]
        
        table_data.append([
            config,
            f'{final_val_acc:.2f}%',
            f'{test_acc:.2f}%',
            f'{final_loss:.4f}',
            total_epochs
        ])
    
    # Создаем фигуру для таблицы
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Заголовки таблицы
    headers = ['Configuration', 'Final Val Acc', 'Test Acc', 'Final Val Loss', 'Epochs']
    
    # Создаем таблицу
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.35, 0.15, 0.15, 0.15, 0.1])
    
    # Стилизация таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Цвета заголовков
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Цвета строк (чередование)
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('#ffffff')
    
    plt.title('Summary: Logit Distillation Experiments', 
             fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Сохранен: {save_path}')
    plt.close()

def print_summary(results):
    """Печать сводной статистики"""
    print('\n' + '='*80)
    print('📊 СВОДНАЯ СТАТИСТИКА')
    print('='*80 + '\n')
    
    for name, data in sorted(results.items()):
        print(f'🔹 {name}')
        
        if 'hyperparameters' in data and name != 'baseline_no_distill':
            print(f'   Temperature: {data["hyperparameters"]["temperature"]}')
            print(f'   Alpha: {data["hyperparameters"]["kd_alpha"]}')
        
        metrics = data['metrics']
        print(f'   Final Val Acc: {metrics["val_accuracy"][-1]:.2f}%')
        print(f'   Test Acc: {data["test_accuracy"]:.2f}%')
        print(f'   Training Budget: {metrics["epochs"][-1]} epochs × {data["train_samples"]} samples = {metrics["epochs"][-1] * data["train_samples"]:,} iterations')
        print()
    
    # Лучший результат
    best = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f'🏆 ЛУЧШИЙ РЕЗУЛЬТАТ:')
    print(f'   {best[0]}')
    print(f'   Test Accuracy: {best[1]["test_accuracy"]:.2f}%')
    
    if 'hyperparameters' in best[1]:
        print(f'   Temperature: {best[1]["hyperparameters"]["temperature"]}')
        print(f'   Alpha: {best[1]["hyperparameters"]["kd_alpha"]}')

def main():
    print('='*80)
    print('📈 ПОСТРОЕНИЕ LEARNING CURVES: LOGIT DISTILLATION')
    print('='*80)
    
    # Загрузка результатов
    results = load_results('../results/logit_distillation_S')
    
    if not results:
        print('❌ Результаты не найдены в ../results/logit_distillation_S!')
        return
    
    print('📊 Построение графиков...\n')
    
    # 1. Основной график - Learning Curves
    plot_main_learning_curve(results, save_path='../plots/logit_distillation_S/learning_curve_main.png')
    
    # 2. Влияние температуры
    plot_temperature_comparison(results,save_path='../plots/logit_distillation_S/temperature_effect.png')
    
    # 3. Влияние alpha
    plot_alpha_comparison(results, save_path='../plots/logit_distillation_S/alpha_effect.png')
    
    # 4. Loss curves
    plot_loss_curves(results, save_path='../plots/logit_distillation_S/loss_curves.png')
    
    # 5. Таблица результатов
    plot_summary_table(results, save_path='../plots/logit_distillation_S/summary_table.png')
    
    # Статистика
    print_summary(results)
    
    print('\n' + '='*80)
    print('🎉 ВСЕ ГРАФИКИ ГОТОВЫ!')
    print('='*80)
    print(f'📁 Графики сохранены в: ../results/logit_distillation_S/')
    print('\nСозданные файлы:')
    print('  • learning_curve_main.png     - Основной график Performance vs Budget')
    print('  • temperature_effect.png      - Влияние температуры')
    print('  • alpha_effect.png            - Влияние веса дистилляции')
    print('  • loss_curves.png             - Кривые loss')
    print('  • summary_table.png           - Сводная таблица результатов')
    print()

if __name__ == '__main__':
    main()