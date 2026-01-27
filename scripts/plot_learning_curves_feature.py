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

def load_results(results_dir='../results/feature_s'):
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
                alpha = data['hyperparameters'].get('alpha', 'N/A')
                temp = data['hyperparameters'].get('temperature', 'N/A')
                use_proj = data['hyperparameters'].get('use_projection', False)
                print(f'    Alpha={alpha}, Temperature={temp}, Projection={use_proj}')
    
    print()
    return results

def plot_main_learning_curve(results, save_path='../plots/feature_distillation/learning_curve_main.png'):
    """
    ОСНОВНОЙ ГРАФИК: Performance vs Training Budget
    
    X-axis: Training Budget (epochs × dataset_size)
    Y-axis: Validation Accuracy (%)
    """
    
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
        if name == 'feature_baseline_no_distill':
            label = '❌ Baseline (No Distillation)'
            style = baseline_style
        else:
            alpha = data['hyperparameters']['alpha']
            temp = data['hyperparameters']['temperature']
            
            # Красивые labels
            if 'alpha' in name and 'proj' in name:
                label = f'α={alpha:.1f}, T={temp:.1f} (Proj)'
            elif 'T2' in name or 'T6' in name:
                label = f'T={temp:.1f}, α={alpha:.1f}'
            else:
                label = f'α={alpha:.1f}, T={temp:.1f}'
            
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
    ax.set_title('Learning Curves: Feature Distillation\nPerformance vs Training Budget', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Легенда
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95, ncol=2,
             edgecolor='black', fancybox=True, shadow=True,
             title='Distillation Configuration', title_fontsize=11)
    
    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(left=0)
    
    # Форматирование оси X
    ax.ticklabel_format(style='plain', axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Сохранен: {save_path}')
    plt.close()

def plot_alpha_comparison(results, save_path='../plots/feature_distillation/alpha_effect.png'):
    """
    График: Влияние веса дистилляции α (при T=4.0 с проекцией)
    """
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Фильтруем эксперименты с проекцией и T=4.0
    alpha_experiments = {name: data for name, data in results.items() 
                        if 'proj' in name and data['hyperparameters']['temperature'] == 4.0}
    
    # Добавляем baseline
    if 'feature_baseline_no_distill' in results:
        alpha_experiments['feature_baseline_no_distill'] = results['feature_baseline_no_distill']
    
    # Цвета для разных alpha
    alpha_colors = {
        0.0: '#7f7f7f',  # серый (baseline)
        0.3: '#8c564b',  # коричневый
        0.5: '#1f77b4',  # синий
        0.7: '#e377c2',  # розовый
        0.9: '#9467bd',  # фиолетовый
    }
    
    for name, data in sorted(alpha_experiments.items()):
        metrics = data['metrics']
        epochs = metrics['epochs']
        val_acc = metrics['val_accuracy']
        
        train_samples = data['train_samples']
        training_budget = [ep * train_samples for ep in epochs]
        
        if name == 'feature_baseline_no_distill':
            label = 'α = 0.0 (No Distillation)'
            alpha_val = 0.0
            linestyle = '--'
            linewidth = 3.5
            marker = 'X'
            markersize = 11
        else:
            alpha_val = data['hyperparameters']['alpha']
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
    ax.set_title('Effect of Feature Distillation Weight (α) on Learning\n(Fixed Temperature=4.0, with Projection)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95,
             edgecolor='black', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Сохранен: {save_path}')
    plt.close()

def plot_temperature_comparison(results, save_path='../plots/feature_distillation/temperature_effect.png'):
    """
    График: Влияние температуры (при α=0.5 с проекцией)
    """
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Фильтруем эксперименты с alpha=0.5
    temp_experiments = {name: data for name, data in results.items() 
                       if data['hyperparameters']['alpha'] == 0.5}
    
    # Добавляем baseline для сравнения
    if 'feature_baseline_no_distill' in results:
        temp_experiments['feature_baseline_no_distill'] = results['feature_baseline_no_distill']
    
    # Цвета для разных температур
    temp_colors = {
        2.0: '#1f77b4',  # синий
        4.0: '#ff7f0e',  # оранжевый
        6.0: '#2ca02c',  # зеленый
        'baseline': '#7f7f7f'  # серый
    }
    
    for name, data in sorted(temp_experiments.items()):
        metrics = data['metrics']
        epochs = metrics['epochs']
        val_acc = metrics['val_accuracy']
        
        train_samples = data['train_samples']
        training_budget = [ep * train_samples for ep in epochs]
        
        if name == 'feature_baseline_no_distill':
            label = 'Baseline (No Feature Distillation)'
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
    ax.set_title('Effect of Temperature on Feature Distillation\n(Fixed α=0.5, with Projection)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95,
             edgecolor='black', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Сохранен: {save_path}')
    plt.close()

def plot_loss_curves(results, save_path='../plots/feature_distillation/loss_curves.png'):
    """
    График: Training & Validation Loss vs Budget (с разделением на feature и classification)
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, data), color in zip(sorted(results.items()), colors):
        metrics = data['metrics']
        epochs = metrics['epochs']
        
        train_samples = data['train_samples']
        training_budget = [ep * train_samples for ep in epochs]
        
        if name == 'feature_baseline_no_distill':
            label = 'Baseline'
            linestyle = '--'
            linewidth = 3
            marker = 'X'
        else:
            alpha = data['hyperparameters']['alpha']
            T = data['hyperparameters']['temperature']
            label = f'α={alpha:.1f}, T={T:.1f}'
            linestyle = '-'
            linewidth = 2.5
            marker = 'o'
        
        # Training Total Loss
        ax1.plot(training_budget, metrics['train_loss'], 
                label=label, color=color, linestyle=linestyle,
                linewidth=linewidth, marker=marker, markersize=7, alpha=0.85)
        
        # Validation Total Loss
        ax2.plot(training_budget, metrics['val_loss'], 
                label=label, color=color, linestyle=linestyle,
                linewidth=linewidth, marker=marker, markersize=7, alpha=0.85)
        
        # Training Feature Loss
        if 'train_feature_loss' in metrics:
            ax3.plot(training_budget, metrics['train_feature_loss'], 
                    label=label, color=color, linestyle=linestyle,
                    linewidth=linewidth, marker=marker, markersize=7, alpha=0.85)
        
        # Validation Feature Loss
        if 'val_feature_loss' in metrics:
            ax4.plot(training_budget, metrics['val_feature_loss'], 
                    label=label, color=color, linestyle=linestyle,
                    linewidth=linewidth, marker=marker, markersize=7, alpha=0.85)
    
    # Training Total Loss
    ax1.set_xlabel('Training Budget', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Training Total Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Training Total Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Validation Total Loss
    ax2.set_xlabel('Training Budget', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Validation Total Loss', fontsize=13, fontweight='bold')
    ax2.set_title('Validation Total Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    # Training Feature Loss
    ax3.set_xlabel('Training Budget', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Training Feature Loss', fontsize=13, fontweight='bold')
    ax3.set_title('Training Feature Loss (Cosine Distance)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    
    # Validation Feature Loss
    ax4.set_xlabel('Training Budget', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Validation Feature Loss', fontsize=13, fontweight='bold')
    ax4.set_title('Validation Feature Loss (Cosine Distance)', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)
    
    fig.suptitle('Loss Curves: Feature Distillation', fontsize=17, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Сохранен: {save_path}')
    plt.close()

def plot_cosine_similarity(results, save_path='../plots/feature_distillation/cosine_similarity.png'):
    """
    График: Cosine Similarity между признаками студента и учителя
    """
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Собираем данные
    exp_names = []
    cosine_sims = []
    colors_list = []
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, data), color in zip(sorted(results.items()), colors):
        # Получаем cosine similarity, если есть, иначе пропускаем
        cosine_sim = data.get('avg_cosine_similarity', None)
        
        # Пропускаем эксперименты без cosine similarity
        if cosine_sim is None:
            continue
        
        if name == 'feature_baseline_no_distill':
            exp_names.append('Baseline')
            colors_list.append('red')
        else:
            alpha = data['hyperparameters']['alpha']
            T = data['hyperparameters']['temperature']
            exp_names.append(f'α={alpha:.1f}\nT={T:.1f}')
            colors_list.append(color)
        
        cosine_sims.append(cosine_sim)
    
    # Если нет данных для отображения
    if not cosine_sims:
        print('⚠️  Нет данных о cosine similarity для построения графика')
        plt.close()
        return
    
    # Создаем bar plot
    bars = ax.bar(range(len(exp_names)), cosine_sims, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Добавляем значения на барах
    for bar, val in zip(bars, cosine_sims):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Experiment Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Cosine Similarity', fontsize=14, fontweight='bold')
    ax.set_title('Feature Space Alignment: Cosine Similarity between Student and Teacher', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Сохранен: {save_path}')
    plt.close()

def plot_summary_table(results, save_path='../plots/feature_distillation/summary_table.png'):
    """
    Таблица: Сводка финальных результатов
    """
    
    # Собираем данные для таблицы
    table_data = []
    
    for name, data in sorted(results.items()):
        metrics = data['metrics']
        
        if name == 'feature_baseline_no_distill':
            config = 'Baseline (No Feature Dist)'
        else:
            alpha = data['hyperparameters']['alpha']
            temp = data['hyperparameters']['temperature']
            use_proj = '✓' if data['hyperparameters'].get('use_projection', False) else '✗'
            config = f'α={alpha:.1f}, T={temp:.1f}, Proj={use_proj}'
        
        final_val_acc = metrics['val_accuracy'][-1]
        test_acc = data['test_accuracy']
        
        # Cosine similarity может отсутствовать
        cosine_sim = data.get('avg_cosine_similarity', None)
        cosine_sim_str = f'{cosine_sim:.4f}' if cosine_sim is not None else 'N/A'
        
        total_epochs = metrics['epochs'][-1]
        
        table_data.append([
            config,
            f'{final_val_acc:.2f}%',
            f'{test_acc:.2f}%',
            cosine_sim_str,
            total_epochs
        ])
    
    # Создаем фигуру для таблицы
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Заголовки таблицы
    headers = ['Configuration', 'Final Val Acc', 'Test Acc', 'Cosine Sim', 'Epochs']
    
    # Создаем таблицу
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.45, 0.13, 0.13, 0.13, 0.1])
    
    # Стилизация таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Цвета заголовков
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#2196F3')
        cell.set_text_props(weight='bold', color='white')
    
    # Цвета строк (чередование)
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('#ffffff')
    
    plt.title('Summary: Feature Distillation Experiments', 
             fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Сохранен: {save_path}')
    plt.close()

def print_summary(results):
    """Печать сводной статистики"""
    print('\n' + '='*80)
    print('📊 СВОДНАЯ СТАТИСТИКА: FEATURE DISTILLATION')
    print('='*80 + '\n')
    
    for name, data in sorted(results.items()):
        print(f'🔹 {name}')
        
        if 'hyperparameters' in data and name != 'feature_baseline_no_distill':
            print(f'   Alpha: {data["hyperparameters"]["alpha"]}')
            print(f'   Temperature: {data["hyperparameters"]["temperature"]}')
            print(f'   Projection: {data["hyperparameters"]["use_projection"]}')
        
        metrics = data['metrics']
        print(f'   Final Val Acc: {metrics["val_accuracy"][-1]:.2f}%')
        print(f'   Test Acc: {data["test_accuracy"]:.2f}%')
        
        # Cosine similarity может отсутствовать
        cosine_sim = data.get('avg_cosine_similarity', None)
        if cosine_sim is not None:
            print(f'   Cosine Similarity: {cosine_sim:.4f}')
        else:
            print(f'   Cosine Similarity: N/A')
        
        print(f'   Training Budget: {metrics["epochs"][-1]} epochs × {data["train_samples"]} samples = {metrics["epochs"][-1] * data["train_samples"]:,} iterations')
        print()
    
    # Лучший результат
    best = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f'🏆 ЛУЧШИЙ РЕЗУЛЬТАТ:')
    print(f'   {best[0]}')
    print(f'   Test Accuracy: {best[1]["test_accuracy"]:.2f}%')
    
    if 'hyperparameters' in best[1]:
        print(f'   Alpha: {best[1]["hyperparameters"]["alpha"]}')
        print(f'   Temperature: {best[1]["hyperparameters"]["temperature"]}')
        cosine_sim = best[1].get("avg_cosine_similarity", None)
        if cosine_sim is not None:
            print(f'   Cosine Similarity: {cosine_sim:.4f}')

def main():
    print('='*80)
    print('📈 ПОСТРОЕНИЕ LEARNING CURVES: FEATURE DISTILLATION')
    print('='*80)
    
    # Загрузка результатов
    results = load_results('../results/feature_t')
    
    if not results:
        print('❌ Результаты не найдены в ../results/feature_t!')
        return
    
    # Создаем папку для графиков
    plots_dir = Path('../plots/feature_distillation')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print('📊 Построение графиков...\n')
    
    # 1. Основной график - Learning Curves
    plot_main_learning_curve(results, save_path='../plots/feature_distillation/learning_curve_main.png')
    
    # 2. Влияние alpha
    plot_alpha_comparison(results, save_path='../plots/feature_distillation/alpha_effect.png')
    
    # 3. Влияние температуры
    plot_temperature_comparison(results, save_path='../plots/feature_distillation/temperature_effect.png')
    
    # 4. Loss curves
    plot_loss_curves(results, save_path='../plots/feature_distillation/loss_curves.png')
    
    # 5. Cosine Similarity
    plot_cosine_similarity(results, save_path='../plots/feature_distillation/cosine_similarity.png')
    
    # 6. Таблица результатов
    plot_summary_table(results, save_path='../plots/feature_distillation/summary_table.png')
    
    # Статистика
    print_summary(results)
    
    print('\n' + '='*80)
    print('🎉 ВСЕ ГРАФИКИ ГОТОВЫ!')
    print('='*80)
    print(f'📁 Графики сохранены в: ../plots/feature_distillation/')
    print('\nСозданные файлы:')
    print('  • learning_curve_main.png     - Основной график Performance vs Budget')
    print('  • alpha_effect.png            - Влияние веса дистилляции α')
    print('  • temperature_effect.png      - Влияние температуры')
    print('  • loss_curves.png             - Кривые loss (total + feature)')
    print('  • cosine_similarity.png       - Cosine similarity между признаками')
    print('  • summary_table.png           - Сводная таблица результатов')
    print()

if __name__ == '__main__':
    main()