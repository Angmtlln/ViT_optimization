import matplotlib.pyplot as plt
import numpy as np

# Данные
models = ['Teacher\n(L14-336)', 'Student-S\n(S16-384)', 'Student-T\n(T16-384)']
params_M = [671.1, 87.2, 69.5]
inference_ms = [None, 104.30, 54.71]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Размер модели
colors = ['#4A90E2', '#50C878', '#F39C12']  # синий, зелёный, оранжевый
ax1.bar(models, params_M, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Параметры (млн)', fontsize=12)
ax1.set_title('Размер модели', fontsize=14, weight='bold', pad=15)
ax1.set_ylim(0, max(params_M) * 1.15)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
for i, v in enumerate(params_M):
    ax1.text(i, v + 15, f'{v:.1f}M', ha='center', fontsize=11, weight='bold')

# Скорость инференса (только студенты)
student_models = models[1:]
student_inference = inference_ms[1:]
ax2.bar(student_models, student_inference, color=colors[1:], edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Время (мс)', fontsize=12)
ax2.set_title('Скорость инференса на CPU', fontsize=14, weight='bold', pad=15)
ax2.set_ylim(0, max(student_inference) * 1.2)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
for i, v in enumerate(student_inference):
    ax2.text(i, v + 4, f'{v:.1f} мс', ha='center', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig('../plots/size_speed_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("Сохранено: plots/size_speed_comparison.png")