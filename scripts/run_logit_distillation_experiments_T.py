import os, sys, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from glob import glob
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

# ==============================
# ИСПРАВЛЕННЫЙ ПУТЬ К PERCEPTION_MODELS
# ==============================

# Так же как в run_all_experiments_logit.py:
sys.path.insert(0, str(Path('../perception_models').resolve()))

try:
    from core.vision_encoder import pe
    from core.vision_encoder import transforms
    print("✅ Модули perception_models импортированы успешно\n")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    # Пробуем альтернативный путь
    sys.path.insert(0, str(Path(__file__).parent.parent / 'perception_models'))
    try:
        from core.vision_encoder import pe
        from core.vision_encoder import transforms
        print("✅ Модули импортированы (альтернативный путь)\n")
    except ImportError as e2:
        print(f"❌ Не удалось импортировать perception_models: {e2}")
        sys.exit(1)

# ==============================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ==============================

# Фиксированные параметры (НЕ МЕНЯТЬ)
SEED = 42
ROOT_DIR = '../data/kvasir-dataset-v2'
BATCH_SIZE = 16
NUM_WORKERS = 6
OPTIMIZER_TYPE = 'AdamW'
WEIGHT_DECAY = 1e-5
SCHEDULER = 'CosineAnnealingLR'

# Архитектуры (фиксированы) - ИСПРАВЛЕНО на S16-384 как в вашем рабочем скрипте
TEACHER_ARCH = 'PE-Core-L14-336'
STUDENT_ARCH = 'PE-Core-T16-384'  # Было T16-384

# Teacher linear probe (фиксировано)
TEACHER_LR = 5e-4
TEACHER_EPOCHS = 10

# Эпохи для сохранения метрик
SAVE_EPOCHS = [1, 2, 5, 10, 20, 30]
TOTAL_EPOCHS = 30

# ==============================
# ВАРЬИРУЕМЫЕ ПАРАМЕТРЫ
# ==============================
 
EXPERIMENTS = [
    # Baseline (без дистилляции)
    {
        'name': 'baseline_no_distill_tiny',
        'temperature': 1.0,
        'kd_alpha': 0.0,  # только CE
        'kd_lr': 1e-4,
    },
    
    # Разные температуры (alpha=0.5)
    {
        'name': 'logit_T1_alpha0.5_tiny',
        'temperature': 1.0,
        'kd_alpha': 0.5,
        'kd_lr': 1e-4,
    },
    {
        'name': 'logit_T3_alpha0.5_tiny',
        'temperature': 3.0,
        'kd_alpha': 0.5,
        'kd_lr': 1e-4,
    },
    {
        'name': 'logit_T5_alpha0.5_tiny',
        'temperature': 5.0,
        'kd_alpha': 0.5,
        'kd_lr': 1e-4,
    },
    {
        'name': 'logit_T10_alpha0.5_tiny',
        'temperature': 10.0,
        'kd_alpha': 0.5,
        'kd_lr': 1e-4,
    },
    
    # Разные alpha (T=4.0)
    {
        'name': 'logit_T4_alpha0.3_tiny',
        'temperature': 4.0,
        'kd_alpha': 0.3,
        'kd_lr': 1e-4,
    },
    {
        'name': 'logit_T4_alpha0.7_tiny',
        'temperature': 4.0,
        'kd_alpha': 0.7,
        'kd_lr': 1e-4,
    },
    {
        'name': 'logit_T4_alpha0.9_tiny',
        'temperature': 4.0,
        'kd_alpha': 0.9,
        'kd_lr': 1e-4,
    },
]

# ==============================
# UTILITY FUNCTIONS
# ==============================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def extract_label_from_path(image_path, root_dir):
    parts = Path(image_path).parts
    try:
        ridx = parts.index(Path(root_dir).name)
        return parts[ridx + 1]
    except (ValueError, IndexError):
        return None

# ==============================
# DATASET
# ==============================

class KDDataset(Dataset):
    def __init__(self, image_ids, image_to_path, image_to_label, label_to_idx, t_transform, s_transform):
        self.ids = image_ids
        self.image_to_path = image_to_path
        self.image_to_label = image_to_label
        self.label_to_idx = label_to_idx
        self.t_transform = t_transform
        self.s_transform = s_transform
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = self.image_to_path[img_id]
        label = self.image_to_label[img_id]
        y = self.label_to_idx[label]
        img = Image.open(path).convert('RGB')
        x_t = self.t_transform(img)
        x_s = self.s_transform(img)
        return x_t, x_s, torch.tensor(y, dtype=torch.long)

# ==============================
# MODELS
# ==============================

class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# ==============================
# LOSS
# ==============================

class KDLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        T = self.T
        kd = self.kl(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1)
        ) * (T * T)
        ce = self.ce(student_logits, labels)
        total = self.alpha * kd + (1.0 - self.alpha) * ce
        return total, kd, ce

# ==============================
# TRAINING FUNCTIONS
# ==============================

def compute_gradient_norm(model):
    """Вычисление нормы градиента"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_teacher_probe(teacher_model, teacher_head, train_loader, val_loader, device):
    """Обучение teacher linear probe"""
    print('\n' + '='*80)
    print('ОБУЧЕНИЕ TEACHER LINEAR PROBE')
    print('='*80 + '\n')
    
    # Заморозка энкодера
    for p in teacher_model.parameters():
        p.requires_grad = False
    
    teacher_head.train()
    opt = torch.optim.AdamW(teacher_head.parameters(), lr=TEACHER_LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()
    best_val_acc = -1.0
    best_state = None
    
    for ep in range(1, TEACHER_EPOCHS + 1):
        teacher_head.train()
        loss_sum, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Teacher Probe Epoch {ep}/{TEACHER_EPOCHS}", ncols=100)
        
        for x_t, _, y in pbar:
            x_t, y = x_t.to(device), y.to(device)
            
            with torch.no_grad():
                feats = teacher_model.encode_image(x_t)
            
            logits = teacher_head(feats)
            loss = ce(logits, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            _, pred = torch.max(logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            loss_sum += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.0*correct/total:.2f}%'})
        
        train_acc = 100.0 * correct / total
        
        # Validation
        teacher_head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_t, _, y in val_loader:
                x_t, y = x_t.to(device), y.to(device)
                feats = teacher_model.encode_image(x_t)
                logits = teacher_head(feats)
                _, pred = torch.max(logits, 1)
                val_total += y.size(0)
                val_correct += (pred == y).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        print(f'Epoch {ep}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = teacher_head.state_dict()
    
    # Загрузка лучшего состояния
    if best_state is not None:
        teacher_head.load_state_dict(best_state)
    
    # Заморозка головы
    for p in teacher_head.parameters():
        p.requires_grad = False
    teacher_head.eval()
    
    print(f'\n✅ Teacher Probe обучен! Best Val Acc: {best_val_acc:.2f}%\n')
    
    return best_val_acc

def validate_student(student_model, student_head, teacher_model, teacher_head, 
                    loader, criterion, device):
    """Валидация студента"""
    student_model.eval()
    student_head.eval()
    teacher_model.eval()
    teacher_head.eval()
    
    total_loss = total_kd = total_ce = 0.0
    correct = total = 0
    
    with torch.no_grad():
        for x_t, x_s, y in loader:
            x_t, x_s, y = x_t.to(device), x_s.to(device), y.to(device)
            
            # Teacher logits
            t_feat = teacher_model.encode_image(x_t)
            t_logits = teacher_head(t_feat)
            
            # Student logits
            s_feat = student_model.encode_image(x_s)
            s_logits = student_head(s_feat)
            
            loss, kd, ce = criterion(s_logits, t_logits, y)
            
            total_loss += loss.item()
            total_kd += kd.item()
            total_ce += ce.item()
            
            _, pred = torch.max(s_logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    
    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_kd = total_kd / n_batches
    avg_ce = total_ce / n_batches
    acc = 100.0 * correct / total
    
    return avg_loss, avg_kd, avg_ce, acc

def run_experiment(exp_config, train_loader, val_loader, test_loader, 
                  teacher_model, teacher_head, num_classes, device):
    """Запуск одного эксперимента"""
    
    print('\n' + '='*80)
    print(f'🚀 ЭКСПЕРИМЕНТ: {exp_config["name"]}')
    print(f'   Temperature: {exp_config["temperature"]}')
    print(f'   Alpha: {exp_config["kd_alpha"]}')
    print(f'   LR: {exp_config["kd_lr"]}')
    print('='*80 + '\n')
    
    # Создаем новую модель студента
    student_model = pe.CLIP.from_config(STUDENT_ARCH, pretrained=True).to(device).float()
    student_dim = student_model.visual.output_dim
    student_head = LinearHead(student_dim, num_classes).to(device)
    
    # Loss и optimizer
    criterion = KDLoss(temperature=exp_config['temperature'], alpha=exp_config['kd_alpha'])
    optimizer = torch.optim.AdamW(
        list(student_model.parameters()) + list(student_head.parameters()),
        lr=exp_config['kd_lr'],
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    
    # Трекер метрик
    metrics_tracker = {
        'epochs': [],
        'train_loss': [],
        'train_kd': [],
        'train_ce': [],
        'val_loss': [],
        'val_kd': [],
        'val_ce': [],
        'val_accuracy': [],
        'gradient_norm': [],
        'learning_rate': []
    }
    
    best_val_acc = -1.0
    
    # Обучение
    for epoch in range(1, TOTAL_EPOCHS + 1):
        student_model.train()
        student_head.train()
        
        loss_sum = kd_sum = ce_sum = grad_norm_sum = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}", ncols=120, leave=False)
        
        for x_t, x_s, y in pbar:
            x_t, x_s, y = x_t.to(device), x_s.to(device), y.to(device)
            
            # Teacher logits (frozen)
            with torch.no_grad():
                t_feat = teacher_model.encode_image(x_t)
                t_logits = teacher_head(t_feat)
            
            # Student logits
            s_feat = student_model.encode_image(x_s)
            s_logits = student_head(s_feat)
            
            loss, kd, ce = criterion(s_logits, t_logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient norm
            grad_norm = compute_gradient_norm(student_model) + compute_gradient_norm(student_head)
            
            optimizer.step()
            
            loss_sum += loss.item()
            kd_sum += kd.item()
            ce_sum += ce.item()
            grad_norm_sum += grad_norm
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'kd': f'{kd.item():.4f}',
                'ce': f'{ce.item():.4f}'
            })
        
        pbar.close()
        
        avg_train_loss = loss_sum / n_batches
        avg_train_kd = kd_sum / n_batches
        avg_train_ce = ce_sum / n_batches
        avg_grad_norm = grad_norm_sum / n_batches
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation
        val_loss, val_kd, val_ce, val_acc = validate_student(
            student_model, student_head, teacher_model, teacher_head,
            val_loader, criterion, device
        )
        
        scheduler.step()
        
        # Сохранение метрик на контрольных эпохах
        if epoch in SAVE_EPOCHS:
            metrics_tracker['epochs'].append(epoch)
            metrics_tracker['train_loss'].append(avg_train_loss)
            metrics_tracker['train_kd'].append(avg_train_kd)
            metrics_tracker['train_ce'].append(avg_train_ce)
            metrics_tracker['val_loss'].append(val_loss)
            metrics_tracker['val_kd'].append(val_kd)
            metrics_tracker['val_ce'].append(val_ce)
            metrics_tracker['val_accuracy'].append(val_acc)
            metrics_tracker['gradient_norm'].append(avg_grad_norm)
            metrics_tracker['learning_rate'].append(current_lr)
            
            print(f'✓ Epoch {epoch}: Val Acc={val_acc:.2f}% | Loss={val_loss:.4f} | GradNorm={avg_grad_norm:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Test evaluation
    student_model.eval()
    student_head.eval()
    test_correct, test_total = 0, 0
    
    with torch.no_grad():
        for _, x_s, y in test_loader:
            x_s, y = x_s.to(device), y.to(device)
            s_feat = student_model.encode_image(x_s)
            logits = student_head(s_feat)
            _, pred = torch.max(logits, 1)
            test_total += y.size(0)
            test_correct += (pred == y).sum().item()
    
    test_acc = 100.0 * test_correct / test_total
    
    # Вычисляем teacher CE на валидации
    teacher_model.eval()
    teacher_head.eval()
    teacher_ce_loss = 0.0
    
    with torch.no_grad():
        for x_t, _, y in val_loader:
            x_t, y = x_t.to(device), y.to(device)
            t_feat = teacher_model.encode_image(x_t)
            t_logits = teacher_head(t_feat)
            loss = F.cross_entropy(t_logits, y)
            teacher_ce_loss += loss.item()
    
    teacher_ce_loss /= len(val_loader)
    
    # Подсчет параметров
    student_params = sum(p.numel() for p in student_model.parameters())
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    
    # Сохранение результатов
    results = {
        'run_id': f"{exp_config['name']}_seed{SEED}",
        'timestamp': datetime.utcnow().isoformat(),
        
        # Dataset
        'dataset_name': 'kvasir-v2',
        'eval_split': 'validation',
        'num_classes': num_classes,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        
        # Models
        'teacher_model_name': TEACHER_ARCH,
        'teacher_parameters': teacher_params,
        'teacher_cross_entropy': float(teacher_ce_loss),
        
        'student_model_name': STUDENT_ARCH,
        'student_parameters': student_params,
        'student_cross_entropy': float(metrics_tracker['val_ce'][-1]),
        'student_accuracy': float(metrics_tracker['val_accuracy'][-1]),
        
        # Distillation
        'distillation_method': 'logit',
        'distillation_tokens': len(train_loader.dataset) * TOTAL_EPOCHS,
        
        # Training config
        'random_seed': SEED,
        'num_epochs': TOTAL_EPOCHS,
        'batch_size': BATCH_SIZE,
        
        # Hyperparameters
        'hyperparameters': {
            'temperature': exp_config['temperature'],
            'kd_alpha': exp_config['kd_alpha'],
            'kd_lr': exp_config['kd_lr'],
            'teacher_epochs': TEACHER_EPOCHS,
            'teacher_lr': TEACHER_LR,
            'weight_decay': WEIGHT_DECAY,
            'optimizer': OPTIMIZER_TYPE,
            'scheduler': SCHEDULER
        },
        
        # Results
        'test_accuracy': float(test_acc),
        'best_val_acc': float(best_val_acc),
        
        # Metrics
        'metrics': metrics_tracker,
        'save_epochs': SAVE_EPOCHS
    }
    
    # Сохранение
    results_dir = Path('../results/logit_distillation_T')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"{exp_config['name']}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Сохранение весов модели
    weights_dir = Path('../model_weights')
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = f"{exp_config['name']}_best.pth"
    model_path = weights_dir / model_name
    
    # Сохраняем best модель (по val_acc)
    torch.save({
        'student_model': student_model.state_dict(),
        'student_head': student_head.state_dict(),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'config': exp_config
    }, model_path)
    
    print(f'   Веса сохранены: {model_path}')
    
    print(f'\n✅ Эксперимент завершен!')
    print(f'   Best Val Acc: {best_val_acc:.2f}%')
    print(f'   Test Acc: {test_acc:.2f}%')
    print(f'   Результаты: {results_file}\n')
    
    return results

# ==============================
# MAIN
# ==============================

def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')
    
    # Загрузка dataset
    print('Подготовка датасета...')
    all_image_paths = glob(os.path.join(ROOT_DIR, '**', '*.jpg'), recursive=True)
    
    image_to_path, image_to_label = {}, {}
    for p in all_image_paths:
        image_id = os.path.splitext(os.path.basename(p))[0]
        label = extract_label_from_path(p, ROOT_DIR)
        if label is not None:
            image_to_path[image_id] = p
            image_to_label[image_id] = label
    
    classes = sorted(list(set(image_to_label.values())))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    
    print(f'Найдено {len(image_to_label)} изображений, {num_classes} классов')
    
    # Сбалансированное подмножество
    images_by_class = defaultdict(list)
    for img_id, lbl in image_to_label.items():
        images_by_class[lbl].append(img_id)
    
    images_per_class = 500
    sampled = []
    for lbl, ids in images_by_class.items():
        k = min(images_per_class, len(ids))
        sampled.extend(random.sample(ids, k))
    random.shuffle(sampled)
    
    # Split
    labels_for_strat = [image_to_label[i] for i in sampled]
    train_val_ids, test_ids = train_test_split(sampled, test_size=0.20, 
                                                stratify=labels_for_strat, random_state=SEED)
    train_labels_for_strat = [image_to_label[i] for i in train_val_ids]
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.20, 
                                          stratify=train_labels_for_strat, random_state=SEED)
    
    print(f'Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}')
    
    # Загрузка моделей
    print('\nЗагрузка моделей...')
    teacher_model = pe.CLIP.from_config(TEACHER_ARCH, pretrained=True).to(device).float().eval()
    teacher_preprocessor = transforms.get_image_transform(teacher_model.image_size)
    teacher_dim = teacher_model.visual.output_dim
    
    # Для студента используем размер 384
    student_preprocessor = transforms.get_image_transform(384)
    
    # Datasets
    train_ds = KDDataset(train_ids, image_to_path, image_to_label, label_to_idx, 
                        teacher_preprocessor, student_preprocessor)
    val_ds = KDDataset(val_ids, image_to_path, image_to_label, label_to_idx, 
                      teacher_preprocessor, student_preprocessor)
    test_ds = KDDataset(test_ids, image_to_path, image_to_label, label_to_idx, 
                       teacher_preprocessor, student_preprocessor)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    # Teacher head
    teacher_head = LinearHead(teacher_dim, num_classes).to(device)
    train_teacher_probe(teacher_model, teacher_head, train_loader, val_loader, device)
    
    # Запуск экспериментов
    all_results = []
    for exp_config in EXPERIMENTS:
        results = run_experiment(
            exp_config, train_loader, val_loader, test_loader,
            teacher_model, teacher_head, num_classes, device
        )
        all_results.append(results)
        
        # Очистка памяти
        torch.cuda.empty_cache()
    
    print('\n' + '='*80)
    print('🎉 ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!')
    print('='*80)
    print(f'\nРезультаты сохранены в: ../results/logit_distillation/')
    print(f'Всего экспериментов: {len(all_results)}\n')

if __name__ == '__main__':
    main()