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
# ПУТЬ К PERCEPTION_MODELS
# ==============================

sys.path.insert(0, str(Path('../perception_models').resolve()))

try:
    from core.vision_encoder import pe
    from core.vision_encoder import transforms
    print("✅ Модули perception_models импортированы успешно\n")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
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

# Архитектуры (фиксированы)
TEACHER_ARCH = 'PE-Core-L14-336'
STUDENT_ARCH = 'PE-Core-T16-384'

# Эпохи для сохранения метрик
SAVE_EPOCHS = [1, 2, 5, 10, 20, 30]
TOTAL_EPOCHS = 30

# ==============================
# ВАРЬИРУЕМЫЕ ПАРАМЕТРЫ
# ==============================

EXPERIMENTS = [
    # Baseline (без дистилляции) - только classification loss
    {
        'name': 'feature_baseline_no_distill_tiny',
        'temperature': 1.0,
        'alpha': 0.0,  # только classification loss
        'learning_rate': 1e-4,
        'projection': False
    },
    
    # Feature distillation с разными alpha (с проекцией)
    {
        'name': 'feature_alpha0.3_proj_tiny',
        'temperature': 4.0,
        'alpha': 0.3,
        'learning_rate': 1e-4,
        'projection': True
    },
    {
        'name': 'feature_alpha0.5_proj_tiny',
        'temperature': 4.0,
        'alpha': 0.5,
        'learning_rate': 1e-4,
        'projection': True
    },
    {
        'name': 'feature_alpha0.7_proj_tiny',
        'temperature': 4.0,
        'alpha': 0.7,
        'learning_rate': 1e-4,
        'projection': True
    },
    {
        'name': 'feature_alpha0.9_proj_tiny',
        'temperature': 4.0,
        'alpha': 0.9,
        'learning_rate': 1e-4,
        'projection': True
    },
    
    # Feature distillation с разными temperature (cosine similarity)
    {
        'name': 'feature_T2_alpha0.5_tiny',
        'temperature': 2.0,
        'alpha': 0.5,
        'learning_rate': 1e-4,
        'projection': True
    },
    {
        'name': 'feature_T6_alpha0.5_tiny',
        'temperature': 6.0,
        'alpha': 0.5,
        'learning_rate': 1e-4,
        'projection': True
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

class FeatureDistillationDataset(Dataset):
    def __init__(self, image_ids, image_to_path, image_to_label, label_to_idx, 
                 t_transform, s_transform):
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

class ClassificationHead(nn.Module):
    """Голова для классификации"""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

class ProjectionHead(nn.Module):
    """Проекционная голова для выравнивания размерностей"""
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.projection = nn.Linear(student_dim, teacher_dim)
    
    def forward(self, x):
        return self.projection(x)

# ==============================
# LOSS
# ==============================

class FeatureDistillationLoss(nn.Module):
    """
    Feature Distillation Loss
    
    Комбинирует:
    1. Feature matching loss (cosine similarity или MSE)
    2. Classification loss (cross-entropy)
    """
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_features, teacher_features, student_logits, labels):
        """
        Args:
            student_features: признаки студента [batch, dim] (после проекции если есть)
            teacher_features: признаки учителя [batch, dim]
            student_logits: логиты студента [batch, num_classes]
            labels: метки [batch]
        
        Returns:
            total_loss, feature_loss, classification_loss
        """
        
        # Classification loss (всегда есть)
        classification_loss = self.ce_loss(student_logits, labels)
        
        # Если alpha = 0, то только classification
        if self.alpha == 0.0:
            return classification_loss, torch.tensor(0.0, device=labels.device), classification_loss
        
        # Feature distillation loss (cosine similarity)
        # Нормализуем признаки
        student_norm = F.normalize(student_features, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
        
        # Cosine similarity: [-1, 1], максимизируем -> минимизируем (1 - cosine)
        cosine_sim = F.cosine_similarity(student_norm, teacher_norm, dim=-1)  # [batch]
        feature_loss = (1.0 - cosine_sim).mean()
        
        # Total loss
        total_loss = self.alpha * feature_loss + (1.0 - self.alpha) * classification_loss
        
        return total_loss, feature_loss, classification_loss

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

def validate_student(student_model, classifier, projection, teacher_model, 
                    loader, criterion, device, use_projection):
    """Валидация студента"""
    student_model.eval()
    classifier.eval()
    if use_projection:
        projection.eval()
    teacher_model.eval()
    
    total_loss = total_feature_loss = total_cls_loss = 0.0
    correct = total = 0
    
    with torch.no_grad():
        for x_t, x_s, y in loader:
            x_t, x_s, y = x_t.to(device), x_s.to(device), y.to(device)
            
            # Teacher features
            teacher_features = teacher_model.encode_image(x_t)
            
            # Student features
            student_features = student_model.encode_image(x_s)
            
            # Projection (если используется)
            if use_projection:
                student_features_proj = projection(student_features)
            else:
                student_features_proj = student_features
            
            # Logits
            logits = classifier(student_features)
            
            # Loss
            loss, feature_loss, cls_loss = criterion(
                student_features_proj, teacher_features, logits, y
            )
            
            total_loss += loss.item()
            total_feature_loss += feature_loss.item()
            total_cls_loss += cls_loss.item()
            
            # Accuracy
            _, pred = torch.max(logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    
    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_feature_loss = total_feature_loss / n_batches
    avg_cls_loss = total_cls_loss / n_batches
    acc = 100.0 * correct / total
    
    return avg_loss, avg_feature_loss, avg_cls_loss, acc

def run_experiment(exp_config, train_loader, val_loader, test_loader,
                  teacher_model, num_classes, device):
    """Запуск одного эксперимента"""
    
    print('\n' + '='*80)
    print(f'🚀 ЭКСПЕРИМЕНТ: {exp_config["name"]}')
    print(f'   Alpha: {exp_config["alpha"]}')
    print(f'   Temperature: {exp_config["temperature"]}')
    print(f'   Projection: {exp_config["projection"]}')
    print(f'   LR: {exp_config["learning_rate"]}')
    print('='*80 + '\n')
    
    # Создаем модель студента
    student_model = pe.CLIP.from_config(STUDENT_ARCH, pretrained=True).to(device).float()
    student_dim = student_model.visual.output_dim
    teacher_dim = teacher_model.visual.output_dim
    
    print(f'   Student dim: {student_dim}, Teacher dim: {teacher_dim}')
    
    # Classifier
    classifier = ClassificationHead(student_dim, num_classes).to(device)
    
    # Projection head (опционально)
    use_projection = exp_config.get('projection', False)
    
    # ВАЖНО: Если alpha > 0 и размерности не совпадают, ТРЕБУЕТСЯ проекция
    if exp_config['alpha'] > 0 and student_dim != teacher_dim and not use_projection:
        print(f'   ⚠️  Размерности не совпадают! Включаем projection автоматически.')
        use_projection = True
    
    if use_projection:
        projection = ProjectionHead(student_dim, teacher_dim).to(device)
        print(f'   ✅ Projection: {student_dim} → {teacher_dim}')
    else:
        projection = None
        print(f'   ❌ Projection: отключена')
    
    # Loss
    criterion = FeatureDistillationLoss(
        temperature=exp_config['temperature'],
        alpha=exp_config['alpha']
    )
    
    # Optimizer
    trainable_params = list(student_model.parameters()) + list(classifier.parameters())
    if use_projection:
        trainable_params += list(projection.parameters())
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=exp_config['learning_rate'],
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    
    # Трекер метрик
    metrics_tracker = {
        'epochs': [],
        'train_loss': [],
        'train_feature_loss': [],
        'train_cls_loss': [],
        'val_loss': [],
        'val_feature_loss': [],
        'val_cls_loss': [],
        'val_accuracy': [],
        'gradient_norm': [],
        'learning_rate': []
    }
    
    best_val_acc = -1.0
    
    # Обучение
    for epoch in range(1, TOTAL_EPOCHS + 1):
        student_model.train()
        classifier.train()
        if use_projection:
            projection.train()
        teacher_model.eval()
        
        loss_sum = feature_loss_sum = cls_loss_sum = grad_norm_sum = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}", ncols=120, leave=False)
        
        for x_t, x_s, y in pbar:
            x_t, x_s, y = x_t.to(device), x_s.to(device), y.to(device)
            
            # Teacher features (frozen)
            with torch.no_grad():
                teacher_features = teacher_model.encode_image(x_t)
            
            # Student features
            student_features = student_model.encode_image(x_s)
            
            # Projection (если используется)
            if use_projection:
                student_features_proj = projection(student_features)
            else:
                student_features_proj = student_features
            
            # Logits
            logits = classifier(student_features)
            
            # Loss
            loss, feature_loss, cls_loss = criterion(
                student_features_proj, teacher_features, logits, y
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient norm
            grad_norm = compute_gradient_norm(student_model) + compute_gradient_norm(classifier)
            if use_projection:
                grad_norm += compute_gradient_norm(projection)
            
            optimizer.step()
            
            loss_sum += loss.item()
            feature_loss_sum += feature_loss.item()
            cls_loss_sum += cls_loss.item()
            grad_norm_sum += grad_norm
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'feat': f'{feature_loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}'
            })
        
        pbar.close()
        
        avg_train_loss = loss_sum / n_batches
        avg_train_feature = feature_loss_sum / n_batches
        avg_train_cls = cls_loss_sum / n_batches
        avg_grad_norm = grad_norm_sum / n_batches
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation
        val_loss, val_feature, val_cls, val_acc = validate_student(
            student_model, classifier, projection, teacher_model,
            val_loader, criterion, device, use_projection
        )
        
        scheduler.step()
        
        # Сохранение метрик на контрольных эпохах
        if epoch in SAVE_EPOCHS:
            metrics_tracker['epochs'].append(epoch)
            metrics_tracker['train_loss'].append(avg_train_loss)
            metrics_tracker['train_feature_loss'].append(avg_train_feature)
            metrics_tracker['train_cls_loss'].append(avg_train_cls)
            metrics_tracker['val_loss'].append(val_loss)
            metrics_tracker['val_feature_loss'].append(val_feature)
            metrics_tracker['val_cls_loss'].append(val_cls)
            metrics_tracker['val_accuracy'].append(val_acc)
            metrics_tracker['gradient_norm'].append(avg_grad_norm)
            metrics_tracker['learning_rate'].append(current_lr)
            
            print(f'✓ Epoch {epoch}: Val Acc={val_acc:.2f}% | Feature={val_feature:.4f} | Cls={val_cls:.4f} | GradNorm={avg_grad_norm:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Test evaluation
    student_model.eval()
    classifier.eval()
    if use_projection:
        projection.eval()
    
    test_correct, test_total = 0, 0
    
    with torch.no_grad():
        for _, x_s, y in test_loader:
            x_s, y = x_s.to(device), y.to(device)
            student_features = student_model.encode_image(x_s)
            logits = classifier(student_features)
            _, pred = torch.max(logits, 1)
            test_total += y.size(0)
            test_correct += (pred == y).sum().item()
    
    test_acc = 100.0 * test_correct / test_total
    
    # Вычисляем cosine similarity между признаками на валидации
    avg_cosine_sim = 0.0
    avg_mse = 0.0
    
    if use_projection:  # ← ДОБАВЛЕНО: проверяем наличие проекции
        n_samples = 0
        with torch.no_grad():
            for x_t, x_s, _ in val_loader:
                x_t, x_s = x_t.to(device), x_s.to(device)
                
                teacher_features = teacher_model.encode_image(x_t)
                student_features = student_model.encode_image(x_s)
                
                # Применяем проекцию
                student_features = projection(student_features)
                
                # Cosine similarity
                teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
                student_norm = F.normalize(student_features, p=2, dim=-1)
                cosine_sim = F.cosine_similarity(teacher_norm, student_norm, dim=-1)
                avg_cosine_sim += cosine_sim.sum().item()
                
                # MSE
                mse = F.mse_loss(student_features, teacher_features, reduction='none').mean(dim=1)
                avg_mse += mse.sum().item()
                
                n_samples += x_t.size(0)
        
        avg_cosine_sim /= n_samples
        avg_mse /= n_samples
    else:
        # Если нет проекции (baseline), устанавливаем значения по умолчанию
        avg_cosine_sim = float('nan')  # или 0.0
        avg_mse = float('nan')  # или 0.0
    
    # Подсчет параметров
    student_params = sum(p.numel() for p in student_model.parameters())
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    classifier_params = sum(p.numel() for p in classifier.parameters())
    projection_params = sum(p.numel() for p in projection.parameters()) if use_projection else 0
    total_student_params = student_params + classifier_params + projection_params
    
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
        
        'student_model_name': STUDENT_ARCH,
        'student_parameters': total_student_params,
        'student_cross_entropy': float(metrics_tracker['val_cls_loss'][-1]),
        'student_accuracy': float(metrics_tracker['val_accuracy'][-1]),
        
        # Distillation
        'distillation_method': 'feature',
        'distillation_tokens': len(train_loader.dataset) * TOTAL_EPOCHS,
        
        # Training config
        'random_seed': SEED,
        'num_epochs': TOTAL_EPOCHS,
        'batch_size': BATCH_SIZE,
        
        # Hyperparameters
        'hyperparameters': {
            'alpha': exp_config['alpha'],
            'temperature': exp_config['temperature'],
            'learning_rate': exp_config['learning_rate'],
            'weight_decay': WEIGHT_DECAY,
            'optimizer': OPTIMIZER_TYPE,
            'scheduler': SCHEDULER,
            'use_projection': use_projection,
            'teacher_dim': teacher_dim,
            'student_dim': student_dim
        },
        
        # Results
        'test_accuracy': float(test_acc),
        'best_val_acc': float(best_val_acc),
        'avg_cosine_similarity': float(avg_cosine_sim) if use_projection else None,  # ← ИЗМЕНЕНО
        'avg_mse_distance': float(avg_mse) if use_projection else None,  # ← ИЗМЕНЕНО
        
        # Metrics
        'metrics': metrics_tracker,
        'save_epochs': SAVE_EPOCHS
    }
    
    # Сохранение
    results_dir = Path('../results/feature_distillation_T')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"{exp_config['name']}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


    weights_dir = Path('../model_weights')
    weights_dir.mkdir(parents=True, exist_ok=True)

    model_name = f"{exp_config['name']}_best.pth"
    model_path = weights_dir / model_name

    torch.save({
        'student_model': student_model.state_dict(),
        'classifier': classifier.state_dict(),
        'projection': projection.state_dict() if use_projection else None,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'config': exp_config,
        'student_arch': STUDENT_ARCH,
        'teacher_arch': TEACHER_ARCH,
        'use_projection': use_projection
    }, model_path)

    print(f'   Веса сохранены: {model_path}')
    
    
    print(f'\n✅ Эксперимент завершен!')
    print(f'   Best Val Acc: {best_val_acc:.2f}%')
    print(f'   Test Acc: {test_acc:.2f}%')
    if use_projection:
        print(f'   Avg Cosine Similarity: {avg_cosine_sim:.4f}')
        print(f'   Avg MSE Distance: {avg_mse:.4f}')
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
    
    student_preprocessor = transforms.get_image_transform(384)
    
    # Datasets
    train_ds = FeatureDistillationDataset(train_ids, image_to_path, image_to_label, label_to_idx,
                                         teacher_preprocessor, student_preprocessor)
    val_ds = FeatureDistillationDataset(val_ids, image_to_path, image_to_label, label_to_idx,
                                       teacher_preprocessor, student_preprocessor)
    test_ds = FeatureDistillationDataset(test_ids, image_to_path, image_to_label, label_to_idx,
                                        teacher_preprocessor, student_preprocessor)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    # Запуск экспериментов
    all_results = []
    for exp_config in EXPERIMENTS:
        results = run_experiment(
            exp_config, train_loader, val_loader, test_loader,
            teacher_model, num_classes, device
        )
        all_results.append(results)
        
        # Очистка памяти
        torch.cuda.empty_cache()
    
    print('\n' + '='*80)
    print('🎉 ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!')
    print('='*80)
    print(f'\nРезультаты сохранены в: ../results/feature_distillation_tiny/')
    print(f'Всего экспериментов: {len(all_results)}\n')

if __name__ == '__main__':
    main()