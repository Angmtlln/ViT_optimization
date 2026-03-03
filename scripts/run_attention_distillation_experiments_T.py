import os, sys, json, random, time, math
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

# Обучение teacher probe
TEACHER_LR = 5e-4
TEACHER_EPOCHS = 5

# ==============================
# ВАРЬИРУЕМЫЕ ПАРАМЕТРЫ
# ==============================

EXPERIMENTS = [
    # Baseline (без attention distillation) - только CE
    {
        'name': 'attention_baseline_no_distill_tiny',
        'alpha': 0.0,  # только classification loss
        'learning_rate': 1e-4,
        'attention_loss_type': 'mse'
    },
    
    # Attention distillation с разными alpha (MSE)
    {
        'name': 'attention_alpha0.3_mse_tiny',
        'alpha': 0.3,
        'learning_rate': 1e-4,
        'attention_loss_type': 'mse'
    },
    {
        'name': 'attention_alpha0.5_mse_tiny',
        'alpha': 0.5,
        'learning_rate': 1e-4,
        'attention_loss_type': 'mse'
    },
    {
        'name': 'attention_alpha0.6_mse_tiny',
        'alpha': 0.6,
        'learning_rate': 1e-4,
        'attention_loss_type': 'mse'
    },
    {
        'name': 'attention_alpha0.7_mse_tiny',
        'alpha': 0.7,
        'learning_rate': 1e-4,
        'attention_loss_type': 'mse'
    },
    
    # Attention distillation с разными типами loss (alpha=0.5)
    {
        'name': 'attention_alpha0.5_l1_tiny',
        'alpha': 0.5,
        'learning_rate': 1e-4,
        'attention_loss_type': 'l1'
    },
    {
        'name': 'attention_alpha0.5_cosine_tiny',
        'alpha': 0.5,
        'learning_rate': 1e-4,
        'attention_loss_type': 'cosine'
    },
    
    # Разные learning rates (alpha=0.6, MSE)
    {
        'name': 'attention_lr5e-5_tiny',
        'alpha': 0.6,
        'learning_rate': 5e-5,
        'attention_loss_type': 'mse'
    },
    {
        'name': 'attention_lr2e-4_tiny',
        'alpha': 0.6,
        'learning_rate': 2e-4,
        'attention_loss_type': 'mse'
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

class AttentionDistillationDataset(Dataset):
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

# ==============================
# ATTENTION MAP EXTRACTION
# ==============================

class ModuleOutputHook:
    """Hook для извлечения промежуточных выходов"""
    def __init__(self, module):
        self.outputs = None
        self.handle = module.register_forward_hook(self._hook)
    
    def _hook(self, module, inputs, output):
        self.outputs = output
    
    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

def get_transformer_module(clip_model):
    """Получение модуля трансформера из CLIP модели"""
    visual = getattr(clip_model, 'visual', None)
    assert visual is not None, 'visual не найден у CLIP-модели'
    transformer = getattr(visual, 'transformer', None)
    assert transformer is not None, 'visual.transformer не найден (ожидается ViT)'
    return transformer

def tokens_to_attention_map(tokens: torch.Tensor) -> torch.Tensor:
    """
    Преобразование токенов трансформера в карту внимания
    
    Args:
        tokens: [B, N, D] или [B, N+1, D] (если есть CLS токен)
    
    Returns:
        attention_map: [B, H, W] - нормированная карта внимания
    """
    assert tokens.dim() == 3, 'Ожидается тензор с формой [B, N, D]'
    B, N, D = tokens.shape
    
    # Определяем размерность сетки патчей
    s = int(math.sqrt(N))
    if s * s == N:
        # CLS токена нет
        patch_tokens = tokens
        H = W = s
    else:
        # Первый токен - CLS, остальные - патчи
        Np = N - 1
        H = W = int(math.sqrt(Np))
        assert H * W == Np, f'Число патч-токенов {Np} не квадрат: не удается построить карту'
        patch_tokens = tokens[:, 1:, :]  # Убираем CLS токен
    
    # Reshape в пространственную сетку
    fmap = patch_tokens.view(B, H, W, D)
    
    # Создаем карту внимания как сумму квадратов по каналам
    attn = fmap.pow(2).sum(dim=-1)  # [B, H, W]
    
    # L2-нормализация по карте для масштаб-инвариантности
    attn = attn / (attn.norm(p=2, dim=(1, 2), keepdim=True) + 1e-6)
    
    return attn

def match_attention_maps(attn_s: torch.Tensor, attn_t: torch.Tensor, 
                        loss_type='mse') -> torch.Tensor:
    """
    Приведение карт внимания к одному разрешению и вычисление loss
    
    Args:
        attn_s: карта студента [B, H_s, W_s]
        attn_t: карта учителя [B, H_t, W_t]
        loss_type: тип loss ('mse', 'l1', 'cosine')
    
    Returns:
        loss (scalar)
    """
    # Приводим к одному размеру (используем размер студента)
    if attn_s.shape != attn_t.shape:
        attn_t_resized = F.interpolate(
            attn_t.unsqueeze(1), 
            size=attn_s.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
    else:
        attn_t_resized = attn_t
    
    # Вычисляем loss
    if loss_type == 'mse':
        return F.mse_loss(attn_s, attn_t_resized)
    elif loss_type == 'l1':
        return F.l1_loss(attn_s, attn_t_resized)
    elif loss_type == 'cosine':
        # Cosine similarity loss (минимизируем 1 - cosine)
        attn_s_flat = attn_s.view(attn_s.size(0), -1)
        attn_t_flat = attn_t_resized.view(attn_t_resized.size(0), -1)
        cosine_sim = F.cosine_similarity(attn_s_flat, attn_t_flat, dim=1)
        return (1.0 - cosine_sim).mean()
    else:
        raise ValueError(f"Неподдерживаемый loss_type: {loss_type}")

# ==============================
# LOSS
# ==============================

class AttentionDistillationLoss(nn.Module):
    """
    Attention Distillation Loss
    
    Комбинирует:
    1. Attention matching loss (MSE/L1/Cosine между картами внимания)
    2. Classification loss (cross-entropy)
    """
    def __init__(self, alpha=0.6, attention_loss_type='mse'):
        super().__init__()
        self.alpha = alpha
        self.attention_loss_type = attention_loss_type
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, labels, attn_student, attn_teacher):
        """
        Args:
            student_logits: логиты студента [batch, num_classes]
            labels: метки [batch]
            attn_student: карта внимания студента [batch, H_s, W_s]
            attn_teacher: карта внимания учителя [batch, H_t, W_t]
        
        Returns:
            total_loss, attention_loss, classification_loss
        """
        # Classification loss (всегда есть)
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Если alpha = 0, то только classification
        if self.alpha == 0.0:
            return ce_loss, torch.tensor(0.0, device=labels.device), ce_loss
        
        # Attention distillation loss
        attention_loss = match_attention_maps(
            attn_student, attn_teacher, 
            loss_type=self.attention_loss_type
        )
        
        # Total loss
        total_loss = self.alpha * attention_loss + (1.0 - self.alpha) * ce_loss
        
        return total_loss, attention_loss, ce_loss

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

def train_teacher_probe(model, head, train_loader, val_loader, epochs, lr, device):
    """Обучение линейной головы учителя (энкодер заморожен)"""
    # Замораживаем энкодер
    for p in model.parameters():
        p.requires_grad = False
    
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()
    
    best_val_acc = -1.0
    best_state = None
    
    print('\nОбучение Teacher Linear Probe...')
    
    for ep in range(1, epochs + 1):
        head.train()
        loss_sum, n_batches, correct, total = 0.0, 0, 0, 0
        
        for x_t, _, y in train_loader:
            x_t = x_t.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.no_grad():
                feats = model.encode_image(x_t)
            
            logits = head(feats)
            loss = ce(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, pred = torch.max(logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            loss_sum += loss.item()
            n_batches += 1
        
        train_acc = 100.0 * correct / total if total > 0 else 0.0
        
        # Validation
        head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_t, _, y in val_loader:
                x_t = x_t.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                feats = model.encode_image(x_t)
                logits = head(feats)
                _, pred = torch.max(logits, 1)
                val_total += y.size(0)
                val_correct += (pred == y).sum().item()
        
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        
        print(f'Epoch {ep}/{epochs} | TrainAcc={train_acc:.2f}% | ValAcc={val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = head.state_dict()
    
    # Загружаем лучшее состояние
    if best_state is not None:
        head.load_state_dict(best_state)
    
    # Замораживаем голову
    for p in head.parameters():
        p.requires_grad = False
    head.eval()
    
    print(f'✓ Teacher probe обучен (лучшая ValAcc={best_val_acc:.2f}%)\n')
    
    return best_val_acc

def validate_attention(teacher_model, student_model,
                      teacher_head, student_head,
                      teacher_hook, student_hook,
                      loader, criterion, device):
    """Валидация студента с attention distillation"""
    teacher_model.eval()
    student_model.eval()
    teacher_head.eval()
    student_head.eval()
    
    total_loss = total_attn = total_ce = 0.0
    correct = total = n_batches = 0
    
    with torch.no_grad():
        for x_t, x_s, y in loader:
            x_t = x_t.to(device, non_blocking=True)
            x_s = x_s.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Teacher forward (активирует hook)
            t_feat = teacher_model.encode_image(x_t)
            _ = teacher_head(t_feat)
            
            # Student forward (активирует hook)
            s_feat = student_model.encode_image(x_s)
            s_logits = student_head(s_feat)
            
            # Извлекаем токены из hooks
            t_tokens = teacher_hook.outputs
            s_tokens = student_hook.outputs
            
            if isinstance(t_tokens, (tuple, list)):
                t_tokens = t_tokens[0]
            if isinstance(s_tokens, (tuple, list)):
                s_tokens = s_tokens[0]
            
            # Строим карты внимания
            attn_t = tokens_to_attention_map(t_tokens)
            attn_s = tokens_to_attention_map(s_tokens)
            
            # Loss
            loss, attn_loss, ce_loss = criterion(s_logits, y, attn_s, attn_t)
            
            total_loss += loss.item()
            total_attn += attn_loss.item()
            total_ce += ce_loss.item()
            
            _, pred = torch.max(s_logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            n_batches += 1
    
    avg_loss = total_loss / max(1, n_batches)
    avg_attn = total_attn / max(1, n_batches)
    avg_ce = total_ce / max(1, n_batches)
    acc = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, avg_attn, avg_ce, acc

def run_experiment(exp_config, train_loader, val_loader, test_loader,
                  teacher_model, teacher_head, num_classes, device):
    """Запуск одного эксперимента"""
    
    print('\n' + '='*80)
    print(f'🚀 ЭКСПЕРИМЕНТ: {exp_config["name"]}')
    print(f'   Alpha: {exp_config["alpha"]}')
    print(f'   Attention Loss Type: {exp_config["attention_loss_type"]}')
    print(f'   LR: {exp_config["learning_rate"]}')
    print('='*80 + '\n')
    
    # Создаем модель студента
    student_model = pe.CLIP.from_config(STUDENT_ARCH, pretrained=True).to(device).float()
    student_dim = student_model.visual.output_dim
    teacher_dim = teacher_model.visual.output_dim
    
    print(f'   Student dim: {student_dim}, Teacher dim: {teacher_dim}')
    
    # Classification head
    student_head = ClassificationHead(student_dim, num_classes).to(device)
    
    # Регистрируем hooks для извлечения токенов трансформера
    teacher_transformer = get_transformer_module(teacher_model)
    student_transformer = get_transformer_module(student_model)
    
    teacher_hook = ModuleOutputHook(teacher_transformer)
    student_hook = ModuleOutputHook(student_transformer)
    
    print(f'   ✅ Зарегистрированы forward-хуки на visual.transformer')
    
    # Loss
    criterion = AttentionDistillationLoss(
        alpha=exp_config['alpha'],
        attention_loss_type=exp_config['attention_loss_type']
    )
    
    # Optimizer (обучаем только студента)
    for p in teacher_model.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        list(student_model.parameters()) + list(student_head.parameters()),
        lr=exp_config['learning_rate'],
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    
    # Трекер метрик
    metrics_tracker = {
        'epochs': [],
        'train_loss': [],
        'train_attention_loss': [],
        'train_ce_loss': [],
        'val_loss': [],
        'val_attention_loss': [],
        'val_ce_loss': [],
        'val_accuracy': [],
        'gradient_norm': [],
        'learning_rate': []
    }
    
    best_val_acc = -1.0
    best_student_state = None
    best_student_head_state = None
    
    # Обучение
    for epoch in range(1, TOTAL_EPOCHS + 1):
        student_model.train()
        student_head.train()
        teacher_model.eval()
        teacher_head.eval()
        
        loss_sum = attn_sum = ce_sum = grad_norm_sum = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}", ncols=120, leave=False)
        
        for x_t, x_s, y in pbar:
            x_t, x_s, y = x_t.to(device), x_s.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Teacher forward (frozen, активирует hook)
            with torch.no_grad():
                t_feat = teacher_model.encode_image(x_t)
                _ = teacher_head(t_feat)
            
            # Извлекаем токены учителя
            t_tokens = teacher_hook.outputs
            if isinstance(t_tokens, (tuple, list)):
                t_tokens = t_tokens[0]
            attn_t = tokens_to_attention_map(t_tokens).detach()
            
            # Student forward (активирует hook)
            s_feat = student_model.encode_image(x_s)
            s_logits = student_head(s_feat)
            
            # Извлекаем токены студента
            s_tokens = student_hook.outputs
            if isinstance(s_tokens, (tuple, list)):
                s_tokens = s_tokens[0]
            attn_s = tokens_to_attention_map(s_tokens)
            
            # Loss
            loss, attn_loss, ce_loss = criterion(s_logits, y, attn_s, attn_t)
            
            loss.backward()
            
            # Gradient norm
            grad_norm = compute_gradient_norm(student_model) + compute_gradient_norm(student_head)
            
            optimizer.step()
            
            loss_sum += loss.item()
            attn_sum += attn_loss.item()
            ce_sum += ce_loss.item()
            grad_norm_sum += grad_norm
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'attn': f'{attn_loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}'
            })
        
        pbar.close()
        
        avg_train_loss = loss_sum / n_batches
        avg_train_attn = attn_sum / n_batches
        avg_train_ce = ce_sum / n_batches
        avg_grad_norm = grad_norm_sum / n_batches
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation
        val_loss, val_attn, val_ce, val_acc = validate_attention(
            teacher_model, student_model,
            teacher_head, student_head,
            teacher_hook, student_hook,
            val_loader, criterion, device
        )
        
        scheduler.step()
        
        # Сохранение метрик на контрольных эпохах
        if epoch in SAVE_EPOCHS:
            metrics_tracker['epochs'].append(epoch)
            metrics_tracker['train_loss'].append(avg_train_loss)
            metrics_tracker['train_attention_loss'].append(avg_train_attn)
            metrics_tracker['train_ce_loss'].append(avg_train_ce)
            metrics_tracker['val_loss'].append(val_loss)
            metrics_tracker['val_attention_loss'].append(val_attn)
            metrics_tracker['val_ce_loss'].append(val_ce)
            metrics_tracker['val_accuracy'].append(val_acc)
            metrics_tracker['gradient_norm'].append(avg_grad_norm)
            metrics_tracker['learning_rate'].append(current_lr)
            
            print(f'✓ Epoch {epoch}: Val Acc={val_acc:.2f}% | Attn={val_attn:.4f} | CE={val_ce:.4f} | GradNorm={avg_grad_norm:.4f}')
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_student_state = student_model.state_dict()
            best_student_head_state = student_head.state_dict()
    
    # Загрузка лучшего состояния для финального тестирования
    if best_student_state is not None:
        student_model.load_state_dict(best_student_state)
        student_head.load_state_dict(best_student_head_state)
    
    # Закрываем hooks
    teacher_hook.close()
    student_hook.close()
    
    # Test evaluation
    student_model.eval()
    student_head.eval()
    
    test_correct, test_total = 0, 0
    
    with torch.no_grad():
        for _, x_s, y in test_loader:
            x_s, y = x_s.to(device), y.to(device)
            student_features = student_model.encode_image(x_s)
            logits = student_head(student_features)
            _, pred = torch.max(logits, 1)
            test_total += y.size(0)
            test_correct += (pred == y).sum().item()
    
    test_acc = 100.0 * test_correct / test_total
    
    # Вычисляем среднее MSE между attention maps
    teacher_hook_test = ModuleOutputHook(get_transformer_module(teacher_model))
    student_hook_test = ModuleOutputHook(get_transformer_module(student_model))
    
    avg_attn_mse = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for x_t, x_s, _ in val_loader:
            x_t, x_s = x_t.to(device), x_s.to(device)
            
            # Teacher
            teacher_model.encode_image(x_t)
            t_tokens = teacher_hook_test.outputs
            if isinstance(t_tokens, (tuple, list)):
                t_tokens = t_tokens[0]
            attn_t = tokens_to_attention_map(t_tokens)
            
            # Student
            student_model.encode_image(x_s)
            s_tokens = student_hook_test.outputs
            if isinstance(s_tokens, (tuple, list)):
                s_tokens = s_tokens[0]
            attn_s = tokens_to_attention_map(s_tokens)
            
            # MSE
            mse = match_attention_maps(attn_s, attn_t, loss_type='mse')
            avg_attn_mse += mse.item() * x_t.size(0)
            n_samples += x_t.size(0)
    
    avg_attn_mse /= n_samples
    
    teacher_hook_test.close()
    student_hook_test.close()
    
    # Подсчет параметров
    student_params = sum(p.numel() for p in student_model.parameters())
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_head_params = sum(p.numel() for p in student_head.parameters())
    total_student_params = student_params + student_head_params
    
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
        'student_cross_entropy': float(metrics_tracker['val_ce_loss'][-1]),
        'student_accuracy': float(metrics_tracker['val_accuracy'][-1]),
        
        # Distillation
        'distillation_method': 'attention',
        'distillation_tokens': len(train_loader.dataset) * TOTAL_EPOCHS,
        
        # Training config
        'random_seed': SEED,
        'num_epochs': TOTAL_EPOCHS,
        'batch_size': BATCH_SIZE,
        
        # Hyperparameters
        'hyperparameters': {
            'alpha': exp_config['alpha'],
            'attention_loss_type': exp_config['attention_loss_type'],
            'learning_rate': exp_config['learning_rate'],
            'weight_decay': WEIGHT_DECAY,
            'optimizer': OPTIMIZER_TYPE,
            'scheduler': SCHEDULER,
            'teacher_dim': teacher_dim,
            'student_dim': student_dim
        },
        
        # Results
        'test_accuracy': float(test_acc),
        'best_val_acc': float(best_val_acc),
        'avg_attention_mse': float(avg_attn_mse),
        
        # Metrics
        'metrics': metrics_tracker,
        'save_epochs': SAVE_EPOCHS
    }
    
    # Сохранение результатов (JSON)
    results_dir = Path('../results/attention_distillation_t')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"{exp_config['name']}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Сохранение весов модели
    weights_dir = Path('../model_weights')
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = f"{exp_config['name']}_best.pth"
    model_path = weights_dir / model_name
    
    # Сохраняем лучшую модель (по val_acc)
    torch.save({
        'student_model': best_student_state if best_student_state is not None else student_model.state_dict(),
        'student_head': best_student_head_state if best_student_head_state is not None else student_head.state_dict(),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'config': exp_config
    }, model_path)
    
    print(f'   Веса сохранены: {model_path}')
    
    print(f'\n✅ Эксперимент завершен!')
    print(f'   Best Val Acc: {best_val_acc:.2f}%')
    print(f'   Test Acc: {test_acc:.2f}%')
    print(f'   Avg Attention MSE: {avg_attn_mse:.6f}')
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
    
    teacher_dim = teacher_model.visual.output_dim
    
    # Datasets
    train_ds = AttentionDistillationDataset(train_ids, image_to_path, image_to_label, label_to_idx,
                                            teacher_preprocessor, student_preprocessor)
    val_ds = AttentionDistillationDataset(val_ids, image_to_path, image_to_label, label_to_idx,
                                         teacher_preprocessor, student_preprocessor)
    test_ds = AttentionDistillationDataset(test_ids, image_to_path, image_to_label, label_to_idx,
                                          teacher_preprocessor, student_preprocessor)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    # Обучаем teacher linear probe ОДИН РАЗ
    print('\n' + '='*80)
    print('ОБУЧЕНИЕ TEACHER LINEAR PROBE (общий для всех экспериментов)')
    print('='*80)
    
    teacher_head = ClassificationHead(teacher_dim, num_classes).to(device)
    teacher_val_acc = train_teacher_probe(
        teacher_model, teacher_head, train_loader, val_loader,
        TEACHER_EPOCHS, TEACHER_LR, device
    )
    
    print(f'Teacher Linear Probe обучен: ValAcc={teacher_val_acc:.2f}%\n')
    
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
    print(f'\nРезультаты сохранены в: ../results/attention_distillation_t/')
    print(f'Всего экспериментов: {len(all_results)}\n')

if __name__ == '__main__':
    main()