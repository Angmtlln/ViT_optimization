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

# Обучение teacher probe
TEACHER_LR = 5e-4
TEACHER_EPOCHS = 5

# Размерность проекции
PROJECTION_DIM = 256

# ==============================
# ВАРЬИРУЕМЫЕ ПАРАМЕТРЫ
# ==============================

EXPERIMENTS = [
    # Baseline (без контрастной дистилляции) - только CE + KD
    {
        'name': 'contrastive_baseline_no_contrast_tiny',
        'temperature_kd': 4.0,
        'temperature_contrast': 0.07,
        'alpha_contrast': 0.0,  # отключаем контрастный loss
        'beta_kd': 0.5,
        'gamma_ce': 0.5,
        'learning_rate': 1e-4
    },
    
    # Контрастная дистилляция с разными alpha_contrast
    {
        'name': 'contrastive_alpha0.3_tiny',
        'temperature_kd': 4.0,
        'temperature_contrast': 0.07,
        'alpha_contrast': 0.3,
        'beta_kd': 0.4,
        'gamma_ce': 0.3,
        'learning_rate': 1e-4
    },
    {
        'name': 'contrastive_alpha0.5_tiny',
        'temperature_kd': 4.0,
        'temperature_contrast': 0.07,
        'alpha_contrast': 0.5,
        'beta_kd': 0.3,
        'gamma_ce': 0.2,
        'learning_rate': 1e-4
    },
    {
        'name': 'contrastive_alpha0.7_tiny',
        'temperature_kd': 4.0,
        'temperature_contrast': 0.07,
        'alpha_contrast': 0.7,
        'beta_kd': 0.2,
        'gamma_ce': 0.1,
        'learning_rate': 1e-4
    },
    
    # Разные temperature_contrast (с фиксированным alpha=0.5)
    {
        'name': 'contrastive_T0.05_tiny',
        'temperature_kd': 4.0,
        'temperature_contrast': 0.05,
        'alpha_contrast': 0.5,
        'beta_kd': 0.3,
        'gamma_ce': 0.2,
        'learning_rate': 1e-4
    },
    {
        'name': 'contrastive_T0.10_tiny',
        'temperature_kd': 4.0,
        'temperature_contrast': 0.10,
        'alpha_contrast': 0.5,
        'beta_kd': 0.3,
        'gamma_ce': 0.2,
        'learning_rate': 1e-4
    },
    
    # Разные temperature_kd (с фиксированным alpha=0.5)
    {
        'name': 'contrastive_TKD2_tiny',
        'temperature_kd': 2.0,
        'temperature_contrast': 0.07,
        'alpha_contrast': 0.5,
        'beta_kd': 0.3,
        'gamma_ce': 0.2,
        'learning_rate': 1e-4
    },
    {
        'name': 'contrastive_TKD6_tiny',
        'temperature_kd': 6.0,
        'temperature_contrast': 0.07,
        'alpha_contrast': 0.5,
        'beta_kd': 0.3,
        'gamma_ce': 0.2,
        'learning_rate': 1e-4
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

class ContrastiveDistillationDataset(Dataset):
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
    """MLP projection head для контрастного обучения"""
    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

# ==============================
# LOSSES
# ==============================

class ContrastiveLoss(nn.Module):
    """
    Контрастный loss (InfoNCE) для дистилляции признаков.
    """
    def __init__(self, temperature=0.07, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: [batch_size, projection_dim]
            teacher_features: [batch_size, projection_dim]
        
        Returns:
            InfoNCE loss (scalar)
        """
        batch_size = student_features.shape[0]
        
        # L2 нормализация для cosine similarity
        if self.normalize:
            student_features = F.normalize(student_features, dim=1)
            teacher_features = F.normalize(teacher_features, dim=1)
        
        # Similarity matrix: [batch_size, batch_size]
        similarity_matrix = torch.matmul(student_features, teacher_features.T) / self.temperature
        
        # Positive pairs на диагонали
        labels = torch.arange(batch_size, device=student_features.device)
        
        # InfoNCE = Cross-Entropy
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class ContrastiveDistillationLoss(nn.Module):
    """
    Комбинированный loss для контрастной дистилляции:
    Total Loss = α·L_contrast + β·L_KD + γ·L_CE
    """
    def __init__(self, temperature_kd=4.0, temperature_contrast=0.07, 
                 alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.T_kd = temperature_kd
        self.alpha = alpha  # weight for contrastive
        self.beta = beta    # weight for KD
        self.gamma = gamma  # weight for CE
        
        self.contrastive_loss = ContrastiveLoss(temperature=temperature_contrast)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_features, teacher_features, 
                student_logits, teacher_logits, labels):
        """
        Args:
            student_features: projected features from student [B, proj_dim]
            teacher_features: projected features from teacher [B, proj_dim]
            student_logits: classification logits from student [B, num_classes]
            teacher_logits: classification logits from teacher [B, num_classes]
            labels: ground truth labels [B]
        
        Returns:
            total_loss, contrast_loss, kd_loss, ce_loss
        """
        # 1. Контрастная дистилляция (InfoNCE на признаках)
        if self.alpha > 0:
            contrast_loss = self.contrastive_loss(student_features, teacher_features)
        else:
            contrast_loss = torch.tensor(0.0, device=labels.device)
        
        # 2. Knowledge Distillation (KL на логитах с температурой)
        if self.beta > 0:
            kd_loss = self.kl_loss(
                F.log_softmax(student_logits / self.T_kd, dim=-1),
                F.softmax(teacher_logits / self.T_kd, dim=-1)
            ) * (self.T_kd ** 2)
        else:
            kd_loss = torch.tensor(0.0, device=labels.device)
        
        # 3. Classification Loss (CE на ground truth)
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Взвешенная сумма
        total_loss = (self.alpha * contrast_loss + 
                     self.beta * kd_loss + 
                     self.gamma * ce_loss)
        
        return total_loss, contrast_loss, kd_loss, ce_loss

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
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
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

def validate_contrastive(teacher_model, student_model,
                         teacher_head, student_head,
                         teacher_projection, student_projection,
                         loader, criterion, device):
    """Валидация студента с контрастной дистилляцией"""
    teacher_model.eval()
    student_model.eval()
    teacher_head.eval()
    student_head.eval()
    teacher_projection.eval()
    student_projection.eval()
    
    total_loss = total_contrast = total_kd = total_ce = 0.0
    correct = total = n_batches = 0
    
    with torch.no_grad():
        for x_t, x_s, y in loader:
            x_t = x_t.to(device, non_blocking=True)
            x_s = x_s.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Teacher forward
            t_features = teacher_model.encode_image(x_t)
            t_projected = teacher_projection(t_features)
            t_logits = teacher_head(t_features)
            
            # Student forward
            s_features = student_model.encode_image(x_s)
            s_projected = student_projection(s_features)
            s_logits = student_head(s_features)
            
            # Loss calculation
            loss, contrast, kd, ce = criterion(
                s_projected, t_projected,
                s_logits, t_logits, y
            )
            
            total_loss += loss.item()
            total_contrast += contrast.item()
            total_kd += kd.item()
            total_ce += ce.item()
            
            _, pred = torch.max(s_logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            n_batches += 1
    
    avg_loss = total_loss / max(1, n_batches)
    avg_contrast = total_contrast / max(1, n_batches)
    avg_kd = total_kd / max(1, n_batches)
    avg_ce = total_ce / max(1, n_batches)
    acc = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, avg_contrast, avg_kd, avg_ce, acc

def run_experiment(exp_config, train_loader, val_loader, test_loader,
                  teacher_model, teacher_head, num_classes, device):
    """Запуск одного эксперимента"""
    
    print('\n' + '='*80)
    print(f'🚀 ЭКСПЕРИМЕНТ: {exp_config["name"]}')
    print(f'   Alpha (contrast): {exp_config["alpha_contrast"]}')
    print(f'   Beta (KD): {exp_config["beta_kd"]}')
    print(f'   Gamma (CE): {exp_config["gamma_ce"]}')
    print(f'   T_contrast: {exp_config["temperature_contrast"]}')
    print(f'   T_KD: {exp_config["temperature_kd"]}')
    print(f'   LR: {exp_config["learning_rate"]}')
    print('='*80 + '\n')
    
    # Создаем модель студента
    student_model = pe.CLIP.from_config(STUDENT_ARCH, pretrained=True).to(device).float()
    student_dim = student_model.visual.output_dim
    teacher_dim = teacher_model.visual.output_dim
    
    print(f'   Student dim: {student_dim}, Teacher dim: {teacher_dim}')
    
    # Classification head
    student_head = ClassificationHead(student_dim, num_classes).to(device)
    
    # Projection heads
    teacher_projection = ProjectionHead(teacher_dim, hidden_dim=512, output_dim=PROJECTION_DIM).to(device)
    student_projection = ProjectionHead(student_dim, hidden_dim=512, output_dim=PROJECTION_DIM).to(device)
    
    print(f'   ✅ Projection heads: {teacher_dim}/{student_dim} → {PROJECTION_DIM}')
    
    # Loss
    criterion = ContrastiveDistillationLoss(
        temperature_kd=exp_config['temperature_kd'],
        temperature_contrast=exp_config['temperature_contrast'],
        alpha=exp_config['alpha_contrast'],
        beta=exp_config['beta_kd'],
        gamma=exp_config['gamma_ce']
    )
    
    # Optimizer (обучаем студента + его головы + обе проекции)
    # Teacher остается замороженным
    for p in teacher_model.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        list(student_model.parameters()) + 
        list(student_head.parameters()) + 
        list(student_projection.parameters()) +
        list(teacher_projection.parameters()),
        lr=exp_config['learning_rate'],
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    
    # Трекер метрик
    metrics_tracker = {
        'epochs': [],
        'train_loss': [],
        'train_contrast_loss': [],
        'train_kd_loss': [],
        'train_ce_loss': [],
        'val_loss': [],
        'val_contrast_loss': [],
        'val_kd_loss': [],
        'val_ce_loss': [],
        'val_accuracy': [],
        'gradient_norm': [],
        'learning_rate': []
    }
    
    best_val_acc = -1.0
    best_student_state = None
    best_student_head_state = None
    best_student_projection_state = None
    best_teacher_projection_state = None
    
    # Обучение
    for epoch in range(1, TOTAL_EPOCHS + 1):
        student_model.train()
        student_head.train()
        student_projection.train()
        teacher_model.eval()
        teacher_head.eval()
        teacher_projection.train()
        
        loss_sum = contrast_sum = kd_sum = ce_sum = grad_norm_sum = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}", ncols=120, leave=False)
        
        for x_t, x_s, y in pbar:
            x_t, x_s, y = x_t.to(device), x_s.to(device), y.to(device)
            
            # Teacher forward (frozen)
            with torch.no_grad():
                t_features = teacher_model.encode_image(x_t)
                t_logits = teacher_head(t_features)
            
            t_projected = teacher_projection(t_features)
            
            # Student forward
            s_features = student_model.encode_image(x_s)
            s_projected = student_projection(s_features)
            s_logits = student_head(s_features)
            
            # Loss
            total_loss, contrast_loss, kd_loss, ce_loss = criterion(
                s_projected, t_projected,
                s_logits, t_logits, y
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient norm
            grad_norm = (compute_gradient_norm(student_model) + 
                        compute_gradient_norm(student_head) +
                        compute_gradient_norm(student_projection) +
                        compute_gradient_norm(teacher_projection))
            
            optimizer.step()
            
            loss_sum += total_loss.item()
            contrast_sum += contrast_loss.item()
            kd_sum += kd_loss.item()
            ce_sum += ce_loss.item()
            grad_norm_sum += grad_norm
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'contrast': f'{contrast_loss.item():.4f}',
                'kd': f'{kd_loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}'
            })
        
        pbar.close()
        
        avg_train_loss = loss_sum / n_batches
        avg_train_contrast = contrast_sum / n_batches
        avg_train_kd = kd_sum / n_batches
        avg_train_ce = ce_sum / n_batches
        avg_grad_norm = grad_norm_sum / n_batches
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation
        val_loss, val_contrast, val_kd, val_ce, val_acc = validate_contrastive(
            teacher_model, student_model,
            teacher_head, student_head,
            teacher_projection, student_projection,
            val_loader, criterion, device
        )
        
        scheduler.step()
        
        # Сохранение метрик на контрольных эпохах
        if epoch in SAVE_EPOCHS:
            metrics_tracker['epochs'].append(epoch)
            metrics_tracker['train_loss'].append(avg_train_loss)
            metrics_tracker['train_contrast_loss'].append(avg_train_contrast)
            metrics_tracker['train_kd_loss'].append(avg_train_kd)
            metrics_tracker['train_ce_loss'].append(avg_train_ce)
            metrics_tracker['val_loss'].append(val_loss)
            metrics_tracker['val_contrast_loss'].append(val_contrast)
            metrics_tracker['val_kd_loss'].append(val_kd)
            metrics_tracker['val_ce_loss'].append(val_ce)
            metrics_tracker['val_accuracy'].append(val_acc)
            metrics_tracker['gradient_norm'].append(avg_grad_norm)
            metrics_tracker['learning_rate'].append(current_lr)
            
            print(f'✓ Epoch {epoch}: Val Acc={val_acc:.2f}% | Contrast={val_contrast:.4f} | KD={val_kd:.4f} | CE={val_ce:.4f} | GradNorm={avg_grad_norm:.4f}')
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_student_state = student_model.state_dict()
            best_student_head_state = student_head.state_dict()
            best_student_projection_state = student_projection.state_dict()
            best_teacher_projection_state = teacher_projection.state_dict()
    
    # Загрузка лучшего состояния для финального тестирования
    if best_student_state is not None:
        student_model.load_state_dict(best_student_state)
        student_head.load_state_dict(best_student_head_state)
        student_projection.load_state_dict(best_student_projection_state)
        teacher_projection.load_state_dict(best_teacher_projection_state)
    
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
    
    # Вычисляем similarity между признаками
    avg_cosine_sim = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for x_t, x_s, _ in val_loader:
            x_t, x_s = x_t.to(device), x_s.to(device)
            
            t_features = teacher_model.encode_image(x_t)
            t_projected = teacher_projection(t_features)
            
            s_features = student_model.encode_image(x_s)
            s_projected = student_projection(s_features)
            
            # Cosine similarity
            t_norm = F.normalize(t_projected, p=2, dim=-1)
            s_norm = F.normalize(s_projected, p=2, dim=-1)
            cosine_sim = F.cosine_similarity(t_norm, s_norm, dim=-1)
            avg_cosine_sim += cosine_sim.sum().item()
            
            n_samples += x_t.size(0)
    
    avg_cosine_sim /= n_samples
    
    # Подсчет параметров
    student_params = sum(p.numel() for p in student_model.parameters())
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_head_params = sum(p.numel() for p in student_head.parameters())
    student_proj_params = sum(p.numel() for p in student_projection.parameters())
    teacher_proj_params = sum(p.numel() for p in teacher_projection.parameters())
    total_student_params = student_params + student_head_params + student_proj_params + teacher_proj_params
    
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
        'distillation_method': 'contrastive',
        'distillation_tokens': len(train_loader.dataset) * TOTAL_EPOCHS,
        
        # Training config
        'random_seed': SEED,
        'num_epochs': TOTAL_EPOCHS,
        'batch_size': BATCH_SIZE,
        
        # Hyperparameters
        'hyperparameters': {
            'alpha_contrast': exp_config['alpha_contrast'],
            'beta_kd': exp_config['beta_kd'],
            'gamma_ce': exp_config['gamma_ce'],
            'temperature_kd': exp_config['temperature_kd'],
            'temperature_contrast': exp_config['temperature_contrast'],
            'learning_rate': exp_config['learning_rate'],
            'weight_decay': WEIGHT_DECAY,
            'optimizer': OPTIMIZER_TYPE,
            'scheduler': SCHEDULER,
            'projection_dim': PROJECTION_DIM,
            'teacher_dim': teacher_dim,
            'student_dim': student_dim
        },
        
        # Results
        'test_accuracy': float(test_acc),
        'best_val_acc': float(best_val_acc),
        'avg_cosine_similarity': float(avg_cosine_sim),
        
        # Metrics
        'metrics': metrics_tracker,
        'save_epochs': SAVE_EPOCHS
    }
    
    # Сохранение результатов (JSON)
    results_dir = Path('../results/contrastive_distillation_t')
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
        'student_projection': best_student_projection_state if best_student_projection_state is not None else student_projection.state_dict(),
        'teacher_projection': best_teacher_projection_state if best_teacher_projection_state is not None else teacher_projection.state_dict(),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'config': exp_config
    }, model_path)
    
    print(f'   Веса сохранены: {model_path}')
    
    print(f'\n✅ Эксперимент завершен!')
    print(f'   Best Val Acc: {best_val_acc:.2f}%')
    print(f'   Test Acc: {test_acc:.2f}%')
    print(f'   Avg Cosine Similarity: {avg_cosine_sim:.4f}')
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
    train_ds = ContrastiveDistillationDataset(train_ids, image_to_path, image_to_label, label_to_idx,
                                              teacher_preprocessor, student_preprocessor)
    val_ds = ContrastiveDistillationDataset(val_ids, image_to_path, image_to_label, label_to_idx,
                                           teacher_preprocessor, student_preprocessor)
    test_ds = ContrastiveDistillationDataset(test_ids, image_to_path, image_to_label, label_to_idx,
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
    print(f'\nРезультаты сохранены в: ../results/contrastive_distillation_t/')
    print(f'Всего экспериментов: {len(all_results)}\n')

if __name__ == '__main__':
    main()