"""
Quantization-Aware Training + Knowledge Distillation (QAT + KD)
Берём лучшие обученные веса → квантизуем → дообучаем с дистилляцией от FP32 teacher
"""

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
import copy

# ==============================
# ПУТЬ К PERCEPTION_MODELS
# ==============================

sys.path.insert(0, str(Path('../../perception_models').resolve()))

try:
    from core.vision_encoder import pe
    from core.vision_encoder import transforms
    print("✅ Модули perception_models импортированы успешно\n")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)

# ==============================
# КОНФИГУРАЦИЯ
# ==============================

SEED = 42
ROOT_DIR = '../../data/kvasir-dataset-v2'
BATCH_SIZE = 16
NUM_WORKERS = 0

ARCH_TINY  = 'PE-Core-T16-384'
ARCH_SMALL = 'PE-Core-S16-384'
TEACHER_ARCH = 'PE-Core-L14-336'  # ✅ Оригинальный Large учитель

# Training параметры
LEARNING_RATE = 1e-4
EPOCHS = 20
WARMUP_EPOCHS = 2

# Пути
MODEL_WEIGHTS_DIR  = Path('../../model_weights')
TEACHER_WEIGHT_FILE = 'teacher_best.pth'  # ✅ Large FP32 учитель

# ==============================
# ЛУЧШИЕ МОДЕЛИ ПО КАЖДОМУ МЕТОДУ ДИСТИЛЛЯЦИИ
# Берём готовые веса → квантизуем → дообучаем с QAT + KD
# (1 лучшая × 4 метода × 2 размера = 8 экспериментов)
# ==============================

EXPERIMENTS = [
    # --- LOGIT DISTILLATION ---
    # Best: T=3, α=0.5, test_acc=93.5%
    {
        'name': 'qat_kd_logit_T3_alpha0.5_small',
        'distillation_type': 'logit',
        'arch': ARCH_SMALL,
        'weight_file': 'small_weights/logit_T3_alpha0.5_small_best.pth',
        'temperature': 3.0,
        'alpha': 0.5,
    },
    # Best: T=5, α=0.5, test_acc=85.375%
    {
        'name': 'qat_kd_logit_T5_alpha0.5_tiny',
        'distillation_type': 'logit',
        'arch': ARCH_TINY,
        'weight_file': 'tiny_weights/logit_T5_alpha0.5_tiny_best.pth',
        'temperature': 5.0,
        'alpha': 0.5,
    },
    # --- ATTENTION DISTILLATION ---
    # Best: α=0.6, mse, lr=5e-5, test_acc=92.0%
    {
        'name': 'qat_kd_attention_lr5e-5_small',
        'distillation_type': 'logit',  # QAT использует logit KD поверх attention-весов
        'arch': ARCH_SMALL,
        'weight_file': 'small_weights/attention_lr5e-5_small_best.pth',
        'temperature': 4.0,
        'alpha': 0.6,
    },
    # Best: α=0.6, mse, lr=5e-5, test_acc=82.875%
    {
        'name': 'qat_kd_attention_lr5e-5_tiny',
        'distillation_type': 'logit',
        'arch': ARCH_TINY,
        'weight_file': 'tiny_weights/attention_lr5e-5_tiny_best.pth',
        'temperature': 4.0,
        'alpha': 0.6,
    },
    # --- CONTRASTIVE DISTILLATION ---
    # Best: α_contrast=0.3, test_acc=91.625%
    {
        'name': 'qat_kd_contrastive_alpha0.3_small',
        'distillation_type': 'logit',
        'arch': ARCH_SMALL,
        'weight_file': 'small_weights/contrastive_alpha0.3_small_best.pth',
        'temperature': 4.0,
        'alpha': 0.3,
    },
    # Best: T_contrast=0.1, test_acc=87.375%
    {
        'name': 'qat_kd_contrastive_T0.10_tiny',
        'distillation_type': 'logit',
        'arch': ARCH_TINY,
        'weight_file': 'tiny_weights/contrastive_T0.10_tiny_best.pth',
        'temperature': 4.0,
        'alpha': 0.5,
    },
    # --- FEATURE DISTILLATION ---
    # Best: α=0.7, proj, test_acc=90.0%
    {
        'name': 'qat_kd_feature_alpha0.7_proj_small',
        'distillation_type': 'feature',
        'arch': ARCH_SMALL,
        'weight_file': 'small_weights/feature_alpha0.7_proj_small_best.pth',
        'temperature': 4.0,
        'alpha': 0.7,
        'use_projection': True,
    },
    # Best: α=0.9, proj, test_acc=81.875%
    {
        'name': 'qat_kd_feature_alpha0.9_proj_tiny',
        'distillation_type': 'feature',
        'arch': ARCH_TINY,
        'weight_file': 'tiny_weights/feature_alpha0.9_proj_tiny_best.pth',
        'temperature': 4.0,
        'alpha': 0.9,
        'use_projection': True,
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

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  Используется CPU")
    return device

# ==============================
# DATASET
# ==============================

class SimpleDataset(Dataset):
    def __init__(self, image_ids, image_to_path, image_to_label, label_to_idx, transform):
        self.ids = image_ids
        self.image_to_path = image_to_path
        self.image_to_label = image_to_label
        self.label_to_idx = label_to_idx
        self.transform = transform
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = self.image_to_path[img_id]
        label = self.image_to_label[img_id]
        y = self.label_to_idx[label]
        img = Image.open(path).convert('RGB')
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.long)

# ==============================
# MODELS
# ==============================

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# ==============================
# QUANTIZATION
# ==============================

class QuantizedLinear(nn.Module):
    """INT8 Linear layer с Fake Quantization для QAT обучения"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
    
    def fake_quantize(self, x):
        scale = x.abs().max() / 127.0
        if scale > 0:
            quantized   = torch.round(x / scale).clamp(-128, 127)
            dequantized = quantized * scale
        else:
            dequantized = x
        return dequantized
    
    def forward(self, x):
        weight_q = self.fake_quantize(self.weight)
        return F.linear(x, weight_q, self.bias)
    
    def to_real_quantized(self):
        with torch.no_grad():
            scale = self.weight.abs().max().item() / 127.0
            if scale > 0:
                weight_int8 = torch.round(self.weight / scale).clamp(-128, 127).to(torch.int8)
            else:
                weight_int8 = torch.zeros_like(self.weight, dtype=torch.int8)
            return weight_int8, scale


class QuantizedLinearInference(nn.Module):
    """INT8 Linear для inference (после QAT обучения)"""
    def __init__(self, in_features, out_features, weight_int8, scale, bias=None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.register_buffer('weight_int8', weight_int8)
        self.register_buffer('weight_scale', torch.tensor(scale))
        if bias is not None:
            self.register_buffer('bias', bias.detach().clone())
        else:
            self.bias = None
    
    @property
    def weight(self):
        return self.weight_int8.float() * self.weight_scale
    
    def forward(self, x):
        weight_fp32 = self.weight_int8.float() * self.weight_scale
        return F.linear(x, weight_fp32, self.bias)


def replace_linear_with_quantized(model):
    """Заменить все nn.Linear на QuantizedLinear + скопировать веса"""
    replaced_count = 0
    
    def recursive_replace(module, name=''):
        nonlocal replaced_count
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child_module, nn.Linear):
                qlinear = QuantizedLinear(
                    child_module.in_features,
                    child_module.out_features,
                    bias=child_module.bias is not None
                )
                qlinear.weight.data.copy_(child_module.weight.data)
                if child_module.bias is not None:
                    qlinear.bias.data.copy_(child_module.bias.data)
                setattr(module, child_name, qlinear)
                replaced_count += 1
                if replaced_count <= 3:
                    print(f'      ✓ {full_name} ({child_module.in_features}→{child_module.out_features})')
            else:
                recursive_replace(child_module, full_name)
    
    recursive_replace(model)
    print(f'      Всего заменено слоёв: {replaced_count}')
    return model


def convert_to_inference_mode(model):
    """Конвертировать QAT модель в реальную INT8 для inference"""
    def recursive_convert(module, name=''):
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, QuantizedLinear):
                weight_int8, scale = child_module.to_real_quantized()
                qlinear_inf = QuantizedLinearInference(
                    child_module.in_features,
                    child_module.out_features,
                    weight_int8, scale,
                    child_module.bias.data if child_module.bias is not None else None
                )
                setattr(module, child_name, qlinear_inf)
            else:
                recursive_convert(child_module, f"{name}.{child_name}" if name else child_name)
    
    recursive_convert(model)
    return model

# ==============================
# TRAINING & EVALUATION
# ==============================

def get_model_size(model):
    device = next(model.parameters()).device
    model_cpu = model.cpu()
    param_size  = sum(p.nelement() * p.element_size() for p in model_cpu.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model_cpu.buffers())
    model.to(device)
    return (param_size + buffer_size) / 1024 / 1024


def evaluate_model(model, loader, device):
    model.eval()
    model = model.to(device)
    correct = total = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Evaluation', leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, pred = torch.max(logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return 100.0 * correct / total


def train_qat_kd(student_wrapper, teacher_wrapper, train_loader, val_loader, test_loader,
                 device, config):
    print(f'\n{"="*80}')
    print(f'🎓 QAT + KD: {config["name"]}')
    print(f'   Тип дистилляции : {config["distillation_type"]}')
    print(f'   Архитектура     : {config["arch"]}')
    print(f'   Temperature     : {config["temperature"]}, Alpha: {config["alpha"]}')
    print(f'{"="*80}\n')
    
    student_wrapper = student_wrapper.to(device)
    teacher_wrapper = teacher_wrapper.to(device)
    teacher_wrapper.eval()
    
    optimizer    = torch.optim.AdamW(student_wrapper.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, (total_steps - step) / (total_steps - warmup_steps))
    
    scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion_ce = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [], 'val_acc': [], 'test_acc': [],
        'best_val_acc': 0, 'best_test_acc': 0
    }
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        student_wrapper.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            student_logits = student_wrapper(x)
            
            with torch.no_grad():
                teacher_logits = teacher_wrapper(x)
            
            if config['distillation_type'] == 'logit':
                T     = config['temperature']
                alpha = config['alpha']
                loss_ce = criterion_ce(student_logits, y)
                loss_kd = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T * T)
                loss = alpha * loss_kd + (1 - alpha) * loss_ce
            
            elif config['distillation_type'] == 'feature':
                student_features = student_wrapper.get_features(x)
                with torch.no_grad():
                    teacher_features = teacher_wrapper.get_features(x)
                loss_feature = F.mse_loss(student_features, teacher_features)
                loss_ce = criterion_ce(student_logits, y)
                loss = config['alpha'] * loss_feature + (1 - config['alpha']) * loss_ce
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_wrapper.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
        
        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate_model(student_wrapper, val_loader, device)
        test_acc = evaluate_model(student_wrapper, test_loader, device)
        
        print(f'   Epoch {epoch+1}: Loss={avg_loss:.4f}, Val={val_acc:.2f}%, Test={test_acc:.2f}%')
        
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc']  = val_acc
            history['best_test_acc'] = test_acc
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save({
                'student_model': student_wrapper.student_model.state_dict(),
                'student_head':  student_wrapper.student_head.state_dict(),
                'projection':    student_wrapper.projection.state_dict() if hasattr(student_wrapper, 'projection') and student_wrapper.projection else None,
                'best_val_acc':  best_val_acc,
                'best_test_acc': test_acc,
                'epoch': epoch + 1,
                'config': config
            }, Path('checkpoints') / f"{config['name']}_best.pth")
    
    print(f'\n   ✅ Лучшая Val Acc : {history["best_val_acc"]:.2f}%')
    print(f'   ✅ Test Acc       : {history["best_test_acc"]:.2f}%\n')
    return history

# ==============================
# WRAPPERS
# ==============================

class LogitDistillationWrapper(nn.Module):
    def __init__(self, clip_model, head):
        super().__init__()
        self.student_model = clip_model
        self.student_head  = head
    
    def forward(self, x):
        return self.student_head(self.student_model.encode_image(x))
    
    def get_features(self, x):
        return self.student_model.encode_image(x)


class FeatureDistillationWrapper(nn.Module):
    def __init__(self, clip_model, head, projection=None):
        super().__init__()
        self.student_model = clip_model
        self.student_head  = head
        self.projection    = projection
    
    def forward(self, x):
        return self.student_head(self.student_model.encode_image(x))
    
    def get_features(self, x):
        features = self.student_model.encode_image(x)
        if self.projection is not None:
            features = self.projection(features)
        return features

# ==============================
# MAIN
# ==============================

def main():
    set_seed(SEED)
    device = get_device()
    print()
    
    # Dataset
    print('Подготовка датасета...')
    all_image_paths = glob(os.path.join(ROOT_DIR, '**', '*.jpg'), recursive=True)
    
    image_to_path, image_to_label = {}, {}
    for p in all_image_paths:
        image_id = os.path.splitext(os.path.basename(p))[0]
        label = extract_label_from_path(p, ROOT_DIR)
        if label is not None:
            image_to_path[image_id] = p
            image_to_label[image_id] = label
    
    classes      = sorted(list(set(image_to_label.values())))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes  = len(classes)
    print(f'Найдено {len(image_to_label)} изображений, {num_classes} классов')
    
    images_by_class = defaultdict(list)
    for img_id, lbl in image_to_label.items():
        images_by_class[lbl].append(img_id)
    
    sampled = []
    for lbl, ids in images_by_class.items():
        sampled.extend(random.sample(ids, min(500, len(ids))))
    random.shuffle(sampled)
    
    labels_for_strat = [image_to_label[i] for i in sampled]
    train_val_ids, test_ids = train_test_split(sampled, test_size=0.20,
                                               stratify=labels_for_strat, random_state=SEED)
    train_labels = [image_to_label[i] for i in train_val_ids]
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.20,
                                          stratify=train_labels, random_state=SEED)
    print(f'Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}')
    
    transform    = transforms.get_image_transform(384)
    train_ds     = SimpleDataset(train_ids, image_to_path, image_to_label, label_to_idx, transform)
    val_ds       = SimpleDataset(val_ids,   image_to_path, image_to_label, label_to_idx, transform)
    test_ds      = SimpleDataset(test_ids,  image_to_path, image_to_label, label_to_idx, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # ==============================
    # Загрузка Teacher (Large FP32)
    # ==============================
    print(f'\n{"="*80}')
    print(f'Загрузка FP32 Teacher: {TEACHER_ARCH}')
    print(f'{"="*80}\n')
    
    teacher_path = MODEL_WEIGHTS_DIR / TEACHER_WEIGHT_FILE
    if not teacher_path.exists():
        print(f'❌ Teacher не найден: {teacher_path}')
        return
    
    teacher_checkpoint = torch.load(teacher_path, map_location='cpu')
    teacher_model = pe.CLIP.from_config(TEACHER_ARCH, pretrained=False).float()
    teacher_dim   = teacher_model.visual.output_dim
    teacher_head  = ClassificationHead(teacher_dim, num_classes)
    
    # Гибкая загрузка весов учителя
    for key in ['student_model', 'model']:
        if key in teacher_checkpoint:
            teacher_model.load_state_dict(teacher_checkpoint[key])
            print(f'✅ Teacher модель из ключа: "{key}"')
            break
    for key in ['student_head', 'head', 'classifier']:
        if key in teacher_checkpoint:
            teacher_head.load_state_dict(teacher_checkpoint[key])
            print(f'✅ Teacher голова из ключа: "{key}"')
            break
    
    teacher_model.eval()
    teacher_head.eval()
    print(f'✅ Teacher загружен: {TEACHER_ARCH} (dim={teacher_dim})')
    
    Path('checkpoints').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('quantized_models').mkdir(exist_ok=True)
    
    all_results = []
    
    for config in EXPERIMENTS:
        print(f'\n{"="*80}')
        print(f'ЭКСПЕРИМЕНТ : {config["name"]}')
        print(f'Архитектура : {config["arch"]}')
        print(f'Веса        : {config["weight_file"]}')
        print(f'{"="*80}')
        
        weight_path = MODEL_WEIGHTS_DIR / config['weight_file']
        if not weight_path.exists():
            print(f'⚠️  Веса не найдены: {weight_path} — пропускаем\n')
            continue
        
        # ✅ Загружаем готовые веса студента
        checkpoint = torch.load(weight_path, map_location='cpu')
        print(f'   Ключи в checkpoint: {list(checkpoint.keys())}')
        
        student_model = pe.CLIP.from_config(config['arch'], pretrained=False).float()
        student_dim   = student_model.visual.output_dim
        student_head  = ClassificationHead(student_dim, num_classes)
        
        # Гибкая загрузка весов студента
        loaded_model = False
        for key in ['student_model', 'model']:
            if key in checkpoint:
                student_model.load_state_dict(checkpoint[key])
                print(f'   ✅ Веса модели из ключа: "{key}"')
                loaded_model = True
                break
        if not loaded_model:
            print('   ⚠️  Не найден ключ для модели — пропускаем\n')
            continue
        
        loaded_head = False
        for key in ['student_head', 'head', 'classifier']:
            if key in checkpoint:
                student_head.load_state_dict(checkpoint[key])
                print(f'   ✅ Веса головы из ключа: "{key}"')
                loaded_head = True
                break
        if not loaded_head:
            print('   ⚠️  Не найден ключ для head — пропускаем\n')
            continue
        
        # ✅ Квантизуем студента (QAT fake-quant)
        print(f'\n   Квантизация student модели (QAT fake-quant)...')
        student_model = replace_linear_with_quantized(student_model)
        
        # Создаём wrappers
        teacher_wrapper = LogitDistillationWrapper(teacher_model, teacher_head)
        
        if config['distillation_type'] == 'feature':
            projection = ProjectionHead(student_dim, teacher_dim) if config.get('use_projection') else None
            if projection:
                projection = replace_linear_with_quantized(projection)
            student_wrapper = FeatureDistillationWrapper(student_model, student_head, projection)
        else:
            student_wrapper = LogitDistillationWrapper(student_model, student_head)
        
        # QAT + KD обучение
        try:
            history = train_qat_kd(
                student_wrapper, teacher_wrapper,
                train_loader, val_loader, test_loader,
                device, config
            )
        except Exception as e:
            print(f'\n❌ Ошибка: {e}')
            import traceback
            traceback.print_exc()
            continue
        
        # Конвертация в реальный INT8
        print('\n   Конвертация в INT8 inference mode...')
        student_wrapper = convert_to_inference_mode(student_wrapper)
        
        # Финальная оценка
        final_val_acc  = evaluate_model(student_wrapper, val_loader, device)
        final_test_acc = evaluate_model(student_wrapper, test_loader, device)
        final_size     = get_model_size(student_wrapper)
        
        print(f'   ✓ INT8: Val={final_val_acc:.2f}%, Test={final_test_acc:.2f}%, Size={final_size:.2f} MB')
        
        results = {
            'model_name':          config['name'],
            'distillation_type':   config['distillation_type'],
            'arch':                config['arch'],
            'source_weights':      config['weight_file'],
            'quantization_method': 'qat_fake_quant',
            'timestamp':           datetime.utcnow().isoformat(),
            'training_history':    history,
            'final_metrics': {
                'val_accuracy':  float(final_val_acc),
                'test_accuracy': float(final_test_acc),
                'model_size_mb': float(final_size)
            },
            'config': config
        }
        
        all_results.append(results)
        
        model_file   = Path('quantized_models') / f"{config['name']}.pth"
        results_file = Path('results') / f"{config['name']}_results.json"
        
        torch.save(student_wrapper.cpu().state_dict(), model_file)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'\n   💾 Модель    : {model_file}')
        print(f'   💾 Результаты: {results_file}')
        
        if torch.cuda.is_available():
            del student_wrapper
            torch.cuda.empty_cache()
    
    if not all_results:
        print('\n❌ Нет результатов.\n')
        return
    
    summary_file = Path('results/qat_kd_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f'\n{"="*80}')
    print('🎉 QAT + KD ЗАВЕРШЕНО!')
    print(f'{"="*80}')
    print(f'\n{"Модель":<45} {"Best Val":<12} {"Test Acc":<12} {"Size MB":<10}')
    print('-' * 80)
    for r in all_results:
        print(f'{r["model_name"]:<45} '
              f'{r["training_history"]["best_val_acc"]:<12.2f} '
              f'{r["training_history"]["best_test_acc"]:<12.2f} '
              f'{r["final_metrics"]["model_size_mb"]:<10.2f}')

if __name__ == '__main__':
    main()