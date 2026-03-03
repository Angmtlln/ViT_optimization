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

# Fine-tuning параметры
LEARNING_RATE = 1e-5
FINETUNE_EPOCHS = 3
WARMUP_STEPS = 50

# Пути
MODEL_WEIGHTS_DIR = Path('../../model_weights')

# ==============================
# ЛУЧШИЕ МОДЕЛИ ПО КАЖДОМУ МЕТОДУ ДИСТИЛЛЯЦИИ
# (1 лучшая × 4 метода × 2 размера = 8 моделей)
# ==============================

MODELS_TO_QUANTIZE = [
    # --- LOGIT DISTILLATION ---
    # Best: T=3, α=0.5, test_acc=93.5%
    {
        'method': 'logit',
        'arch': ARCH_SMALL,
        'weight_file': 'small_weights/logit_T3_alpha0.5_small_best.pth',
        'name': 'logit_T3_alpha0.5_small_progressive_int8'
    },
    # Best: T=5, α=0.5, test_acc=85.375%
    {
        'method': 'logit',
        'arch': ARCH_TINY,
        'weight_file': 'tiny_weights/logit_T5_alpha0.5_tiny_best.pth',
        'name': 'logit_T5_alpha0.5_tiny_progressive_int8'
    },
    # --- ATTENTION DISTILLATION ---
    # Best: α=0.6, loss=mse, lr=5e-5, test_acc=92.0%
    {
        'method': 'attention',
        'arch': ARCH_SMALL,
        'weight_file': 'small_weights/attention_lr5e-5_small_best.pth',
        'name': 'attention_lr5e-5_small_progressive_int8'
    },
    # Best: α=0.6, loss=mse, lr=5e-5, test_acc=82.875%
    {
        'method': 'attention',
        'arch': ARCH_TINY,
        'weight_file': 'tiny_weights/attention_lr5e-5_tiny_best.pth',
        'name': 'attention_lr5e-5_tiny_progressive_int8'
    },
    # --- CONTRASTIVE DISTILLATION ---
    # Best: α_contrast=0.3, β_kd=0.4, γ_ce=0.3, T_kd=4.0, T_c=0.07, test_acc=91.625%
    {
        'method': 'contrastive',
        'arch': ARCH_SMALL,
        'weight_file': 'small_weights/contrastive_alpha0.3_small_best.pth',
        'name': 'contrastive_alpha0.3_small_progressive_int8'
    },
    # Best: α_contrast=0.5, β_kd=0.3, γ_ce=0.2, T_kd=4.0, T_c=0.1, test_acc=87.375%
    {
        'method': 'contrastive',
        'arch': ARCH_TINY,
        'weight_file': 'tiny_weights/contrastive_T0.10_tiny_best.pth',
        'name': 'contrastive_T0.10_tiny_progressive_int8'
    },
    # --- FEATURE DISTILLATION ---
    # Best: α=0.7, T=4.0, proj=True, test_acc=90.0%
    {
        'method': 'feature',
        'arch': ARCH_SMALL,
        'weight_file': 'small_weights/feature_alpha0.7_proj_small_best.pth',
        'name': 'feature_alpha0.7_proj_small_progressive_int8'
    },
    # Best: α=0.9, T=4.0, proj=True, test_acc=81.875%
    {
        'method': 'feature',
        'arch': ARCH_TINY,
        'weight_file': 'tiny_weights/feature_alpha0.9_proj_tiny_best.pth',
        'name': 'feature_alpha0.9_proj_tiny_progressive_int8'
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

# ==============================
# QUANTIZATION FUNCTIONS
# ==============================

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, weight, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        weight_float = weight.detach().clone()
        scale = weight_float.abs().max().item() / 127.0
        
        if scale > 0:
            quantized_weight = torch.round(weight_float / scale).clamp(-128, 127).to(torch.int8)
        else:
            quantized_weight = torch.zeros_like(weight_float, dtype=torch.int8)
        
        self.register_buffer('weight_int8', quantized_weight)
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
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'


def get_all_linear_layers(model):
    linear_layers = []
    
    def find_linear_recursive(module, name=''):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child_module, (nn.Linear, QuantizedLinear)):
                linear_layers.append((full_name, child_module))
            else:
                find_linear_recursive(child_module, full_name)
    
    find_linear_recursive(model)
    return linear_layers


def quantize_layer(model, layer_name):
    parts = layer_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    layer = getattr(parent, parts[-1])
    
    if isinstance(layer, nn.Linear):
        quantized = QuantizedLinear(
            layer.in_features,
            layer.out_features,
            layer.weight.data,
            layer.bias.data if layer.bias is not None else None
        )
        setattr(parent, parts[-1], quantized)
        return True
    return False


def get_model_size(model):
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def evaluate_model(model, loader, device='cpu'):
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


def finetune_step(model, train_loader, val_loader, device='cpu', epochs=3, lr=1e-5):
    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Fine-tune Epoch {epoch+1}/{epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_acc = evaluate_model(model, val_loader, device)
        print(f'   Epoch {epoch+1}: Val Acc = {val_acc:.2f}%')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc

# ==============================
# PROGRESSIVE QUANTIZATION
# ==============================

def progressive_quantization(model, train_loader, val_loader, test_loader, device='cpu'):
    print('\n📊 Начало прогрессивной квантизации...\n')
    
    print('   Оценка исходной FP32 модели...')
    fp32_val_acc  = evaluate_model(model, val_loader, device)
    fp32_test_acc = evaluate_model(model, test_loader, device)
    fp32_size     = get_model_size(model)
    print(f'   ✓ FP32: Val={fp32_val_acc:.2f}%, Test={fp32_test_acc:.2f}%, Size={fp32_size:.2f} MB\n')
    
    all_layers = list(reversed(get_all_linear_layers(model)))
    print(f'   Найдено {len(all_layers)} Linear слоёв')
    
    history = []
    num_steps = 5
    layers_per_step = max(1, len(all_layers) // num_steps)
    quantized_count = 0
    
    for step in range(num_steps):
        print(f'\n{"="*80}')
        print(f'ШАГ {step+1}/{num_steps}: Квантизация {layers_per_step} слоёв')
        print(f'{"="*80}')
        
        start_idx = step * layers_per_step
        end_idx   = min(start_idx + layers_per_step, len(all_layers))
        
        for i in range(start_idx, end_idx):
            layer_name, _ = all_layers[i]
            if quantize_layer(model, layer_name):
                quantized_count += 1
                if quantized_count <= 3:
                    print(f'      ✓ Квантизован: {layer_name}')
        
        print(f'      Всего квантизовано: {quantized_count}/{len(all_layers)} слоёв')
        
        curr_val_acc = evaluate_model(model, val_loader, device)
        curr_size    = get_model_size(model)
        
        print(f'\n   После квантизации (без fine-tuning):')
        print(f'      Val Acc: {curr_val_acc:.2f}% (drop: {fp32_val_acc - curr_val_acc:.2f}%)')
        print(f'      Size: {curr_size:.2f} MB (compression: {fp32_size / curr_size:.2f}x)')
        
        if step < num_steps - 1:
            print(f'\n   Fine-tuning для восстановления точности...')
            finetuned_val_acc = finetune_step(
                model, train_loader, val_loader,
                device=device, epochs=FINETUNE_EPOCHS, lr=LEARNING_RATE
            )
            print(f'\n   После fine-tuning:')
            print(f'      Val Acc: {finetuned_val_acc:.2f}% (восстановлено: +{finetuned_val_acc - curr_val_acc:.2f}%)')
        else:
            finetuned_val_acc = curr_val_acc
        
        history.append({
            'step': step + 1,
            'quantized_layers': quantized_count,
            'val_acc_before_ft': float(curr_val_acc),
            'val_acc_after_ft': float(finetuned_val_acc),
            'model_size_mb': float(curr_size),
            'compression_ratio': float(fp32_size / curr_size)
        })
    
    print(f'\n{"="*80}')
    print('ФИНАЛЬНАЯ ОЦЕНКА')
    print(f'{"="*80}\n')
    
    final_val_acc  = evaluate_model(model, val_loader, device)
    final_test_acc = evaluate_model(model, test_loader, device)
    final_size     = get_model_size(model)
    
    print(f'   ✓ Final INT8: Val={final_val_acc:.2f}%, Test={final_test_acc:.2f}%, Size={final_size:.2f} MB')
    print(f'   ✓ Compression: {fp32_size/final_size:.2f}x ({fp32_size:.2f} MB → {final_size:.2f} MB)')
    print(f'   ✓ Accuracy drop: {fp32_val_acc - final_val_acc:.2f}%')
    
    return {
        'fp32': {
            'val_accuracy':  float(fp32_val_acc),
            'test_accuracy': float(fp32_test_acc),
            'model_size_mb': float(fp32_size)
        },
        'int8': {
            'val_accuracy':  float(final_val_acc),
            'test_accuracy': float(final_test_acc),
            'model_size_mb': float(final_size)
        },
        'improvements': {
            'accuracy_drop_percent': float(fp32_val_acc - final_val_acc),
            'compression_ratio':     float(fp32_size / final_size)
        },
        'history': history,
        'config': {
            'num_steps':              num_steps,
            'layers_per_step':        layers_per_step,
            'total_layers_quantized': quantized_count,
            'finetune_epochs':        FINETUNE_EPOCHS,
            'learning_rate':          LEARNING_RATE
        }
    }, model

# ==============================
# MAIN
# ==============================

def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n')
    
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
    num_classes  = len(classes)
    print(f'Найдено {len(image_to_label)} изображений, {num_classes} классов')
    
    images_by_class = defaultdict(list)
    for img_id, lbl in image_to_label.items():
        images_by_class[lbl].append(img_id)
    
    images_per_class = 500
    sampled = []
    for lbl, ids in images_by_class.items():
        k = min(images_per_class, len(ids))
        sampled.extend(random.sample(ids, k))
    random.shuffle(sampled)
    
    labels_for_strat = [image_to_label[i] for i in sampled]
    train_val_ids, test_ids = train_test_split(sampled, test_size=0.20,
                                               stratify=labels_for_strat, random_state=SEED)
    train_labels_for_strat = [image_to_label[i] for i in train_val_ids]
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.20,
                                          stratify=train_labels_for_strat, random_state=SEED)
    
    print(f'Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}')
    
    transform    = transforms.get_image_transform(384)
    train_ds     = SimpleDataset(train_ids, image_to_path, image_to_label, label_to_idx, transform)
    val_ds       = SimpleDataset(val_ids,   image_to_path, image_to_label, label_to_idx, transform)
    test_ds      = SimpleDataset(test_ids,  image_to_path, image_to_label, label_to_idx, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    Path('results').mkdir(exist_ok=True)
    Path('quantized_models').mkdir(exist_ok=True)
    
    all_results = []
    
    for model_config in MODELS_TO_QUANTIZE:
        weight_path = MODEL_WEIGHTS_DIR / model_config['weight_file']
        
        if not weight_path.exists():
            print(f'\n⚠️  Веса не найдены: {weight_path} — пропускаем\n')
            continue
        
        print(f'\n{"="*80}')
        print(f'МОДЕЛЬ: {model_config["name"]}')
        print(f'Метод дистилляции: {model_config["method"]}')
        print(f'Архитектура: {model_config["arch"]}')
        print(f'{"="*80}')
        
        checkpoint    = torch.load(weight_path, map_location='cpu')
        print(f'   Ключи в checkpoint: {list(checkpoint.keys())}')

        # Используем arch из конфига — не глобальный STUDENT_ARCH
        student_model = pe.CLIP.from_config(model_config['arch'], pretrained=False).float()
        student_dim   = student_model.visual.output_dim
        student_head  = ClassificationHead(student_dim, num_classes)
        
        # Гибкая загрузка весов модели
        for key in ['student_model', 'model']:
            if key in checkpoint:
                student_model.load_state_dict(checkpoint[key])
                print(f'   ✅ Веса модели из ключа: "{key}"')
                break
        
        # Гибкая загрузка весов головы
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
        
        class SimpleWrapper(nn.Module):
            def __init__(self, clip_model, head):
                super().__init__()
                self.clip_model = clip_model
                self.head = head
            def forward(self, x):
                features = self.clip_model.encode_image(x)
                return self.head(features)
        
        wrapped_model = SimpleWrapper(student_model, student_head)
        wrapped_model.eval()
        
        try:
            results, quantized_model = progressive_quantization(
                wrapped_model, train_loader, val_loader, test_loader, device
            )
            
            results['model_name']           = model_config['name']
            results['distillation_method']  = model_config['method']
            results['arch']                 = model_config['arch']
            results['quantization_method']  = 'progressive_int8'
            results['timestamp']            = datetime.utcnow().isoformat()
            
            all_results.append(results)
            
            results_file = Path('results') / f"{model_config['name']}_results.json"
            model_file   = Path('quantized_models') / f"{model_config['name']}.pth"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            torch.save(quantized_model.state_dict(), model_file)
            
            print(f'\n   💾 Результаты: {results_file}')
            print(f'   💾 Модель: {model_file}')
        
        except Exception as e:
            print(f'\n❌ Ошибка: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print('\n❌ Нет результатов.\n')
        return
    
    summary_file = Path('results/progressive_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f'\n{"="*80}')
    print('🎉 ПРОГРЕССИВНАЯ КВАНТИЗАЦИЯ ЗАВЕРШЕНА!')
    print(f'{"="*80}')
    print(f'\n{"Модель":<45} {"FP32 Acc":<12} {"INT8 Acc":<12} {"Drop":<10} {"Compression":<15}')
    print('-' * 95)
    for r in all_results:
        print(f'{r["model_name"]:<45} '
              f'{r["fp32"]["val_accuracy"]:<12.2f} '
              f'{r["int8"]["val_accuracy"]:<12.2f} '
              f'{r["improvements"]["accuracy_drop_percent"]:<10.2f} '
              f'{r["improvements"]["compression_ratio"]:<15.2f}x')

if __name__ == '__main__':
    main()