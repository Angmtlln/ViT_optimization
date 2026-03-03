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

ARCH_TINY = 'PE-Core-T16-384'
ARCH_SMALL = 'PE-Core-S16-384'

MODEL_WEIGHTS_DIR = Path('../../model_weights')

# ==============================
# ЛУЧШИЕ МОДЕЛИ ПО КАЖДОМУ МЕТОДУ ДИСТИЛЛЯЦИИ
# (1 лучшая конфигурация × 4 метода × 2 размера = 8 моделей)
# ==============================

MODELS_TO_QUANTIZE = [
    # --- LOGIT DISTILLATION ---
    # Best: T=3, α=0.5, test_acc=93.5%
    {
        'method': 'logit',
        'weight_file': 'small_weights/logit_T3_alpha0.5_small_best.pth',
        'name': 'logit_T3_alpha0.5_small_ptq_int8'
    },
    # Best: T=5, α=0.5, test_acc=85.375%
    {
        'method': 'logit',
        'weight_file': 'tiny_weights/logit_T5_alpha0.5_tiny_best.pth',
        'name': 'logit_T5_alpha0.5_tiny_ptq_int8'
    },
    # --- ATTENTION DISTILLATION ---
    # Best: α=0.6, loss=mse, lr=5e-5, test_acc=92.0%
    {
        'method': 'attention',
        'weight_file': 'small_weights/attention_lr5e-5_small_best.pth',
        'name': 'attention_lr5e-5_small_ptq_int8'
    },
    # Best: α=0.6, loss=mse, lr=5e-5, test_acc=82.875%
    {
        'method': 'attention',
        'weight_file': 'tiny_weights/attention_lr5e-5_tiny_best.pth',
        'name': 'attention_lr5e-5_tiny_ptq_int8'
    },
    # --- CONTRASTIVE DISTILLATION ---
    # Best: α_contrast=0.3, β_kd=0.4, γ_ce=0.3, T_kd=4.0, T_c=0.07, test_acc=91.625%
    {
        'method': 'contrastive',
        'weight_file': 'small_weights/contrastive_alpha0.3_small_best.pth',
        'name': 'contrastive_alpha0.3_small_ptq_int8'
    },
    # Best: α_contrast=0.5, β_kd=0.3, γ_ce=0.2, T_kd=4.0, T_c=0.1, test_acc=87.375%
    {
        'method': 'contrastive',
        'weight_file': 'tiny_weights/contrastive_T0.10_tiny_best.pth',
        'name': 'contrastive_T0.10_tiny_ptq_int8'
    },
    # --- FEATURE DISTILLATION ---
    # Best: α=0.7, T=4.0, proj=True, test_acc=90.0%
    {
        'method': 'feature',
        'weight_file': 'small_weights/feature_alpha0.7_proj_small_best.pth',
        'name': 'feature_alpha0.7_proj_small_ptq_int8'
    },
    # Best: α=0.9, T=4.0, proj=True, test_acc=81.875%
    {
        'method': 'feature',
        'weight_file': 'tiny_weights/feature_alpha0.9_proj_tiny_best.pth',
        'name': 'feature_alpha0.9_proj_tiny_ptq_int8'
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

def detect_architecture(weight_filename):
    filename_lower = weight_filename.lower()
    if '_small_' in filename_lower:
        return ARCH_SMALL, 384
    elif '_tiny_' in filename_lower:
        return ARCH_TINY, 192
    else:
        print(f'      ⚠️ Не удалось определить архитектуру из имени файла. Используем Tiny по умолчанию.')
        return ARCH_TINY, 192

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
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'scale={self.weight_scale.item():.6f}')


def apply_int8_quantization(model):
    model.eval()
    
    quantized_layers_count = 0
    total_quantized_params = 0
    
    def replace_linear_recursive(module, name=''):
        nonlocal quantized_layers_count, total_quantized_params
        
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child_module, nn.Linear):
                num_params = child_module.weight.numel()
                if child_module.bias is not None:
                    num_params += child_module.bias.numel()
                total_quantized_params += num_params
                
                quantized_layer = QuantizedLinear(
                    child_module.in_features,
                    child_module.out_features,
                    child_module.weight.data,
                    child_module.bias.data if child_module.bias is not None else None
                )
                setattr(module, child_name, quantized_layer)
                quantized_layers_count += 1
                
                if quantized_layers_count <= 5:
                    print(f'      ✓ {full_name} ({child_module.in_features}→{child_module.out_features})')
            else:
                replace_linear_recursive(child_module, full_name)
    
    print('   Замена Linear слоёв на QuantizedLinear...')
    
    total_params_before = sum(p.numel() for p in model.parameters())
    replace_linear_recursive(model)
    
    print(f'      Всего квантизовано слоёв: {quantized_layers_count}')
    print(f'      Квантизовано параметров: {total_quantized_params:,} из {total_params_before:,} ({100*total_quantized_params/total_params_before:.1f}%)')
    
    return model


def get_model_size(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024


def get_model_size_detailed(model):
    size_by_dtype = defaultdict(float)
    total_params = 0
    for name, param in model.named_parameters():
        size_mb = param.nelement() * param.element_size() / 1024 / 1024
        dtype = str(param.dtype).replace('torch.', '')
        size_by_dtype[dtype] += size_mb
        total_params += param.nelement()
    for name, buffer in model.named_buffers():
        size_mb = buffer.nelement() * buffer.element_size() / 1024 / 1024
        dtype = str(buffer.dtype).replace('torch.', '')
        size_by_dtype[dtype] += size_mb
    return dict(size_by_dtype), total_params


def analyze_non_quantized_layers(model):
    non_quantized_params = defaultdict(int)
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            non_quantized_params['LayerNorm'] += sum(p.numel() for p in module.parameters())
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            non_quantized_params['BatchNorm'] += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.Embedding):
            non_quantized_params['Embedding'] += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.Conv2d):
            non_quantized_params['Conv2d'] += sum(p.numel() for p in module.parameters())
    return dict(non_quantized_params)


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


def measure_inference_time(model, loader, device='cpu', num_batches=50):
    model.eval()
    model = model.to(device)
    times = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            x = x.to(device)
            if i < 5:
                _ = model(x)
                continue
            start = time.time()
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)
    return np.mean(times), np.std(times)

# ==============================
# MAIN PTQ PIPELINE
# ==============================

def apply_ptq(model_config, calibration_loader, val_loader, test_loader,
              student_model_fp32, student_head_fp32, device):

    print(f'\n{"="*80}')
    print(f'📊 PTQ для модели: {model_config["name"]}')
    print(f'   Метод дистилляции: {model_config["method"]}')
    print(f'{"="*80}\n')

    class SimpleWrapper(nn.Module):
        def __init__(self, clip_model, head):
            super().__init__()
            self.clip_model = clip_model
            self.head = head
        def forward(self, x):
            features = self.clip_model.encode_image(x)
            logits = self.head(features)
            return logits

    wrapped_model_fp32 = SimpleWrapper(student_model_fp32, student_head_fp32)

    print('   Оценка FP32 модели...')
    fp32_val_acc = evaluate_model(wrapped_model_fp32, val_loader, device='cpu')
    fp32_test_acc = evaluate_model(wrapped_model_fp32, test_loader, device='cpu')
    fp32_time, fp32_std = measure_inference_time(wrapped_model_fp32, val_loader, device='cpu')
    fp32_size = get_model_size(wrapped_model_fp32)
    fp32_size_by_dtype, fp32_params = get_model_size_detailed(wrapped_model_fp32)

    print(f'   ✓ FP32: Val Acc={fp32_val_acc:.2f}%, Test Acc={fp32_test_acc:.2f}%')
    print(f'   ✓ FP32: Size={fp32_size:.2f} MB, Time={fp32_time:.2f}±{fp32_std:.2f} ms')
    print(f'   ✓ FP32: Parameters={fp32_params:,}, Size by dtype: {fp32_size_by_dtype}')

    print('\n   Применение INT8 Quantization...')
    quantized_model = copy.deepcopy(wrapped_model_fp32)
    quantized_model = apply_int8_quantization(quantized_model)

    non_quant_params = analyze_non_quantized_layers(quantized_model)
    if non_quant_params:
        print(f'\n   📋 Неквантизованные слои:')
        for layer_type, count in non_quant_params.items():
            print(f'      {layer_type}: {count:,} параметров ({100*count/fp32_params:.1f}%)')

    int8_size = get_model_size(quantized_model)
    int8_size_by_dtype, int8_params = get_model_size_detailed(quantized_model)

    print(f'\n   📊 Размер модели:')
    print(f'      FP32: {fp32_size:.2f} MB → INT8: {int8_size:.2f} MB')
    print(f'      Сжатие: {fp32_size / int8_size:.2f}x')

    print('\n   Оценка INT8 модели...')
    int8_val_acc = evaluate_model(quantized_model, val_loader, device='cpu')
    int8_test_acc = evaluate_model(quantized_model, test_loader, device='cpu')
    int8_time, int8_std = measure_inference_time(quantized_model, val_loader, device='cpu')

    print(f'   ✓ INT8: Val Acc={int8_val_acc:.2f}%, Test Acc={int8_test_acc:.2f}%')
    print(f'   ✓ INT8: Size={int8_size:.2f} MB, Time={int8_time:.2f}±{int8_std:.2f} ms')

    accuracy_drop = fp32_val_acc - int8_val_acc
    compression_ratio = fp32_size / int8_size if int8_size > 0 else 0
    speedup = fp32_time / int8_time if int8_time > 0 else 0

    print(f'\n   📈 Итог:')
    print(f'      Compression: {compression_ratio:.2f}x | Speedup: {speedup:.2f}x | Accuracy drop: {accuracy_drop:.2f}%')

    results = {
        'model_name': model_config['name'],
        'distillation_method': model_config['method'],
        'quantization_method': 'int8_weights_fp32_activations',
        'timestamp': datetime.utcnow().isoformat(),
        'fp32': {
            'val_accuracy': float(fp32_val_acc),
            'test_accuracy': float(fp32_test_acc),
            'model_size_mb': float(fp32_size),
            'inference_time_ms': float(fp32_time),
            'inference_time_std_ms': float(fp32_std),
            'num_parameters': int(fp32_params),
            'size_by_dtype': {k: float(v) for k, v in fp32_size_by_dtype.items()}
        },
        'int8': {
            'val_accuracy': float(int8_val_acc),
            'test_accuracy': float(int8_test_acc),
            'model_size_mb': float(int8_size),
            'inference_time_ms': float(int8_time),
            'inference_time_std_ms': float(int8_std),
            'num_parameters': int(int8_params),
            'size_by_dtype': {k: float(v) for k, v in int8_size_by_dtype.items()}
        },
        'improvements': {
            'accuracy_drop_percent': float(accuracy_drop),
            'compression_ratio': float(compression_ratio),
            'speedup': float(speedup)
        },
        'config': {
            'seed': SEED,
            'quantization_type': 'weight_only_int8_storage',
        }
    }

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"{model_config['name']}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    models_dir = Path('quantized_models')
    models_dir.mkdir(exist_ok=True)
    model_file = models_dir / f"{model_config['name']}.pth"
    torch.save(quantized_model.state_dict(), model_file)

    print(f'\n   💾 Результаты: {results_file}')
    print(f'   💾 Модель: {model_file}')

    return results

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
    num_classes = len(classes)
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

    calibration_ids = random.sample(train_ids, min(500, len(train_ids)))
    print(f'Calibration: {len(calibration_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}')

    transform = transforms.get_image_transform(384)

    calibration_ds = SimpleDataset(calibration_ids, image_to_path, image_to_label, label_to_idx, transform)
    val_ds         = SimpleDataset(val_ids,         image_to_path, image_to_label, label_to_idx, transform)
    test_ds        = SimpleDataset(test_ids,        image_to_path, image_to_label, label_to_idx, transform)

    calibration_loader = DataLoader(calibration_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_loader         = DataLoader(val_ds,         batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader        = DataLoader(test_ds,        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    Path('results').mkdir(exist_ok=True)
    Path('quantized_models').mkdir(exist_ok=True)

    all_results = []

    for model_config in MODELS_TO_QUANTIZE:
        weight_path = MODEL_WEIGHTS_DIR / model_config['weight_file']

        if not weight_path.exists():
            print(f'\n⚠️  Веса не найдены: {weight_path} — пропускаем\n')
            continue

        print(f'\n{"="*80}')
        print(f'Загрузка модели: {weight_path}')
        print(f'{"="*80}')

        checkpoint = torch.load(weight_path, map_location='cpu')
        print(f'   Ключи в checkpoint: {list(checkpoint.keys())}')

        student_arch, expected_dim = detect_architecture(model_config['weight_file'])
        print(f'   Архитектура: {student_arch} (dim={expected_dim})')

        student_model = pe.CLIP.from_config(student_arch, pretrained=False).float()
        student_dim   = student_model.visual.output_dim
        student_head  = ClassificationHead(student_dim, num_classes)
        print(f'   Размерность модели: {student_dim}')

        # Загрузка весов модели
        if 'student_model' in checkpoint:
            student_model.load_state_dict(checkpoint['student_model'])
        elif 'model' in checkpoint:
            student_model.load_state_dict(checkpoint['model'])
        else:
            student_model.load_state_dict(checkpoint)

        # Загрузка весов головы
        if 'student_head' in checkpoint:
            student_head.load_state_dict(checkpoint['student_head'])
        elif 'head' in checkpoint:
            student_head.load_state_dict(checkpoint['head'])
        elif 'classifier' in checkpoint:
            student_head.load_state_dict(checkpoint['classifier'])
        else:
            print(f'   ⚠️  Не найден ключ для head — пропускаем\n')
            continue

        student_model.eval()
        student_head.eval()

        try:
            results = apply_ptq(
                model_config, calibration_loader, val_loader, test_loader,
                student_model, student_head, device
            )
            all_results.append(results)
        except Exception as e:
            print(f'\n❌ Ошибка при квантизации {model_config["name"]}: {e}')
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print('\n❌ Нет результатов для сохранения.')
        return

    summary_file = Path('results/ptq_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\n{"="*80}')
    print(f'🎉 ВСЕ PTQ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ! Всего квантизовано: {len(all_results)}')
    print(f'{"="*80}')
    print(f'\n{"Метод":<35} {"FP32 Acc":<12} {"INT8 Acc":<12} {"Drop":<10} {"Compression":<15} {"Speedup":<10}')
    print('-' * 95)
    for r in all_results:
        print(f'{r["model_name"]:<35} '
              f'{r["fp32"]["val_accuracy"]:<12.2f} '
              f'{r["int8"]["val_accuracy"]:<12.2f} '
              f'{r["improvements"]["accuracy_drop_percent"]:<10.2f} '
              f'{r["improvements"]["compression_ratio"]:<15.2f}x '
              f'{r["improvements"]["speedup"]:<10.2f}x')

if __name__ == '__main__':
    main()