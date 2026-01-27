import os
import sys
import subprocess
import itertools
from pathlib import Path

# Параметры экспериментов для Feature Distillation
teacher_epochs_list = [4, 5]
student_epochs_list = [5, 6,]

experiments = list(itertools.product(teacher_epochs_list, student_epochs_list))

print(f'🔬 ЗАПУСК {len(experiments)} ЭКСПЕРИМЕНТОВ (FEATURE DISTILLATION)')
print('='*70)

for i, (t_ep, s_ep) in enumerate(experiments, 1):
    run_id = f"feature_dist_T{t_ep}ep_S{s_ep}ep"
    
    print(f'\n[{i}/{len(experiments)}] 🚀 Запуск: {run_id}')
    print(f'  Teacher Epochs: {t_ep}')
    print(f'  Student Epochs: {s_ep}')
    print('-'*70)
    
    script_content = f"""
import os, sys, math, random, json, time
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

# Параметры
TEACHER_EPOCHS = {t_ep}
KD_EPOCHS = {s_ep}
RUN_ID = "{run_id}"
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 0
FEATURE_ALPHA = 0.5  # Вес для feature loss
KD_LR = 1e-4
TEACHER_LR = 5e-4
ROOT_DIR = '../data/kvasir-dataset-v2'
SAVE_BEST = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {{device}}')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sys.path.insert(0, str(Path('../perception_models').resolve()))
from core.vision_encoder import pe, transforms

teacher_model = pe.CLIP.from_config('PE-Core-L14-336', pretrained=True).to(device).float().eval()
student_model = pe.CLIP.from_config('PE-Core-S16-384', pretrained=True).to(device).float()

teacher_preprocessor = transforms.get_image_transform(teacher_model.image_size)
student_preprocessor = transforms.get_image_transform(student_model.image_size)
teacher_dim = teacher_model.visual.output_dim
student_dim = student_model.visual.output_dim

# Датасет (тот же код)
def extract_label_from_path(image_path, root_dir):
    parts = Path(image_path).parts
    try:
        ridx = parts.index(Path(root_dir).name)
        return parts[ridx + 1]
    except (ValueError, IndexError):
        return None

all_image_paths = glob(os.path.join(ROOT_DIR, '**', '*.jpg'), recursive=True)
image_to_path, image_to_label = {{}}, {{}}
for p in all_image_paths:
    image_id = os.path.splitext(os.path.basename(p))[0]
    label = extract_label_from_path(p, ROOT_DIR)
    if label is not None:
        image_to_path[image_id] = p
        image_to_label[image_id] = label

classes = sorted(list(set(image_to_label.values())))
label_to_idx = {{c: i for i, c in enumerate(classes)}}
idx_to_label = {{i: c for c, i in label_to_idx.items()}}

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
train_val_ids, test_ids = train_test_split(sampled, test_size=0.20, stratify=labels_for_strat, random_state=SEED)
train_labels_for_strat = [image_to_label[i] for i in train_val_ids]
train_ids, val_ids = train_test_split(train_val_ids, test_size=0.20, stratify=train_labels_for_strat, random_state=SEED)

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

train_ds = KDDataset(train_ids, image_to_path, image_to_label, label_to_idx, teacher_preprocessor, student_preprocessor)
val_ds = KDDataset(val_ids, image_to_path, image_to_label, label_to_idx, teacher_preprocessor, student_preprocessor)
test_ds = KDDataset(test_ids, image_to_path, image_to_label, label_to_idx, teacher_preprocessor, student_preprocessor)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

# Головы
class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

# Projection layer для выравнивания размерностей
class FeatureProjection(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.projection = nn.Linear(student_dim, teacher_dim)
    def forward(self, x):
        return self.projection(x)

num_classes = len(classes)
teacher_head = LinearHead(teacher_dim, num_classes).to(device)
student_head = LinearHead(student_dim, num_classes).to(device)
feature_projector = FeatureProjection(student_dim, teacher_dim).to(device)

# Обучение teacher
print(f'\\n🎓 Обучение Teacher на {{TEACHER_EPOCHS}} эпох...')
for p in teacher_model.parameters():
    p.requires_grad = False

teacher_head.train()
opt = torch.optim.AdamW(teacher_head.parameters(), lr=TEACHER_LR, weight_decay=1e-5)
ce = nn.CrossEntropyLoss()

for ep in range(1, TEACHER_EPOCHS + 1):
    for x_t, _, y in tqdm(train_loader, desc=f'Teacher Epoch {{ep}}/{{TEACHER_EPOCHS}}'):
        x_t = x_t.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            feats = teacher_model.encode_image(x_t)
        logits = teacher_head(feats)
        loss = ce(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

for p in teacher_head.parameters():
    p.requires_grad = False
teacher_head.eval()

os.makedirs('../model_weights', exist_ok=True)
torch.save({{'state_dict': teacher_head.state_dict(), 'num_classes': num_classes}}, 
           f'../model_weights/teacher_feature_probe_{{RUN_ID}}.pth')

# Feature Distillation Loss
class FeatureDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, student_features, teacher_features, student_logits, labels):
        # Feature matching loss (MSE)
        feature_loss = self.mse(student_features, teacher_features)
        
        # Classification loss
        ce_loss = self.ce(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * feature_loss + (1.0 - self.alpha) * ce_loss
        
        return total_loss, feature_loss, ce_loss

fd_criterion = FeatureDistillationLoss(alpha=FEATURE_ALPHA)

# Обучение студента с Feature Distillation
print(f'\\n📚 Обучение Student с Feature Distillation на {{KD_EPOCHS}} эпох...')
optimizer = torch.optim.AdamW(
    list(student_model.parameters()) + 
    list(student_head.parameters()) + 
    list(feature_projector.parameters()), 
    lr=KD_LR, 
    weight_decay=1e-5
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=KD_EPOCHS)

history = {{
    'train_loss': [], 
    'val_loss': [], 
    'val_acc': [],
    'feature_loss': [],
    'ce_loss': []
}}
best_val_acc = -1.0

for epoch in range(1, KD_EPOCHS + 1):
    student_model.train()
    student_head.train()
    feature_projector.train()
    teacher_model.eval()
    teacher_head.eval()
    
    loss_sum = 0.0
    feature_loss_sum = 0.0
    ce_loss_sum = 0.0
    n_batches = 0
    
    for x_t, x_s, y in tqdm(train_loader, desc=f'Feature KD Epoch {{epoch}}/{{KD_EPOCHS}}'):
        x_t = x_t.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Teacher features (frozen)
        with torch.no_grad():
            t_feat = teacher_model.encode_image(x_t)
        
        # Student features
        s_feat = student_model.encode_image(x_s)
        s_feat_projected = feature_projector(s_feat)
        s_logits = student_head(s_feat)
        
        # Compute loss
        loss, feat_loss, ce_loss = fd_criterion(s_feat_projected, t_feat, s_logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        feature_loss_sum += feat_loss.item()
        ce_loss_sum += ce_loss.item()
        n_batches += 1
    
    scheduler.step()
    
    # Валидация
    student_model.eval()
    student_head.eval()
    feature_projector.eval()
    correct, total = 0, 0
    val_loss_sum = 0.0
    val_feature_loss_sum = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for x_t, x_s, y in val_loader:
            x_t = x_t.to(device, non_blocking=True)
            x_s = x_s.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            t_feat = teacher_model.encode_image(x_t)
            s_feat = student_model.encode_image(x_s)
            s_feat_projected = feature_projector(s_feat)
            s_logits = student_head(s_feat)
            
            loss, feat_loss, _ = fd_criterion(s_feat_projected, t_feat, s_logits, y)
            val_loss_sum += loss.item()
            val_feature_loss_sum += feat_loss.item()
            val_batches += 1
            
            _, pred = torch.max(s_logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    
    val_acc = 100.0 * correct / total if total > 0 else 0.0
    val_loss = val_loss_sum / max(1, val_batches)
    val_feature_loss = val_feature_loss_sum / max(1, val_batches)
    
    history['train_loss'].append(loss_sum / max(1, n_batches))
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['feature_loss'].append(feature_loss_sum / max(1, n_batches))
    history['ce_loss'].append(ce_loss_sum / max(1, n_batches))
    
    print(f'Epoch {{epoch}}: Val Loss={{val_loss:.4f}}, Feature Loss={{val_feature_loss:.4f}}, Val Acc={{val_acc:.2f}}%')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({{
            'student_model': student_model.state_dict(),
            'student_head': student_head.state_dict(),
            'feature_projector': feature_projector.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc
        }}, f"../model_weights/student_feature_best_{{RUN_ID}}.pth")

# Тестирование
student_model.eval()
student_head.eval()
correct, total = 0, 0
with torch.no_grad():
    for _, x_s, y in test_loader:
        x_s = x_s.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        s_feat = student_model.encode_image(x_s)
        logits = student_head(s_feat)
        _, pred = torch.max(logits, 1)
        total += y.size(0)
        correct += (pred == y).sum().item()

test_acc = 100.0 * correct / total if total > 0 else 0.0

# Сохранение результатов
results = {{
    "run_id": RUN_ID,
    "student_accuracy": float(history['val_acc'][-1]),
    "test_accuracy": float(test_acc),
    "final_feature_loss": float(history['feature_loss'][-1]),
    "final_ce_loss": float(history['ce_loss'][-1]),
    "num_epochs": KD_EPOCHS,
    "history": {{
        "train_loss": [float(x) for x in history['train_loss']],
        "val_loss": [float(x) for x in history['val_loss']],
        "val_acc": [float(x) for x in history['val_acc']],
        "feature_loss": [float(x) for x in history['feature_loss']],
        "ce_loss": [float(x) for x in history['ce_loss']]
    }},
    "hyperparameters": {{
        "teacher_epochs": TEACHER_EPOCHS,
        "feature_alpha": FEATURE_ALPHA,
        "kd_lr": KD_LR
    }}
}}

os.makedirs('../results', exist_ok=True)
with open(f'../results/{{RUN_ID}}.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\\n✅ Эксперимент {{RUN_ID}} завершен!')
print(f'Feature Loss: {{history["feature_loss"][-1]:.4f}}')
print(f'Test Acc: {{test_acc:.2f}}%')
"""
    
    temp_script = Path(f'../scripts/temp_feature_exp_{run_id}.py')
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    try:
        subprocess.run([sys.executable, str(temp_script)], check=True)
        print(f'✅ Эксперимент {run_id} завершен успешно!')
    except subprocess.CalledProcessError as e:
        print(f'❌ Ошибка в эксперименте {run_id}: {e}')
    finally:
        if temp_script.exists():
            temp_script.unlink()

print('\n' + '='*70)
print('🎉 ВСЕ FEATURE DISTILLATION ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!')
print('='*70)