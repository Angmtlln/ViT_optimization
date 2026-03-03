import sys, torch, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path('../perception_models').resolve()))
from core.vision_encoder import pe, transforms

TEACHER_ARCH = 'PE-Core-L14-336'
STUDENT_ARCH = 'PE-Core-S16-384'  # или S16-384

device = torch.device('cpu')

# Teacher
teacher = pe.CLIP.from_config(TEACHER_ARCH, pretrained=True).to(device).float().eval()
t_transform = transforms.get_image_transform(teacher.image_size)

# Student
student = pe.CLIP.from_config(STUDENT_ARCH, pretrained=True).to(device).float().eval()
s_transform = transforms.get_image_transform(384)

# Dummy input
dummy_img = torch.randn(1, 3, 384, 384).to(device)

# Warmup
for _ in range(10):
    _ = student.encode_image(dummy_img)

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = student.encode_image(dummy_img)
    times.append(time.perf_counter() - start)

avg_time_ms = np.mean(times) * 1000
teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6  # M
student_params = sum(p.numel() for p in student.parameters()) / 1e6  # M

print(f'Teacher: {teacher_params:.1f}M params')
print(f'Student: {student_params:.1f}M params')
print(f'Student Inference (CPU): {avg_time_ms:.2f} ms')