import json
from pathlib import Path

ROOT = Path(r"c:\Users\amirn\OneDrive\Рабочий стол\ViT_opti\ViT_optimization")
RESULT_DIRS = [
    ROOT / "results",
]

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out

rows = []
for base in RESULT_DIRS:
    if not base.exists():
        continue
    for p in base.rglob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        flat = flatten(data)
        flat["__file"] = str(p.relative_to(ROOT))
        rows.append(flat)

# собрать все ключи
all_keys = sorted({k for r in rows for k in r.keys()})
# вывод в CSV
out_csv = ROOT / "results" / "all_metrics.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)

with open(out_csv, "w", encoding="utf-8") as f:
    f.write(",".join(all_keys) + "\n")
    for r in rows:
        f.write(",".join(str(r.get(k, "")) for k in all_keys) + "\n")

print(f"Saved: {out_csv}")