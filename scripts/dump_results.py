import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

folders = {
    "logit":       ("logit_distillation_S",       "logit_distillation_T"),
    "feature":     ("feature_distillation_S",     "feature_distillation_T"),
    "attention":   ("attention_distillation_S",   "attention_distillation_t"),
    "contrastive": ("contrastive_distillation_S", "contrastive_distillation_T"),
}

for method, (fs, ft) in folders.items():
    print(f"\n=== {method.upper()} ===")

    s_data = {}
    for jf in sorted((RESULTS / fs).glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            s_data[jf.stem] = json.load(f).get("test_accuracy")

    t_data = {}
    for jf in sorted((RESULTS / ft).glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            t_data[jf.stem] = json.load(f).get("test_accuracy")

    # Нормализуем stem: убираем суффикс _small/_tiny
    s_norm = {k.replace("_small", ""): v for k, v in s_data.items()}
    t_norm = {k.replace("_tiny",  ""): v for k, v in t_data.items()}

    all_keys = sorted(set(list(s_norm) + list(t_norm)))
    for k in all_keys:
        sv = s_norm.get(k)
        tv = t_norm.get(k)
        sv_str = f"{sv:.2f}%" if sv is not None else "—"
        tv_str = f"{tv:.2f}%" if tv is not None else "—"
        print(f"  {k:<50}  S={sv_str:<10}  T={tv_str}")