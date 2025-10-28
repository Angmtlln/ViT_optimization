# ViT_optimization

Repository of completed experiments exploring optimization strategies for Vision Transformers (ViT). The experiments cover multiple approaches (distillation, quantization, pruning, weight-sharing, architecture search) and include reproducible notebooks, utilities, checkpoints and evaluation artifacts. Sections marked TODO are placeholders to be filled with numerical results and notes.

## Summary
- Goal: Reduce model size, latency and energy while retaining accuracy for ViT models.
- Approaches implemented and evaluated: attention distillation, task distillation, post-training quantization, quantization-aware training, unstructured & structured pruning, weight sharing and block-level architecture search.
- All experiments were run on the dataset configured in experiments (ROOT_DIR). See experiment notebooks for dataset splits and preprocessing.

## Experiments (completed)
1. Attention Distillation
   - Notebook: experiments/Attention_distillation_Small.ipynb
   - Objective: Attention-map matching + CE classification loss.
   - Status: Completed
   - Results: Top-1: TODO, Params: TODO, Latency: TODO

2. Task Distillation (logit / feature)
   - Notebook: experiments/Task_distillation.ipynb
   - Objective: Match teacher logits/features with student.
   - Status: Completed
   - Results: Top-1: TODO, Params: TODO, Latency: TODO

3. Quantization
   - Notebooks: experiments/Quantization_PTQ.ipynb, experiments/Quantization_QAT.ipynb
   - Methods: Symmetric/asymmetric INT8, per-channel weight quant, QAT with straight-through estimator.
   - Status: Completed
   - Results: Accuracy drop: TODO, Size reduction: TODO

4. Pruning
   - Notebooks: experiments/Pruning_Unstructured.ipynb, experiments/Pruning_Structured.ipynb
   - Methods: Magnitude pruning, structured attention-head/channel pruning, iterative unstructured pruning.
   - Status: Completed
   - Results: Sparsity vs accuracy curve: TODO

5. Weight Sharing & Parameter Factorization
   - Notebook: experiments/Weight_sharing.ipynb
   - Objective: Reduce parameter redundancy via shared projection layers and low-rank factorization.
   - Status: Completed
   - Results: Params reduced: TODO, Perf: TODO

6. Architecture Search (block-level)
   - Notebook: experiments/Arch_search_BlockLevel.ipynb
   - Objective: Search for reduced-depth/width block structures under FLOPs/latency budget.
   - Status: Completed
   - Results: Pareto front saved: experiments/arch_pareto_*.csv

7. Latency & Energy Evaluation
   - Notebook: experiments/Latency_energy_eval.ipynb
   - Devices: NVIDIA GPU (CUDA) and CPU (Windows).
   - Status: Completed
   - Results: Latency and energy table: TODO

## Results Summary
- Full numeric tables and plots are in experiments/results_summary.md and experiments/figures/.
- Key artifacts:
  - Checkpoints: artifacts/checkpoints/
  - Plots: artifacts/figures/
  - Logs & metrics: artifacts/logs/
- Fill TODOs in each notebook results cell and in experiments/results_summary.md.

## Reproduce (quick)
Open the repository root in VS Code or Jupyter on Windows:

```powershell
cd "c:\Users\amirn\OneDrive\Рабочий стол\ViT_opti\ViT_optimization"
code .
# Or run a specific notebook
jupyter notebook experiments/Attention_distillation_Small.ipynb
```

General install (example):

```powershell
python -m pip install -r requirements.txt
# if requirements.txt not present:
python -m pip install torch torchvision numpy scikit-learn pillow matplotlib
```

Notes:
- Use CUDA where possible. Set NUM_WORKERS = 0 on Windows if DataLoader issues occur.
- Each notebook includes a reproducibility cell with seeds, dataset paths and exact hyperparameters.

## Repo structure
- experiments/           - Notebooks for each experiment
- artifacts/             - Checkpoints, figures, logs (experiment outputs)
- scripts/               - Utility scripts for training/eval/export
- core/                  - Helper modules (token extraction, attention utils, hooks)
- README.md              - This file

## Checkpoints & filenames
Common filenames produced by experiments (examples):
- artifacts/checkpoints/teacher_*.pth
- artifacts/checkpoints/student_*.pth
- artifacts/figures/*.png
- experiments/results_summary.md

## How to add results
- Open the notebook for the experiment.
- Add numeric values to the "Results" cells.
- Update experiments/results_summary.md with consolidated tables.
- Commit artifacts/ entries (avoid large binaries in Git; prefer an artifact storage or Git LFS).

## License & Citation
- License: TODO (add LICENSE file)
- If you use these experiments in a publication, cite this repository and the ViT/Perception-Models sources used.

## Contributing
- Create an issue for new experiments or reproducibility fixes.
- Add notebooks under experiments/ with clear README sections and save outputs under artifacts/.
