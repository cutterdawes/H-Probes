# H-Probes

Tools and notebooks for hierarchical probe experiments on tree-traversal and GSM8K-style math tasks. The repo covers dataset generation, model response collection, probe training/evaluation, and ablation-style interventions.

## Repository layout
- `notebooks/`: analysis and visualization notebooks (geometry, statistics, comparisons).
- `scripts/`: command-line pipelines for dataset creation, generation, probing, and interventions.
- `utils/`: shared utilities for datasets, prompting, probing, plotting, and model helpers.
- `environment.yml`: conda environment definition (Python 3.10 + PyTorch + HF stack).

## Quickstart
```bash
conda env create -f environment.yml
conda activate hprobes
python scripts/pipeline.py --setting tree
```

## Common entry points
- `scripts/create_dataset.py`: build tree or GSM8K datasets.
- `scripts/generate_responses.py`: generate model outputs + embeddings.
- `scripts/evaluate_probe.py`: train/evaluate probes.
- `scripts/intervene.py`: probe-based ablations and interventions.
- `scripts/visualize.py`: summary plots and report figures.
