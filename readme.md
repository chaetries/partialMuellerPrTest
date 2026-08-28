# Physical Realizability Prediction for Mueller Matrices

This repository contains utilities, notebooks, and experiment definitions for predicting whether measured Mueller matrices are physically realizable. It supports analytical physical-realizability tests and machine-learning models that operate on full or partial Mueller-matrix measurements.

The project was developed for polarimetric imaging experiments on cervix, brain, and AFM Mueller-matrix microscopy data. Data, trained models, publication files, and generated results are intentionally kept outside Git because they are large or environment-specific.

## What is included

- Analytical realizability checks based on coherency-matrix eigenvalues and characteristic-polynomial criteria.
- Batch classification and preprocessing of per-pixel Mueller matrices.
- XGBoost, CatBoost, and PyTorch MLP model-loading helpers.
- Full-matrix and partial-measurement experiment configurations.
- Evaluation helpers for regression metrics, SSIM, and reconstructed images.
- Mueller-matrix and Lu-Chipman decomposition visualizations.

## Experiment variants

| Name | Matrix entries used | Notebook |
| --- | --- | --- |
| `full` | Complete 4 x 4 matrix | Training workflow |
| `3x3` | Partial 3 x 3 measurement | `pr_partial_3x3.ipynb` |
| `4x3` | Partial 4 x 3 measurement | `pr_partial_4x3.ipynb` |
| `4x1_lastcol` | Last-column partial measurement | `pr_partial_4x1_lastcol.ipynb` |

Experiment metadata and path resolution are centralized in `src/utils/experiments.py`.

## Repository layout

```text
partialPr/
├── notebooks/
│   ├── training/          # model-training workflow
│   ├── experiments/       # partial-measurement experiments
│   └── testing/           # evaluation and sample inference
├── src/utils/
│   ├── physical_realizability.py
│   ├── experiments.py
│   ├── file_paths.py
│   ├── testing_utils/
│   ├── utils.py
│   ├── visualization.py
│   └── lu_chipman.py
└── archive/               # retained legacy workflows
```

The directories `data/`, `model/`, `results/`, `publication/`, and `temporary/` are local working directories and are ignored by Git.

## Installation

Python 3.10 or newer is recommended. Create a virtual environment and install the scientific Python dependencies used by the notebooks:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy matplotlib seaborn scikit-learn scikit-image jupyter
```

For machine-learning experiments, also install the required model libraries:

```bash
python -m pip install torch xgboost catboost
```

Run commands from the repository root so imports such as `src.utils` resolve correctly.

## Data layout

The default local structure is:

```text
data/
├── raw/
├── interim/
├── processed/
└── test/
```

By default, processed data is read from `data/processed`. Set `PARTIAL_PR_PROCESSED_PATH` to use another location:

```bash
export PARTIAL_PR_PROCESSED_PATH=/path/to/processed/data
```

The path helper also recognizes the original external-drive layout when it is mounted.

## Basic usage

Test one Mueller matrix analytically:

```python
import numpy as np

from src.utils.physical_realizability import charpoly

matrix = np.eye(4)
is_realizable = charpoly(matrix)
print(is_realizable)
```

Test a batch of matrices:

```python
from src.utils.physical_realizability import charpoly_vectorized

# matrices has shape (n_samples, 4, 4)
mask = charpoly_vectorized(matrices)
```

Resolve paths for a registered experiment:

```python
from src.utils.file_paths import file_paths

model_path = file_paths.experiment_model_path("4x3", "xgb")
results_dir = file_paths.experiment_results_path("4x3")
notebook_path = file_paths.experiment_notebook_path("4x3")
```

Supported model-family names are `xgb`, `catboost`, and `pixel_mlp`.

## Typical workflow

1. Place raw or processed Mueller-matrix data under `data/`, or configure the processed-data environment variable.
2. Use `notebooks/data_processing.ipynb` to prepare per-pixel matrices and labels.
3. Train full-matrix models with `notebooks/training/training.ipynb`.
4. Run a notebook under `notebooks/experiments/` for a partial-measurement variant.
5. Evaluate models with the testing notebooks or robustness scripts.
6. Write generated figures and metrics to `results/`.

## Notes

- `src/utils/pr_test.py` and `src/utils/visualisation.py` are compatibility wrappers for older notebooks.
- Trained model files are resolved first from `model/experiments/<experiment>/`, with support for the earlier flat model layout.
- Tissue image dimensions are defined in `src/utils/file_paths.py` and currently cover cervix, brain, AFMMM, and simulated samples.

## Repository

GitHub: <https://github.com/chaetries/pr_prediction>
