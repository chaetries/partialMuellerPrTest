"""
Robustness Testing for the 4x3 Partial Mueller Matrix XGBoost Model.

Testing I  – Isolated held-out samples (brain / cervix / AFMMM)
Testing II – Monte Carlo simulated samples (225 µm, 250 µm, 300 µm)

Run from the project root:
    python notebooks/experiments/robustness_4x3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.file_paths import file_paths
from src.utils.pr_test import charpoly_vectorized

INTERIM_BASE = Path("/Volumes/ep_ssd/database/partialPr/data/interim")
XGB_PATH = file_paths.model_save_path / "pixel_xgb_4x3.json"
CAT_PATH = file_paths.model_save_path / "pixel_catboost_4x3.cbm"
MLP_PATH = file_paths.model_save_path / "best_pixel_mlp_4x3.pth"
OUT_DIR      = file_paths.experiment_results_path("4x3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Held-out sample IDs (same as used in the 3x3 and 3x4 robustness testing).
ISOLATED_SAMPLES: dict[str, list[str]] = {
    "brain": [
        "2022-02-16_T_HORAO-1-C_FR_15Z_3",
        "2022-03-16_T_HORAO-4-D_FR_1_2",
    ],
    "afmmm": [
        "AFMMM_sample_he9_Data",
        "AFMMM_sample_bg11_Data",
        "AFMMM_sample_bg5_Data",
    ],
    "cervix": [
        "Sample25_550_Data",
        "Sample2_550_Data",
        "Sample23_550_Data",
        "Sample19_550_Data",
    ],
}

SIMULATED_FILES = [
    "mm_225um_raw.npy",
    "mm_250um_raw.npy",
    "mm_300um_raw.npy",
]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_4x3(arr_hw16: np.ndarray) -> np.ndarray:
    """Return the first 3 columns of each 4×4 Mueller matrix (4×3 block).

    Parameters
    ----------
    arr_hw16 : ndarray, shape (H, W, 16)
        Flattened 4×4 Mueller matrices (row-major).

    Returns
    -------
    ndarray, shape (H*W, 12)
    """
    H, W, _ = arr_hw16.shape
    M = arr_hw16.reshape(H, W, 4, 4)
    return M[:, :, :, :3].reshape(-1, 12)


class PixelMLP(nn.Module):
    def __init__(self, in_features: int = 12, hidden_sizes: tuple = (128, 64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(0.30)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_xgb() -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(str(XGB_PATH))
    return bst


def load_cat() -> CatBoostClassifier:
    cat = CatBoostClassifier()
    cat.load_model(str(CAT_PATH))
    return cat


def load_mlp() -> PixelMLP:
    torch.set_num_threads(1)  # prevent OpenMP deadlock in Jupyter on macOS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mlp = PixelMLP().to(device)
    mlp.load_state_dict(torch.load(str(MLP_PATH), map_location=device, weights_only=False))
    mlp.eval()
    return mlp


def predict_xgb(model: xgb.Booster, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probs = model.predict(xgb.DMatrix(X))
    return (probs > threshold).astype(int)


def predict_cat(model: CatBoostClassifier, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probs = model.predict_proba(X)[:, 1]
    return (probs > threshold).astype(int)


def predict_mlp(model: PixelMLP, X: np.ndarray, batch_size: int = 8192, threshold: float = 0.5) -> np.ndarray:
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy().ravel()
            preds.append(probs)
    return (np.concatenate(preds) > threshold).astype(int)


# ---------------------------------------------------------------------------
# Testing I – Isolated held-out samples
# ---------------------------------------------------------------------------

def run_isolated_testing(xgb_model: xgb.Booster) -> pd.DataFrame:
    """Predict on each held-out sample and return per-tissue summary metrics."""
    rows = []

    for tissue, sample_ids in ISOLATED_SAMPLES.items():
        interim_dir = INTERIM_BASE / tissue
        tissue_rows = []

        for sid in sample_ids:
            fpath = interim_dir / f"{sid}_combined.npy"
            if not fpath.exists():
                print(f"  [SKIP] not found: {fpath}")
                continue

            arr = np.load(fpath)            # (H, W, 17)
            X16 = arr[..., :16]             # (H, W, 16) – Mueller matrix
            y_true = arr[..., 16].ravel().astype(int)  # GT PR label

            X12 = extract_4x3(X16)          # (H*W, 12)
            y_pred = predict_xgb(xgb_model, X12)

            tissue_rows.append({
                "tissue":    tissue,
                "sample":    sid,
                "accuracy":  accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall":    recall_score(y_true, y_pred, zero_division=0),
                "f1":        f1_score(y_true, y_pred, zero_division=0),
            })
            print(f"  {sid}: acc={tissue_rows[-1]['accuracy']:.4f}  f1={tissue_rows[-1]['f1']:.4f}")

        rows.extend(tissue_rows)

    df_all = pd.DataFrame(rows)

    if df_all.empty:
        print("  [WARNING] No samples found — check that the external drive is mounted.")
        return df_all

    # Per-tissue averages (what goes in the paper table)
    df_avg = (
        df_all
        .groupby("tissue")[["accuracy", "precision", "recall", "f1"]]
        .mean()
        .reset_index()
    )

    df_all.to_csv(OUT_DIR / "isolated_sample_metrics_4x3.csv", index=False)
    df_avg.to_csv(OUT_DIR / "isolated_sample_avg_metrics_4x3.csv", index=False)
    print("\nIsolated sample averages:")
    print(df_avg.to_string(index=False))
    return df_avg


# ---------------------------------------------------------------------------
# Testing II – Simulated samples
# ---------------------------------------------------------------------------

def normalize_by_m11(arr_hw16: np.ndarray) -> np.ndarray:
    """Divide all 16 MM elements by M11 (element index 0)."""
    m11 = arr_hw16[..., 0:1]
    eps = np.where(m11 == 0, 1e-12, 0)
    return arr_hw16 / (m11 + eps)


def compute_gt_mask(arr_hw16: np.ndarray) -> np.ndarray:
    """CCP ground-truth PR test. Returns boolean mask (H*W,)."""
    H, W, _ = arr_hw16.shape
    M44 = arr_hw16.reshape(-1, 4, 4)
    return charpoly_vectorized(M44)


def run_simulated_testing(
    xgb_model: xgb.Booster,
    cat_model: CatBoostClassifier,
    mlp_model: PixelMLP,
) -> pd.DataFrame:
    """Run PR test on all simulated samples with all three 4×3 models."""
    candidates = [
        Path("/Users/chaechae/Desktop/EP_Code/MonteCarloProcessing/output_data"),
        Path("/Volumes/ep_ssd/database/partialPr/data/test/simulated"),
        file_paths.simulated_test_path,
    ]
    sim_dir = next((p for p in candidates if p.exists()), candidates[-1])
    rows = []

    for fname in SIMULATED_FILES:
        fpath = sim_dir / fname
        if not fpath.exists():
            print(f"  [SKIP] not found: {fpath}")
            continue

        thickness = fname.split("_")[1]
        arr      = np.load(fpath)            # (100, 100, 16)
        arr_norm = normalize_by_m11(arr)
        y_true   = compute_gt_mask(arr_norm).astype(int)
        X12      = extract_4x3(arr_norm)

        acc_xgb = accuracy_score(y_true, predict_xgb(xgb_model, X12))
        acc_cat = accuracy_score(y_true, predict_cat(cat_model, X12))
        acc_mlp = accuracy_score(y_true, predict_mlp(mlp_model, X12))

        rows.append({"sample": thickness, "CAT": acc_cat, "XGB": acc_xgb, "MLP": acc_mlp})
        print(f"  {fname}: XGB={acc_xgb:.4f}  CAT={acc_cat:.4f}  MLP={acc_mlp:.4f}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("  [WARNING] No simulated samples found — check that the path is accessible.")
        return df
    df.to_csv(OUT_DIR / "simulated_metrics_4x3.csv", index=False)
    print("\nSimulated sample results:")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading 4×3 models...")
    xgb_model = load_xgb()
    cat_model = load_cat()
    mlp_model = load_mlp()

    print("\n=== Testing I: Isolated Held-Out Samples ===")
    df_isolated = run_isolated_testing(xgb_model)

    print("\n=== Testing II: Simulated Samples ===")
    df_simulated = run_simulated_testing(xgb_model, cat_model, mlp_model)

    print(f"\nResults saved to: {OUT_DIR}")
