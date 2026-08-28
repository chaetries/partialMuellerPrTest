"""Verify ML PR-classifier accuracy on the simulated phantom DETECTOR pixels.

Corrected normalization (Option B): the simulated .npy files store the 15 non-M11
Mueller elements already normalized by M11, with the [0,0] slot holding the raw
backscattered reflectance. The correct preparation is therefore to SET M11 = 1
(not divide every element by the tiny stored M11, which double-normalizes).

The background corners (M11_raw == 0, no photons collected) are excluded; metrics
are reported over the circular detector (signal) pixels only, which is the
physically meaningful region.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
        python notebooks/testing/verify_simulated_detector_accuracy.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.file_paths import file_paths
from src.utils.physical_realizability import charpoly_vectorized


class PixelMLP(nn.Module):
    """MLP matching the saved checkpoints (Linear-BatchNorm-ReLU-Dropout blocks)."""

    def __init__(self, input_dim, hidden=(128, 64, 32), dropout=0.30):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# scenario -> (xgb file, catboost file, mlp file, n_features, element selector from (...,4,4))
CFG = {
    "3x3": ("pixel_xgb_3x3.json", "pixel_catboost_3x3.cbm", "best_pixel_mlp_3x3.pth", 9,
            lambda M: M[:, :3, :3]),
    "3x4": ("pixel_xgb.json", "pixel_catboost.cbm", "best_pixel_mlp.pth", 12,
            lambda M: M[:, :3, :]),
    "4x3": ("pixel_xgb_4x3.json", "pixel_catboost_4x3.cbm", "best_pixel_mlp_4x3.pth", 12,
            lambda M: M[:, :, :3]),
}
SAMPLES = ["mm_225um_raw.npy", "mm_250um_raw.npy", "mm_300um_raw.npy"]


def option_b(raw):
    """Set M11 = 1, keep the already-normalized elements (corrected normalization)."""
    f = raw.copy().astype(float)
    f[:, :, 0] = 1.0
    return f


def main():
    mp = file_paths.model_save_path
    print("DETECTOR-ONLY accuracy (background excluded). GT over detector = 100% PR.")
    for scenario, (xf, cf, mf, n_feat, selector) in CFG.items():
        bst = xgb.Booster(); bst.load_model(str(mp / xf))
        cat = CatBoostClassifier(); cat.load_model(str(mp / cf))
        mlp = PixelMLP(n_feat)
        mlp.load_state_dict(torch.load(mp / mf, map_location="cpu", weights_only=False))
        mlp.eval()

        print(f"--- {scenario} input ---")
        for fn in SAMPLES:
            raw = np.load(file_paths.simulated_test_path / fn).astype(float)
            H, W, _ = raw.shape
            detector = (raw[:, :, 0] > 0).reshape(-1)        # circular FOV; corners are background

            mm = option_b(raw)
            M44 = mm.reshape(-1, 4, 4)
            gt = charpoly_vectorized(M44).astype(int)         # CCP ground truth
            X = selector(M44).reshape(len(gt), -1).astype("float32")

            xa = (bst.predict(xgb.DMatrix(X, nthread=1)) > 0.5).astype(int)
            ca = (cat.predict_proba(X)[:, 1] > 0.5).astype(int)
            with torch.no_grad():
                ma = (torch.sigmoid(mlp(torch.tensor(X))).numpy().ravel() > 0.5).astype(int)

            g = gt[detector]
            print(f"  {fn[3:8]}: det={int(detector.sum())}  GT_PR={int(g.sum())}/{int(detector.sum())}  "
                  f"XGB={(xa[detector] == g).mean():.4f}  "
                  f"Cat={(ca[detector] == g).mean():.4f}  "
                  f"MLP={(ma[detector] == g).mean():.4f}")


if __name__ == "__main__":
    main()
