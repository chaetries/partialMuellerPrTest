"""Common data-processing helpers used by notebooks and scripts."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim

from src.utils.physical_realizability import build_eigen_matrix


def load_dataset(file_path: Path | str) -> np.ndarray:
    """Load a flattened Mueller CSV into an ``(N, 4, 4)`` array."""
    csv_path = Path(file_path)
    logging.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path, header=None)
    data_np = df.to_numpy().reshape((-1, 4, 4))
    logging.info("Loaded dataset shape: %s", data_np.shape)
    return data_np


def save_data(
    matrices: np.ndarray,
    indices: list[int] | np.ndarray,
    save_dir: Path | str,
    file_name_prefix: str,
) -> None:
    """Persist matrix rows and original indices as CSV files."""
    if len(matrices) == 0:
        logging.warning("No matrices to save for %s", file_name_prefix)
        return

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrices_df = pd.DataFrame(matrices.reshape(len(matrices), -1))
    indices_df = pd.DataFrame(indices, columns=["Original_Index"])

    matrices_file_path = output_dir / f"{file_name_prefix}_matrices.csv"
    indices_file_path = output_dir / f"{file_name_prefix}_indices.csv"

    matrices_df.to_csv(matrices_file_path, index=False, header=False)
    indices_df.to_csv(indices_file_path, index=False, header=False)

    logging.info("Saved matrices to %s", matrices_file_path)
    logging.info("Saved indices to %s", indices_file_path)


def classify_and_save_matrices(
    dataset: np.ndarray,
    check_function: Callable[[np.ndarray], bool],
    save_directory: Path | str,
    file_name_prefix: str,
):
    """Run PR classification and save realizable/non-realizable subsets."""
    start_time = time.time()
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    realizable_indices: list[int] = []
    non_realizable_indices: list[int] = []

    for index, matrix in enumerate(dataset):
        if check_function(matrix):
            realizable_indices.append(index)
        else:
            non_realizable_indices.append(index)

    realizable_matrices = dataset[realizable_indices]
    non_realizable_matrices = dataset[non_realizable_indices]

    print("Realizable matrices:", realizable_matrices.shape)
    print("Non-realizable matrices:", non_realizable_matrices.shape)

    realizable_df, realizable_indices_df = _save_data_and_indices(
        realizable_matrices,
        realizable_indices,
        "physically_realizable",
        save_dir,
        file_name_prefix,
    )

    _save_data_and_indices(
        non_realizable_matrices,
        non_realizable_indices,
        "not_physically_realizable",
        save_dir,
        file_name_prefix,
    )

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return realizable_df, realizable_indices_df


def _save_data_and_indices(
    matrices: np.ndarray,
    indices: list[int],
    suffix: str,
    save_dir: Path,
    prefix: str,
):
    """Internal helper for writing matrix subsets and index files."""
    if len(matrices) == 0:
        print(f"No {suffix} matrices to save.")
        return None, None

    df = pd.DataFrame(matrices.reshape(len(matrices), -1)).astype(np.float64)
    file_path = save_dir / f"{prefix}_{suffix}.csv"
    df.to_csv(file_path, index=False, header=False, float_format="%.10f")

    indices_df = pd.DataFrame(indices, columns=["Original_Index"]).astype(int)
    indices_file_path = save_dir / f"{prefix}_{suffix}_indices.csv"
    indices_df.to_csv(indices_file_path, index=False, header=False)

    print(f"Saved {suffix} matrices to {file_path}")
    print(f"Saved {suffix} indices to {indices_file_path}")

    return df, indices_df


def save_coherency_matrices(coherency_matrices_np: np.ndarray, file_path: Path | str) -> None:
    """Save coherency matrices in flattened CSV format."""
    flattened_matrices = coherency_matrices_np.reshape(coherency_matrices_np.shape[0], -1)
    pd.DataFrame(flattened_matrices).to_csv(file_path, index=False)


def repopulate_pixels(
    realizable_df: pd.DataFrame,
    realizable_indices_df: pd.DataFrame,
    num_rows: int,
    num_cols: int,
    file_prefix: str,
    file_directory: Path | str,
    num_components: int = 16,
) -> pd.DataFrame:
    """Rebuild a dense pixel matrix from sparse realizable rows and indices."""
    populated_dataset = np.zeros((num_rows * num_cols, num_components), dtype=np.float64)
    realizable_data = realizable_df.to_numpy(dtype=np.float64)
    realizable_indices = realizable_indices_df.to_numpy().flatten().astype(int)

    if realizable_indices.size == 0:
        raise ValueError("No realizable indices were provided.")

    max_index = int(realizable_indices.max())
    if max_index >= num_rows * num_cols:
        raise ValueError("Indices exceed the number of pixels in the output dataset")

    populated_dataset[realizable_indices] = realizable_data

    populated_df = pd.DataFrame(populated_dataset)
    output_dir = Path(file_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_prefix}_populated.csv"
    populated_df.to_csv(output_path, index=False)

    return populated_df


def evaluate_model_performance(
    dataset_name: str,
    model,
    X,
    y_actual,
    tolerance: float = 0.01,
    height: int = 100,
    width: int = 100,
    batch_size: int = 4096,
):
    """Evaluate model predictions and return per-output metrics.

    ``tolerance``, ``height`` and ``width`` are kept for backward compatibility
    with existing notebook signatures.
    """
    del tolerance, height, width

    try:
        y_pred = model.predict(X, batch_size=batch_size)
    except TypeError:
        y_pred = model.predict(X)

    if y_pred.shape != y_actual.shape:
        raise ValueError("The shape of predicted values does not match the shape of actual values.")

    mse = mean_squared_error(y_actual, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y_actual, y_pred, multioutput="raw_values")
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred, multioutput="raw_values")

    if np.any(r2 < 0):
        print("Warning: Negative R2 detected; the model may be performing worse than the mean.")

    results_df = pd.DataFrame({"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"\n{dataset_name} Performance Metrics:")
    print(results_df)
    return y_pred, results_df


def calculate_ssim_index(image_a: np.ndarray, image_b: np.ndarray, height: int, width: int) -> float:
    """Compute SSIM between two flattened components."""
    image_a = image_a.reshape((height, width))
    image_b = image_b.reshape((height, width))
    data_range = max(image_a.max() - image_a.min(), image_b.max() - image_b.min())
    if data_range == 0:
        data_range = 1.0
    score, _ = ssim(image_a, image_b, data_range=data_range, full=True)
    print(f"SSIM: {score}")
    return float(score)


def compute_ssim_scores(
    pred_populated: pd.DataFrame,
    original_populated: pd.DataFrame,
    height: int,
    width: int,
) -> pd.DataFrame:
    """Compute SSIM for the last 4 components of prediction vs reference."""
    pred_last = pred_populated.iloc[:, -4:].values
    orig_last = original_populated.iloc[:, -4:].values
    ssim_scores = [
        calculate_ssim_index(orig_last[:, i], pred_last[:, i], height, width)
        for i in range(orig_last.shape[1])
    ]
    return pd.DataFrame(ssim_scores, columns=["SSIM"])


def process_and_save_coherency_matrices(
    combined_np: np.ndarray,
    interim_path: Path | str,
    prefix: str,
) -> np.ndarray:
    """Convert Mueller matrices to coherency matrices and save as CSV."""
    coherency_matrices_np = np.array([build_eigen_matrix(matrix) for matrix in combined_np])
    output_dir = Path(interim_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_coherency_matrices.csv"
    save_coherency_matrices(coherency_matrices_np, output_path)
    print(f"Saved coherency matrices to {output_path}")
    return coherency_matrices_np
