# utils/utils.py

import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import sys
import matplotlib.colors as colors

# Project-specific imports
root_dir = os.path.abspath("../")
sys.path.append(root_dir)

from src.utils.pr_test import build_eigen_matrix

########################
# Function: load_dataset
# Loads Mueller matrices from a CSV file into a NumPy array.
########################

def load_dataset(file_path):
    """
    Loads a dataset from a CSV file and reshapes it into Mueller matrices.

    Parameters:
    - file_path (Path): Path to the CSV file.

    Returns:
    - np.ndarray: Numpy array of shape (-1, 4, 4) containing Mueller matrices.
    """
    logging.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path, header=None)
    data_np = df.to_numpy().reshape((-1, 4, 4))
    logging.info(f"Loaded dataset shape: {data_np.shape}")
    return data_np

########################
# Function: save_data
# Saves matrices and their indices as CSV files.
########################
def save_data(matrices, indices, save_dir, file_name_prefix):
    """
    Saves matrices and their indices to CSV files.

    Parameters:
    - matrices (np.ndarray): Numpy array of matrices.
    - indices (List[int]): List of original indices.
    - save_dir (Path): Directory to save the files.
    - file_name_prefix (str): Prefix for the file names.
    """
    if len(matrices) == 0:
        logging.warning(f"No matrices to save for {file_name_prefix}")
        return

    matrices_df = pd.DataFrame(matrices.reshape(len(matrices), -1))
    indices_df = pd.DataFrame(indices, columns=['Original_Index'])

    matrices_file_path = save_dir / f"{file_name_prefix}_matrices.csv"
    indices_file_path = save_dir / f"{file_name_prefix}_indices.csv"

    matrices_df.to_csv(matrices_file_path, index=False, header=False)
    indices_df.to_csv(indices_file_path, index=False, header=False)

    logging.info(f"Saved matrices to {matrices_file_path}")
    logging.info(f"Saved indices to {indices_file_path}")


########################
# Function: classify_and_save_matrices
# Classifies and saves realizable and non-realizable matrices.
########################

def classify_and_save_matrices(dataset, check_function, save_directory, file_name_prefix):
    """
    Classifies matrices based on the physical realizability test and saves the results.

    Parameters:
    - dataset (np.ndarray): Numpy array of shape (-1, 4, 4) containing Mueller matrices.
    - check_function (callable): Function to perform the physical realizability test.
    - save_directory (Path): Directory to save the results.
    - file_name_prefix (str): Prefix for the saved file names.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: DataFrames of realizable matrices and their indices.
    """

    start_time = time.time()

    def save_data_and_indices(matrices, indices, suffix, save_dir, prefix):
        if len(matrices) == 0:
            print(f"No {suffix} matrices to save.")
            return None, None

        # Convert matrices to float64 to ensure precision
        df = pd.DataFrame(matrices.reshape(len(matrices), -1)).astype(np.float64)
        file_path = save_dir / f'{prefix}_{suffix}.csv'

        # Save matrices with high precision (e.g., 10 decimal places)
        df.to_csv(file_path, index=False, header=False, float_format='%.10f')

        # Convert indices to integers
        indices_df = pd.DataFrame(indices, columns=['Original_Index']).astype(int)
        indices_file_path = save_dir / f'{prefix}_{suffix}_indices.csv'

        # Save indices without decimal places
        indices_df.to_csv(indices_file_path, index=False, header=False)
        # Alternatively, specify float_format='%.0f' if needed:
        # indices_df.to_csv(indices_file_path, index=False, header=False, float_format='%.0f')

        print(f"Saved {suffix} matrices to {file_path}")
        print(f"Saved {suffix} indices to {indices_file_path}")

        return df, indices_df

    save_dir = Path(save_directory)

    # Classify matrices
    realizable_indices = [i for i, matrix in enumerate(dataset) if check_function(matrix)]
    non_realizable_indices = [i for i, matrix in enumerate(dataset) if not check_function(matrix)]

    realizable_matrices = dataset[realizable_indices]
    non_realizable_matrices = dataset[non_realizable_indices]

    print("Realizable matrices: ", realizable_matrices.shape)
    print("Non-realizable matrices: ", non_realizable_matrices.shape)

    # Save realizable matrices and indices
    realizable_df, realizable_indices_df = save_data_and_indices(
        realizable_matrices,
        realizable_indices,
        'physically_realizable',
        save_dir,
        file_name_prefix
    )

    # Save non-realizable matrices and indices
    non_realizable_df, non_realizable_indices_df = save_data_and_indices(
        non_realizable_matrices,
        non_realizable_indices,
        'not_physically_realizable',
        save_dir,
        file_name_prefix
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return realizable_df, realizable_indices_df

########################
# Function: save_coherency_matrices
# Saves coherency matrices to a CSV file.
########################

def save_coherency_matrices(coherency_matrices_np, file_path):
    flattened_matrices = coherency_matrices_np.reshape(coherency_matrices_np.shape[0], -1)
    pd.DataFrame(flattened_matrices).to_csv(file_path, index=False)


########################
# Function: repopulate_pixels
# Repopulates a dataset based on realizable indices and saves it.
########################
def repopulate_pixels(realizable_df, realizable_indices_df, num_rows, num_cols, file_prefix, file_directory,
                      num_components=16):
    """
    Takes in a DataFrame of realizable data and a DataFrame of indices, populates an array of
    specified size based on these indices, and saves the populated array as a CSV file with a prefix.

    Parameters:
    - realizable_df: DataFrame containing realizable data to be populated into the dataset.
    - realizable_indices_df: DataFrame containing the indices at which to populate the data.
    - num_rows: The number of rows in the output dataset.
    - num_cols: The number of columns in the output dataset.
    - file_prefix: Prefix for the filename under which the CSV will be saved.
    - file_directory: Directory where the CSV file will be saved.
    - num_components: Optional, the number of components each pixel has (default is 16).

    Returns:
    - DataFrame of the populated dataset. This DataFrame is also saved as a CSV at the specified file path with the given prefix.
    """

    # Initialize an empty numpy array filled with zeros
    # Assuming each pixel has 16 components (hence the last dimension is fixed as 16)
    populated_dataset = np.zeros((num_rows * num_cols, num_components))

    # Convert the realizable_df to a numpy array if it's not already
    realizable_data = realizable_df.to_numpy()

    # Ensure realizable_indices_df is a numpy array for iteration
    realizable_indices = realizable_indices_df.to_numpy().flatten().astype(int)

    # Debug prints
    print(f"Shape of realizable_data: {realizable_data.shape}")
    print(f"Shape of realizable_indices: {realizable_indices.shape}")
    print(f"Max index value: {realizable_indices.max()}")
    print(f"Min index value: {realizable_indices.min()}")
    print(f"Dataset size: {populated_dataset.shape[0]}")

    # Check if the indices are within the valid range
    if realizable_indices.max() >= num_rows * num_cols:
        raise ValueError("Indices exceed the number of pixels in the output dataset")

    # Populate the dataset
    for index, realizable_index in enumerate(realizable_indices):
        if 0 <= realizable_index < num_rows * num_cols:
            populated_dataset[realizable_index] = realizable_data[index]
        else:
            print(f"Index out of range: {realizable_index}")

    # Convert the repopulated dataset to a DataFrame for any further operations or saving
    populated_df = pd.DataFrame(populated_dataset)

    # Construct file path with prefix
    file_path = f"{file_directory}/{file_prefix}_populated.csv"

    # Save the DataFrame to a CSV file at the specified path
    populated_df.to_csv(file_path, index=False)

    return populated_df

########################
# Function: evaluate_model_performance
# Evaluates model predictions and outputs performance metrics.
########################
def evaluate_model_performance(dataset_name, model, X, y_actual, tolerance=0.01, height=100, width=100, batch_size=4096):
    """
    Evaluates model performance and returns predictions and metrics.
    """
    y_pred = model.predict(X, batch_size=batch_size)
    if y_pred.shape != y_actual.shape:
        raise ValueError("The shape of predicted values does not match the shape of actual values.")

    mse = mean_squared_error(y_actual, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_actual, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred, multioutput='raw_values')
    if np.any(r2 < 0):
        print("Warning: Negative R² detected; the model may be performing worse than the mean.")

    results_df = pd.DataFrame({'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    print(f"\n{dataset_name} Performance Metrics:")
    print(results_df)
    return y_pred, results_df


########################
# Function: calculate_ssim_index
# Computes the SSIM between two images.
########################
def calculate_ssim_index(imageA, imageB, height, width):
    """
    Calculates the SSIM between two images (reshaped to the given height and width).
    """
    imageA = imageA.reshape((height, width))
    imageB = imageB.reshape((height, width))
    data_range = max(imageA.max() - imageA.min(), imageB.max() - imageB.min())
    score, _ = ssim(imageA, imageB, data_range=data_range, full=True)
    print(f"SSIM: {score}")
    return score

########################
# Function: compute_ssim_scores
# Computes SSIM scores for the target components of two datasets.
########################
def compute_ssim_scores(pred_populated, original_populated, height, width):
    """
    Computes SSIM for the last 4 columns (assumed to be the target components) of two populated matrices.
    Returns a DataFrame of SSIM scores.
    """
    pred_last = pred_populated.iloc[:, -4:].values
    orig_last = original_populated.iloc[:, -4:].values
    ssim_scores = [calculate_ssim_index(orig_last[:, i], pred_last[:, i], height, width)
                   for i in range(orig_last.shape[1])]
    return pd.DataFrame(ssim_scores, columns=['SSIM'])


########################
# Function: process_and_save_coherency_matrices
# Processes Mueller matrices into coherency matrices and saves them.
########################
def process_and_save_coherency_matrices(combined_np, interim_path, prefix):
    coherency_matrices = [build_eigen_matrix(M) for M in combined_np]
    coherency_matrices_np = np.array(coherency_matrices)
    coherency_matrices_file_path = interim_path / f'{prefix}_coherency_matrices.csv'
    save_coherency_matrices(coherency_matrices_np, coherency_matrices_file_path)
    print(f"Saved coherency matrices to {coherency_matrices_file_path}")
    coherency_matrices_df = pd.read_csv(coherency_matrices_file_path)
    coherency_matrices_np = coherency_matrices_df.to_numpy().reshape(-1, 4, 4)
    return coherency_matrices_np




