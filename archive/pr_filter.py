"""
Simple PR Filter and Visualization Script
Uses the exact same logic as the notebook for physical realizability testing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.io import loadmat

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.physical_realizability import charpoly
from src.utils.visualization import visualize_save_mueller

#===============================================================================
# CONFIGURATION - EDIT THESE
#===============================================================================
# For .mat files:
INPUT_FILE = "/Volumes/ep_ssd/database/partialMueller/data/raw/cervix/Sample23/550_Mueller.mat"
TISSUE_TYPE = "cervix"  # Options: "cervix", "brain", "afmmm"

# Or for .npz files:
# INPUT_FILE = "/Volumes/ep_ssd/database/cervix_dataset/npz/sample1/550nm/MM.npz"
# TISSUE_TYPE = "cervix"

# Or for Brain samples:
# INPUT_FILE = "/path/to/brain/sample 1/MM.mat"
# TISSUE_TYPE = "brain"

# Or for AFMMM samples:

# INPUT_FILE = "/path/to/afmmm/sample_he1/FinalMM.mat"
# TISSUE_TYPE = "afmmm"

OUTPUT_DIR = "/Users/chaechae/Desktop/pr_publication"
SAMPLE_NAME = "sample23_550nm"

# Visualization options
USE_BLACK_BACKGROUND = True  # True = black for filtered pixels, False = white

# Dimensions are set automatically based on tissue type
TISSUE_DIMENSIONS = {
    'cervix': {'num_rows': 600, 'num_cols': 800},
    'brain': {'num_rows': 388, 'num_cols': 516},
    'afmmm': {'num_rows': 500, 'num_cols': 500}
}
#===============================================================================


def load_mueller_matrix(file_path, tissue_type):
    """
    Load Mueller matrix from .mat or .npz file

    Args:
        file_path: Path to .mat or .npz file
        tissue_type: "cervix", "brain", or "afmmm"

    Returns:
        mueller_matrix: numpy array of shape (H, W, 16)
    """
    file_path = Path(file_path)

    print(f"Loading {tissue_type} data from {file_path.name}...")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check file type
    if file_path.suffix == '.npz':
        # Load from .npz file
        data = np.load(file_path)
        print(f"NPZ keys: {list(data.keys())}")

        # Try common keys
        for key in ['nM', 'MM', 'mueller_matrix', 'M']:
            if key in data:
                mueller_matrix = data[key]
                print(f"Using key '{key}'")
                break
        else:
            mueller_matrix = data[list(data.keys())[0]]
            print(f"Using first key: {list(data.keys())[0]}")

        print(f"Loaded shape from .npz: {mueller_matrix.shape}")

        # Ensure correct shape
        dims = TISSUE_DIMENSIONS[tissue_type]
        H, W = dims['num_rows'], dims['num_cols']

        if len(mueller_matrix.shape) == 2 and mueller_matrix.shape[1] == 16:
            # Reshape from (H*W, 16) to (H, W, 16)
            mueller_matrix = mueller_matrix.reshape(H, W, 16)
            print(f"Reshaped to: {mueller_matrix.shape}")

        return mueller_matrix

    elif file_path.suffix == '.mat':
        # Load from .mat file - EXACT notebook logic
        sample = loadmat(file_path)

        if tissue_type == "cervix":
            # Cervix: Load from 'MM' key
            data = sample.get('MM')
            if data is None:
                raise ValueError("'MM' key not found in cervix .mat file")
            mueller_matrix = data  # Shape: (600, 800, 16)
            print(f"Loaded cervix data shape: {mueller_matrix.shape}")

        elif tissue_type == "brain":
            # Brain: Load from 'MM' key
            data = sample.get('MM')
            if data is None:
                raise ValueError("'MM' key not found in brain .mat file")
            mueller_matrix = data  # Shape: (388, 516, 16)
            print(f"Loaded brain data shape: {mueller_matrix.shape}")

        elif tissue_type == "afmmm":
            # AFMMM: Stack individual components - EXACT notebook logic
            keys = ['FinalM11', 'FinalM12', 'FinalM13', 'FinalM14',
                    'FinalM21', 'FinalM22', 'FinalM23', 'FinalM24',
                    'FinalM31', 'FinalM32', 'FinalM33', 'FinalM34',
                    'FinalM41', 'FinalM42', 'FinalM43', 'FinalM44']

            matrices = []
            for key in keys:
                if key in sample:
                    matrices.append(sample[key][:500, :500])
                else:
                    raise ValueError(f"Key {key} not found in AFMMM .mat file")

            if len(matrices) == 16:
                mueller_matrix = np.stack(matrices, axis=-1)  # Shape: (500, 500, 16)
                print(f"Loaded AFMMM data shape: {mueller_matrix.shape}")
            else:
                raise ValueError("Not all Mueller matrix components found in AFMMM file")

        else:
            raise ValueError(f"Unknown tissue type: {tissue_type}")

        return mueller_matrix

    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}. Use .mat or .npz")


def classify_matrices_with_pr(mueller_matrix, tissue_type):
    """
    Apply PR test to Mueller matrix - EXACT notebook logic

    Args:
        mueller_matrix: array of shape (H, W, 16) or (H*W, 16)
        tissue_type: "cervix", "brain", or "afmmm"

    Returns:
        realizable_indices: list of indices that pass PR test
        non_realizable_indices: list of indices that fail PR test
        H, W: dimensions
    """
    # Get dimensions for this tissue type
    dims = TISSUE_DIMENSIONS[tissue_type]
    H, W = dims['num_rows'], dims['num_cols']

    # Reshape to (N, 16) if needed
    original_shape = mueller_matrix.shape
    if len(mueller_matrix.shape) == 3:
        mueller_flat = mueller_matrix.reshape(-1, 16)
    else:
        mueller_flat = mueller_matrix

    # Reshape to (N, 4, 4) for PR test
    mueller_matrices = mueller_flat.reshape(-1, 4, 4)
    print(f"Testing {len(mueller_matrices)} Mueller matrices...")

    # Apply charpoly to each matrix - EXACT notebook approach
    realizable_indices = []
    non_realizable_indices = []

    for i, M in enumerate(mueller_matrices):
        if charpoly(M):  # Uses the exact function from pr_test.py
            realizable_indices.append(i)
        else:
            non_realizable_indices.append(i)

        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1:,} / {len(mueller_matrices):,}")

    print(f"\nPR Test Results:")
    print(f"  Realizable: {len(realizable_indices):,} ({len(realizable_indices)/len(mueller_matrices)*100:.2f}%)")
    print(f"  Non-realizable: {len(non_realizable_indices):,} ({len(non_realizable_indices)/len(mueller_matrices)*100:.2f}%)")

    return realizable_indices, non_realizable_indices, H, W


def repopulate_pixels(mueller_matrix, realizable_indices, H, W):
    """
    Create populated dataset with only realizable pixels - EXACT notebook logic
    """
    # Flatten mueller matrix
    if len(mueller_matrix.shape) == 3:
        mueller_flat = mueller_matrix.reshape(-1, 16)
    else:
        mueller_flat = mueller_matrix

    # Create empty dataset
    populated_dataset = np.zeros((H * W, 16))

    # Fill in realizable pixels
    realizable_data = mueller_flat[realizable_indices]
    for idx, realizable_idx in enumerate(realizable_indices):
        populated_dataset[realizable_idx] = realizable_data[idx]

    return pd.DataFrame(populated_dataset)


def visualize_mueller_pdf(mueller_df, output_path, sample_name, num_rows, num_cols,
                          use_black_bg=True, label_suffix=""):
    """
    Create PDF visualization of Mueller matrix

    Args:
        mueller_df: DataFrame with Mueller matrix data (zeros for filtered pixels)
        output_path: Path to save PDF
        sample_name: Sample name for title
        num_rows, num_cols: Dimensions
        use_black_bg: If True, use black for filtered pixels; if False, use white
        label_suffix: Additional label for filename (e.g., "BEFORE", "AFTER")
    """
    # Reshape data
    data_array = mueller_df.values.reshape(num_rows, num_cols, 4, 4)

    # Calculate figsize
    aspect_ratio = float(num_cols) / float(num_rows)
    base_height = 20
    base_width = base_height * aspect_ratio
    figsize = (base_width, base_height)

    fig, axes = plt.subplots(4, 4, figsize=figsize)
    axes = axes.flatten()

    # Prepare colormap
    cmap = plt.cm.jet.copy()
    if use_black_bg:
        cmap.set_bad(color='black')  # Black for filtered/NaN pixels
    else:
        cmap.set_bad(color='white')  # White for filtered/NaN pixels

    # Convert zeros to NaN for visualization
    data_to_show = data_array.copy()
    data_to_show[data_to_show == 0] = np.nan

    diagonal_indices = [0, 5, 10, 15]

    # Determine global vmin and vmax
    global_vmin = np.nanmin(data_to_show)
    global_vmax = np.nanmax(data_to_show)
    if global_vmin == global_vmax:
        global_vmin -= 1e-5
        global_vmax += 1e-5

    # Plot each component
    for i in range(16):
        row, col = divmod(i, 4)
        component_data = data_to_show[:, :, row, col]

        # Determine color scale
        if i in diagonal_indices:
            vmin, vmax = 0, 1
        else:
            vmin, vmax = -0.1, 0.1

        # Protect against narrow range
        component_vmin = np.nanmin(component_data)
        component_vmax = np.nanmax(component_data)
        if component_vmin == component_vmax:
            component_vmin -= 1e-5
            component_vmax += 1e-5

        im = axes[i].imshow(
            component_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='upper',
            aspect='equal',
            interpolation='none'
        )
        axes[i].axis('off')

    # Adjust layout
    plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.85)

    # Create colorbar
    cbar_ax = fig.add_axes([0.858, 0.11, 0.032, 0.77])
    norm = plt.cm.colors.Normalize(vmin=global_vmin, vmax=global_vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('')
    cbar.ax.set_yticklabels([])
    cbar.ax.tick_params(axis='y', which='both', length=0)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1)

    # Save
    plt.savefig(output_path, bbox_inches='tight', transparent=False, format='pdf')
    plt.close(fig)
    print(f"Saved PDF: {output_path}")


def create_before_after_gif(mueller_df, output_path, sample_name, num_rows, num_cols, use_black_bg=True):
    """Create GIF showing before and after PR filtering"""
    from PIL import Image

    print(f"Creating before/after GIF...")

    # Reshape data
    data_array = mueller_df.values.reshape(num_rows, num_cols, 4, 4)

    # Create two frames
    images = []

    for frame_type in ['BEFORE', 'AFTER']:
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()

        # Prepare colormap
        cmap = plt.cm.jet.copy()
        if use_black_bg:
            cmap.set_bad(color='black')
        else:
            cmap.set_bad(color='white')

        # Convert zeros to NaN
        data_to_show = data_array.copy()
        data_to_show[data_to_show == 0] = np.nan

        if frame_type == 'BEFORE':
            # For BEFORE, only show zero-filtered
            zero_mask = np.isclose(data_array[:, :, 0, 0], 0, atol=1e-8)
            data_to_show[zero_mask] = np.nan
            title = f'{sample_name} - BEFORE PR Filtering'
        else:
            # For AFTER, all zeros are already NaN
            title = f'{sample_name} - AFTER PR Filtering'

        diagonal_indices = [0, 5, 10, 15]

        for i in range(16):
            row, col = divmod(i, 4)
            component_data = data_to_show[:, :, row, col]

            if i in diagonal_indices:
                vmin, vmax = 0, 1
            else:
                vmin, vmax = -0.1, 0.1

            axes[i].imshow(component_data, cmap=cmap, vmin=vmin, vmax=vmax,
                          origin='upper', aspect='equal', interpolation='none')
            axes[i].axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.85, top=0.95)
        fig.suptitle(title, fontsize=24, fontweight='bold')

        # Add colorbar
        cbar_ax = fig.add_axes([0.858, 0.11, 0.032, 0.77])
        global_vmin = np.nanmin(data_to_show)
        global_vmax = np.nanmax(data_to_show)
        norm = plt.cm.colors.Normalize(vmin=global_vmin, vmax=global_vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

        # Convert to image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_array = img_array[:, :, :3]
        images.append(Image.fromarray(img_array))
        plt.close(fig)

    # Save GIF
    images[0].save(output_path, save_all=True, append_images=images[1:],
                   duration=1500, loop=0)
    print(f"Saved GIF: {output_path}")


def main():
    # Get dimensions for this tissue type
    dims = TISSUE_DIMENSIONS[TISSUE_TYPE]
    H, W = dims['num_rows'], dims['num_cols']

    print("="*80)
    print("PR Filter and Visualization")
    print("="*80)
    print(f"Tissue Type: {TISSUE_TYPE}")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Dimensions: {H} x {W}")
    print("="*80)
    print()

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Mueller matrix from .mat file - EXACT notebook logic
    print("Step 1: Loading Mueller matrix from .mat file...")
    mueller_matrix = load_mueller_matrix(INPUT_FILE, TISSUE_TYPE)
    print()

    # 2. Apply PR test - EXACT notebook logic
    print("Step 2: Applying PR test (using charpoly from pr_test.py)...")
    realizable_indices, non_realizable_indices, H, W = classify_matrices_with_pr(mueller_matrix, TISSUE_TYPE)
    print()

    # 3. Create populated dataset - EXACT notebook logic
    print("Step 3: Creating populated dataset with PR-filtered pixels...")
    populated_df = repopulate_pixels(mueller_matrix, realizable_indices, H, W)
    print(f"Populated dataframe shape: {populated_df.shape}")
    print()

    # 4. Save filtered data as .npy file
    print("Step 4: Saving filtered Mueller matrix as .npy file...")
    filtered_npy_path = output_dir / f'{SAMPLE_NAME}_filtered.npy'
    filtered_array = populated_df.values.reshape(H, W, 16)

    # ------------------------------------------------------------------
    # FORCE M11 = 1 FOR ALL NON-ZERO (PR-PASSING) PIXELS — NPZ ONLY
    # ------------------------------------------------------------------
    flat = filtered_array.reshape(-1, 16)

    pr_mask = ~np.isclose(flat[:, 0], 0, atol=1e-8)  # PR pixels only
    flat[pr_mask, 0] = 1.0  # FORCE M11 = 1

    filtered_array = flat.reshape(H, W, 16)

    print(
        f"M11 forced to 1 for {pr_mask.sum():,} PR pixels "
        f"({pr_mask.sum() / (H * W) * 100:.2f}%)"
    )

    np.save(filtered_npy_path, filtered_array)

    np.save(filtered_npy_path, filtered_array)
    print(f"Saved filtered data: {filtered_npy_path}")
    print()

    # 5. Save visualizations
    print("Step 5: Creating visualizations...")

    # Create BEFORE PDF (original data with zeros removed)
    print("  Creating BEFORE PDF (original data)...")
    mueller_flat_original = mueller_matrix.reshape(-1, 16)
    original_df = pd.DataFrame(mueller_flat_original)
    # Filter only zeros from original
    for i in range(len(original_df)):
        if np.isclose(original_df.iloc[i, 0], 0, atol=1e-8):  # M11 near zero
            original_df.iloc[i] = 0

    before_pdf_path = output_dir / f'{SAMPLE_NAME}_BEFORE.pdf'
    visualize_mueller_pdf(original_df, before_pdf_path, SAMPLE_NAME, H, W,
                         use_black_bg=USE_BLACK_BACKGROUND, label_suffix="BEFORE")

    # Create AFTER PDF (PR filtered)
    print("  Creating AFTER PDF (PR filtered)...")
    after_pdf_path = output_dir / f'{SAMPLE_NAME}_AFTER.pdf'
    visualize_mueller_pdf(populated_df, after_pdf_path, SAMPLE_NAME, H, W,
                         use_black_bg=USE_BLACK_BACKGROUND, label_suffix="AFTER")

    # Before/After GIF
    print("  Creating GIF...")
    gif_path = output_dir / f'{SAMPLE_NAME}_before_after.gif'
    create_before_after_gif(populated_df, gif_path, SAMPLE_NAME, H, W,
                           use_black_bg=USE_BLACK_BACKGROUND)

    # Save PR mask
    print("  Creating PR mask...")
    pr_mask = np.zeros(H * W, dtype=bool)
    pr_mask[realizable_indices] = True
    pr_mask_2d = pr_mask.reshape(H, W)

    plt.figure(figsize=(12, 8))
    plt.imshow(pr_mask_2d, cmap='RdYlGn', origin='upper')
    plt.colorbar(label='Physically Realizable')
    plt.title(f'{SAMPLE_NAME} - PR Mask\nPR Coverage: {len(realizable_indices)/(H*W)*100:.2f}%',
              fontsize=14, fontweight='bold')
    plt.axis('off')
    mask_path = output_dir / f'{SAMPLE_NAME}_pr_mask.png'
    plt.savefig(mask_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved PR mask: {mask_path}")

    print()
    print("="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Output files in: {output_dir}")
    print(f"  1. {SAMPLE_NAME}_filtered.npy - Filtered Mueller matrix (non-PR pixels as zeros)")
    print(f"  2. {SAMPLE_NAME}_BEFORE.pdf - Original data (zeros filtered)")
    print(f"  3. {SAMPLE_NAME}_AFTER.pdf - PR filtered data")
    print(f"  4. {SAMPLE_NAME}_before_after.gif - Animated before/after")
    print(f"  5. {SAMPLE_NAME}_pr_mask.png - PR mask visualization")
    bg_color = "BLACK" if USE_BLACK_BACKGROUND else "WHITE"
    print(f"\nFiltered pixels shown as: {bg_color}")
    print("="*80)


if __name__ == '__main__':
    main()
