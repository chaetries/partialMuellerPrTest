import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from fractions import Fraction
import seaborn as sns


def visualize_save_mueller(
    data,
    visualisation_path,
    sample_number,
    wavelength,
    label="Sample",
    figsize=None,          
    num_rows=500,
    num_cols=500,
    filter_zeros=True,
    show_fig=True,
    crop_square=False       # <-- Option to decide whether you want to crop
):
    """
    Visualize and save a Mueller matrix as a 4x4 grid of subplots using imshow.

    This function takes a Mueller matrix dataset, visualizes it as a 4x4 grid of subplots,
    and saves the resulting image. It handles data normalization, zero filtering, and
    creates a single colorbar for the entire figure.

    Parameters:
    - data (pandas.DataFrame): The Mueller matrix data.
    - visualisation_path (str or Path): Path to save the visualization.
    - sample_number (str): Identifier for the sample.
    - wavelength (str): Wavelength of the measurement.
    - label (str, optional): Additional label for the filename. Defaults to "Sample".
    - figsize (tuple, optional): Figure size. Defaults to None (auto-compute).
    - num_rows (int, optional): Number of rows in the data. Defaults to 500.
    - num_cols (int, optional): Number of columns in the data. Defaults to 500.
    - filter_zeros (bool, optional): Whether to filter out zero values. Defaults to True.
    - show_fig (bool, optional): Whether to display the figure. Defaults to True.
    - crop_square (bool, optional): Whether to crop the data to a square. Defaults to True.

    Returns:
    None. The function saves the visualization as a PDF file.
    """

    # Ensure the directory exists
    visualisation_path = Path(visualisation_path)
    visualisation_path.mkdir(parents=True, exist_ok=True)

    # 1. Reshape the data from a flat array of shape (num_rows * num_cols, 16)
    reshaped_data = data.values.reshape(num_rows, num_cols, 4, 4)

    # 2. Optionally crop to the largest possible square
    if crop_square:
        crop_size = min(num_rows, num_cols)
        row_start = (num_rows - crop_size) // 2
        row_end   = row_start + crop_size
        col_start = (num_cols - crop_size) // 2
        col_end   = col_start + crop_size

        reshaped_data = reshaped_data[row_start:row_end, col_start:col_end, :, :]

        # Update num_rows and num_cols to reflect the cropped size
        num_rows, num_cols = crop_size, crop_size

    # 3. Filter out zero values (M11 near zero)
    if filter_zeros:
        zero_mask = np.isclose(reshaped_data[:, :, 0, 0], 0, atol=1e-8)
        reshaped_data[zero_mask] = np.nan

    # 4. Automatically compute figsize if not provided
    if figsize is None:
        base_height = 20  # or any other base size you prefer
        # If it's now cropped to a square, num_cols == num_rows, so aspect_ratio=1.0
        aspect_ratio = float(num_cols) / float(num_rows)
        base_width = base_height * aspect_ratio
        figsize = (base_width, base_height)

    # Prepare the plot
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    axes = axes.flatten()

    # Define diagonal indices for color scaling
    diagonal_indices = [0, 5, 10, 15]

    # Define a colormap and set NaN values to appear as white
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color='white')  # NaNs will be white

    # Determine global vmin and vmax for consistent color scaling
    global_vmin = np.nanmin(reshaped_data)
    global_vmax = np.nanmax(reshaped_data)
    if global_vmin == global_vmax:
        global_vmin -= 1e-5
        global_vmax += 1e-5

    # 5. Plot each of the 16 Mueller components
    for i in range(16):
        row, col = divmod(i, 4)
        component_data = reshaped_data[:, :, row, col]

        # Decide color scale based on diagonal or off-diagonal
        if i in diagonal_indices:
            vmin, vmax = 0, 1
        else:
            vmin, vmax = -0.1, 0.1

        # Protect against narrow data range in this component
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
            aspect='equal',   # preserves the data's aspect in each subplot
            interpolation='none'
        )
        axes[i].axis('off')  # Hide axes

    # Adjust layout for colorbar
    plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.85)

    # Create a single colorbar for the entire figure
    norm = colors.Normalize(vmin=global_vmin, vmax=global_vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.858, 0.11, 0.032, 0.77])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('')
    cbar.ax.set_yticklabels([])
    cbar.ax.tick_params(axis='y', which='both', length=0)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1)

    # Save the figure
    filename = f"{label}_Mueller_{sample_number}_{wavelength}.pdf"
    save_path = visualisation_path / filename
    try:
        plt.savefig(save_path, bbox_inches='tight', transparent=False)
        print(f"Saved PDF successfully to {save_path}")
        if show_fig:
            plt.show()
    except Exception as e:
        print(f"Failed to save PDF: {e}")
    finally:
        plt.close(fig)



def visualize_and_save_lu_chipman(
    data,
    file_path,
    file_name: str = 'Visualization.png',
    figsize=(15, 10),
    cmap='jet',
    title='',
    xlabel='',
    ylabel='',
    vmin=None,
    vmax=None,
    step=None,
    use_pi_notation=False,
    show_legend=True,
    zero_color=None      # ← color to use for exact-0 **and** exact-1 pixels
):
    """
    Plots and saves a 2D matrix as an image, with an optional forced color
    for the two “edge‐case” values 0 and 1.

    Parameters
    ----------
    data : np.ndarray
        The 2D matrix to plot.
    file_path : str or Path
        Directory where the plot should be saved.
    file_name : str
        Name of file to save.
    figsize : tuple
    cmap : str
    title : str
    xlabel, ylabel : str
    vmin, vmax : float
    step : float
    use_pi_notation : bool
    show_legend : bool
    zero_color : str or None
        If not None, color all exact zeros **and** all exact ones
        in this color (e.g. 'black' or 'white'); all other values
        are shown in `cmap`.
    """
    figure_path = Path(file_path)
    figure_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=figsize)

    if zero_color is not None:
        # mask both 0 and 1 so they become “bad” in the colormap
        mask = (data == 0) | (data == 1)
        plot_data = np.ma.masked_array(data, mask=mask)

        im = plt.imshow(plot_data, aspect='auto',
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation='nearest')
        # force the masked (0 & 1) pixels into your chosen color
        im.cmap.set_bad(zero_color)
    else:
        im = plt.imshow(data, aspect='auto',
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation='nearest')

    if show_legend:
        cbar = plt.colorbar(im)
        if step is not None and (vmin is not None and vmax is not None):
            ticks = np.arange(vmax, vmin - step/2, -step)
            cbar.set_ticks(ticks)
            if use_pi_notation:
                def format_pi(x):
                    if abs(x) < 1e-10:
                        return "0"
                    frac = Fraction(x/np.pi).limit_denominator(20)
                    if frac == Fraction(0,1):
                        return "0"
                    if frac.numerator == 1 and frac.denominator == 1:
                        return "π"
                    if frac.numerator == -1 and frac.denominator == 1:
                        return "-π"
                    if frac.denominator == 1:
                        return f"{frac.numerator}π"
                    return f"{frac.numerator}π/{frac.denominator}"
                cbar.set_ticklabels([format_pi(t) for t in ticks])
            else:
                cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
        cbar.ax.tick_params(labelsize=30)

    plt.title(title, fontsize=30)
    plt.axis('off')
    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)

    out_file = figure_path / file_name
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

    print(f"Plot saved successfully at: {out_file}")
    
def plot_differential_histogram(diff_data, file_path, file_name='Histogram.png', bins=50,
                                title='Histogram of Differences', xlabel='Difference', ylabel='Frequency'):
    figure_path = Path(file_path)
    if not figure_path.exists():
        figure_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.hist(diff_data.ravel(), bins=bins, color='blue', edgecolor='black')
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.savefig(figure_path / file_name, bbox_inches='tight', pad_inches=0)
    plt.show()
    print(f"Histogram saved successfully at: {figure_path / file_name}")

# Function to plot differential images with enhanced color scale
def visualize_differential(diff_data, file_path, file_name='Differential.png', figsize=(15, 10), cmap='jet', title='Differential Visualization', xlabel='', ylabel='', vmin=None, vmax=None):
    figure_path = Path(file_path)
    if not figure_path.exists():
        figure_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=figsize)
    plt.imshow(diff_data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.savefig(figure_path / file_name, bbox_inches='tight', pad_inches=0)
    print(f"Plot saved successfully at: {figure_path / file_name}")
    plt.show()


def plot_training_predictions(
    y_true,
    y_pred,
    target_labels=None,
    scaler=None,
    save_format='png',
    figure_path='.',  # Default to current directory
):
    """
    Plots actual vs. predicted values for all target variables in a 2x2 grid
    and saves the plot in the specified directory. By default, the plot is
    saved as PNG, but it can be optionally saved as PDF by setting
    save_format='pdf'.

    Parameters:
    - y_true: array-like, actual values.
    - y_pred: array-like, predicted values.
    - target_labels: list of strings, optional labels for target variables.
    - scaler: scaler object with an inverse_transform method, optional.
    - save_format: string, 'png' (default) or 'pdf'.
    - figure_path: path-like, directory to save the figure (default: current dir).
    """

    # Convert figure_path to a Path object if it's not already
    figure_path = Path(figure_path)
    # Ensure the directory exists (creates it if it doesn't)
    figure_path.mkdir(parents=True, exist_ok=True)

    # Optionally inverse-transform the data if a scaler is provided
    if scaler:
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)

    # Create the figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('', fontsize=35, fontname='Times New Roman', fontstyle='italic', y=1.02)

    # Loop over subplots and plot
    for i, ax in enumerate(axs.flat):
        # Only plot if the number of target columns is at least i+1
        if i < y_true.shape[1]:
            ax.scatter(
                y_true[:, i],
                y_pred[:, i],
                label='Predictions',
                color='blue',
                alpha=0.5
            )
            ax.plot(
                [y_true[:, i].min(), y_true[:, i].max()],
                [y_true[:, i].min(), y_true[:, i].max()],
                'k--',
                lw=2,
                label='Ideal'
            )

            # Label each subplot
            label = f'$M_{{4,{i + 1}}}$' if target_labels is None else target_labels[i]
            ax.set_title(label, fontsize=28, fontname='Times New Roman', fontstyle='italic')
            ax.set_xlabel('Actual Values', fontsize=22, labelpad=15)
            ax.set_ylabel('Predicted Values', fontsize=22, labelpad=15)

            # Format tick labels
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # Adjust tick parameters
            ax.tick_params(
                axis='both', which='major', labelsize=18, length=10, width=2,
                direction='in', top=True, right=True
            )
            ax.tick_params(
                axis='both', which='minor', labelsize=18, length=5, width=1,
                direction='in', top=True, right=True
            )

            # Adjust offset text size
            ax.xaxis.get_offset_text().set_fontsize(16)
            ax.yaxis.get_offset_text().set_fontsize(16)

            # Make axes spines thicker
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)

            # Optionally turn off the background grid
            ax.grid(False)

    plt.tight_layout()

    # Construct the final file path
    file_name = f'predictions_plot.{save_format.lower()}'
    final_path = figure_path / file_name

    # Save the plot in the desired format
    if save_format.lower() == 'pdf':
        with PdfPages(final_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
    else:
        plt.savefig(final_path, bbox_inches='tight', pad_inches=0)

    print(f"Plot saved successfully at: {final_path}")

    # Show the plot and close the figure to free up resources
    plt.show()
    plt.close(fig)



def plot_pred_histograms(
    y_true,
    y_pred,
    figure_path='.',
    file_name='pred_histograms.png'
):
    """
    Plots histograms of the prediction differences for each target variable
    and saves the figure to the specified path.

    Parameters:
    - y_true: array-like, true values (shape: [n_samples, n_targets]).
    - y_pred: array-like, predicted values (same shape as y_true).
    - figure_path: path-like, directory to save the figure (default: current dir).
    - file_name: string, the filename to use when saving the figure (default: 'pred_histograms.png').
    """
    # Convert figure_path to a Path object if it's not already
    figure_path = Path(figure_path)
    # Ensure the directory exists (creates it if it doesn't)
    figure_path.mkdir(parents=True, exist_ok=True)

    # Calculate differences
    differences = y_pred - y_true
    num_variables = y_true.shape[1]

    # Create the figure
    fig, axs = plt.subplots(1, num_variables, figsize=(5 * num_variables, 4), squeeze=False)

    for i in range(num_variables):
        axs[0, i].hist(differences[:, i], bins=30, color='gray', alpha=0.75)
        axs[0, i].set_title(f'M(4, {i+1})')
        axs[0, i].set_xlabel('Prediction Difference')
        axs[0, i].set_ylabel('Frequency')

        # Add a vertical line at x=0
        axs[0, i].axvline(x=0, color='red', linestyle='--', linewidth=1.5)

    plt.tight_layout()

    # Construct the full file path
    save_path = figure_path / file_name

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"Plot saved successfully at: {save_path}")

    # Show and close
    plt.show()
    plt.close(fig)


def plot_pred_distributions(
        y_true,
        y_pred,
        labels=None,
        alpha=0.5,
        figure_path='.',
        file_name='pred_distributions.png'
):
    """
    Plots overlaid density curves for actual and predicted values of multiple targets
    to compare their distributions, and saves the resulting plot.

    Parameters:
    - y_true (np.array): Actual values (expected to have multiple columns, one for each target).
    - y_pred (np.array): Predicted values (shape should match y_true).
    - labels (list of str): Labels for each target variable.
    - alpha (float): Transparency for the density plots.
    - figure_path (str or Path): Directory to save the figure (default: current directory).
    - file_name (str): Filename for the saved figure (default: 'pred_distributions.png').
    """
    # Convert figure_path to a Path object if not already one
    figure_path = Path(figure_path)
    # Ensure the directory exists
    figure_path.mkdir(parents=True, exist_ok=True)

    num_targets = y_true.shape[1]
    plt.figure(figsize=(15, num_targets * 3))

    for i in range(num_targets):
        plt.subplot(num_targets, 1, i + 1)
        sns.kdeplot(y_true[:, i], color='blue', label='Actual', alpha=alpha)
        sns.kdeplot(y_pred[:, i], color='red', label='Predicted', alpha=alpha)
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)  # Add vertical line at x=0

        plt.legend()
        title_text = f'Distribution of Actual vs. Predicted Values for {labels[i]}' if labels else f'Target {i + 1}'
        plt.title(title_text)
        plt.xlabel('Values')
        plt.ylabel('Density')

    plt.tight_layout()

    # Construct the full file path
    save_path = figure_path / file_name
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"Plot saved successfully at: {save_path}")

    plt.show()
    plt.close()


def visualize_save_mueller_with_mask(data, raw_path, visualisation_path, sample_number, wavelength, label="Sample", figsize=(15, 10), num_rows=600, num_cols=800):
    """
    Visualizes the components of a Mueller matrix from a CSV file in a 4x4 grid, overlays masks, and saves the grid.
    Normalizes the matrix by M(1,1) if M(1,1) is greater than 1.
    Uses the rainbow colormap for visualization.

    Parameters:
    - data: DataFrame containing the Mueller matrix data.
    - raw_path: Path object to the directory containing the mask data.
    - visualisation_path: Path object indicating where to save the figure.
    - sample_number: Sample number to be included in the saved filename.
    - wavelength: Wavelength to be included in the saved filename.
    - label: A custom label to be included in the saved filename.
    - figsize: Optional. A tuple representing the figure size for the grid.
    - num_rows: The number of rows for reshaping each Mueller matrix component. Default is 600.
    - num_cols: The number of columns for reshaping each Mueller matrix component. Default is 800.
    """

    # Ensure the visualization path exists
    visualisation_path.mkdir(parents=True, exist_ok=True)

    # Normalize each matrix by M(1,1) if M(1,1) is greater than 1
    m11_index = 0  # Assuming M(1,1) is the first element in the flattened matrix
    for index, row in data.iterrows():
        if row[m11_index] > 1:
            data.iloc[index] = row / row[m11_index]

    # Load mask data
    def load_mask(path):
        with open(path, 'rb') as mask_file:
            mask_data = np.fromfile(mask_file, dtype=np.uint32, count=num_rows * num_cols)
        mask_data = np.reshape(mask_data, [num_cols, num_rows])
        mask_data = np.flip(mask_data, axis=1)
        mask_data = np.rot90(mask_data)
        return np.where(mask_data >= 1, 1, 0)

    diseased_mask_data = load_mask(raw_path / f'Meredith/Sample{sample_number}/diseased_ZoneMask.dat')
    healthy_mask_data = load_mask(raw_path / f'Meredith/Sample{sample_number}/healthy_ZoneMask.dat')

    # Generate Mueller matrix component labels
    labels = [f"M({i},{j})" for i in range(1, 5) for j in range(1, 5)]

    # Display in a 4x4 grid format
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    axes = axes.flatten()  # Flatten the 2D grid to 1D for easy indexing

    for i in range(min(16, data.shape[1])):
        # Reshape column to 2D array maintaining original spatial structure
        component_data = data.iloc[:, i].values.reshape(num_rows, num_cols)

        # Set zero values to NaN for visualization purposes
        component_data[component_data == 0] = np.nan

        # Set color scale based on diagonal or off-diagonal
        if i % 5 == 0:  # Diagonal elements
            vmin, vmax = 0, 1
        else:  # Off-diagonal elements
            vmin, vmax = -0.1, 0.1

        # Display the Mueller matrix component data
        im = axes[i].imshow(component_data, cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)

        # Overlay the masks
        overlay = np.zeros((*component_data.shape, 4), dtype=np.uint8)  # RGBA image
        overlay[..., 0] = 255 * diseased_mask_data  # Red channel for diseased
        overlay[..., 1] = 255 * healthy_mask_data  # Green channel for healthy
        overlay[..., 3] = 255 * (diseased_mask_data + healthy_mask_data)  # Alpha channel for transparency

        axes[i].imshow(overlay, aspect='auto')
        axes[i].axis('off')

    # Adjust layout to add more space between the plots and the color bar
    plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.85)

    # Create a single colorbar that encompasses the full range
    cbar_ax = fig.add_axes([0.9, 0.11, 0.032, 0.77])  # Adjusted position and height for the unified colorbar
    norm = colors.Normalize(vmin=-0.1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

    # Hide the tick labels on the color bar
    cbar.ax.yaxis.set_ticks([])

    # Save the entire 4x4 grid as a single image
    filename = f"{label}_Mueller_{sample_number}_{wavelength}_csv.png"
    save_path = visualisation_path / filename
    plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close(fig)  # Close the figure to free up memory


def visualize_last_row(flat_data, visualisation_path, sample_number, figsize=(15, 10), num_rows=600, num_cols=800,
                       vmin=None, vmax=None):
    """
    Visualizes the components of the last row of a Mueller matrix reshaped into the original grid format.

    Parameters:
    - flat_data: numpy array of shape (num_rows * num_cols, num_components), where each component is a flat array.
    - visualisation_path: Path object indicating where to save the figure.
    - sample_number: Sample number to be included in the filename for identification.
    - figsize: Optional. A tuple representing the figure size for the grid.
    - num_rows: The number of rows to reshape each component into. Default is 600.
    - num_cols: The number of columns to reshape each component into. Default is 800.
    - vmin: Optional. The minimum value for the color scale. Default is the global minimum of the data.
    - vmax: Optional. The maximum value for the color scale. Default is the global maximum of the data.
    """
    # Ensure the visualization path exists
    visualisation_path = Path(visualisation_path)
    visualisation_path.mkdir(parents=True, exist_ok=True)

    # Number of components to visualize
    num_components = flat_data.shape[1]

    # Display in a grid format
    fig, axes = plt.subplots(1, num_components, figsize=figsize)

    if num_components == 1:
        axes = [axes]  # Make it iterable if there's only one subplot

    # Labels for components
    labels = [f"4, {i + 1}" for i in range(num_components)]

    # Determine the min and max values for consistent color scaling across components
    global_min = flat_data.min() if vmin is None else vmin
    global_max = flat_data.max() if vmax is None else vmax

    # Visualize each component
    for i, ax in enumerate(axes):
        subplot_title = labels[i]
        ax.set_title(subplot_title)

        # Reshape component data to 2D array for spatial structure visualization
        component_data = flat_data[:, i].reshape(num_rows, num_cols)

        # Visualization with jet color map
        im = ax.imshow(component_data, cmap='jet', aspect='equal', vmin=global_min, vmax=global_max)

        ax.axis('off')

    plt.tight_layout()

    # Create a colorbar with the global scale
    sm = plt.cm.ScalarMappable(cmap='jet', norm=colors.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='horizontal', pad=0.01)

    # Save the visualized image
    filename = f"LastRow_Components_{sample_number}.pdf"
    save_path = visualisation_path / filename
    plt.savefig(save_path, bbox_inches='tight')

    plt.show()