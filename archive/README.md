# PR Filter Script

One simple script that applies physical realizability testing and creates visualizations.
**Loads directly from original .mat files - same as notebook.**

## Usage

1. Edit the configuration in `pr_filter.py`:

```python
# For Cervix samples:
INPUT_FILE = "/Volumes/ep_ssd/database/cervix_dataset/Sample23/550_Mueller.mat"
TISSUE_TYPE = "cervix"

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

# Dimensions are set automatically based on tissue type:
# Cervix: 600x800, Brain: 388x516, AFMMM: 500x500
```

2. Run:

```bash
python pr_filter.py
```

## What It Does

1. **Loads Mueller matrix from .mat or .npz file**:
   - **.mat files** (exact same logic as notebook):
     - **Cervix**: Loads `550_Mueller.mat`, accesses 'MM' key
     - **Brain**: Loads `MM.mat`, accesses 'MM' key
     - **AFMMM**: Loads `FinalMM.mat`, stacks 16 components
   - **.npz files**: Auto-detects key (tries 'nM', 'MM', 'mueller_matrix', 'M')
2. Applies PR test using `charpoly` from `src/utils/pr_test.py` - **exact same logic as notebook**
3. Creates populated dataset with only realizable pixels
4. Generates visualizations:
   - **2 PDFs**: BEFORE (original) and AFTER (PR filtered)
   - **GIF**: Before/After animation
   - **PNG**: PR mask

## Output Files

In the output directory you'll get **4 files**:
1. `{SAMPLE_NAME}_BEFORE.pdf` - Original Mueller matrix (zeros filtered only)
2. `{SAMPLE_NAME}_AFTER.pdf` - PR-filtered Mueller matrix
3. `{SAMPLE_NAME}_before_after.gif` - Animated before/after comparison
4. `{SAMPLE_NAME}_pr_mask.png` - PR mask showing realizable pixels

**Filtered pixels** are shown as:
- **BLACK** if `USE_BLACK_BACKGROUND = True` (default)
- **WHITE** if `USE_BLACK_BACKGROUND = False`

## How PR Test Works

The script uses the **exact same approach** as your notebook:

```python
for i, M in enumerate(mueller_matrices):
    if charpoly(M):  # Uses charpoly from pr_test.py
        realizable_indices.append(i)
    else:
        non_realizable_indices.append(i)
```

This is identical to the notebook's `classify_and_save_matrices` function.

## Notes

- **Loads from original .mat files** - same extraction logic as your notebook
- Uses `charpoly` from `src/utils/pr_test.py` (same as notebook)
- Uses `visualize_save_mueller` from `src/utils/visualisation.py` (same as notebook)
- Processing time: ~2-3 minutes for cervix sample (600×800 = 480,000 pixels)

## File Paths for Common Samples

**Cervix:**
```
/Volumes/ep_ssd/database/cervix_dataset/Sample{1-25}/550_Mueller.mat
```

**Brain:**
```
/path/to/brain/sample {1-N}/MM.mat
```

**AFMMM:**
```
/path/to/afmmm/sample_he{1-N}/FinalMM.mat
/path/to/afmmm/sample_bg{1-N}/FinalMM.mat
/path/to/afmmm/sample_bw{1-N}/FinalMM.mat
```
