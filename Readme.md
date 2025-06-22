# Cortical Surface Processing Pipeline

Two-step pipeline to process brain surfaces using spherical harmonics.

## What each script does

### Script 1: `full_preprocessing.py`
**Algorithm:** Complete data preparation
1. Create fsaverage template and compute its coefficients
2. Load brain mesh (vertices + triangles) for each subject
3. Resample to standard sphere template (same number of points for all brains)
4. Smooth surfaces and compute spherical harmonic coefficients
5. Center coordinates and save everything

**Input:** `lh_vertices.mat`, `lh_faces.mat` (raw brain mesh)  
**Output:** `lh_resampled.npz`, `coeffs_lh.pkl`, `lh_center.npz` (processed data)

### Script 2: `reconstruction.py` 
**Algorithm:** Mathematical reconstruction and quality analysis
1. Load preprocessed data from step 1
2. **Reconstruct:** Rebuild surface from coefficients 
3. **Validate:** Compare original vs reconstructed (distance errors, curvature)
4. **Analyze:** Generate quality statistics and visualizations

**Input:** `lh_resampled.npz` + coefficients from step 1  
**Output:** Analysis results and quality metrics

## How spherical harmonics work

Think of it like JPEG compression but for 3D brain shapes:
- **Low orders** (0-10): Global brain shape
- **Medium orders** (10-30): Major folds and curves  
- **High orders** (30+): Fine details

The algorithm finds the "recipe" (coefficients) to rebuild any brain shape.

## Usage

### Step 0: Copy data to your working directory
**Important:** Copy your brain data to your working folder because the pipeline will add new files (harmonics, coefficients).

```bash
# Copy your original data
cp -r /original/brain/data /your/working/directory/
```

### Step 1: Complete preprocessing (replaces old cortical_resample.py)
```bash
python full_preprocessing.py --data_dir /path/to/your/subjects --templates_dir /path/to/templates
```

### Step 2: Analyze and reconstruct
```bash
python reconstruction.py --base_dir /path/to/your/subjects --data_path /path/to/templates
```

### Process one subject only
```bash
python full_preprocessing.py --data_dir /path/to/your/subjects --templates_dir /path/to/templates --subject sub-001
python reconstruction.py --base_dir /path/to/your/subjects --data_path /path/to/templates --subject sub-001 --visualize
```

## File flow

```
Raw brain mesh → Complete preprocessing → Analysis
lh_vertices.mat → lh_resampled.npz + coeffs_lh.pkl → Quality metrics
```

## Key parameters

### Full Preprocessing
- `--data_dir` - Your subjects folder (required)
- `--templates_dir` - Harmonics templates folder (required)
- `--subject sub-001` - Process one subject only
- `--lmax 80` - Detail level (higher = more detail)
- `--skip_fsaverage` - Skip template creation

### Reconstruction
- `--lmax 30` - How much detail to use for reconstruction
- `--visualize` - Show before/after comparison
- `--save_surface` - Save reconstructed surface

## Data structure

```
your_working_directory/
├── data/
│   ├── Anatomy_data_CAM_CAN/   # Your brain data (copied here)
│   │   └── sub-001/
│   │       ├── lh_vertices.mat        # Input: raw mesh
│   │       ├── lh_faces.mat  
│   │       ├── rh_vertices.mat
│   │       ├── rh_faces.mat
│   │       ├── lh_resampled.npz       # Generated: clean mesh
│   │       ├── rh_resampled.npz
│   │       ├── coeffs_lh.pkl          # Generated: SH coefficients
│   │       ├── coeffs_rh.pkl
│   │       ├── lh_center.npz          # Generated: hemisphere centers
│   │       └── rh_center.npz
│   └── fsaverage/                     # Generated: template
│       ├── lh_resampled.npz
│       ├── rh_resampled.npz
│       ├── coeffs_lh.pkl
│       └── coeffs_rh.pkl
└── src/
    └── cortical_transformation/
        ├── full_preprocessing.py      # Step 1: Complete preprocessing
        ├── reconstruction.py          # Step 2: Analysis
        └── data/                      # Math templates
            ├── Y_lh.npz               # Left hemisphere harmonics
            └── Y_rh.npz               # Right hemisphere harmonics
```