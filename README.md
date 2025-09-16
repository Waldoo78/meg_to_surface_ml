
## Table of Contents

- [Overview](#-overview)
- [Pipeline Components](#-pipeline-components)
- [Installation](#-installation)
- [Usage](#-usage)
- [Example Results](#-example-results)
- [Mathematical Framework](#-mathematical-framework)
- [Performance Tuning](#-performance-tuning)
- [Dependencies](#-dependencies)
- [Citation](#-citation)
- [License](#-license)

## Overview

This project implements a comprehensive pipeline for neuroimaging data analysis that combines cortical surface processing with Bayesian source reconstruction. The approach allows for efficient analysis of brain structure and function by standardizing cortical representation while maintaining anatomical validity.

### Key Features

- **Cortical Surface Standardization**: Transforms diverse cortical meshes into a common representation
- **Spherical Harmonic Decomposition**: Provides a compact, frequency-domain representation of cortical geometry
- **Bayesian Source Estimation**: Implements variational free energy optimization for source localization
- **Cross-Subject Analysis Support**: Facilitates group-level comparisons via standardized representations
- **Quality Assessment Tools**: Quantifies reconstruction accuracy and source estimation reliability

## Pipeline Components

The pipeline consists of three main stages, each implemented in dedicated modules:



### 1️⃣ Cortical Surface Resampling (`cortical_resample.py`)

Transforms raw cortical meshes into standardized representations with consistent vertex density and topology:

- Loads cortical meshes from MATLAB format
- Applies controlled smoothing to reduce noise
- Projects surfaces onto a sphere and resamples to standard topology
- Evaluates mesh quality and corrects issues
- Outputs standardized surfaces (.npz format)

```python
# Example: Resampling a subject's cortical surface
from cortical_resample import process_surfaces

results = process_surfaces(
    main_folder="/path/to/subjects",
    subject_id="sub-CC110033",
    save_results=True,
    preprocess=True
)
```

### 2️⃣ Surface Reconstruction with Spherical Harmonics (`reconstruction.py`)

Transforms resampled surfaces into a compact mathematical representation:

- Computes spherical harmonic coefficients up to specified order
- Reconstructs surfaces from coefficients with configurable detail level
- Merges left and right hemisphere reconstructions
- Calculates quality metrics (distances, angles, curvature)
- Saves coefficients and reconstructed surfaces

```python
# Example: Reconstructing surfaces using spherical harmonics
from reconstruction import process_subjects

results = process_subjects(
    base_dir="/path/to/subjects",
    data_path="/path/to/harmonics",
    subject_id="sub-CC110033",
    sigma=1e-4,
    lmax=30,
    max_order_coeff=80,
    visualize_first=True
)
```

### 3️⃣ Bayesian Source Reconstruction

Applies variational Bayesian inference to estimate neural sources from MEG/EEG data:

- Performs SVD dimensionality reduction for computational efficiency
- Implements a probabilistic generative model with free energy optimization
- Estimates source activity and uncertainty on the cortical surface
- Provides posterior distributions of key parameters

```python
# Example: Running Bayesian source estimation
import numpy as np
from source_estimation import run_inference_qe

# Load data
Y = np.load("meg_data.npy")
leadfield = np.load("leadfield.npy")
Qe = np.eye(Y.shape[0])  # Identity noise precision matrix

# Run inference
results = run_inference_qe(
    Y_reduced=Y,
    leadfield=leadfield,
    Qe=Qe,
    n_components=20,
    num_steps=200,
    learning_rate=0.3
)

# Access source estimates
J = results["J"]  # Source activity estimates
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)

### Option 1: Install Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/cortical-transformation.git
cd cortical-transformation

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Install Using conda

```bash
# Clone the repository
git clone https://github.com/yourusername/cortical-transformation.git
cd cortical-transformation

# Create conda environment
conda env create -f environment.yml
conda activate cortical-env
```

### Download Required Data Files

```bash
# Create data directory
mkdir -p src/cortical_transformation/data

# Download spherical harmonic basis files (example - replace with actual commands)
# Note: These files are not included in the repository
# Contact the author for access to the harmonic basis files
# Y_lh.npz and Y_rh.npz are necessary for the pipeline to function
```

## Usage

### Complete Pipeline Example

```python
from cortical_resample import process_surfaces
from reconstruction import process_subjects
from source_estimation import run_inference_qe
import numpy as np

# 1. Resample cortical surfaces
subject_results = process_surfaces(
    main_folder="/path/to/subjects",
    subject_id="sub-CC110033",
    save_results=True,
    preprocess=True
)

# 2. Calculate spherical harmonic coefficients and reconstruct surfaces
sh_results = process_subjects(
    base_dir="/path/to/subjects",
    data_path="/path/to/data",
    subject_id="sub-CC110033",
    sigma=1e-4,
    lmax=30,
    max_order_coeff=80,
    visualize_first=True
)

# 3. Perform Bayesian source estimation (if MEG/EEG data is available)
Y = np.load("/path/to/subjects/sub-CC110033/meg_data.npy")
leadfield = np.load("/path/to/subjects/sub-CC110033/leadfield.npy")
Qe = np.eye(Y.shape[0])

source_results = run_inference_qe(
    Y_reduced=Y,
    leadfield=leadfield,
    Qe=Qe,
    n_components=20,
    num_steps=200
)

# Save source estimates
np.savez(
    "/path/to/subjects/sub-CC110033/source_estimates.npz",
    J=source_results["J"],
    gamma=source_results["gamma"]["mean"],
    beta=source_results["beta"]["mean"]
)
```

### Command-Line Interface

```bash
# Resample a subject's cortical surface
python cortical_resample.py --subject sub-CC110033 --data_dir /path/to/subjects

# Reconstruct a subject's cortical surface
python reconstruction.py --subject sub-CC110033 --data_dir /path/to/subjects --harmonic_dir /path/to/data

# Run source estimation
python source_estimation.py --subject sub-CC110033 --data_dir /path/to/subjects
```

## 📊 Example Results

The pipeline provides comprehensive evaluation metrics for both surface reconstruction quality and source estimation accuracy. Results are generated for each subject and can be used to assess the quality of the processing at each stage.

### Surface Reconstruction Quality

Surface reconstruction quality is assessed using metrics including point-to-point distances, normal angles, and curvature differences between the original and reconstructed surfaces.

### Source Reconstruction

Source reconstruction results include posterior estimates of key parameters (gamma, beta) and source activity maps that can be visualized on the cortical surface.

## Mathematical Framework

### Spherical Harmonic Representation

A cortical surface with coordinates **r** = (*x*, *y*, *z*) is decomposed as:

$$\mathbf{r}(\theta, \phi) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} Y_{lm}(\theta, \phi)$$

where:
- $Y_{lm}(\theta, \phi)$ are the spherical harmonic basis functions
- $c_{lm}$ are the spherical harmonic coefficients
- $L$ is the maximum harmonic degree (order)

### Bayesian Source Model

The MEG/EEG source estimation problem is formulated as:

$$\mathbf{Y} = \mathbf{L}\mathbf{J} + \mathbf{E}$$

where:
- $\mathbf{Y}$ is the MEG/EEG data
- $\mathbf{L}$ is the leadfield matrix (forward model)
- $\mathbf{J}$ are the source currents
- $\mathbf{E}$ is measurement noise

The variational free energy (negative ELBO) is optimized:

$$\mathcal{F} = \mathbb{E}_{q}[\log q(\theta) - \log p(\theta, \mathbf{Y})]$$

where:
- $q(\theta)$ is the variational posterior
- $p(\theta, \mathbf{Y})$ is the joint distribution

## Performance Tuning

### Spherical Harmonic Order Selection

The level of detail in cortical reconstruction can be controlled by adjusting the `lmax` parameter:

- **Low Order (lmax=10-15)**: Captures global shape but smooths out gyri and sulci
- **Medium Order (lmax=20-30)**: Good balance between detail and noise suppression
- **High Order (lmax=40-60)**: Captures fine details but may include noise
- **Very High Order (lmax>60)**: Primarily for specialized applications requiring extreme detail

### GPU Acceleration

The Bayesian inference component uses JAX, which automatically utilizes GPU when available. To enable GPU acceleration:

```python
# Before importing JAX or NumPyro:
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Then import JAX packages
import jax
import jax.numpy as jnp
import numpyro
```

### Memory Optimization

For large datasets, consider:

- Processing subjects sequentially rather than in parallel
- Reducing SVD components for source estimation (n_components=10-15)
- Using a lower harmonic order for initial tests

## Dependencies

Primary dependencies:

- **NumPy/SciPy**: Core numerical operations
- **JAX/JAX NumPy**: GPU-accelerated array processing and automatic differentiation
- **NumPyro**: Probabilistic programming and variational inference
- **PyVista**: 3D mesh processing and visualization
- **Matplotlib**: Visualization and plotting

For a complete list, see `requirements.txt`.




## 👥 Contributors

- Walid Bouainouche (@Waldoo78) - Project Lead

## Acknowledgments

- The Montreal Neurological Institute (The Neuro) at McGill University for support
- The open-source neuroimaging community
