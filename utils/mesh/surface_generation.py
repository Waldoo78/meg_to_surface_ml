import numpy as np
import utils.mesh.spherical_harmonics as SH
from utils.mathutils import dist_Haussdorf

def generate_surface_from_coefficients(preprocessed_data, coefficient_data, sigma=0, n_jobs=-1):
    """Generates a surface from precomputed spherical harmonics coefficients
    
    Args:
        preprocessed_data (dict): Output from preprocess_surface
        coefficient_data (dict): Output from compute_coefficients
        sigma (float): Smoothing parameter
        n_jobs (int): Number of parallel jobs
        
    Returns:
        dict: Generated surface data including coordinates and error metrics
    """
    # Generate surface
    coords_recon = SH.generate_surface(
        preprocessed_data['theta'],
        preprocessed_data['phi'],
        coefficient_data['lmax'],
        sigma,
        coefficient_data['organized_coeffs'],
        n_jobs=n_jobs
    )
    
    # Add back the center
    coords_recon += preprocessed_data['center']
    
    # Compute errors
    error = np.sqrt(np.sum(
        (preprocessed_data['orig_coords'] - coords_recon)**2, 
        axis=1
    ))
    error_hausdorff = dist_Haussdorf(
        preprocessed_data['orig_coords'], 
        coords_recon
    )
    
    return {
        'reconstructed_coords': coords_recon,
        'error': error,
        'error_hausdorff': error_hausdorff,
        'triangles': preprocessed_data['sphere_tris']
    }
