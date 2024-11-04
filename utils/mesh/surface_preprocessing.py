import numpy as np
from utils.mathutils import cart_to_sph, solve_coefficients_svd
import utils.mesh.spherical_harmonics as SH
from utils.mesh.Projection_onto_sphere import spherical_projection_40962
from s3pipe.utils.utils import get_sphere_template
from s3pipe.utils.interp_numpy import resampleSphereSurf

def preprocess_surface(surface_coords, surface_tris, hemisphere='lh'):
    """Preprocesses a surface for spherical harmonics analysis
    
    Args:
        surface_coords (ndarray): Surface coordinates
        surface_tris (ndarray): Surface triangles
        hemisphere (str): 'lh' or 'rh'
        
    Returns:
        dict: Preprocessed data including spherical coordinates and angles
    """
    # Project to sphere
    sphere_coords, sphere_tris, orig_coords, orig_tris = spherical_projection_40962(
        (surface_coords, surface_tris), hemisphere=hemisphere
    )
    
    # Center coordinates
    center = np.mean(sphere_coords, axis=0)
    sphere_coords_centered = sphere_coords - center
    orig_coords_centered = orig_coords - center
    
    # Convert to spherical coordinates
    _, theta, phi = cart_to_sph(sphere_coords_centered)
    
    return {
        'theta': theta,
        'phi': phi,
        'center': center,
        'orig_coords': orig_coords,
        'orig_tris': orig_tris,
        'sphere_coords': sphere_coords,
        'sphere_tris': sphere_tris
    }

def compute_template_projection(template_mesh, template_name):
    """
    Project template mesh to sphere and save results
    
    Args:
        template_mesh (tuple): (vertices, faces) of the template mesh
        template_name (str): Name for the output file
    """
    # Project the sphere
    sphere_coords, sphere_tris, orig_coords, orig_tris = spherical_projection_40962(
        template_mesh, 
        hemisphere='lh'
    )

    # Calculate center from sphere coordinates
    center = np.mean(sphere_coords, axis=0)
    
    # Center coordinates before computing spherical coordinates
    sphere_coords_centered = sphere_coords - center
    
    # Convert to spherical coordinates
    _, theta, phi = cart_to_sph(sphere_coords_centered)
    
    # Save results
    np.savez(template_name+'_template.npz', 
        theta=theta,
        phi=phi,
        sphere_coords=sphere_coords,
        sphere_tris=sphere_tris,
        center=center,  
        orig_coords=template_mesh[0],
        orig_tris=template_mesh[1]
    )

def compute_coefficients(preprocessed_data, lmax, lambda_reg=0, n_jobs=-1):
    """Computes spherical harmonics coefficients for a preprocessed surface
    
    Args:
        preprocessed_data (dict): Output from preprocess_surface
        lmax (int): Maximum degree of spherical harmonics
        lambda_reg (float): Regularization parameter
        n_jobs (int): Number of parallel jobs
        
    Returns:
        dict: Computed coefficients and related data
    """
    # Compute spherical harmonics basis
    Y = SH.compute_Y(
        preprocessed_data['theta'], 
        preprocessed_data['phi'], 
        lmax, 
        n_jobs=n_jobs
    )
    
    # Solve for coefficients
    coeffs = solve_coefficients_svd(
        Y, 
        preprocessed_data['orig_coords'] - preprocessed_data['center'], 
        lambda_reg
    )
    
    return {
        'coefficients': coeffs,
        'organized_coeffs': SH.organize_coeffs(coeffs, lmax),
        'lmax': lmax
    }

def resample_inner_surface(surf_data, target_n_vertices=40962):
    """Resample inner surface to target resolution using spherical interpolation
    
    Args:
        surf_data (dict): Output from preprocess_surface containing original coordinates and faces
        target_n_vertices (int): Target number of vertices (default 40962)
    
    Returns:
        dict: Resampled surface data
    """
    template_surf = get_sphere_template(target_n_vertices)
    
    resampled_features = resampleSphereSurf(
        surf_data['sphere_coords'],
        template_surf['vertices'],
        surf_data['orig_coords'],
        faces=surf_data['orig_tris'][:, 1:]
    )
    
    return {
        'vertices': resampled_features[:, -3:],
        'faces': template_surf['faces']
    }