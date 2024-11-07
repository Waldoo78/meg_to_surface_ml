import numpy as np
from utils.mathutils import cart_to_sph, solve_coefficients_svd
import utils.mesh.spherical_harmonics as SH
from utils.mesh.Projection_onto_sphere import get_spherical_projection, get_resampled_inner_surface
from s3pipe.utils.utils import get_sphere_template
from s3pipe.utils.interp_numpy import resampleSphereSurf
from utils.mesh.surface_generation import compute_surface_metrics

#Spherical projection for the template

def get_spherical_params(resampled_surface_coords, resampled_surface_tris, hemisphere):
    """Gets spherical projection parameters for a surface using S3MAP
    
    Args:
        resampled_surface_coords (ndarray): Resampled surface coordinates (N_vertices, 3)
        resampled_surface_tris (ndarray): Resampled Surface triangles
        hemisphere (str): 'lh' or 'rh'
        
    Returns:
        dict: Spherical parameters including angles and coordinates at 40962 vertices
    """
    if hemisphere not in ['lh', 'rh']:
        raise ValueError("hemisphere must be 'lh' or 'rh'")
    
    sphere_coords, sphere_tris = get_spherical_projection(
        (resampled_surface_coords, resampled_surface_tris),
        hemisphere
    )

    center = np.mean(sphere_coords, axis=0)
    sphere_coords_centered = sphere_coords - center
    _, theta, phi = cart_to_sph(sphere_coords_centered)

    return {
        'theta': theta,
        'phi': phi,
        'center': center,
        'coords': resampled_surface_coords,
        'tris': resampled_surface_tris,
        'sphere_coords': sphere_coords,
        'sphere_tris': sphere_tris,
    }

def compute_coefficients(Y, template_projection, resampled_surface, lmax, lambda_reg=0):
    """Computes spherical harmonics coefficients with the grid imposed by the template
    
    Args:
        Y: Spherical harmonics basis computed from template sphere coordinates
        template_projection: Template sphere coordinates
        resampled_surface: Tuple (coords, triangles) of the target surface
        lmax (int): Maximum degree of spherical harmonics
        lambda_reg (float): Regularization parameter
        
    Returns:
        dict: Coefficients organized for use with generate_surface()
    """
    target_coords, target_tris = resampled_surface
    
    # Use template center for consistency
    template_center = np.mean(template_projection, axis=0)
    
    # Center both surfaces relative to template center
    coeffs = solve_coefficients_svd(
        Y,
        target_coords - template_center,  # Center target surface using template center
        lambda_reg
    )

    # Organize coefficients
    organized_coeffs = SH.organize_coeffs(coeffs, lmax)
    
    return {
        'organized_coeffs': organized_coeffs,
        'lmax': lmax,
        'template_center': template_center  
    }


def compute_template_projection(template_coords, template_tris, output_path, hemisphere):
    """Projects template mesh to sphere and saves results"""

    sphere_data = get_spherical_params(
        template_coords,
        template_tris,
        hemisphere
    )
    
    # Save using consistent format
    np.savez(f'{output_path}_{hemisphere}_template.npz',
        theta=sphere_data['theta'],
        phi=sphere_data['phi'],
        sphere_coords=sphere_data['sphere_coords'],
        sphere_tris=sphere_data['sphere_tris'],
        center=sphere_data['center'],
        coords=sphere_data['coords'],
        tris=sphere_data['tris'],
    )