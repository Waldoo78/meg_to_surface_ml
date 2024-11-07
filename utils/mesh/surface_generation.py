import numpy as np
import utils.mesh.spherical_harmonics as SH
from s3pipe.surface.surf import Surface
import s3pipe.surface.prop as sprop
from utils.mathutils import dist_Haussdorf

def generate_surface_from_coefficients(Y,coefficients, lmax, sigma=0):
    # """Generates a surface from spherical harmonics coefficients.
    
    # Args:
    #     theta (ndarray): Polar angles in radians [0, π]
    #     phi (ndarray): Azimuthal angles in radians [0, 2π]
    #     coefficients (dict): Dictionary of coefficients organized by degree
    #     lmax (int): Maximum degree of spherical harmonics
    #     sigma (float, optional): Smoothing parameter. Defaults to 0.
        
    # Returns:
    #     ndarray: Surface coordinates centered at origin (N_points, 3)
    # """
    # return SH.generate_surface(Y, lmax, sigma, coefficients)
    pass


#To improve
def compute_surface_metrics(generated_coords, reference_coords, triangles):
    """Computes comparison metrics between generated and reference brain surfaces."""
    # Center surfaces
    center = np.mean(reference_coords, axis=0)
    ref_centered = reference_coords - center
    gen_centered = generated_coords - center
    error = np.sqrt(np.sum((ref_centered-gen_centered)**2, axis=1))

    # Prepare surfaces
    s3map_triangles = np.column_stack((np.full(len(triangles), 3), triangles))
    ref_surf = Surface(ref_centered, s3map_triangles)
    gen_surf = Surface(gen_centered, s3map_triangles)
    
    # Compute basic geometric properties
    ref_curvature = sprop.computeMeanCurvature(ref_surf, dist=0.5)  
    gen_curvature = sprop.computeMeanCurvature(gen_surf, dist=0.5)
    ref_areas = sprop.computeVertexArea(ref_surf)
    gen_areas = sprop.computeVertexArea(gen_surf)
    
    return {
        'raw_metrics': {
            'reference_curvature': ref_curvature,
            'generated_curvature': gen_curvature,
            'reference_areas': ref_areas,
            'generated_areas': gen_areas,
            'error': error
        },
        'distance_stats': {
            'error_hausdorff': float(dist_Haussdorf(ref_centered, gen_centered)),
            'error': error,
            'mean_error': float(np.mean(error)),
            'max_error': float(np.max(error)),
            'std_error': float(np.std(error))
        },
        'shape_stats': {
            'mean_curvature_diff': float(np.mean(np.abs(ref_curvature - gen_curvature))),
        },
        'area_stats': {
            'total_area_diff': float(np.abs(np.sum(ref_areas) - np.sum(gen_areas)))
        }
    }