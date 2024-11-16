# Mathutils.py
import numpy as np
import s3pipe.surface.prop as sprop
from scipy.spatial.distance import directed_hausdorff
from s3pipe.surface.surf import Surface


def cart_to_sph(coords):
    """Converts Cartesian coordinates to spherical coordinates
    
    Args:
        coords (ndarray): Cartesian coordinates (N_points, 3)
        
    Returns:
        tuple: (r, theta, phi) where:
            r (ndarray): radial distance
            theta (ndarray): azimuthal angle [0, 2π]
            phi (ndarray): polar angle [0, π]
    """
    x, y, z = coords.T
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # [-π, π] -> [0, 2π]
    theta = np.where(theta < 0, theta + 2*np.pi, theta)
    phi = np.arccos(np.clip(z / r, -1, 1))
    
    return r, theta, phi

def sph_to_cart(r, theta, phi):
    """Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        r (ndarray): Radial distance from origin
        theta (ndarray): Azimuthal angle in radians [0, 2π] in x-y plane from x-axis (longitude)
        phi (ndarray): Polar angle in radians [0, π] from z-axis (colatitude)
        
    Returns:
        ndarray: Array of shape (n, 3) containing the x, y, z Cartesian coordinates
        
    Note:
        The convention used is:
        x = r * cos(theta) * sin(phi)
        y = r * sin(theta) * sin(phi)
        z = r * cos(phi)
    """
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    
    return np.column_stack((x, y, z))

def hausdorff_distance(array1, array2):
    """Compute the Hausdorff distance between two 3D point clouds.
    
    Args:
        array1 (ndarray): First array of 3D points
        array2 (ndarray): Second array of 3D points
        
    Returns:
        float: Hausdorff distance between the point clouds
    """
    distance_1 = directed_hausdorff(array1, array2)[0]
    distance_2 = directed_hausdorff(array2, array1)[0]
    return max(distance_1, distance_2)


def compute_surface_metrics(generated_coords, reference_coords, triangles):
    """Computes comparison metrics between generated and reference brain surfaces."""
    
    center = np.mean(reference_coords, axis=0)
    ref_centered = reference_coords - center
    gen_centered = generated_coords - center
    error = np.sqrt(np.sum((ref_centered-gen_centered)**2, axis=1))
    

    s3map_triangles = np.column_stack((np.full(len(triangles), 3), triangles))
    ref_surf = Surface(ref_centered, s3map_triangles)
    gen_surf = Surface(gen_centered, s3map_triangles)
    

    ref_curvature = sprop.computeMeanCurvature(ref_surf, dist=0.5)
    gen_curvature = sprop.computeMeanCurvature(gen_surf, dist=0.5)
    ref_areas = sprop.computeVertexArea(ref_surf)
    gen_areas = sprop.computeVertexArea(gen_surf)
    
    # Construction directe du dictionnaire
    return {
        'raw_metrics': {
            'reference_curvature': ref_curvature,
            'generated_curvature': gen_curvature,
            'reference_areas': ref_areas,
            'generated_areas': gen_areas,
            'error': error
        },
        'distance_stats': {
            'error_hausdorff': float(hausdorff_distance(ref_centered, gen_centered)),
            'error': error,
            'mean_error': float(np.mean(error)),
            'max_error': float(np.max(error)),
            'std_error': float(np.std(error))
        },
        'shape_stats': {
            'mean_curvature_diff': float(np.mean(np.abs(ref_curvature - gen_curvature)))
        },
        'area_stats': {
            'total_area_diff': float(np.abs(np.sum(ref_areas) - np.sum(gen_areas)))
        }
    }