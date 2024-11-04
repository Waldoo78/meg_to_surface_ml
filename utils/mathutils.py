# Mathutils.py
import numpy as np
from scipy.spatial.distance import directed_hausdorff

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

# def solve_coefficients_svd(Y, coords, lambda_reg):
#     """Resolves the linear system to find the coefficients for a given surface
    
#     Args:
#         Y (ndarray): Spherical harmonics matrix (N_points, N_coeffs)
#         coords (ndarray): Surface coordinates (N_points, 3)
#         lambda_reg (float): Regularization parameter
        
#     Returns:
#         ndarray: Coefficients matrix (N_coeffs, 3)
#     """
#     if lambda_reg < 0:
#         raise ValueError("lambda_reg must be non-negative")
    
#     N_coeffs = Y.shape[1]
#     coeffs = np.zeros((N_coeffs, 3), dtype=np.complex128)
    
#     # Construct regularized system
#     A = np.vstack([Y, np.sqrt(lambda_reg) * np.eye(N_coeffs)])
    
#     # Solve for each coordinate
#     for i in range(3):
#         b = np.concatenate([coords[:, i], np.zeros(N_coeffs)])
#         coeffs[:, i] = np.linalg.lstsq(A, b, rcond=None)[0]
    
#     return coeffs

def solve_coefficients_svd(Y, coords, lambda_reg):
    """Resolves the linear system to find the coefficients
    
    Args:
        Y (ndarray): Spherical harmonics matrix (N_points, N_coeffs)
        coords (ndarray): Coordinates or differences to fit. Shape can be:
            - (N_points, 2) for angular deformations (θ, φ)
            - (N_points, 1) for radial deformations (r)
            - (N_points, 3) for full 3D coordinates
        lambda_reg (float): Regularization parameter
        
    Returns:
        ndarray: Coefficients matrix (N_coeffs, n_dims)
            where n_dims matches the second dimension of coords
    """
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be non-negative")
    
    # Handle different input dimensions
    coords = np.atleast_2d(coords)
    if coords.shape[1] not in [1, 2, 3]:
        raise ValueError("coords must have 1, 2 or 3 components (radial, angular, or full 3D)")
    
    N_coeffs = Y.shape[1]
    n_dims = coords.shape[1]
    coeffs = np.zeros((N_coeffs, n_dims), dtype=np.complex128)
    
    # Construct regularized system
    A = np.vstack([Y, np.sqrt(lambda_reg) * np.eye(N_coeffs)])
    
    # Solve for each coordinate
    for i in range(n_dims):
        b = np.concatenate([coords[:, i], np.zeros(N_coeffs)])
        coeffs[:, i] = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return coeffs

def dist_Haussdorf(array1,array2): #3D arrays
    distance_1 = directed_hausdorff(array1, array2)[0]
    distance_2 = directed_hausdorff(array2, array1)[0]
    dist_Hauss=max(distance_1,distance_2)
    return dist_Hauss