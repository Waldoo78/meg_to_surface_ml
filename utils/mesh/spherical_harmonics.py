import numpy as np
from scipy.special import sph_harm
from utils.mathutils import cart_to_sph

def compute_Y(theta, phi, lmax):
    N = len(theta)
    M = (lmax + 1)**2
    Y = np.zeros((N, M), dtype=complex)
 
    idx = 0
    for l in range(lmax + 1):
        for m in range(0, l + 1):
            ylm = sph_harm(m, l, theta, phi)[:, np.newaxis]
            if m == 0:
                Y[:, idx] = ylm.flatten()
                idx += 1
            else:
                Y[:, idx] = ylm.flatten()
                idx += 1
                Y[:, idx] = (-1)**m * np.conjugate(ylm.flatten())
                idx += 1           
    return Y

def organize_coeffs(coeffs, lmax):
    orders = {}
    start_idx = 0
    for l in range(lmax + 1):
        size = 2 * l + 1
        orders[l] = coeffs[start_idx:start_idx + size, :]
        start_idx += size
    return orders

def generate_surface(Y, lmax, sigma, orders):
    N_points = Y.shape[0]
    xyz_total = np.zeros((N_points, 3), dtype=np.complex128)
    scales = np.array([np.exp(-l * (l + 1) * sigma) for l in range(1, lmax + 1)])
    for l in range(1, lmax + 1):
        start_idx = l * l
        size = 2 * l + 1
        Y_block = Y[:, start_idx:start_idx + size]
        xyz_total += scales[l-1] * (Y_block @ orders[l])
    
    return np.real(xyz_total)

def get_spherical_params(sphere_coords,sphere_tris):
    center = np.mean(sphere_coords, axis=0)
    _, theta, phi = cart_to_sph(sphere_coords - center)

    return {
        'theta': theta, 'phi': phi,
        'coords': sphere_coords, 'tris': sphere_tris,
    }

def compute_coefficients(Y, template_projection, resampled_surface, lmax, lambda_reg=0):
    """Computes spherical harmonics coefficients with the grid imposed by the template"""
    target_coords, target_tris = resampled_surface
    template_center = np.mean(template_projection, axis=0)
    
    coeffs = np.linalg.lstsq(Y, target_coords - template_center, rcond=lambda_reg)[0]
    
    return {
        'organized_coeffs': organize_coeffs(coeffs, lmax),
        'lmax': lmax,
        'template_center': template_center  
    }