import numpy as np
from scipy.special import sph_harm

def compute_Y(theta, phi, lmax, n_jobs=None):
    """Computes the spherical harmonics basis matrix"""
    N = (lmax + 1)**2
    Y = np.zeros((theta.shape[0], N), dtype=np.complex128)
    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y[:, idx] = sph_harm(m, l, theta, phi)
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

def generate_surface_partial(Y, l, sigma, ordres):
    """Generates surface coordinates for a single degree l, with Y the matrix of spherical harmonics"""
    N_points = Y.shape[0]
    xyz = np.zeros((N_points, 3), dtype=np.complex128)
    scale = np.exp(-l * (l + 1) * sigma)
    
    start_idx = l * l  # Index de départ pour le degré l
    for m_idx, m in enumerate(range(-l, l + 1)):
        Ylm = Y[:, start_idx + m_idx]  # On ajoute m_idx à l'index de départ
        xyz += scale * ordres[l][m_idx, :] * Ylm.reshape(-1, 1)
    
    return xyz

def generate_surface(Y,lmax, sigma, orders):
    """Generates surface coordinates from spherical harmonics coefficients.
    
    Args:
        phi (ndarray): Polar angle in radians [0, π]
        theta (ndarray): Azimuthal angle in radians [0, 2π]
        lmax (int): Maximum degree
        sigma (float): Smoothing parameter
        orders (dict): Coefficients dictionary by degree
        
    Returns:
        ndarray: Real surface coordinates (N_points, 3)
    """
    return np.real(sum(generate_surface_partial(Y,l, sigma, orders) 
                      for l in range(lmax + 1)))