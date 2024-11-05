import numpy as np
from scipy.special import sph_harm

def compute_sph_harm(l, m, theta, phi):
    """Computes a single spherical harmonic
    
    Args:
        l (int): Degree of spherical harmonic
        m (int): Order of spherical harmonic
        theta (ndarray): Azimuthal angle in radians [0, 2π]
        phi (ndarray): Polar angle in radians [0, π]
        
    Returns:
        ndarray: Complex spherical harmonic values
    """
    return sph_harm(m, l, theta, phi)

def compute_Y(theta, phi, lmax, n_jobs=None):
    """Computes the spherical harmonics basis matrix
    
    Args:
        theta (ndarray): Azimuthal angles in radians [0, 2π]
        phi (ndarray): Polar angles in radians [0, π]
        lmax (int): Maximum degree of spherical harmonics
        n_jobs (int): Number of parallel jobs (non utilisé dans la version vectorisée)
        
    Returns:
        ndarray: Matrix of spherical harmonic values (N_points, (lmax+1)²)
    """
    N = (lmax + 1)**2
    Y = np.zeros((theta.shape[0], N), dtype=np.complex128)
    
    # Création des indices l et m
    l_vals = []
    m_vals = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            l_vals.append(l)
            m_vals.append(m)
    l_vals = np.array(l_vals)
    m_vals = np.array(m_vals)
    
    # Vectorized computations
    result = sph_harm(m_vals[:, np.newaxis], l_vals[:, np.newaxis], 
                     theta[np.newaxis, :], phi[np.newaxis, :])
    for idx, (l, m) in enumerate(zip(l_vals, m_vals)):
        j = l * (l + 1) + m
        Y[:, j] = result[idx]
    
    return Y

def organize_coeffs(coeffs, lmax):
    """Organizes coefficients by degree l
    
    Args:
        coeffs (ndarray): Coefficient matrix (N_coeffs, 3)
        lmax (int): Maximum degree
        
    Returns:
        dict: Dictionary mapping l to its coefficients
    """
    ordres = {}
    start_idx = 0
    for l in range(lmax + 1):
        size = 2 * l + 1
        ordres[l] = coeffs[start_idx:start_idx + size, :]
        start_idx += size
    return ordres

def generate_surface_partial(l, sigma, theta, phi, ordres):
    """Generates surface coordinates for a single degree l"""
    N_points = theta.shape[0]
    xyz = np.zeros((N_points, 3), dtype=np.complex128)
    scale = np.exp(-l * (l + 1) * sigma)
    for m_idx, m in enumerate(range(-l, l + 1)):
        Ylm = compute_sph_harm(l, m, theta, phi)
        xyz += scale * ordres[l][m_idx, :] * Ylm.reshape(-1, 1)
        
    return xyz

def generate_surface(theta, phi, lmax, sigma, ordres, n_jobs=None):
    """Generates surface coordinates from spherical harmonics coefficients
    
    Args:
        theta (ndarray): Azimuthal angles
        phi (ndarray): Polar angles
        lmax (int): Maximum degree
        sigma (float): Smoothing parameter
        ordres (dict): Coefficients by degree
        n_jobs (int): Number of parallel jobs (non utilisé dans la version vectorisée)
        
    Returns:
        ndarray: Surface coordinates (N_points, 3)
    """
    return np.real(sum(generate_surface_partial(l, sigma, theta, phi, ordres) 
                      for l in range(lmax + 1)))