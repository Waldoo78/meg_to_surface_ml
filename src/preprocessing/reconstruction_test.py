import numpy as np
from nilearn.datasets import fetch_surf_fsaverage
from nilearn import surface
from utils.mesh import surface_preprocessing as sp
from utils.mesh import surface_generation as sg
from utils.mesh.visualization import show_comparison

def compute_surface_coefficients(surface_mesh, template_preprocessed, lmax=35, lambda_reg=0, n_jobs=-1):
    """Compute spherical harmonics coefficients of a surface relative to template"""   
    surface_coords, surface_tris = surface_mesh
    
    preprocessed = sp.preprocess_surface(surface_coords, surface_tris, hemisphere='lh')
    
    coeffs = sp.compute_coefficients({
        'theta': template_preprocessed['theta'],
        'phi': template_preprocessed['phi'],
        'center': preprocessed['center'],
        'orig_coords': preprocessed['orig_coords']
    }, lmax, lambda_reg, n_jobs)
    
    return {
        'coefficients': coeffs,
        'preprocessed': preprocessed
    }

def reconstruct_surface_from_coefficients(coefficients, template_preprocessed, sigma=0, n_jobs=-1):
    """Generate surface reconstruction from spherical harmonics coefficients"""
    reconstruction = sg.generate_surface_from_coefficients(
        template_preprocessed,
        coefficients, 
        sigma,
        n_jobs
    )
    
    return (reconstruction['reconstructed_coords'],
            reconstruction['error'],
            reconstruction['error_hausdorff'],
            template_preprocessed['sphere_tris'])

def load_template_data(template_path):
    """Load template data from NPZ file"""
    template_data = np.load(template_path)
    return {
        'theta': template_data['theta'],
        'phi': template_data['phi'],
        'sphere_coords': template_data['sphere_coords'],
        'sphere_tris': template_data['sphere_tris'],
        'orig_coords': template_data['orig_coords'],
        'orig_tris': template_data['orig_tris'],
        'center': template_data['center']  
    }

def main():
    """Main function to run surface reconstruction"""
    # Load data
    template_preprocessed = load_template_data('fsaverage_template.npz')
    
    fsaverage7 = fetch_surf_fsaverage(mesh='fsaverage7')
    surf = surface.load_surf_mesh(fsaverage7['pial_left'])
    surface_mesh = (surf[0], surf[1])
    
    # Compute reconstruction
    coeffs_data = compute_surface_coefficients(
        surface_mesh,
        template_preprocessed,
        lmax=3,
        lambda_reg=1e-7
    )
    
    coords_recon, error, error_hausdorff, template_tris = reconstruct_surface_from_coefficients(
        coeffs_data['coefficients'],
        template_preprocessed,
        sigma=0
    )
    
    # Print metrics
    print(f"Hausdorff Error: {error_hausdorff:.4f}")
    print(f"Mean Error: {np.mean(error):.4f}")
    print(f"Max Error: {np.max(error):.4f}")
    
    # Visualize results - sans window_size
    show_comparison(
        orig_coords=coeffs_data['preprocessed']['orig_coords'],
        recon_coords=coords_recon,
        triangles=template_tris,
        error=error
    )

if __name__ == "__main__":
    main()