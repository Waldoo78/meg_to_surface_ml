import numpy as np
from nilearn.datasets import fetch_surf_fsaverage
from nilearn import surface
import pyvista as pv
from utils.file_manip.Matlab_to_array import load_faces, load_vertices
from utils.mesh import surface_preprocessing as sp
from utils.mesh import surface_generation as sg
from utils.mesh.Projection_onto_sphere import get_resampled_inner_surface
from utils.mesh.visualization import show_comparison
import utils.mesh.spherical_harmonics as SH

def load_template_data(template_path):
    """Load template data from NPZ file"""
    data = np.load(template_path)
    return {
        'theta': data['theta'],
        'phi': data['phi'],
        'sphere_coords': data['sphere_coords'],
        'sphere_tris': data['sphere_tris'],
        'coords': data['coords'],
        'tris': data['tris'],
        'center': data['center']
    }

# Template generation
fsaverage6 = fetch_surf_fsaverage(mesh='fsaverage6')

# Compute Y matrices for both hemispheres once
lmax = 30
# Generate templates if needed
# for hemi, surf_file in [('lh', fsaverage6['pial_left']), ('rh', fsaverage6['pial_right'])]:
#     print(f"\nGenerating {hemi} template...")
#     surf = surface.load_surf_mesh(surf_file)
#     sp.compute_template_projection(surf[0], surf[1], "src/preprocessing/fsaverage", hemi)

# Load template projections
template_projection_lh = load_template_data('src/preprocessing/fsaverage_lh_template.npz')
template_projection_rh = load_template_data('src/preprocessing/fsaverage_rh_template.npz')


# Compute Y matrices
Y_lh = SH.compute_Y(
    template_projection_lh['theta'],
    template_projection_lh['phi'],
    lmax,
)

Y_rh = SH.compute_Y(
    template_projection_rh['theta'],
    template_projection_rh['phi'],
    lmax,
)


def compute_surface_coefficients(surface_mesh, hemisphere, lmax, lambda_reg=0):
    """Compute spherical harmonics coefficients of a surface relative to template"""
    surface_coords, surface_tris = surface_mesh
    
    # Select appropriate template and Y matrix
    if hemisphere == "lh":
        Y = Y_lh
        template_projection = template_projection_lh
    else:
        Y = Y_rh
        template_projection = template_projection_rh

    # Get resampled surface
    resampled_surface = get_resampled_inner_surface((surface_coords, surface_tris), hemisphere)
    
    # Compute coefficients using template sphere coordinates
    coeffs = sp.compute_coefficients(
        Y,
        template_projection['sphere_coords'],  # Utiliser les coordonnées de la sphère template
        resampled_surface,
        lmax, 
        lambda_reg,
    )
    
    return {
        'organized_coeffs': coeffs['organized_coeffs'],
        'lmax': lmax,
        'resampled_surface': resampled_surface,
        'template_center': np.mean(template_projection['sphere_coords'], axis=0)  
    }

def reconstruct_surface_from_coefficients(coefficients_data, hemisphere, sigma=0):
    """Generate surface reconstruction from spherical harmonics coefficients"""
    # Select appropriate template
    template_data = template_projection_lh if hemisphere == 'lh' else template_projection_rh
    Y=Y_lh if hemisphere=="lh" else Y_rh

    reconstruction_coords=SH.generate_surface(Y, coefficients_data['lmax'], sigma, coefficients_data['organized_coeffs'],)

    # Recentrer la reconstruction
    if 'template_center' in coefficients_data:
        reconstruction_coords += coefficients_data['template_center']
    
    # Get reference coordinates and template triangulation
    target_coords, _ = coefficients_data['resampled_surface']
    template_tris = template_data['sphere_tris']
    
    # Compute metrics
    metrics =sg.compute_surface_metrics(
        reconstruction_coords,
        target_coords,
        template_tris
    )
    
    return {
        'reconstructed_coords': reconstruction_coords,
        'triangles': template_tris,
        'metrics': metrics
    }
def process_hemisphere(surface_mesh, hemisphere, lmax, lambda_reg=1e-7, sigma=1e-5):
    """Process single hemisphere"""
    print(f"\nProcessing {hemisphere} hemisphere...")
    
    # Compute coefficients
    coeffs_data = compute_surface_coefficients(
        surface_mesh,
        hemisphere,
        lmax=lmax,
        lambda_reg=lambda_reg
    )
    
    # Reconstruct surface
    reconstruction = reconstruct_surface_from_coefficients(
        coeffs_data,
        hemisphere,
        sigma=sigma
    )

    # Visualization
    target_coords, target_tris = coeffs_data['resampled_surface']
    show_comparison(
        target_coords,
        reconstruction['reconstructed_coords'],
        target_tris,
        reconstruction['metrics'],  # Pass the metrics correctly
        f"{hemisphere} Hemisphere Reconstruction"
    )
    
    return {
        'coeffs_data': coeffs_data,
        'reconstruction': reconstruction['reconstructed_coords'],
        'metrics': reconstruction['metrics']
    }


def main_freesurfer():
    """Process FreeSurfer surface with comprehensive metrics and visualization"""
    print("\n=== Starting Surface Reconstruction ===")
    
    # Load surface
    fsaverage7 = fetch_surf_fsaverage(mesh='fsaverage7')
    surf = surface.load_surf_mesh(fsaverage7['pial_left'])
    print(f"Original surface: {surf[0].shape} vertices")
    

    # Process left hemisphere with increased parameters
    result = process_hemisphere(
        (surf[0], surf[1]),
        'lh',
        lmax=30,
        lambda_reg=1e-6,
        sigma=1e-4
    )
    
    # Print metrics
    metrics = result['metrics']
    print("\n=== Reconstruction Metrics ===")
    print(f"Mean Error: {metrics['distance_stats']['mean_error']:.4f} mm")
    print(f"Max Error: {metrics['distance_stats']['max_error']:.4f} mm")
    print(f"Hausdorff Error: {metrics['distance_stats']['error_hausdorff']:.4f} mm")
    print(f"Mean Curvature Difference: {metrics['shape_stats']['mean_curvature_diff']:.4f}")
    print(f"Total Area Difference: {metrics['area_stats']['total_area_diff']:.4f} mm²")
    
    return result

if __name__ == "__main__":
    results = main_freesurfer()

