import numpy as np
from nilearn.datasets import fetch_surf_fsaverage
from nilearn import surface
from utils.mesh import surface_preprocessing as sp
from utils.mesh import surface_generation as sg
from utils.mesh.visualization import show_comparison

def compute_surface_coefficients(surface_mesh, template_preprocessed, hemisphere, lmax=35, lambda_reg=0, n_jobs=-1):
    """Compute spherical harmonics coefficients of a surface relative to template"""   
    surface_coords, surface_tris = surface_mesh
    
    preprocessed = sp.preprocess_surface(surface_coords, surface_tris, hemisphere)
    
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

def main_freesurfer():
    """Main function to run surface reconstruction for both hemispheres"""
    
    # 1. Création des templates pour les deux hémisphères
    print("Creating templates for both hemispheres...")
    fsaverage6 = fetch_surf_fsaverage(mesh='fsaverage6')
    
    # Gauche
    surf_lh = surface.load_surf_mesh(fsaverage6['pial_left'])
    surf_lh_mesh = (surf_lh[0], surf_lh[1])
    template_name_lh = "fsaverage_lh"
    print("\nGenerating left hemisphere template...")
    sp.compute_template_projection(surf_lh_mesh, template_name_lh, hemisphere="lh")
    
    # Droite
    surf_rh = surface.load_surf_mesh(fsaverage6['pial_right'])
    surf_rh_mesh = (surf_rh[0], surf_rh[1])
    template_name_rh = "fsaverage_rh"
    print("\nGenerating right hemisphere template...")
    sp.compute_template_projection(surf_rh_mesh, template_name_rh, hemisphere="rh")
    
    # 2. Chargement des templates générés
    print("\nLoading generated templates...")
    template_preprocessed_lh = load_template_data('fsaverage_lh_template.npz')
    template_preprocessed_rh = load_template_data('fsaverage_rh_template.npz')
    
    # 3. Chargement des surfaces à traiter
    print("\nLoading surfaces to process...")
    fsaverage7 = fetch_surf_fsaverage(mesh='fsaverage7')
    surf_lh = surface.load_surf_mesh(fsaverage7['pial_left'])
    surf_rh = surface.load_surf_mesh(fsaverage7['pial_right'])
    surface_mesh_lh = (surf_lh[0], surf_lh[1])
    surface_mesh_rh = (surf_rh[0], surf_rh[1])
    
    # 4. Traitement hémisphère gauche
    print("\nProcessing left hemisphere...")
    coeffs_data_lh = compute_surface_coefficients(
        surface_mesh_lh,
        template_preprocessed_lh,
        "lh",
        lmax=15,
        lambda_reg=1e-7
    )
    
    coords_recon_lh, error_lh, error_hausdorff_lh, template_tris_lh = reconstruct_surface_from_coefficients(
        coeffs_data_lh['coefficients'],
        template_preprocessed_lh,
        sigma=1e-5
    )
    
    # 5. Traitement hémisphère droit
    print("\nProcessing right hemisphere...")
    coeffs_data_rh = compute_surface_coefficients(
        surface_mesh_rh,
        template_preprocessed_rh,
        "rh",
        lmax=15,
        lambda_reg=1e-7
    )
    
    coords_recon_rh, error_rh, error_hausdorff_rh, template_tris_rh = reconstruct_surface_from_coefficients(
        coeffs_data_rh['coefficients'],
        template_preprocessed_rh,
        sigma=1e-5
    )
    
    # 6. Affichage des métriques
    print("\nResults for left hemisphere:")
    print(f"Hausdorff Error: {error_hausdorff_lh:.4f}")
    print(f"Mean Error: {np.mean(error_lh):.4f}")
    print(f"Max Error: {np.max(error_lh):.4f}")

    print("\nResults for right hemisphere:")
    print(f"Hausdorff Error: {error_hausdorff_rh:.4f}")
    print(f"Mean Error: {np.mean(error_rh):.4f}")
    print(f"Max Error: {np.max(error_rh):.4f}")
    
    # 7. Visualisation
    print("\nVisualizing results...")
    show_comparison(
        orig_coords=coeffs_data_lh['preprocessed']['orig_coords'],
        recon_coords=coords_recon_lh,
        triangles=template_tris_lh,
        error=error_lh,
    )

    show_comparison(
        orig_coords=coeffs_data_rh['preprocessed']['orig_coords'],
        recon_coords=coords_recon_rh,
        triangles=template_tris_rh,
        error=error_rh,
    )
    
    return {
        'left': {
            'coeffs_data': coeffs_data_lh,
            'reconstruction': coords_recon_lh,
            'error': error_lh,
            'hausdorff': error_hausdorff_lh
        },
        'right': {
            'coeffs_data': coeffs_data_rh,
            'reconstruction': coords_recon_rh,
            'error': error_rh,
            'hausdorff': error_hausdorff_rh
        }
    }



def main_matlab():
    from utils.file_manip.Matlab_to_array import load_faces, load_vertices
    import pyvista as pv
    
    def extract_hemisphere(mesh, hemisphere):
        """
        Extract left or right hemisphere from a brain mesh.
        Y positif = gauche
        Y négatif = droit
        """
        points = mesh.points
        
        if hemisphere == 'left':
            indices = points[:, 1] >= 0  # positive y for left
        else:
            indices = points[:, 1] < 0   # negative y for right
            
        points = points[indices]
        old_to_new = np.cumsum(indices) - 1
        faces = mesh.faces.reshape(-1, 4)
        valid_faces = []
        
        for face in faces[:, 1:]:
            if all(indices[face]):
                new_face = old_to_new[face]
                valid_faces.append([3] + list(new_face))
                
        valid_faces = np.array(valid_faces).flatten()
        return pv.PolyData(points, valid_faces)
    
    template_preprocessed = load_template_data('fsaverage_template.npz')

    faces = load_faces(r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\data\Faces_1.mat")
    vertices = load_vertices(r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\data\Vertices_1.mat")
    pv_mesh_full = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)))
    hemisphere_mesh=extract_hemisphere(pv_mesh_full)

    vertices = hemisphere_mesh.points
    faces = hemisphere_mesh.faces.reshape(-1, 4)[:, 1:]
    surface_mesh = (vertices, faces)

    # Compute reconstruction
    coeffs_data = compute_surface_coefficients(
        surface_mesh,
        template_preprocessed,
        lmax=15,
        lambda_reg=1e-7
    )
    
    coords_recon, error, error_hausdorff, template_tris = reconstruct_surface_from_coefficients(
        coeffs_data['coefficients'],
        template_preprocessed,
        sigma=1e-5
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
    main_freesurfer()