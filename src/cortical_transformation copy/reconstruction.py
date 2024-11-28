import numpy as np
from nilearn.datasets import fetch_surf_fsaverage
from nilearn import surface
import pyvista as pv
import pickle
import os 
from utils.file_manip.Matlab_to_array import load_faces, load_vertices
from utils.mathutils import compute_surface_metrics, cart_to_sph
from utils.cortical import spherical_harmonics as SH 
from utils.cortical.visualization import show_comparison
from utils.cortical.visualization import convert_triangles_to_pyvista


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

#Parameters
sigma=0
lambda_reg=1e-9
lmax = 27


template_projection = pickle.load(open(r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\cortical_transformation\data\spherical_template.pkl", 'rb'))
pre_computed_folder=r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\cortical_transformation\data"

# #Save Y if not done 
# Y = SH.compute_Y(template_projection['theta'], template_projection['phi'], lmax=120)
# Y_file=os.path.join(pre_computed_folder,"Y_120.npz")
# np.savez(Y_file, Y=Y)

Y=np.load(os.path.join(pre_computed_folder, "Y_120.npz"))['Y']
Y_lh=Y[:,:(lmax+1)**2]

#Surface reconstructions
def main_freesurfer():
    print("\n=== Starting Surface Reconstruction ===")

    # Load surface
    fsaverage6 = fetch_surf_fsaverage(mesh='fsaverage6')
    orig_vertices, orig_triangles = surface.load_surf_mesh(fsaverage6['pial_left'])

    # Load resampled surface from NPZ
    # resampled_data = np.load('output/resampled_surface.npz')
    resampled_surface = (orig_vertices, orig_triangles)

    # 2. Compute spherical harmonics coefficients
    coeffs = SH.compute_coefficients(Y_lh, template_projection['sphere_coords'], 
                                    resampled_surface, lmax, lambda_reg)

    # 3. Reconstruct surface from coefficients
    reconstruction_coords = SH.generate_surface(Y_lh, lmax, sigma, orders=coeffs['organized_coeffs'])

    # 4. Compute quality metrics
    metrics = compute_surface_metrics(reconstruction_coords, resampled_surface[0], 
                                        template_projection['sphere_tris'])

    # 5. Visualize results
    show_comparison(resampled_surface[0], reconstruction_coords, 
                    template_projection['sphere_tris'], metrics)

    print("\n=== Reconstruction Metrics ===")
    print(f"Mean Error: {metrics['distance_stats']['mean_error']:.4f} mm")
    print(f"Max Error: {metrics['distance_stats']['max_error']:.4f} mm")
    print(f"Hausdorff Error: {metrics['distance_stats']['error_hausdorff']:.4f} mm")
    print(f"Mean Curvature Difference: {metrics['shape_stats']['mean_curvature_diff']:.4f}")
    print(f"Total Area Difference: {metrics['area_stats']['total_area_diff']:.4f} mm²")

    return {
        'resampled_surface': resampled_surface,
        'reconstruction': reconstruction_coords,
        'metrics': metrics,
        'coefficients': coeffs
    }

def main_matlab():
    left_faces_file =r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN\sub-CC712027\lh_faces.mat"
    left_vertices_file =r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN\sub-CC712027\lh_vertices.mat"

    # Load and center original surface
    left_faces = load_faces(left_faces_file)
    left_vertices = load_vertices(left_vertices_file)
    left_vertices = left_vertices - left_vertices.mean(axis=0)

    print("\n=== Starting Surface Reconstruction ===")

    # Load and center resampled surface
    resampled_surface = np.load(r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN\sub-CC712027\lh_resampled.npz")
    r_coords, r_tris = resampled_surface["coords"], resampled_surface["tris"]
    r_coords = r_coords-r_coords.mean(axis=0)
    resampled_surface = (r_coords, r_tris)

    # 2. Compute spherical harmonics coefficients
    coeffs = SH.compute_coefficients(Y_lh, template_projection['sphere_coords'], 
                                    resampled_surface, lmax, lambda_reg)

    # 3. Reconstruct surface (starting from l=1)
    reconstruction_coords = SH.generate_surface(Y_lh, lmax, sigma, orders=coeffs['organized_coeffs'])
    
    tris1 = convert_triangles_to_pyvista(left_faces)
    tris2 = convert_triangles_to_pyvista(r_tris)
    tris3 = tris2

    p = pv.Plotter(shape=(1,3))

    p.subplot(0,0)
    mesh1 = pv.PolyData(left_vertices, tris1)
    p.add_mesh(mesh1, show_edges=True, color="blue")
    p.add_text("Original (centered)", position='upper_edge')

    p.subplot(0,1)
    mesh2 = pv.PolyData(r_coords, tris2)
    p.add_mesh(mesh2, show_edges=True, color="green")
    p.add_text("Resampled (centered)", position='upper_edge')

    p.subplot(0,2)
    mesh3 = pv.PolyData(reconstruction_coords, tris3)
    p.add_mesh(mesh3, show_edges=True, color="red")
    p.add_text("Reconstructed", position='upper_edge')

    p.link_views()
    p.show()


    # 4. Compute quality metrics
    metrics =compute_surface_metrics(reconstruction_coords, resampled_surface[0], 
                                        template_projection['sphere_tris'])

    # 5. Visualize results
    show_comparison(resampled_surface[0], reconstruction_coords, 
                    template_projection['sphere_tris'], metrics)

    print("\n=== Reconstruction Metrics ===")
    print(f"Mean Error: {metrics['distance_stats']['mean_error']:.4f} mm")
    print(f"Max Error: {metrics['distance_stats']['max_error']:.4f} mm")
    print(f"Hausdorff Error: {metrics['distance_stats']['error_hausdorff']:.4f} mm")
    print(f"Mean Curvature Difference: {metrics['shape_stats']['mean_curvature_diff']:.4f}")
    print(f"Total Area Difference: {metrics['area_stats']['total_area_diff']:.4f} mm²")

    return {
        'resampled_surface': resampled_surface,
        'reconstruction': reconstruction_coords,
        'metrics': metrics,
        'coefficients': coeffs
    }

if __name__ == "__main__":
    results = main_matlab()