import numpy as np
from nilearn.datasets import fetch_surf_fsaverage
from nilearn import surface
import pyvista as pv
import pickle
from utils.file_manip.Matlab_to_array import load_faces, load_vertices
from utils.mathutils import compute_surface_metrics
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

#Paramaters
sigma=0
lambda_reg=1e-9
lmax = 42

template_projection = pickle.load(open(r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\data\spherical_template.pkl", 'rb'))
Y=np.load(r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\data\Y_120.npz")['Y']
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
    print(np.mean(reconstruction_coords, axis=0))

    print(left_vertices.shape, r_coords.shape, reconstruction_coords.shape)
    
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


# def main_matlab_test():
#     left_faces = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\data\lh_faces.mat"
#     left_vertices = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\data\lh_vertices.mat"

#     left_faces1 = load_faces(left_faces)
#     left_vertices1 = load_vertices(left_vertices)

#     left_faces_file = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN\sub-CC710679\lh_faces.mat"
#     left_vertices_file = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN\sub-CC710679\lh_vertices.mat"

#     left_faces = load_faces(left_faces_file)
#     left_vertices = load_vertices(left_vertices_file)

#     # Visualisation des surfaces originales

#     p1 = pv.Plotter(shape=(1,2))
    
#     # Surface de référence originale
#     p1.subplot(0,0)
#     tris1_orig = np.column_stack((np.full(len(left_faces1), 3), left_faces1))
#     mesh1_orig = pv.PolyData(left_vertices1, tris1_orig)
#     p1.add_mesh(mesh1_orig, color='lightgray', show_edges=False)

#     # Surface avec erreur originale
#     p1.subplot(0,1)
#     tris_orig = np.column_stack((np.full(len(left_faces), 3), left_faces))
#     mesh_orig = pv.PolyData(left_vertices, tris_orig)
#     p1.add_mesh(mesh_orig,
#                 show_edges=False,
#                 )
#     p1.link_views()
#     p1.show()

#     # Visualisation des surfaces après resampling
#     r1_coords, r1_tris = get_resampled_inner_surface((left_vertices1, left_faces1),'lh')
#     r_coords, r_tris = get_resampled_inner_surface((left_vertices, left_faces),'lh')

#     error = np.sqrt(np.sum((r_coords-r1_coords)**2, axis=1))

#     p2 = pv.Plotter(shape=(1,2))
    
#     # Surface de référence resample
#     p2.subplot(0,0)
#     tris1 = np.column_stack((np.full(len(r1_tris), 3), r1_tris))
#     mesh1 = pv.PolyData(r1_coords, tris1)
#     p2.add_mesh(mesh1, color='lightgray', show_edges=False)
    
#     # Surface avec erreur resample
#     p2.subplot(0,1)
#     tris = np.column_stack((np.full(len(r_tris), 3), r_tris))
#     mesh = pv.PolyData(r_coords, tris)
#     mesh.point_data['error'] = error
#     p2.add_mesh(mesh, 
#               scalars='error',
#               cmap='jet',  
#               show_edges=False,
#               scalar_bar_args={'title': 'Resampled Error (mm)'}
#     )
    
#     p2.link_views()  
#     p2.show()


# def main_matlab_test2():
#     resampled721532=np.load(r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN\sub-CC721532\lh_resampled.npz")
#     resampled720358=np.load(r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN\sub-CC720358\lh_resampled.npz")
#     coords_721532,tris_721532=resampled721532["coords"], resampled721532["tris"]
#     coords_720358,tris_720358=resampled720358["coords"], resampled720358["tris"]
#     print(tris_720358==tris_721532)
#     # mesh1=pv.PolyData(coords_721532, convert_to_poly(tris_721532))
#     # mesh2=pv.PolyData(coords_720358, convert_to_poly(tris_720358))
#     # p=pv.Plotter(shape=(1,2))

#     # p.subplot(0,0)
#     # p.add_mesh(mesh1, show_edges=True)

#     # p.subplot(0,1)
#     # p.add_mesh(mesh2,show_edges=True)

#     # p.link_views()
#     # p.show()
    

   
if __name__ == "__main__":
    results = main_matlab()